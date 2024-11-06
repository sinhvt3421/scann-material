import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers


class AddToken(keras.layers.Layer):
    def __init__(self, dim=128, **kwargs):
        super(AddToken, self).__init__()
        self.dim = dim
        self.token = tf.Variable(tf.ones((1, 1, self.dim)), dtype="float32", name="prop_token")

    def call(self, centers):
        bs = tf.shape(centers)[0]
        tokens = tf.repeat(self.token, (bs, 1, 1))
        return tf.concat([tokens, centers], 1)


class ResidualNorm(keras.layers.Layer):
    def __init__(self, dim=128, dropout_rate=0.1, **kwargs):
        super(ResidualNorm, self).__init__()
        self.dim = dim
        self.dropout = dropout_rate

        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dim, activation="swish", kernel_regularizer=regularizers.l2(1e-4)),
                tf.keras.layers.Dense(dim, kernel_regularizer=regularizers.l2(1e-4)),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )

        self.add = tf.keras.layers.Add()

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super(ResidualNorm, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "dropout": self.dropout,
            }
        )
        return config


class LocalAttention(keras.layers.Layer):
    """
    Implements a local attention block
    """

    def __init__(
        self,
        dim=128,
        num_head=8,
        v_proj=True,
        scale=0.5,
        activation="swish",
        kq_proj=True,
        dropout=False,
        g_update="dv",
        **kwargs
    ):
        """_summary_

        Args:
            dim (int, optional): Dimension of projection for query and key attention. Defaults to 128.
            num_head (int, optional): Number of head attention use. head dim will be dim // num_head. Defaults to 8.
            v_proj (bool, optional): A Boolen for whether using value project or not. Defaults to True.
            scale (float, optional): A scalar for normalization attention value (default to Transformer paper). Defaults to 0.5.
            g_update (str, optional): A str for whether using geometrical update type in SCANN+ or not. Defaults to dv - distance+voronoi.
            name (str, optional):  Defaults to 'LA_layer'.
        """
        super(LocalAttention, self).__init__(**kwargs)

        # Init hyperparameter
        self.dim = dim
        self.hdim = dim // num_head

        self.scale = scale
        self.num_head = num_head

        self.v_proj = v_proj
        self.kq_proj = kq_proj

        self.dropout = dropout

        # Linear projection before attention
        self.proj_q = tf.keras.layers.Dense(dim, name="query", kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k = tf.keras.layers.Dense(dim, name="key", kernel_regularizer=regularizers.l2(1e-4))

        if self.v_proj:
            self.proj_v = tf.keras.layers.Dense(dim, name="value", kernel_regularizer=regularizers.l2(1e-4))

        # Filter gaussian distance - Distance embedding
        self.g_update = g_update
        self.filter_geo = tf.keras.layers.Dense(
            dim,
            name="filter_geo",
            activation=activation,
            kernel_regularizer=regularizers.l2(1e-4),
        )

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        if self.g_update:
            self.layer_norm_g = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        if self.dropout:
            self.drop_out = tf.keras.layers.Dropout(0.05)

    def call(self, atom_query, atom_neighbor, neighbor_geometry, mask, neighbor_weight=None, neighbor_cos_angles=None):
        """
        Args:
            atom_query:     A tensor of size [batch_size, len_atom_centers, dim]. Center representation
                            for all local structure
            atom_neighbor:  A tensor of size [batch_size,len_atom_centers, num_neighbors, dim].
                            Representation for all neighbor of center atoms
            neighbor_geometry: A tensor of size [batch_size, len_atom_centers, num_neighbors, 1]
                            neighbor_geometry of neighbor around center atoms (distance or distance * weight embedding)
            mask:           A Boolen tensor for masking different number of neighbors for each center atoms
        """

        sh = tf.shape(atom_neighbor)
        bs = sh[0]
        qlen = sh[1]
        nlen = sh[2]

        # Get neighbor vector from indices, size [B, M, N, d] and  Multiply weight Voronoi with neighbor
        atom_neighbor = tf.gather_nd(atom_query, atom_neighbor)

        # Shape neighbor_weighted [B, M, N, embedding_dim ]
        atom_neighbor = tf.reshape(atom_neighbor, [bs, qlen, nlen, self.dim])

        if self.g_update == "dv":
            geometry_update = self.filter_geo(
                tf.concat(
                    [
                        tf.repeat(tf.expand_dims(atom_query, 2), nlen, 2),
                        neighbor_geometry,
                        atom_neighbor,
                    ],
                    -1,
                )
            )

            neighbor_geometry = self.layer_norm_g(geometry_update + neighbor_geometry)

        elif self.g_update == "dvc":
            geometry_update = self.filter_geo(
                tf.concat(
                    [
                        tf.repeat(tf.expand_dims(atom_query, 2), nlen, 2),
                        neighbor_geometry,
                        atom_neighbor,
                    ],
                    -1,
                )
            )

            geometry_update = tf.einsum("bmkd,bmnkd->bmnd", geometry_update, neighbor_cos_angles)
            neighbor_geometry = self.layer_norm_g(geometry_update + neighbor_geometry)

        else:
            neighbor_geometry = self.filter_geo(neighbor_geometry) * neighbor_weight

        atom_neighbor_geometry = atom_neighbor * neighbor_geometry

        # Query centers atoms shape [bs, len_atom_centers, dim]
        query = self.proj_q(atom_query)

        # Key neighbor atoms shape [bs, len_atom_centers, num_neighbors, dim]
        key = self.proj_k(atom_neighbor_geometry)

        if self.v_proj:
            value_k = self.proj_v(atom_neighbor)
            value_q = self.proj_v(atom_query)

        # shape query_t [bs, len_atom_centers, heads dim]
        query_t = tf.reshape(query, [bs, -1, self.num_head, self.hdim])

        # shape key [bs, len_atom_centers, num_neighbors, heads dim]
        key = tf.reshape(key, [bs, -1, nlen, self.num_head, self.hdim])

        if self.v_proj:
            value_k = tf.reshape(value_k, [bs, -1, nlen, self.num_head, self.hdim])

        # shape query_t [bs, len_atom_centers, heads, dim] * [bs, len_atom_centers, num_neighbors, heads, dim]
        # shape energy [bs, heads, len_atom_centers, num_neighbors]
        dk = tf.cast(tf.shape(key)[-1], tf.float32) ** (-self.scale)
        query_t = tf.multiply(query_t, dk)

        energy = tf.einsum("bchd,bcnhd->bhcn", query_t, key)

        # shape attn [bs, heads, len_atom_centers, num_neighbors] -> softmax over num_neighbors
        mask_scaled = (1.0 - tf.expand_dims(mask, 1)) * -1e9
        energy += mask_scaled

        attn = tf.nn.softmax(energy, -1)

        if self.dropout:
            attn = self.drop_out(attn)

        if self.v_proj:
            v = value_k
            q = value_q

        elif self.kq_proj:
            v = key
            q = query

        else:
            v = tf.reshape(atom_neighbor, [bs, -1, nlen, self.num_head, self.hdim])
            q = atom_query

        context = tf.einsum("bcn, bcnhd -> bcnhd", mask, tf.einsum("bhcn, bcnhd -> bcnhd", attn, v))

        context = tf.reshape(context, [bs, qlen, nlen, self.dim])

        # Taking sum over weighted neighbor representation and query representation for center representation

        context = tf.reduce_sum(context, 2) + q

        context = self.layer_norm(context)

        return attn, context, neighbor_geometry

    def get_config(self):
        config = super(LocalAttention, self).get_config()
        config.update(
            {
                "dim": self.dim,
                "scale": self.scale,
                "num_head": self.num_head,
                "v_proj": self.v_proj,
                "kq_proj": self.kq_proj,
                "g_update": self.g_update,
                "dropout": self.dropout,
            }
        )
        return config


class GlobalAttention(keras.layers.Layer):
    """
    Implements a global attention block
    """

    def __init__(self, dim=128, v_proj=False, kq_proj=True, norm=True, **kwargs):
        """

        Args:
            dim (int, optional): Dimension of projection for query and key attention. Defaults to 16.
            num_head (int, optional): Number of head attention use. Total dim will be dim * num_head. Defaults to 8.
            v_proj (bool, optional): A Boolen for whether using value project or not. Defaults to True.
            scale (float, optional): A scalar for normalization attention value. Defaults to 0.5.
            norm (bool, optional): A Boolen for whether using normalization for aggreation attention. Defaults to False.
            name (str, optional):  Defaults to 'GA_layer'.
        """
        super(GlobalAttention, self).__init__()

        # Setup
        self.dim = dim
        self.norm = norm

        self.v_proj = v_proj
        self.kq_proj = kq_proj

        # Linear proj. before attention
        self.proj_q = tf.keras.layers.Dense(dim, name="query", kernel_regularizer=regularizers.l2(1e-4))

        self.proj_k = tf.keras.layers.Dense(dim, name="key", kernel_regularizer=regularizers.l2(1e-4))

        if self.v_proj:
            self.proj_v = tf.keras.layers.Dense(dim, name="value", kernel_regularizer=regularizers.l2(1e-4))

    def call(self, atom_query, mask):
        # Query centers atoms shape [bs, len_atom_centers, dim]
        query = self.proj_q(atom_query)

        # Key centers atoms shape [bs, len_atom_centers, dim]
        key = self.proj_k(atom_query)

        if self.v_proj:
            value = self.proj_v(atom_query)

        # shape energy [bs, len_atom_centers, len_atom_centers]
        # Multiply key vs query and take the sum over query index (axis -1)
        energy = tf.einsum("bkd , bqd -> bkq", tf.multiply(mask, key), tf.multiply(mask, query))

        # Calculate attention from other atom except for the center
        identity = tf.eye(tf.shape(energy)[1], batch_shape=[tf.shape(energy)[0]], dtype="bool")
        mask_center = tf.cast(tf.math.logical_not(identity), tf.float32)

        energy = tf.multiply(mask_center, energy)

        # Taking the sum of attention from all local structures
        # shape transform_energy [bs, len_atom_centers, 1]
        agg_attention = tf.reduce_sum(energy, -1)
        agg_attention = tf.reshape(agg_attention, [tf.shape(atom_query)[0], -1, 1])

        agg_attention = tf.multiply(mask, agg_attention)

        # Normalize the score for better softmax behaviors
        if self.norm:
            # Normalize score
            agg_attention, _ = tf.linalg.normalize(agg_attention, ord="euclidean", axis=1, name=None)

        mask_scale = (1.0 - mask) * -1e9
        agg_attention += mask_scale

        attn = tf.nn.softmax(agg_attention, 1)

        if self.v_proj:
            v = value

        elif self.kq_proj:
            v = key

        else:
            v = atom_query

        # Multiply the attention score and local structure representation
        context = tf.multiply(mask, tf.multiply(attn, v))

        context = tf.reduce_sum(context, 1)

        return attn, context

    def get_config(self):
        config = super(GlobalAttention, self).get_config()

        config.update(
            {
                "dim": self.dim,
                "norm": self.norm,
                "v_proj": self.v_proj,
                "kq_proj": self.kq_proj,
            }
        )
        return config
