import gc
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import yaml
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, Lambda, Multiply
from tensorflow.keras.models import load_model

from scann.layers import *
from scann.layers import _CUSTOM_OBJECTS
from scann.utils.datagenerator import DataIterator
from scann.utils.general import load_dataset, split_data


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def r2_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.lr(self.model.optimizer.iterations))
        print("current_lr=", lr)


class SCANN:
    """
    Implements main SCANN
    """

    def __init__(self, config=None, pretrained="", mode="train"):
        """
        Args:
            config:
                    use_ring: Whether using extra ring information for molecule data
                    use_attn_norm: Whether using normalization layer for local attention as in Transformer

                    n_attention: Number of local attention layer
                    n_embedding: Embedding dims for atomic number of atoms

                    dense_embed: Dims for computing linear layer
                    dim: dim for computing atttention layer
                    num_head: total head for attention as in Transformer, set to 1 will not using head attention
                    scale: attention scale factor, default to 0.5.
        """
        self.config = config
        self.model = None
        self.mean, self.std = 0, 1
        self.trainIter, self.validIter, self.testIter = None, None, None

        if "target_mean" in self.config["hyper"]:
            self.mean = float(self.config["hyper"]["target_mean"])
            self.std = float(self.config["hyper"]["target_std"])

        self.model = create_model(self.config)
        if pretrained:
            self.model.load_weights(pretrained)

    @classmethod
    def load_model_infer(cls, path):
        model = load_model(path, custom_objects=_CUSTOM_OBJECTS)

        return model

    def prepare_dataset(self, split=True):
        data_energy, data_neighbor = load_dataset(
            use_ref=self.config["hyper"]["use_ref"],
            use_ring=self.config["model"]["use_ring"],
            dataset=self.config["hyper"]["data_energy_path"],
            dataset_neighbor=self.config["hyper"]["data_nei_path"],
            target_prop=self.config["hyper"]["target"],
        )

        if self.config["hyper"]["scaler"]:
            target = [d[1] for d in data_energy]
            self.mean, self.std = np.mean(target, dtype="float32"), np.std(target, dtype="float32")
            print("Normalize dataset property with mean: ", self.mean, " , std: ", self.std, "\n")
            data_energy[:, 1] = (data_energy[:, 1] - self.mean) / self.std

        self.config["hyper"]["target_mean"] = str(self.mean)
        self.config["hyper"]["target_std"] = str(self.std)

        self.config["hyper"]["data_size"] = len(data_energy)
        if split:
            train, valid, test, extra = split_data(
                len_data=len(data_energy),
                test_percent=self.config["hyper"]["test_percent"],
                train_size=self.config["hyper"]["train_size"],
                test_size=self.config["hyper"]["test_size"],
            )

            assert len(extra) == 0, "Split was inexact {} {} {} {}".format(
                len(train), len(valid), len(test), len(extra)
            )

            print(
                "Number of train data : ",
                len(train),
                " , Number of valid data: ",
                len(valid),
                " , Number of test data: ",
                len(test),
                "\n",
            )

            self.trainIter, self.validIter, self.testIter = [
                DataIterator(
                    batch_size=self.config["hyper"]["batch_size"],
                    data_neighbor=data_neighbor[indices],
                    data_energy=data_energy[indices],
                    use_ring=self.config["model"]["use_ring"],
                    shuffle=(len(indices) == len(train)),
                    feature=self.config["model"]["feature"],
                    g_update=self.config["model"]["g_update"],
                    converter=self.config["hyper"]["target_scale"],
                )
                for indices in (train, valid, test)
            ]
            return train, valid, test

        else:
            self.dataIter = DataIterator(
                batch_size=self.config["hyper"]["batch_size"],
                data_neighbor=data_neighbor,
                data_energy=data_energy,
                use_ring=self.config["model"]["use_ring"],
                feature=self.config["model"]["feature"],
                g_update=self.config["model"]["g_update"],
                converter=self.config["hyper"]["target_scale"],
            )

    def create_callbacks(self):
        callbacks = []
        callbacks.append(
            ModelCheckpoint(
                filepath="{}_{}/models/model_{}.h5".format(
                    self.config["hyper"]["save_path"],
                    self.config["hyper"]["target"],
                    self.config["hyper"]["target"],
                ),
                monitor="val_predict_property_mae",
                save_weights_only=False,
                verbose=2,
                save_best_only=True,
            )
        )

        callbacks.append(EarlyStopping(monitor="val_predict_property_mae", patience=200))

        if self.config["hyper"]["scheduler"] == "sgdr":
            lr = SGDRC(
                lr_min=self.config["hyper"]["min_lr"],
                lr_max=self.config["hyper"]["lr"],
                t0=50,
                tmult=2,
                lr_max_compression=1.2,
                trigger_val_mae=300,
            )
            sgdr = LearningRateScheduler(lr.lr_scheduler)

            callbacks.append(lr)
            callbacks.append(sgdr)
        else:
            callbacks.append(LearningRateLoggingCallback())

        return callbacks

    def train(self, epochs=1000):
        if self.config["hyper"]["scheduler"] == "sgdr":
            lr = self.config["hyper"]["lr"]
        else:
            lr = tf.keras.optimizers.schedules.CosineDecay(
                self.config["hyper"]["lr"],
                0.5 * len(self.trainIter) * epochs,
                alpha=self.config["hyper"]["min_lr"] / self.config["hyper"]["lr"],
                name=None,
            )

        self.model.compile(
            loss=[root_mean_squared_error, None, None],
            optimizer=tf.keras.optimizers.Adam(lr, decay=1e-5),
            metrics=[["mae", r2_square], None, None],
        )

        if not os.path.exists(
            "{}_{}/models/".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"])
        ):
            os.makedirs("{}_{}/models/".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"]))

        callbacks = self.create_callbacks()

        yaml.safe_dump(
            self.config,
            open(
                "{}_{}/config.yaml".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"]),
                "w",
            ),
            default_flow_style=False,
        )

        self.hist = self.model.fit(
            self.trainIter,
            epochs=epochs,
            validation_data=self.validIter,
            callbacks=callbacks,
            verbose=2,
            shuffle=False,
            use_multiprocessing=True,
            workers=4,
        )

        tf.keras.backend.clear_session()
        del self.model
        gc.collect()

    def evaluate(self):
        # Predict for testdata
        if not hasattr(self, "model"):
            print("Load best validation weight for predicting testset", "\n")
            self.model = load_model(
                "{}_{}/models/model_{}.h5".format(
                    self.config["hyper"]["save_path"],
                    self.config["hyper"]["target"],
                    self.config["hyper"]["target"],
                ),
                custom_objects=_CUSTOM_OBJECTS,
            )

        data = self.dataIter if hasattr(self, "dataIter") else self.testIter

        y_predict = []
        y = []
        for i in range(len(data)):
            inputs, target = data.__getitem__(i)
            output, _, _ = self.model.predict(inputs)

            y.extend(list(target))
            y_predict.extend(list(np.squeeze(output)))
            if i % 10 == 0:
                print(f"{i}/{len(data)}")

        print(
            "Result for testset ",
            self.config["hyper"]["target"],
            " : R2 score: ",
            r2_score(y, y_predict),
            " and MAE: ",
            mean_absolute_error(y, y_predict) * self.std,
        )
        with open(
            "{}_{}/report.txt".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"]),
            "w",
        ) as f:
            f.write(
                "Test MAE: "
                + str(mean_absolute_error(y, y_predict) * self.std)
                + ", Test R2: "
                + str(r2_score(y, y_predict))
            )

        if hasattr(self, "hist"):
            save_data = [y_predict, y, self.hist.history]

            np.save(
                "{}_{}/hist_data.npy".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"]),
                np.array(save_data, dtype=object),
            )

            with open(
                "{}_{}/report.txt".format(self.config["hyper"]["save_path"], self.config["hyper"]["target"]),
                "w",
            ) as f:
                f.write("Training MAE: " + str(min(self.hist.history["predict_property_mae"]) * self.std) + "\n")
                f.write("Val MAE: " + str(min(self.hist.history["val_predict_property_mae"]) * self.std) + "\n")
                f.write(
                    "Test MAE: "
                    + str(mean_absolute_error(y, y_predict) * self.std)
                    + ", Test R2: "
                    + str(r2_score(y, y_predict))
                )

            print("Saved model record for dataset")

    def predict_ga(self, ip):
        prop, ga, rep = self.model.predict(ip)
        return prop / self.config["hyper"]["target_scale"] * self.std + self.mean, ga

    def predict_rep(self, ip):
        prop, ga, rep = self.model.predict(ip)
        return prop / self.config["hyper"]["target_scale"] * self.std + self.mean, ga, rep


def create_model_pretrained(pretrained):
    model = load_model(pretrained, custom_objects=_CUSTOM_OBJECTS)
    model.summary()

    return model


def create_model(config):
    cfm = config["model"]
    if cfm["feature"] == "atomic":
        shape = (None,)
    if cfm["feature"] == "cgcnn":
        shape = (None, 92)

    # Atom Inputs
    atomic = Input(name="atomic", shape=shape, dtype="int32")
    atom_mask = Input(shape=[None, 1], name="atom_mask")

    # Neighbor Inputs
    neighbor = Input(name="neighbors", shape=(None, None), dtype="int32")
    neighbor_mask = Input(name="neighbor_mask", shape=(None, None), dtype="float32")

    neighbor_weight = Input(name="neighbor_weight", shape=(None, None), dtype="float32")
    neighbor_distance = Input(name="neighbor_distance", shape=(None, None), dtype="float32")

    inputs = [
        atomic,
        atom_mask,
        neighbor,
        neighbor_mask,
        neighbor_weight,
        neighbor_distance,
    ]
    if cfm["use_ring"]:
        ring_info = Input(name="ring_aromatic", shape=(None, 2), dtype="float32")
        inputs.append(ring_info)

    # Embedding atom and extra information as ring, aromatic
    if cfm["feature"] == "atomic":
        centers = Embedding(cfm["n_atoms"], cfm["embedding_dim"], name="embed_atom", dtype="float32")(atomic)

    if cfm["feature"] == "cgcnn":
        centers = Dense(cfm["embedding_dim"], name="embed_atom", dtype="float32")(atomic)

    if cfm["use_ring"]:
        ring_embed = Dense(10, name="extra_embed", dtype="float32")(ring_info)

        # Shape embed_atom [B, M, n_embedding + 10]
        centers = tf.concat([centers, ring_embed], -1)

    centers = Dense(cfm["local_dim"], activation="swish", name="dense_embed", dtype="float32")(centers)
    centers = Dropout(0.1)(centers)

    neighbor_indices = Lambda(gather_shape, name="get_neighbor")(neighbor)

    neighbor_distance = GaussianExpansion(np.linspace(0, cfm["gaussian_d"], 20, dtype="float32"))(neighbor_distance)

    if cfm["g_update"] == "dv":
        neighbor_distance = Dense(cfm["local_dim"], activation="swish", name="neighbor_d", dtype="float32")(
            neighbor_distance
        )
        neighbor_weight = GaussianExpansion(np.linspace(0, np.pi * 2, 20, dtype="float32"))(neighbor_weight)

        neighbor_weight = Dense(cfm["local_dim"], activation="swish", name="neighbor_w", dtype="float32")(
            neighbor_weight
        )
        geometry_features = Multiply(name="geometry_features")([neighbor_distance, neighbor_weight])

    elif cfm["g_update"] == "dvc":
        neighbor_cos_angles = Input(name="neighbor_cos_angles", shape=(None, None, None), dtype="float32")
        inputs.append(neighbor_cos_angles)

        neighbor_cos_angles = GaussianExpansion(np.linspace(-1, 1, 20, dtype="float32"))(neighbor_cos_angles)

        neighbor_cos_angles = Dense(
            cfm["local_dim"],
            activation="swish",
            name="neighbor_a",
            dtype="float32",
        )(neighbor_cos_angles)

        neighbor_distance = Dense(cfm["local_dim"], activation="swish", name="neighbor_d", dtype="float32")(
            neighbor_distance
        )
        neighbor_weight = GaussianExpansion(np.linspace(0, np.pi * 2, 20, dtype="float32"))(neighbor_weight)

        neighbor_weight = Dense(cfm["local_dim"], activation="swish", name="neighbor_w", dtype="float32")(
            neighbor_weight
        )

        geometry_features = Multiply(name="geometry_features")([neighbor_distance, neighbor_weight])
    else:

        neighbor_weight = tf.expand_dims(neighbor_weight, -1)

    def local_attention_block(c, n_in, n_g, n_m, n_w=None, n_c=None):
        # Local attention for local structure representation
        attn_local, context, g_f = LocalAttention(
            v_proj=False,
            kq_proj=True,
            dim=cfm["local_dim"],
            num_head=cfm["num_head"],
            activation="swish",
            dropout=cfm["use_drop"],
            g_update=cfm["g_update"],
        )(c, n_in, n_g, n_m, n_w, n_c)
        if cfm["use_attn_norm"]:
            # 2 Forward Norm layers
            centers = ResidualNorm(cfm["local_dim"])(context)
        else:
            centers = context

        return centers, attn_local, g_f

    # Local Attention recursive layers
    for i in range(cfm["n_attention"]):

        if cfm["g_update"] == "dvc":
            arg = (centers, neighbor_indices, geometry_features, neighbor_mask, None, neighbor_cos_angles)
        elif cfm["g_update"] == "dv":
            arg = (centers, neighbor_indices, geometry_features, neighbor_mask)
        else:
            arg = (centers, neighbor_indices, neighbor_distance, neighbor_mask, neighbor_weight)

        centers, attn_local, geometry_features = local_attention_block(*arg)

    # Dense layer after Local Attention -> representation for each local structure [B, M, d]
    centers = Dense(
        cfm["global_dim"],
        activation="swish",
        name="after_Lc",
        kernel_regularizer=regularizers.l2(1e-4),
    )(centers)

    # Using weighted attention score for combining structures representation
    attn_global, struc_rep = GlobalAttention(
        v_proj=False, kq_proj=True, dim=cfm["global_dim"], norm=cfm["use_ga_norm"]
    )(centers, atom_mask)

    # Shape struct representation [B, d]
    struc_rep = Dense(
        cfm["dense_out"],
        activation="swish",
        name="bf_property",
        kernel_regularizer=regularizers.l2(1e-4),
    )(struc_rep)

    # Shape predict_property [B, 1]
    # predict_property = Dense(
    #     1, name="predict_property", activation=mrelu if config["hyper"]["target"] == "e_b" else None
    # )(struc_rep)
    predict_property = Dense(1, name="predict_property", activation=None)(struc_rep)
    model = tf.keras.Model(inputs=inputs, outputs=[predict_property, attn_global, struc_rep])

    model.summary()

    return model
