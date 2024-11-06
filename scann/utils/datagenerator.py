from math import ceil

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from .general import pad_nested_sequences, pad_sequence, pad_distance_vectors
from scann.utils.dataset import atomic_features


class DataIterator(Sequence):
    """
    Create Data interator over dataset
    """

    def __init__(
        self,
        data_energy,
        data_neighbor,
        batch_size=32,
        converter=1.0,
        use_ring=False,
        shuffle=False,
        feature="atomic",
        g_update="dv",
    ):
        """
        Args:
            data_energy (list): _description_
            data_neighbor (list): _description_
            indices (dict): Index for train, valid, test
            batch_size (int, optional): Defaults to 32.
            converter (bool, optional): Scale target value. Defaults to 1.0.
            use_ring (bool, optional): Use ring aromatic information. Defaults to False.
            centers (list, optional): Gaussian expand centers. Defaults to np.linspace(0, 4, 80).
            feature (str, optional): Feature for atoms representation. Atomic numbers or CGCNN features
        """

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.data_neighbor = data_neighbor
        self.data_energy = data_energy

        self.use_ring = use_ring

        self.weight_index = 3  # Using angle normalized weights information
        self.distance_index = -2
        self.c_index = -1

        if g_update == "dv":
            self.weight_index = 2  # Using angle information

        self.g_update = g_update

        self.feature = feature

        if converter > 1.0:
            self.converter = 1000  # meV
        else:
            self.converter = 1.0  # eV

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data_energy))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return ceil(len(self.data_energy) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_nei = self.data_neighbor[indexes]
        batch_atom = self.data_energy[indexes]

        max_length_center = max(len(c) for c in batch_nei)
        max_length_neighbor = max(len(n) for c in batch_nei for n in c)

        energy = np.array([float(p[1]) * self.converter for p in batch_atom], "float32")

        # Padding neighbor for each atoms
        local_neighbor = [[[n[1] for n in lc] for lc in p] for p in batch_nei]
        pad_local = pad_nested_sequences(
            local_neighbor,
            max_length_neighbor,
            max_length_center,
            value=1000,
            dtype="int32",
        )
        mask_local = pad_local != 1000
        pad_local[pad_local == 1000] = 0

        # Padding local weight and distance
        local_weight = [[[n[self.weight_index] for n in lc] for lc in p] for p in batch_nei]

        local_distance = [[[n[self.distance_index] for n in lc] for lc in p] for p in batch_nei]

        pad_local_weight = pad_nested_sequences(local_weight, max_length_neighbor, max_length_center, dtype="float32")

        pad_local_distance = pad_nested_sequences(
            local_distance, max_length_neighbor, max_length_center, dtype="float32"
        )

        # Padding atomic numbers of atom
        atomics = [center[0] for center in batch_atom]
        pad_atom = pad_sequence(atomics, padding="post", maxlen=max_length_center, value=0, dtype="int32")

        mask_atom = pad_atom != 0

        if self.feature == "cgcnn":
            pad_atom = np.array([[atomic_features[str(x)] for x in ats] for ats in pad_atom], "float32")

        if self.use_ring:
            extra_info = [center[2] for center in batch_atom]

            pad_extra = pad_sequence(
                extra_info,
                padding="post",
                maxlen=max_length_center,
                value=0,
                dtype="int32",
            )
        if self.g_update == "dvc":
            # Get angle-based information
            local_dvector = [[[n[self.c_index] for n in lc] for lc in p] for p in batch_nei]
            pad_local_dvector = pad_distance_vectors(
                local_dvector,
                max_length_neighbor,
                max_length_center,
                value=0,
                dtype="float32",
            )
            neighbor_cos_angles = np.einsum(
                "bmij,bmkj-> bmik",
                pad_local_dvector,
                pad_local_dvector,
            )
        inputs = {
            "atomic": pad_atom,
            "atom_mask": np.expand_dims(mask_atom, -1),
            "neighbors": pad_local,
            "neighbor_mask": mask_local,
            "neighbor_weight": pad_local_weight,
            "neighbor_distance": pad_local_distance,
        }
        if self.g_update == "dvc":
            inputs["neighbor_cos_angles"] = neighbor_cos_angles

        if self.use_ring:
            inputs["ring_aromatic"] = pad_extra

        return (inputs, energy)
