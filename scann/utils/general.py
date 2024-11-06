import os

import numpy as np

from openbabel import pybel
from pymatgen.core import Molecule, Structure
from sklearn.model_selection import train_test_split

from scann.utils.dataset import atomic_numbers, atoms_symbol

from .voronoi_neighbor import compute_voronoi_neighbor


def pad_distance_vectors(vectors, max_len_1, max_len_2, dtype="float32", value=0):
    pad_sq = []
    for v in vectors:
        if len(v) == 0:
            x = np.full((1, max_len_1, 3), value, dtype=dtype)
            pad_sq.append(x)
            continue
        x = np.full((len(v), max_len_1, 3), value, dtype=dtype)
        for i, s in enumerate(v):
            trunc = s[-max_len_1:]
            if len(trunc) > 0:
                trunc = np.asarray(trunc, dtype=dtype)
                x[i, : len(trunc)] = trunc
        pad_sq.append(x)
    x = np.full((len(pad_sq), max_len_2, max_len_1, 3), value, dtype=dtype)
    for i, s in enumerate(pad_sq):
        trunc = s[-max_len_2:]
        trunc = np.asarray(trunc, dtype=dtype)
        x[i, : len(trunc) :] = trunc
    return x


def pad_sequence(sequences, maxlen=None, dtype="int32", value=0, padding="post"):
    num_samples = len(sequences)
    sample_shape = ()

    if maxlen is None:
        lengths = []
        for x in sequences:
            lengths.append(len(x))
        maxlen = np.max(lengths)

    sample_shape = np.asarray(sequences[0]).shape[1:]

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)

    for idx, s in enumerate(sequences):
        trunc = s[-maxlen:]
        trunc = np.asarray(trunc, dtype=dtype)
        x[idx, : len(trunc)] = trunc
    return x


def pad_nested_sequences(sequences, max_len_1, max_len_2, dtype="int32", value=0):
    """
        Pad 3D array
    Args:
        sequences (list): 3D array axis=(0,1,2)
        max_len_1 (int): Max length inside axis = 2
        max_len_2 (int): Max length outside axis = 1
        dtype (str, optional):  Defaults to 'int32'.
        value (int, optional): Padding value. Defaults to 0.

    Returns:
        np.ndarray: Padded sequences
    """
    pad_sq = [pad_sequence(sq, padding="post", maxlen=max_len_1, value=value, dtype=dtype) for sq in sequences]
    pad_sq = pad_sequence(pad_sq, padding="post", maxlen=max_len_2, value=value, dtype=dtype)
    return pad_sq


class GaussianDistance:
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """

    def __init__(self, centers: np.ndarray = np.linspace(0, 4, 20)):
        """
        Args:
            centers: (np.array) centers for the Gaussian basis
            width: (float) width of Gaussian basis
        """
        self.centers = centers
        self.width = np.diff(self.centers).mean()

    def convert(self, d: np.ndarray):
        """
        expand distance vector d with given parameters
        Args:
            d: (1d array) distance array
        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        return np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width**2)


def split_data(len_data, test_percent=0.1, train_size=None, test_size=None):
    """
        Split data training
    Args:
        len_data (int): Length of dataset.
        test_percent (float, optional): Percentage of data for testing. Defaults to 0.1.
        train_size (int, optional): If provided, fix the number of train and test size
        test_size (int, optional): If provided, fix the number of train and test size
    Returns:
        _type_: _description_
    """
    if train_size:
        N_train = train_size
        N_test = test_size
    else:
        N_train = int(len_data * (1 - test_percent * 2))
        N_test = int(len_data * test_percent)

    N_val = len_data - N_train - N_test

    data_perm = np.random.permutation(len_data)
    train, valid, test, extra = np.split(data_perm, [N_train, N_train + N_val, N_train + N_val + N_test])
    return train, valid, test, extra


def load_dataset(dataset, dataset_neighbor, target_prop, use_ref=False, use_ring=True):
    """
        Load dataset and neighbor information
    Args:
        dataset (str): Path to dataset
        dataset_neighbor (str): Path to dataset neighbor
        target_prop (str): Target property for training
        use_ref (bool, optional): Use reference energy. Defaults to False.
        use_ring (bool, optional): Use ring aromatic information. Defaults to True.

    Returns:
        data_energy (list)
        data_neighbor (list)
    """

    data_full = np.load(dataset, allow_pickle=True)

    if use_ref:
        print("Using reference energy optimization", "\n")

    if use_ring:
        print("Using ring aromatic information", "\n")

    data_energy = [
        (
            [d["Atomic"], float(d["Properties"][target_prop]), np.stack([d["Features"][x] for x in d["Features"]], -1)]
            if use_ring
            else (
                [
                    d["Atomic"],
                    float(d["Properties"][target_prop]) - float(d["Properties"]["Ref_energy"]),
                ]
                if use_ref
                else [d["Atomic"], float(d["Properties"][target_prop])]
            )
        )
        for d in data_full
    ]

    data_energy = np.array(data_energy, dtype="object")

    data_neighbor = np.load(dataset_neighbor, allow_pickle=True)
    data_neighbor = np.array(data_neighbor, dtype="object")

    return data_energy, data_neighbor


def process_xyz_pmt(file):
    """
        The xyz file may contain Lattice="..." information for crystalline structures
    Args:
        file (str): path to xyz file

    Returns:
        Molecule/Structure: structure from pymatgen
    """
    with open(file) as f:
        lines = f.readlines()
        if len(lines[1].split()) < 9:
            lattice = None
        else:
            lattice = [[float(s) for s in lines[1].split('"')[1].split()[i : i + 3]] for i in range(0, 9, 3)]
        atoms = []
        coords = []
        for line in lines[2:]:
            data = line.split()
            atom = data[0]
            x, y, z = map(float, data[1:])
            atoms.append(atom)
            coords.append([x, y, z])

        struct = {"Atoms": atoms, "Coords": coords}
        if lattice:
            struct["Latiice"] = lattice

    return struct


def load_file(file, mol=False):
    """

    Args:
        file (str): path to file. All formats supported by Pymatgen (cif/POSCAR/xyz/mol)
        mol (bool, optional): Whether the file contains molecule structure. Defaults to False.

    Returns:
        Structure (Pymatgen): Structure class from Pymatgen
    """
    try:
        if mol:
            struct = Molecule.from_file(file)
            coord = struct.cart_coords
            a = max(10, max(coord[:, 0]) - min(coord[:, 0]) + 0.1)
            b = max(10, max(coord[:, 1]) - min(coord[:, 1]) + 0.1)
            c = max(10, max(coord[:, 2]) - min(coord[:, 2]) + 0.1)

            struct = struct.get_boxed_structure(a, b, c, reorder=False)
        else:
            struct = Structure.from_file(file)

        return struct
    except:
        print("Can not read file using Pymatgen. Please check the file format")
        return None


def prepare_input_pmt(struct, d_t=4.0, w_t=0.4, angle=True):
    """

    Args:
        struct (pmt.Molecule/Structure): structure from pymatgen
        d_t (float, optional): Cut off threshold for distance. Defaults to 4.0 A.
        w_t (float, optional): Cut off threshold for Voronoi weight. Defaults to 0.2.

    Returns:
        dict: input dictionary for model input
    """

    neighbors = compute_voronoi_neighbor(struct, d_thresh=d_t, w_thresh=w_t)

    local_neighbor = np.array([pad_sequence([[n[1] for n in lc] for lc in neighbors], value=1000)], "int32")
    mask_local = local_neighbor != 1000
    local_neighbor[local_neighbor == 1000] = 0

    local_weight = np.array([pad_sequence([[n[2 if angle else 3] for n in lc] for lc in neighbors], dtype="float32")])
    local_distance = np.array(
        [
            pad_sequence(
                [[n[-1] for n in lc] for lc in neighbors],
                dtype="float32",
            )
        ]
    )

    atomics = np.array([struct.atomic_numbers], "int32")
    mask_atom = atomics != 0

    inputs = {
        "atomic": atomics,
        "atom_mask": np.expand_dims(mask_atom, -1),
        "neighbors": local_neighbor,
        "neighbor_mask": mask_local,
        "neighbor_weight": local_weight,
        "neighbor_distance": local_distance,
    }

    return inputs
