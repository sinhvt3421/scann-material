import os
import numpy as np
from sklearn.model_selection import train_test_split
from openbabel import pybel
from utils.dataset import atoms_symbol
from .voronoi_neighbor import compute_voronoi_neighbor


def pad_sequence(sequences, maxlen=None, dtype='int32', value=0, padding='post'):

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


def pad_nested_sequences(sequences, max_len_1, max_len_2, dtype='int32', value=0):
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
    pad_sq = [pad_sequence(
        sq, padding='post', maxlen=max_len_1, value=value,dtype=dtype) for sq in sequences]
    pad_sq = pad_sequence(pad_sq, padding='post',
                          maxlen=max_len_2, value=value,dtype=dtype)
    return pad_sq

class GaussianDistance():
    """
    Expand distance with Gaussian basis sit at centers and with width 0.5.
    """

    def __init__(self, centers: np.ndarray = np.linspace(0, 4, 20), width=0.5):
        """
        Args:
            centers: (np.array) centers for the Gaussian basis
            width: (float) width of Gaussian basis
        """
        self.centers = centers
        self.width = width

    def convert(self, d: np.ndarray):
        """
        expand distance vector d with given parameters
        Args:
            d: (1d array) distance array
        Returns
            (matrix) N*M matrix with N the length of d and M the length of centers
        """
        d = np.array(d)
        return np.exp(-((d[:, None] - self.centers[None, :]) ** 2) / self.width ** 2)


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
        N_train = int(len_data * (1-test_percent*2))
        N_test = int(len_data * test_percent)

    N_val = len_data - N_train - N_test

    data_perm = np.random.permutation(len_data)
    train, valid, test, extra = np.split(
        data_perm, [N_train, N_train+N_val, N_train + N_val + N_test])
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

    data_energy = []
    if use_ref:
        print('Using reference energy optimization')

    if use_ring:
        print('Using ring aromatic information')

    for i, d in enumerate(data_full):
        if use_ring:
                data_energy.append([d['Atomic'], d['Properties'][target_prop],
                                    d['Ring'], d['Aromatic']])
        else:
            if use_ref:
                data_energy.append([d['Atomic'],
                                    d['Properties'][target_prop]-d['Properties']['Ref_energy']])
            else:
                data_energy.append([d['Atomic'], d['Properties'][target_prop]])

    data_energy = np.array(data_energy, dtype='object')

    data_neighbor = np.load(dataset_neighbor, allow_pickle=True)
    data_neighbor = np.array(data_neighbor, dtype='object')

    return data_energy, data_neighbor


def process_xyz(file):
    mol = next(pybel.readfile("xyz", file))
    atoms = [x.OBAtom for x in mol.atoms]

    coordinates = np.array([x.coords for x in mol.atoms], dtype='float32')

    atomics = [x.GetAtomicNum() for x in atoms]
    atomic_symbols = [atoms_symbol[str(x)] for x in atomics]

    ring_info = [1 if at.IsInRing() else 0 for at in atoms]
    aromatic = [1 if at.IsAromatic() else 0 for at in atoms]

    nstruct = {'Atoms': atomic_symbols, 'Atomic': atomics,
                'Coords': coordinates, 'Ring': ring_info,
                'Aromatic': aromatic}

    return nstruct


def prepare_input(struct, d_t=4.0, w_t=0.2, centers=np.linspace(0, 4, 20), use_ring=True):

    system_atom = struct['Atoms']
    system_coord = np.array(struct['Coords'], dtype='float32')
    lattice = None

    neighbors = compute_voronoi_neighbor(system_atom, system_coord, lattice, d_t, w_t)

    GD = GaussianDistance(centers)

    local_neighbor = np.array([pad_sequence([[n[1] for n in lc] for lc in neighbors],value=1000)],'int32')
    mask_local = (local_neighbor != 1000)
    local_neighbor[local_neighbor == 1000] = 0

    local_weight = np.array([pad_sequence([[n[2] for n in lc] for lc in neighbors],dtype='float32')])
    local_distance =  np.array([pad_sequence([GD.convert([float(n[3]) for n in lc])
                           for lc in neighbors],dtype='float32')])

    atomics = np.array([struct['Atomic']],'int32')
    mask_atom = (atomics != 0)

    if use_ring:
        extra_info = [np.stack([struct['Ring'], struct['Aromatic']], -1)]
            

    inputs = {'atomic': atomics, 'atom_mask': np.expand_dims(mask_atom, -1),
            'neighbors': local_neighbor, 'neighbor_mask': mask_local,
            'neighbor_weight': np.expand_dims(local_weight, -1),
            'neighbor_distance': local_distance}

    if use_ring:
        inputs['ring_aromatic'] = np.array(extra_info,'int32')

    return inputs
