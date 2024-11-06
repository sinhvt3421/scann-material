import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pymatgen as pmt
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.core import Molecule, Structure


def compute_voronoi_neighbor(struct, cutoff=7, d_thresh=4.0, w_thresh=0.4, max_cutoff=30):
    """
        Compute Voronoi neighbors and weight from solid angle using VoronoiNN package
    Args:
        struct (pmt.Structure/Molecule): pymatgen structure or molecule
        cutoff (float, optional):   Cut off threshold for VoronoiNN. Defaults to 13.0 A.
        d_thresh (float, optional): Cut off threshold for distance. Defaults to 4.0 A.
        w_thresh (float, optional): Cut off threshold for Voronoi angle. Defaults to pi/8.

    Returns:
        list:  Neighbors information for each Local structure:
                Ex: [['H', 1, 0.4, 3.5],...]
    """

    # Initialize VoronoiNN
    voronoi = VoronoiNN(weight="solid_angle", cutoff=cutoff, allow_pathological=True, compute_adj_neighbors=False)
    error_occurred = False

    # Filter atoms neighbors with high Voronoi weights: solid_angle/max(solid_angle)
    local_xyz = []

    for i in range(len(struct)):
        while not error_occurred:
            try:
                nns = voronoi.get_voronoi_polyhedra(struct, i).values()
                max_weight = max(nn["solid_angle"] for nn in nns)

                local_xyz.append(
                    [
                        [
                            nn["site"].species_string,
                            nn["site"].index,
                            nn["solid_angle"],
                            nn["solid_angle"] / max_weight,  # Include the normalized weight
                            np.linalg.norm(struct[i].coords - nn["site"].coords),
                            # for updating graph with angle-based information
                            (struct[i].coords - nn["site"].coords)
                            / np.linalg.norm(struct[i].coords - nn["site"].coords),
                        ]
                        for nn in nns
                        if nn["solid_angle"] >= w_thresh
                        and nn["solid_angle"] / max_weight >= 0.2
                        and np.linalg.norm(struct[i].coords - nn["site"].coords) <= d_thresh
                    ]
                )
                break
            except:
                cutoff += 5.0  # Increase the cutoff distance
                print("Error Voronoi, increase cutoff to ", cutoff)
                if cutoff > max_cutoff:
                    print("Error Voronoi, max cutoff")
                    break
                voronoi = VoronoiNN(weight="solid_angle", cutoff=cutoff, allow_pathological=True)
    return local_xyz


# define a function to compute the voronoi neighbor using a single structure
def compute_voronoi_neighbor_wrapper(s, d_t, w_t, box=10):
    system_atom = s["Atoms"]
    system_coord = np.array(s["Coords"], dtype="float32")

    if "Lattice" in s:
        lattice = s["Lattice"]

        cart = s["Cartesian"] if "Cartesian" in s else True

        struct = Structure(
            lattice=lattice,
            coords=system_coord,
            species=system_atom,
            coords_are_cartesian=cart,
        )
        cutoff = 7
    else:
        struct = Molecule(system_atom, system_coord)
        a = max(box, max(system_coord[:, 0]) - min(system_coord[:, 0]) + 0.1)
        b = max(box, max(system_coord[:, 1]) - min(system_coord[:, 1]) + 0.1)
        c = max(box, max(system_coord[:, 2]) - min(system_coord[:, 2]) + 0.1)

        struct = struct.get_boxed_structure(a, b, c, reorder=False)
        cutoff = 7

    return compute_voronoi_neighbor(struct, cutoff, d_t, w_t)


def parallel_compute_neighbor(dataset_path, save_path, d_t=4.0, w_t=0.2, pool=8):
    """
        Parallel compute Voronoi neighbors
    Args:
        dataset (dict): dictionary of  a structure with format
                    {'Atoms':[....],'Coords':[....],'Lattice':[....],'Properties':[...]}
        savepath (str): Path to save neighbor information for dataset'
        d_t (float, optional): Cut off distance for compute neighbor. Defaults to 4.0.
        w_t (float, optional): Cut off Voronoi weight for compute neighbor. Defaults to 0.4 (pi/8).
        pool (int, optional): Parallel process for computing. Defaults to 8.
    """

    dataset = np.load(dataset_path, allow_pickle=True)

    all_data = []
    print(
        "Computing Voronoi neighbor for dataset ",
        dataset_path,
        ", parallel process: ",
        pool,
        ", saving to: ",
        save_path,
    )

    # Using multiprocess for faster computing, change the number process based on system capacity
    with ProcessPoolExecutor(pool) as executor:
        for i in range(0, len(dataset), pool):
            if i % (10 * pool) == 0:
                print(i)

            batch = dataset[i : i + pool]
            batch_futures = [executor.submit(compute_voronoi_neighbor_wrapper, s, d_t, w_t) for s in batch]
            batch_results = [future.result() for future in batch_futures]
            all_data.extend(batch_results)

    print("Saving data")
    np.save(save_path, np.asarray(all_data, dtype="object"))
    print("Finished computing Voronoi neighbor for dataset ", dataset_path)
