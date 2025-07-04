"""
This script generates KDE models for the chemical losses in PyCLOPS, based on the data.
This basically just fits KDEs to the data. It is here solely for record-keeping purposes.
"""
from typing import Dict
import os

import torch
import mdtraj as md

from pyclops.torchkde.modules import KernelDensity

mol_names = ['Carboxylic-Carbo', 'Cys-Arg', 'Cys-Carboxyl', 'Disulfide', 'Lys-Arg', 'Lys-Tyr', 'Amide']

mol_tetra_dict = {
        'Amide': ['N1', 'C1', 'O1', 'C2'],
        'Carboxylic-Carbo': ['N1', 'N2', 'C1', 'C11'],
        'Cys-Carboxyl': ['S1', 'C3', 'O1', 'C1'],
        'Disulfide': ['S1', 'S2', 'C1', 'C2'],
        'Lys-Arg': ['N1', 'N2', 'N3', 'N4'],
        'Lys-Tyr': ['N1', 'C3', 'O1', 'C4'],
    }

mol_tetra_vals: Dict[str, torch.Tensor] = {}

for mol_name in mol_names:
    stride: int = 1
    frac_to_exclude: float = 0.2 # initial fraction to exclude from the start of the trajectory

    mol_dir = os.path.join('/home/bfd21/rds/hpc-work/tbg/jobs/md-jobs', mol_name)
    results_dir = os.path.join(mol_dir, 'results/')

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory does not exist: {results_dir}")

    seeds = range(10)
    top_fname = os.path.join(results_dir, 'filtered_system.pdb')
    traj_fnames = [os.path.join(results_dir, f'traj_seed_{s}_no_water.dcd') for s in seeds]

    # Load and truncate each trajectory individually
    processed_trajs = []
    for fname in traj_fnames:
        t = md.load(fname, top=top_fname, stride=stride)
        n = t.n_frames
        start = int(frac_to_exclude * n)
        processed_trajs.append(t[start:])

    # Concatenate all truncated trajectories
    traj = processed_trajs[0].join(processed_trajs[1:])

    # Get atom indices for the tetrahedron
    atom_names = mol_tetra_dict[mol_name]
    atom_indices = [traj.topology.select(f"name {name}")[0] for name in atom_names]

    # Define the 6 unique edges of the tetrahedron
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 2), (1, 3),
        (2, 3)
    ]
    edges_idx = [(atom_indices[i], atom_indices[j]) for i, j in edges]

    # Compute distances for all 6 edges across frames
    distances = md.compute_distances(traj, edges_idx)  # shape: (n_frames, 6)
    distances *= 10  # Convert from nm to Angstroms

    print(f'mol_name: {mol_name}')
    print(f"Computed {distances.shape[1]} edge distances across {distances.shape[0]} frames.")
    print("Example (first frame):", distances[0])

    mol_tetra_vals[mol_name] = distances

for mol_name in mol_names:
    kde = KernelDensity(bandwidth=1.0, kernel='cauchy') # bandwidth is in angstroms
    X = torch.from_numpy(mol_tetra_vals[mol_name]).to('cuda')


    kde.fit(X)

    save_path = os.path.join(os.path.dirname(__file__), f'{mol_name}_kde.pt')
    torch.save(kde, save_path)