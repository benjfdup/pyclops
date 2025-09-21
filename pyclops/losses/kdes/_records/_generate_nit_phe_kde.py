"""
This script generates KDE models for the chemical losses in PyCLOPS, based on the data.
This basically just fits KDEs to the data. It is here solely for record-keeping purposes.
"""
from typing import Dict
import os

import torch
import mdtraj as md

from pyclops.torchkde.modules import KernelDensity
from pyclops.losses.nit_phe import NitPhe


mol_verts = {
    'nit-phe': NitPhe._atom_idxs_keys, # N1, 
                                       # C1, 
                                       # C3, 
                                       # C2
}

verts_to_pdb_atoms = {
    NitPhe._atom_idxs_keys: ["N1", "C1", "C3", "C4"],
}

mol_tetra_dict = {
        key: verts_to_pdb_atoms[mol_verts[key]] for key in mol_verts.keys()
    }

mol_tetra_vals: Dict[str, torch.Tensor] = {}

for mol_name in mol_verts.keys():
    stride: int = 1
    frac_to_exclude: float = 0.2 # initial fraction to exclude from the start of the trajectory

    mol_dir = os.path.join('/home/ubuntu/scratch/monorepo/pyclops/pyclops/losses/kdes/_records/_simulations', mol_name)
    results_dir = os.path.join(mol_dir)
    save_dir = '/home/ubuntu/scratch/monorepo/pyclops/pyclops/losses/kdes'

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
    distances = md.compute_distances(traj, edges_idx, periodic=True)  # shape: (n_frames, 6)
    distances *= 10  # Convert from nm to Angstroms

    print(f'mol_name: {mol_name}')
    print(f"Computed {distances.shape[1]} edge distances across {distances.shape[0]} frames.")
    print("Example (first frame):", distances[0])

    mol_tetra_vals[mol_name] = distances

for mol_name in mol_verts.keys():
    X = torch.from_numpy(mol_tetra_vals[mol_name]).to('cuda')
    # fit the bandwidth better here...
    # Note: in the future, I will make for better bandwidth handling (DxD matrix),
    # but for now, 1.0 is fine... it just very likely oversmooths the data.
    kde = KernelDensity(bandwidth=0.8, kernel='cauchy') # bandwidth is in angstroms
    kde.fit(X)

    save_path = os.path.join(save_dir, f'{mol_name}_kde.pt')
    torch.save(kde, save_path)