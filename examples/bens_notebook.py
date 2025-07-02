import torch
from pyclops import ChemicalLossHandler

chig_pdb = "/Users/bendupontjr/mphil_files/all_mphil_code/pyclops/examples/pdbs/chignolin.pdb"
n_atoms = 166
n_batch = 10

chem_loss_handler = ChemicalLossHandler.from_pdb_file(chig_pdb, units_factor = 1.0)

positions = torch.randn(n_batch,n_atoms, 3)

print(chem_loss_handler(positions))

print(chem_loss_handler._call_explicit(positions))