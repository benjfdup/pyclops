import pyclops
from pyclops.core.loss_handler.chemical_loss_handler import ChemicalLossHandler

chem_handler = ChemicalLossHandler.from_pdb_file(pdb_file="/Users/bendupontjr/mphil_files/all_mphil_code/pyclops/examples/pdbs/CysAla5Cys.pdb", units_factor=1.0)