from pyclops import ChemicalLossHandler
from pyclops.structure import StructureMaker

cys_ala_5_cys_pdb = "/Users/bendupontjr/mphil_files/all_mphil_code/pyclops/examples/pdbs/CysAla5Cys.pdb"

chem_loss_handler = ChemicalLossHandler.from_pdb_file(cys_ala_5_cys_pdb, units_factor = 1.0)

print(chem_loss_handler.summary)

struct_mkr = StructureMaker(chem_loss_handler)

head2tail_loss = chem_loss_handler.chemical_losses[0]

struct_mkr.make_structure(head2tail_loss)