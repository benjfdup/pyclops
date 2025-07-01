from pyclops import ChemicalLossHandler
from pyclops.structure import StructureMaker

chig_pdb_file = '/Users/bendupontjr/mphil_files/all_mphil_code/pyclops/examples/pdbs/chignolin.pdb'

chem_loss_handler = ChemicalLossHandler.from_pdb_file(chig_pdb_file, units_factor = 1.0)

print(chem_loss_handler.summary)

glu_cterm = chem_loss_handler.chemical_losses[-2]

struct_mkr = StructureMaker(chem_loss_handler)

mol = struct_mkr.make_structure(glu_cterm)

#AllChem.EmbedMolecule(mol, AllChem.ETKDG())
#AllChem.MMFFOptimizeMolecule(mol)

mol