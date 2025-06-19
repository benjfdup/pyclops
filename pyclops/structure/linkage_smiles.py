from rdkit import Chem
from ..losses.standard_file_locations import LINKAGE_PDB_FILENAMES

LINKAGE_SMILES = {
    loss_type: Chem.MolToSmiles(Chem.MolFromPDBFile(pdb_filename))
    for loss_type, pdb_filename in LINKAGE_PDB_FILENAMES.items()
}