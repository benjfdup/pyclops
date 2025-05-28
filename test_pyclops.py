from pyclops.core.chemical_loss_handler import ChemicalLossHandler

handler = ChemicalLossHandler.from_pdb(
    pdb_path='/Users/bendupontjr/mphil_files/all_mphil_code/pyclops/tests/peptides/chignolin.pdb',
    units_factor=1.0
) 

print(handler.summary())