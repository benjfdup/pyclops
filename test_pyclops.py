from pyclops.core.chemical_loss_handler import ChemicalLossHandler

handler = ChemicalLossHandler.from_pdb(
    pdb_path='/home/bfd21/rds/rds-ab_non_specific-7ZL1FWpHG4k/peptide-md-data/chignolin_dft/data/chignolin.pdb',
    units_factor=1.0,
    debug=True
) 

print(handler.summary)