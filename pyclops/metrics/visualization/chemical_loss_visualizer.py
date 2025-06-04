from ...core.chemical_loss_handler import ChemicalLossHandler

class ChemicalLossVisualizer:
    """
    This class is used to help visualize a chemical loss handler for validation purposes
    in various ways.
    """
    def __init__(self, 
                 chemical_loss_handler: ChemicalLossHandler,
                 ):
        self._chemical_loss_handler = chemical_loss_handler

    # we will want a method to visualize a 2d representaion of the compound as
    # well as all atoms the chemical loss handler is considering

    # we will want a method to visualize, given a set of coordinates, the atoms considered
    # by the smallest chemical loss for each frame (smallest out of the resonances too).