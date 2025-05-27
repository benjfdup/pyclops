from typing import Dict, Set

class IndexesMethodPair:
    def __init__(self, 
                 indexes: Dict[str, int], 
                 method: str, 
                 pair: Set[int],
                 ):
        """
        A structured container for representing a specific instance of a chemical cyclization motif.

        This class groups:
        - the atom indices required for evaluating a particular chemical loss function,
        - a descriptive string indicating the cyclization method and atom choices (e.g., specific resonance),
        - the residue indices of the two amino acids involved in the cyclization.

        Parameters
        ----------
        indexes : Dict[str, int]
            A mapping from atom role keys (e.g., 'O1', 'C2') to global atom indices within the trajectory.
            These are used to extract atomic coordinates for loss calculation.

        method : str
            A descriptive label summarizing the chemical motif and specific permutation (e.g., "AspGlu (OD1-OE2)").

        pair : Set[int]
            A set containing the two residue indices (usually amino acids) involved in the cyclization. 
            Used for bookkeeping and resonance calculations.

        Raises
        ------
        ValueError
            If `pair` is provided but is not a `set` of exactly two integers.

        Example
        -------
        IndexesMethodPair(
            indexes={"O1": 123, "O2": 456, "C1": 789, "C2": 101},
            method="Disulfide, CYS 3 -> CYS 14",
            pair={5, 9}
        )
        """

        self.indexes = indexes
        self.method = method
        self.pair = pair # perhaps explicitly check that pair is a set of ints