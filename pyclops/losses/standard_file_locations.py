"""
Module for managing KDE and linkage PDB file locations in the PyCLOPS package.

This module provides functions and constants to consistently find KDE files
regardless of whether the package is being run from source or has been installed.
"""

import os
import importlib.resources

# Dictionary mapping loss type keys to KDE filenames
KDE_FILENAMES = {
    'amide': 'Amide_kde.pt',
    'carboxylic-carbo': 'Carboxylic-Carbo_kde.pt',
    'cysteine-carbo': 'Cys-Carboxyl_kde.pt',
    'disulfide': 'Disulfide_kde.pt',
    'lys-arg': 'Lys-Arg_kde.pt',
    'lys-tyr': 'Lys-Tyr_kde.pt',
}

LINKAGE_PDB_FILENAMES = {
    'disulfide': 'disulfide_linkage.pdb',
    'amide': 'amide_linkage.pdb',
    'carboxylic-carbo': 'carboxylic-carbo_linkage.pdb',
    'cysteine-carbo': 'cysteine-carbo_linkage.pdb',
    'lys-arg': 'lys-arg_linkage.pdb',
    'lys-tyr': 'lys-tyr_linkage.pdb',
}

def get_kde_path(kde_type: str) -> str:
    """
    Get the absolute path to a KDE file based on its type key.
    
    This function works whether the package is installed or being run from source.
    
    Parameters
    ----------
    kde_type : str
        Key for the KDE file type (e.g., 'amide', 'disulfide', etc.)
        
    Returns
    -------
    str
        Absolute path to the KDE file
        
    Raises
    ------
    ValueError
        If the kde_type is not recognized
    FileNotFoundError
        If the KDE file cannot be found
    """
    if kde_type not in KDE_FILENAMES:
        raise ValueError(f"Unknown KDE type: {kde_type}. Available types: {list(KDE_FILENAMES.keys())}")
    
    filename = KDE_FILENAMES[kde_type]
    
    # Approach 1: Try importlib.resources (Python 3.7+) - works for installed packages
    try:
        with importlib.resources.path('pyclops.losses.kdes', filename) as path:
            return str(path)
    except (ImportError, ModuleNotFoundError):
        # Package might be running from source
        pass
    
    # Approach 2: Look for the 'kdes' folder in the same directory as this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    kdes_path = os.path.join(module_dir, 'kdes', filename)
    
    if os.path.exists(kdes_path):
        return kdes_path
    
    # If we can't find the file, raise an informative error
    raise FileNotFoundError(
        f"Could not find KDE file '{filename}' for type '{kde_type}'. "
        f"Searched in: \n"
        f"- Package resources (pyclops.losses.kdes)\n"
        f"- {kdes_path}\n"
        "Please ensure the data files are correctly installed with the package."
    )

# Create a dictionary that maps keys to actual file paths
# This allows lazy loading of paths when they're actually needed
class _KDEPathDict(dict):
    def __getitem__(self, key):
        # Get the path dynamically only when requested
        return get_kde_path(key)
    
def get_linkage_pdb_path(linkage_type: str) -> str:
    if linkage_type not in LINKAGE_PDB_FILENAMES:
        raise ValueError(f"Unknown linkage type: {linkage_type}. Available types: {list(LINKAGE_PDB_FILENAMES.keys())}")
    
    filename = LINKAGE_PDB_FILENAMES[linkage_type]
    
    # Approach 1: Try importlib.resources (Python 3.7+) - works for installed packages
    try:
        with importlib.resources.path('pyclops.losses.linkage_pdbs', filename) as path:
            return str(path)
    except (ImportError, ModuleNotFoundError):
        # Package might be running from source
        pass
    
    # Approach 2: Look for the 'pdbs' folder in the same directory as this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    pdbs_path = os.path.join(module_dir, 'linkage_pdbs', filename)
    
    if os.path.exists(pdbs_path):
        return pdbs_path
    
    # If we can't find the file, raise an informative error
    raise FileNotFoundError(
        f"Could not find linkage PDB file '{filename}' for type '{linkage_type}'. "
        f"Searched in: \n"
        f"- Package resources (pyclops.losses.linkage_pdbs)\n"
        f"- {pdbs_path}\n"
        "Please ensure the data files are correctly installed with the package."
    )

class _LinkagePDBPathDict(dict):
    def __getitem__(self, key):
        # Get the path dynamically only when requested
        return get_linkage_pdb_path(key)
    
# Public API: Dictionary-like object for accessing KDE file paths
STANDARD_KDE_LOCATIONS = _KDEPathDict()
STANDARD_LINKAGE_PDB_LOCATIONS = _LinkagePDBPathDict()