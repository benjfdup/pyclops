"""
Module for managing KDE file locations in the PyClops package.

This module provides functions and constants to consistently find KDE files
regardless of whether the package is being run from source or has been installed.
"""

import os
import importlib.resources
from typing import Dict, Optional

# Dictionary mapping loss type keys to KDE filenames
KDE_FILENAMES = {
    'amide': 'Amide_kde.pt',
    'carboxylic-carbo': 'Carboxylic-Carbo_kde.pt',
    'cysteine-carbo': 'Cys-Carboxyl_kde.pt',
    'disulfide': 'Disulfide_kde.pt',
    'lys-arg': 'Lys-Arg_kde.pt',
    'lys-tyr': 'Lys-Tyr_kde.pt',
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
    """
    if kde_type not in KDE_FILENAMES:
        raise ValueError(f"Unknown KDE type: {kde_type}. Available types: {list(KDE_FILENAMES.keys())}")
    
    filename = KDE_FILENAMES[kde_type]
    
    # Approach 1: Try importlib.resources (Python 3.7+) - works for installed packages
    try:
        # Import the package's kdes module
        import pyclops.losses.kdes
        
        # Get the path to the resource within the package
        with importlib.resources.path(pyclops.losses.kdes, filename) as path:
            return str(path)
    except (ImportError, ModuleNotFoundError):
        # Package might be running from source or data module structure is different
        pass
    
    # Approach 2: Look for the 'kdes' folder in the same directory as this module
    module_dir = os.path.dirname(os.path.abspath(__file__))
    kdes_path = os.path.join(module_dir, 'kdes', filename)
    
    if os.path.exists(kdes_path):
        return kdes_path
    
    # Approach 3: Check if we're in a development environment with a different structure
    alt_data_path = os.path.join(module_dir, 'kde_files', filename)
    if os.path.exists(alt_data_path):
        return alt_data_path
    
    # If we can't find the file, raise an informative error
    raise FileNotFoundError(
        f"Could not find KDE file '{filename}' for type '{kde_type}'. "
        f"Searched in: \n"
        f"- Package resources (pyclops.losses.kdes)\n"
        f"- {kdes_path}\n"
        f"- {alt_data_path}\n"
        "Please ensure the data files are correctly installed with the package."
    )

# Create a dictionary that maps keys to actual file paths
# This allows lazy loading of paths when they're actually needed
class _KDEPathDict(dict):
    def __getitem__(self, key):
        # Get the path dynamically only when requested
        return get_kde_path(key)

# Public API: Dictionary-like object for accessing KDE file paths
STANDARD_KDE_LOCATIONS = _KDEPathDict()