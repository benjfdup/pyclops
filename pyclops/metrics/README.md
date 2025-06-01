# PyCLOPS Metrics

This module provides various metrics for evaluating PyCLOPS & protein structures. The module includes several optional dependencies that enable additional functionality:

## Optional Dependencies

- **Visualization Tools**:

  - `matplotlib`: Required for plotting metrics and visualizations
  - `nglview`: Required for 3D structure visualization
  - `mdanalysis`: Required for advanced molecular dynamics analysis

- **Structure Analysis**:
  - `rosetta`: Required for Rosetta-based structure scoring and analysis
  - `openmm`: Required for OpenMM-based structure scoring and relaxation
  - `sklearn`: Required for machine learning-based metrics and analysis

## Installation

To install with all optional dependencies:

```bash
pip install pyclops[all]
```

To install with specific optional dependencies:

```bash
pip install pyclops[visualization]  # For matplotlib, nglview, and mdanalysis
pip install pyclops[structure]      # For rosetta and openmm
pip install pyclops[ml]            # For sklearn
```

## Features

- Structure scoring and evaluation
- Ramachandran plot generation
- PDB visualization and analysis
- Structure relaxation and optimization
- Machine learning-based metrics (when sklearn is installed)

Note: Some features may require specific optional dependencies to be installed. The code will raise informative errors if required dependencies are missing.

Includes metrics by which you can evaluate your PyCLOPS & protein structures.

We likely want to add an OpenMM based way of scoring too?

And lastly, a way to take a structure which is run through the conditioning and then add the cyclic "ligand" it has collapsed into in the topology (based on the toy example?) -- This may be fairly complex.

We also probably want a module or something that helps us easily plot ramachandran, pdbs, etc.

We also probably want some ML based way of doing all of this, but I am dubious of this, as it will very likely never have seen any of this...

Alex had a great idea today -- to test how plausible a given cyclic sample is, just add the ligand into the pdb and relax it over say 5 steps...
