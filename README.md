# PyCLOPS: Cyclic Loss for the Optimization of Peptide Structures

PyCLOPS is a Python package for evaluating and optimizing cyclic peptide structures through specialized loss functions. It provides tools for detecting and scoring potential cyclization sites in peptide structures, computing energetic penalties based on molecular geometry, and guiding the optimization of cyclic peptide conformations.

## Overview

Cyclic peptides are important molecular structures in drug discovery and biochemistry due to their enhanced stability, binding specificity, and resistance to proteolytic degradation. PyClops helps computational chemists and structural biologists by:

- Automatically detecting potential cyclization sites in peptide structures
- Evaluating the energetic favorability of different cyclization chemistries
- Providing differentiable loss functions for peptide conformation optimization
- Supporting time-dependent loss schedules for simulated annealing-like approaches
- Visualizing protein structure properties through specialized plotting tools

## Key Features

- **Automatic Detection**: Identify cyclization opportunities in standard PDB structures
- **Multiple Cyclization Chemistries**: Support for disulfide bonds, amide bonds, and various non-standard linkages
- **Kernel Density Estimation**: Statistical potential functions derived from empirical structural data
- **Batched Operations**: Efficient evaluation of many conformations in parallel using PyTorch
- **Flexible Time-Dependent Losses**: Customize how losses change during optimization trajectories
- **Extensible Framework**: Easily define custom cyclization chemistries and loss functions
- **Structure Visualization**: Tools for analyzing protein structure properties through Ramachandran plots and other visualizations

## Installation

### From Source with Conda

Clone the repository and install the package in development mode:

```bash
# Clone the repository
git clone https://github.com/benfdup/pyclops.git
cd pyclops

# Create a conda environment
conda create -n pyclops python=3.9
conda activate pyclops

# Install dependencies
conda install -c conda-forge mdtraj
conda install pytorch -c pytorch

# Install PyClops in development mode
pip install -e .
```

### Optional Dependencies

PyCLOPS includes several optional dependencies that enable additional functionality:

#### Visualization Tools

```bash
pip install pyclops[visualization]  # Installs matplotlib, nglview, and MDAnalysis
```

- `matplotlib`: Required for plotting metrics and visualizations
- `nglview`: Required for 3D structure visualization
- `MDAnalysis`: Required for advanced molecular dynamics analysis

#### Structure Analysis

```bash
pip install pyclops[structure]  # Installs rosetta and openmm
```

- `rosetta`: Required for Rosetta-based structure scoring and analysis
- `openmm`: Required for OpenMM-based structure scoring and relaxation

#### PCA Analysis of Phi and Psi

```bash
pip install pyclops[ml]  # Installs scikit-learn
```

- `scikit-learn`: Required for machine learning-based metrics and analysis

#### All Optional Dependencies

```bash
pip install pyclops[all]  # Installs all optional dependencies
```

Note: The `rosetta` package may require special installation instructions as it's not available through PyPI. Please refer to the Rosetta documentation for installation details.

### From Source with Pip

```bash
# Clone the repository
git clone https://github.com/benfdup/pyclops.git
cd pyclops

# Install in development mode
pip install -e .  # Basic installation
pip install -e ".[all]"  # Install with all optional dependencies
```

### Requirements

Core Requirements:

- Python >=3.9
- torch >=2.5.1
- mdtraj >=1.10.3
- numpy >=1.26.3

Optional Requirements (see above for installation instructions):

- MDAnalysis >=2.0.0
- matplotlib >=3.0.0
- nglview >=3.0.0
- openmm >=7.0.0
- scikit-learn >=1.0.0
- rosetta (version depends on your installation)

## Quick Start

```python
import torch
from pyclops.core.loss_handler import ChemicalLossHandler
from pyclops.metrics.visualization import ramachandran_plot

# Create a handler from a PDB file
handler = ChemicalLossHandler.from_pdb(
    pdb_path="my_peptide.pdb",
    units="angstrom",
    temp=300.0
)

# Print summary of detected cyclization sites
print(handler.summary())

# Evaluate loss for a batch of conformations
positions = torch.rand(32, 100, 3)  # 32 batch, 100 atoms, 3D coordinates
loss = handler(positions)  # Returns [32] tensor of losses

# Generate a Ramachandran plot for the first frame
fig, ax = ramachandran_plot(
    coordinates=positions,
    pdb_file="my_peptide.pdb",
    frame_idx=0,
    save_path="ramachandran.png"
)
```

## Advanced Usage

```python
# Create time-dependent loss with coefficients
from pyclops.core.loss_coeff import PseudoGaussian, Constant
from pyclops.core.loss_handler import ConditioningLossHandler, GyrationLossHandler

# Create individual loss components
cyclic_handler = ChemicalLossHandler.from_pdb("peptide.pdb", units="angstrom")
gyration_handler = GyrationLossHandler(units_factor=1.0, squared=False)

# Combine with time-dependent coefficients
time_dep_handler = ConditioningLossHandler(
    l_cyc=cyclic_handler,
    gamma=PseudoGaussian(mu=0.5, s=0.1, coeff=0.2),
    l_gyr=gyration_handler
)

# Evaluate at different timesteps
timesteps = torch.linspace(0, 1, 10)
for t in timesteps:
    loss_t = time_dep_handler(positions, t)
```

## Supported Cyclization Chemistries

PyClops supports various cyclization chemistries, including:

1. **Disulfide Bonds**: Between cysteine residues
2. **Amide Bonds**:
   - Head-to-tail backbone cyclization
   - Side chain-to-side chain (Lys-Glu, Lys-Asp, Orn-Glu)
   - Side chain-to-terminus (Lys-Head, Arg-Head, Lys-Tail, Orn-Tail)
3. **Carboxylic Interactions**:
   - Asp-Glu, Asp-Asp, Glu-Glu
   - Asp-C-terminus, Glu-C-terminus
4. **Cysteine-Carboxyl**:
   - Cys-C-terminus
   - Cys-Asp, Cys-Glu
5. **Specialty Cyclizations**:
   - Lys-Arg
   - Lys-Tyr

## Project Structure

```
pyclops/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── chemical_loss.py          # Core chemical loss function infrastructure
│   ├── chemical_loss_handler.py  # Handler for chemical loss functions
│   ├── gyration_loss_handler.py  # Handler for gyration-based losses
│   ├── loss_coeff.py            # Time-dependent loss coefficients
│   ├── loss_handler.py          # Base loss handler functionality
│   ├── meta_loss_handler.py     # Meta-level loss handling
│   └── motif_loss_handler.py    # Motif-specific loss handling
├── metrics/
│   ├── __init__.py
│   ├── openmm/                  # OpenMM-specific metrics
│   ├── rosetta/                 # Rosetta-specific metrics
│   └── visualization/           # Structure visualization tools
│       ├── __init__.py
│       └── ramachandran.py      # Ramachandran plot functionality
├── utils/
│   ├── __init__.py
│   ├── constants.py             # Physical constants and parameters
│   ├── default_strategies.py    # Default loss computation strategies
│   ├── indexing.py              # Atom and residue indexing utilities
│   └── utils.py                 # General utility functions
├── losses/
│   ├── __init__.py
│   ├── kdes/                    # Kernel density estimation models
│   │   ├── __init__.py
│   │   ├── Amide_kde.pt                # Amide bond KDE model
│   │   ├── Carboxylic-Carbo_kde.pt     # Carboxylic KDE model
│   │   ├── Cys-Arg_kde.pt             # Cysteine-Arginine KDE model
│   │   ├── Cys-Carboxyl_kde.pt        # Cysteine-Carboxyl KDE model
│   │   ├── Disulfide_kde.pt           # Disulfide bond KDE model
│   │   ├── Lys-Arg_kde.pt             # Lysine-Arginine KDE model
│   │   ├── Lys-Tyr_kde.pt             # Lysine-Tyrosine KDE model
│   │   ├── Sulfur-Mediated-Amide_kde.pt # Sulfur-mediated amide KDE model
│   │   └── generate_kde.py            # KDE model generation script
│   ├── carboxylic_carbo.py      # Carboxyl-based cyclization chemistries
│   ├── disulfide.py            # Disulfide bond formation
│   ├── amide_losses.py         # Amide bond cyclization
│   ├── cysteine_carbo.py       # Cysteine-carboxyl interactions
│   ├── lys_arg.py              # Lysine-arginine interactions
│   ├── lys_tyr.py              # Lysine-tyrosine interactions
│   └── standard_kde_locations.py # KDE model file paths
├── torchkde/                    # Kernel density estimation implementation
│   ├── __init__.py
│   ├── algorithms.py           # KDE algorithms implementation
│   ├── bandwidths.py          # Bandwidth selection methods
│   ├── kernels.py             # Kernel function implementations
│   ├── modules.py             # PyTorch modules for KDE
│   ├── readme.md              # KDE implementation documentation
│   └── utils.py               # KDE utility functions
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── conftest.py            # PyTest configuration
│   ├── pyclops_test.ipynb     # Interactive testing notebook
│   ├── README.md              # Testing documentation
│   ├── test_chemical_loss_handler.py # Chemical loss handler tests
│   └── peptides/              # Test peptide structures
│       ├── __init__.py
│       ├── README.md          # Peptide test data documentation
│       ├── CDEKCG.pdb         # Test peptide structure
│       ├── DDEEKKCGLCGR.pdb   # Test peptide structure
│       ├── EQKCGDCTY.pdb      # Test peptide structure
│       ├── KADGLYQ.pdb        # Test peptide structure
│       ├── KDGEQRNCTYKA.pdb   # Test peptide structure
│       ├── RKGEYH.pdb         # Test peptide structure
│       ├── chignolin.pdb      # Test peptide structure
│       └── pd1_binder.pdb     # Test peptide structure
├── setup.py                    # Package installation configuration
└── README.md                   # This documentation file
```

## Citation

If you use PyClops in your research, please cite:

```
@software{pyclops2025,
  author = {Benjamin du Pont},
  title = {PyClops: Cyclic Loss Operations for Peptide Structures},
  year = {2023},
  url = {https://github.com/yourusername/pyclops}
}
```

## License

MIT License

## Acknowledgments

PyClops builds upon advances in computational chemistry, statistical potentials, and deep learning. We thank the open-source communities behind PyTorch, MDTraj, and related projects.
