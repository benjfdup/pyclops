# PyClops: Cyclic Loss Operations for Peptide Structures

PyClops is a Python package for evaluating and optimizing cyclic peptide structures through specialized loss functions. It provides tools for detecting and scoring potential cyclization sites in peptide structures, computing energetic penalties based on molecular geometry, and guiding the optimization of cyclic peptide conformations.

## Overview

Cyclic peptides are important molecular structures in drug discovery and biochemistry due to their enhanced stability, binding specificity, and resistance to proteolytic degradation. PyClops helps computational chemists and structural biologists by:

- Automatically detecting potential cyclization sites in peptide structures
- Evaluating the energetic favorability of different cyclization chemistries
- Providing differentiable loss functions for peptide conformation optimization
- Supporting time-dependent loss schedules for simulated annealing-like approaches

## Key Features

- **Automatic Detection**: Identify cyclization opportunities in standard PDB structures
- **Multiple Cyclization Chemistries**: Support for disulfide bonds, amide bonds, and various non-standard linkages
- **Kernel Density Estimation**: Statistical potential functions derived from empirical structural data
- **Batched Operations**: Efficient evaluation of many conformations in parallel using PyTorch
- **Flexible Time-Dependent Losses**: Customize how losses change during optimization trajectories
- **Extensible Framework**: Easily define custom cyclization chemistries and loss functions

## Installation

### From Source with Conda

Clone the repository and install the package in development mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/pyclops.git
cd pyclops

# Create a conda environment
conda create -n pyclops python=3.9
conda activate pyclops

# Install dependencies
conda install -c conda-forge mdtraj
conda install pytorch -c pytorch
pip install torchkde  # or other required packages not in conda

# Install PyClops in development mode
pip install -e .
```

### From Source with Pip

```bash
# Clone the repository
git clone https://github.com/yourusername/pyclops.git
cd pyclops

# Install in development mode
pip install -e .
```

### Requirements

- Python >=3.9
- torch >=2.5.1
- mdtraj >=1.10.3
- numpy >=1.26.3
- torch-kde >=0.1.4

## Quick Start

```python
import torch
from pyclops.core.loss_handler import ChemicalLossHandler

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

# Get information about which cyclization chemistry is energetically favorable
best_loss_methods = handler.get_smallest_loss_methods(positions)
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
PyClops/
├── pyclops/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chemical_loss.py  # Core loss function infrastructure
│   │   ├── loss_handler.py   # Management of multiple loss functions
│   │   └── loss_coeff.py     # Time-dependent loss coefficients
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py      # Physical constants and parameters
│   │   ├── geometry.py       # Geometric calculations and transformations
│   │   └── indexing.py       # Atom and residue indexing utilities
│   └── losses/
│       ├── __init__.py
│       ├── carboxylic_carbo.py # Carboxyl-based cyclization chemistries
│       ├── disulfide.py      # Disulfide bond formation
│       ├── amide_losses.py   # Amide bond cyclization
│       ├── cysteine_carbo.py # Cysteine-carboxyl interactions
│       ├── lys_arg.py        # Lysine-arginine interactions
│       ├── lys_tyr.py        # Lysine-tyrosine interactions
│       └── standard_kde_locations.py # KDE model file paths
└── examples/
    ├── basic_usage.py        # Simple usage examples
    └── custom_losses.py      # How to define custom loss functions
```

## Citation

If you use PyClops in your research, please cite:

```
@software{pyclops2023,
  author = {Your Name},
  title = {PyClops: Cyclic Loss Operations for Peptide Structures},
  year = {2023},
  url = {https://github.com/yourusername/pyclops}
}
```

## License

MIT License

## Acknowledgments

PyClops builds upon advances in computational chemistry, statistical potentials, and deep learning. We thank the open-source communities behind PyTorch, MDTraj, and related projects.