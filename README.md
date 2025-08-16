```
██████╗ ██╗   ██╗ ██████╗██╗      ██████╗ ██████╗ ███████╗
██╔══██╗╚██╗ ██╔╝██╔════╝██║     ██╔═══██╗██╔══██╗██╔════╝
██████╔╝ ╚████╔╝ ██║     ██║     ██║   ██║██████╔╝███████╗
██╔═══╝   ╚██╔╝  ██║     ██║     ██║   ██║██╔═══╝ ╚════██║
██║        ██║   ╚██████╗███████╗╚██████╔╝██║     ███████║
╚═╝        ╚═╝    ╚═════╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝
```

# A Python Cyclic Loss for the Optimization of Peptide Structures 👁️ 🧬

A Python library for conditioning Boltzmann generators to design cyclic peptides; developed by the Knowles Lab @ the University of Cambridge.

## What is PyCLOPS?

PyCLOPS is a python package to help you design better cyclic peptides. It is built to condition Boltzmann generators to sample from approximately cyclic space, even when trained on exclusively linear data.

PyCLOPS comes packaged with the **6** unique cyclization chemistries by default, constituting **18** unique inter amino acid pairings to be considered in parallel, far more than any alternative available in the literature; the framework can be trivially extended to accomodate many more. PyCLOPS is largely built in torch, meaning it is **fully compatable** with its native **gradient propagation and GPU acceleration**

PyCLOPS provides:

- **Chemical Loss Functions**: KDE-based loss functions which represent the constraints imposed by particular cyclic loss chemistries
- **Convienient Optimization**: Automatically identify possible cyclizations and consider them in parallel before choosing a structure to collapse into.
- **Topology Modification**: Create new chemical bonds based on losses to asses their impact on conformational dynamics.
- **Scoring**: Evaluate protein structures using MD-based scoring pipelines.

## Installation

### Clone from Source:

```
git clone https://github.com/benjfdup/pyclops.git
cd pyclops
pip install .
```

### Pip Installation:

```
Coming soon
```

## Citation

If you use PyCLOPS in your research, please cite:

```
Insert Paper.
```

## Key Components

### Loss Handlers

- **ChemicalLossHandler**: Main handler for chemical interactions (amide bonds, disulfides, etc.)
- **MotifLossHandler**: Structural deviation from reference motifs.
- **GyrationLossHandler**: Radius of gyration constraints.
- **MetaLossHandler**: Combine multiple loss handlers for convenience.

### Structure Tools

- **StructureMaker**: Modify protein topologies to create new chemical bonds
- **Scoring Functions**: Physics-based structure evaluation

## Dependencies

- PyTorch
- NumPy
- MDAnalysis
- MDTraj
- RDKit

### Project Structure:

```
pyclops/
├── pyclops/                    # Main package directory
│   ├── core/                   # Core functionality
│   │   ├── chemical_loss/      # Chemical loss implementations
│   │   └── loss_handler/       # Loss handler management
│   ├── losses/                 # Loss function implementations
│   │   ├── kdes/              # Kernel density estimation utilities
│   │   ├── utils/             # Loss utility functions
│   │   ├── amide_losses.py    # Amide bond loss functions
│   │   ├── cysteine_carbo.py  # Cysteine-carboxylic acid losses
│   │   ├── carboxylic_carbo.py # Carboxylic acid losses
│   │   ├── disulfide.py       # Disulfide bond losses
│   │   ├── lys_tyr.py         # Lysine-tyrosine losses
│   │   └── lys_arg.py         # Lysine-arginine losses
│   ├── structure/              # Structure manipulation tools
│   │   ├── topology/          # Topology modification
│   │   └── relaxation/        # Structure relaxation
│   ├── metrics/                # Evaluation and scoring
│   │   ├── scoring/           # Structure scoring functions
│   │   └── validation/        # Validation metrics
│   ├── visualization/          # Visualization tools
│   │   └── nglview/           # NGLView integration
│   ├── torchkde/              # PyTorch KDE implementation
│   │   ├── algorithms.py      # KDE algorithms
│   │   ├── bandwidths.py      # Bandwidth selection
│   │   ├── kernels.py         # Kernel functions
│   │   ├── modules.py         # PyTorch modules
│   │   └── utils.py           # KDE utilities
│   └── utils/                  # Utility functions
│       ├── constants.py        # Physical constants
│       └── utils.py            # General utilities
├── examples/                    # Example notebooks and scripts
│   ├── pdbs/                  # Example PDB files
│   ├── Example1_loss_id.ipynb # Loss identification example
│   ├── Example2_structure_opt.ipynb # Structure optimization
│   ├── Example3_topology_mod.ipynb # Topology modification
│   ├── Example4_scoring.ipynb # Structure scoring
│   └── bens_notebook.py       # Additional examples
├── setup.py                    # Package installation configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

#### Key Modules:

- **`core/`**: Central loss handling and chemical interaction management
- **`losses/`**: Implementation of 6 unique cyclization chemistries with 18 inter-amino acid pairings
- **`structure/`**: Tools for modifying protein topologies and relaxing structures
- **`metrics/`**: Physics-based scoring and validation functions
- **`torchkde/`**: Custom PyTorch-based kernel density estimation for loss calculations
- **`visualization/`**: Interactive molecular visualization tools
- **`examples/`**: Comprehensive tutorials and use cases
