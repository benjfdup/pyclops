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

### Todos:

- REDO YOUR TOY MODEL SIMULATIONS. AND REFIT YOUR KDES TO THEM...
- Implement OpenMM scoring properly, with handling of the implicit H20 as a solvant.
  - as a correlary, retrain your models on the chignolin data & validate correctly.
- Correct and finish the structure maker

### Project Structure:

To come!
