import pytest
import torch
import mdtraj as md
from pathlib import Path
import numpy as np

from pyclops.core.chemical_loss_handler import ChemicalLossHandler
from pyclops.utils.constants import KB

# Test markers
pytestmark = [
    pytest.mark.unit,  # Mark all tests in this file as unit tests
    pytest.mark.chemical_loss  # Mark all tests as related to chemical loss
]

@pytest.fixture(params=["chignolin", "pd1_binder"])
def peptide_pdb(request, peptide_pdb_files):
    """Create a fixture that provides each peptide PDB file in turn."""
    if request.param not in peptide_pdb_files:
        pytest.skip(f"{request.param}.pdb not found in peptides directory")
    return peptide_pdb_files[request.param]

@pytest.fixture
def chemical_loss_handler(peptide_pdb):
    """Create a ChemicalLossHandler instance for testing using the current peptide PDB."""
    return ChemicalLossHandler.from_pdb(
        pdb_path=peptide_pdb,
        units="angstroms",
        temp=300.0,
        alpha=-3.0
    )

@pytest.mark.initialization
def test_initialization(chemical_loss_handler, peptide_pdb):
    """Test basic initialization of ChemicalLossHandler."""
    assert chemical_loss_handler.pdb_path == peptide_pdb
    assert chemical_loss_handler.temp == 300.0
    assert chemical_loss_handler.alpha == -3.0
    assert isinstance(chemical_loss_handler.device, torch.device)

@pytest.mark.error_handling
def test_invalid_pdb_path():
    """Test that initialization fails with invalid PDB path."""
    with pytest.raises(FileNotFoundError):
        ChemicalLossHandler.from_pdb(
            pdb_path="nonexistent.pdb",
            units="angstroms"
        )

@pytest.mark.error_handling
def test_invalid_units():
    """Test that initialization fails with invalid units."""
    with pytest.raises(ValueError, match="Unknown unit"):
        ChemicalLossHandler.from_pdb(
            pdb_path="test.pdb",
            units="invalid_unit"
        )

@pytest.mark.error_handling
def test_units_ambiguity():
    """Test that initialization fails when both units and units_factor are provided."""
    with pytest.raises(ValueError, match="Provide either 'units' or 'units_factor'"):
        ChemicalLossHandler.from_pdb(
            pdb_path="test.pdb",
            units="angstroms",
            units_factor=1.0
        )

@pytest.mark.error_handling
def test_missing_units():
    """Test that initialization fails when neither units nor units_factor is provided."""
    with pytest.raises(ValueError, match="Either 'units' or 'units_factor' must be provided"):
        ChemicalLossHandler.from_pdb(
            pdb_path="test.pdb"
        )

@pytest.mark.core_functionality
def test_loss_evaluation(chemical_loss_handler):
    """Test that loss evaluation returns correct shape and type."""
    # Create a batch of positions
    n_atoms = chemical_loss_handler.traj.n_atoms
    batch_size = 2
    positions = torch.randn(batch_size, n_atoms, 3)
    
    # Evaluate loss
    loss = chemical_loss_handler(positions)
    
    # Check output shape and type
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == (batch_size,)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()

def test_get_smallest_loss(chemical_loss_handler):
    """Test getting the smallest loss for each structure."""
    n_atoms = chemical_loss_handler.traj.n_atoms
    batch_size = 2
    positions = torch.randn(batch_size, n_atoms, 3)
    
    smallest_losses = chemical_loss_handler.get_smallest_loss(positions)
    
    assert len(smallest_losses) == batch_size
    for loss in smallest_losses:
        assert hasattr(loss, 'method')
        assert hasattr(loss, 'atom_idxs')

def test_get_smallest_loss_methods(chemical_loss_handler):
    """Test getting the method names of smallest losses."""
    n_atoms = chemical_loss_handler.traj.n_atoms
    batch_size = 2
    positions = torch.randn(batch_size, n_atoms, 3)
    
    methods = chemical_loss_handler.get_smallest_loss_methods(positions)
    
    assert len(methods) == batch_size
    assert all(isinstance(method, str) for method in methods)

def test_get_all_losses(chemical_loss_handler):
    """Test getting all individual loss values."""
    n_atoms = chemical_loss_handler.traj.n_atoms
    batch_size = 2
    positions = torch.randn(batch_size, n_atoms, 3)
    
    all_losses = chemical_loss_handler.get_all_losses(positions)
    
    assert isinstance(all_losses, torch.Tensor)
    assert len(all_losses.shape) == 2
    assert all_losses.shape[0] == batch_size
    assert not torch.isnan(all_losses).any()
    assert not torch.isinf(all_losses).any()

def test_summary(chemical_loss_handler):
    """Test that summary method returns a non-empty string."""
    summary = chemical_loss_handler.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "ChemicalLossHandler Summary" in summary

def test_validate_configuration(chemical_loss_handler):
    """Test configuration validation."""
    is_valid = chemical_loss_handler.validate_configuration()
    assert isinstance(is_valid, bool)

def test_inspect_losses(chemical_loss_handler):
    """Test loss inspection functionality."""
    n_atoms = chemical_loss_handler.traj.n_atoms
    positions = torch.randn(2, n_atoms, 3)
    
    # This should not raise any exceptions
    chemical_loss_handler.inspect_losses(positions, top_k=3)

def test_mask_functionality(peptide_pdb):
    """Test that masking residues works correctly."""
    # Create handler with masked residues
    handler = ChemicalLossHandler.from_pdb(
        pdb_path=peptide_pdb,
        units="angstroms",
        mask={0}  # Mask first residue
    )
    
    # Verify that masked residues are not included in losses
    for loss in handler.losses:
        atom_indices = list(loss.atom_idxs.values())
        for idx in atom_indices:
            # Check that no atom from masked residue is used
            assert handler.traj.topology.atom(idx).residue.index != 0

def test_peptide_specific_properties(peptide_pdb):
    """Test properties specific to each peptide structure."""
    handler = ChemicalLossHandler.from_pdb(
        pdb_path=peptide_pdb,
        units="angstroms"
    )
    
    # Test that the handler loaded the correct number of atoms
    traj = md.load(str(peptide_pdb))
    assert handler.traj.n_atoms == traj.n_atoms
    
    # Test that the handler found some valid cyclization sites
    assert len(handler.losses) > 0
    
    # Test that the summary contains the correct PDB name
    summary = handler.summary()
    assert peptide_pdb.name in summary

def test_peptide_comparison(peptide_pdb_files):
    """Compare properties between different peptide structures."""
    handlers = {}
    for name, pdb_file in peptide_pdb_files.items():
        handlers[name] = ChemicalLossHandler.from_pdb(
            pdb_path=pdb_file,
            units="angstroms"
        )
    
    # Compare number of atoms
    n_atoms = {name: h.traj.n_atoms for name, h in handlers.items()}
    assert len(set(n_atoms.values())) > 1, "All peptides have the same number of atoms"
    
    # Compare number of cyclization sites
    n_losses = {name: len(h.losses) for name, h in handlers.items()}
    assert len(set(n_losses.values())) > 1, "All peptides have the same number of cyclization sites"
    
    # Compare loss values for the same random positions
    batch_size = 2
    positions = torch.randn(batch_size, max(n_atoms.values()), 3)
    
    losses = {}
    for name, handler in handlers.items():
        # Pad or truncate positions to match the number of atoms
        pos = positions[:, :handler.traj.n_atoms, :]
        losses[name] = handler(pos)
    
    # Check that different peptides give different loss values
    loss_values = {name: loss.mean().item() for name, loss in losses.items()}
    assert len(set(loss_values.values())) > 1, "All peptides give the same loss values" 