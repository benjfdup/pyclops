import pytest
import torch
import mdtraj as md
from pathlib import Path
import numpy as np

from pyclops.core.chemical_loss_handler import ChemicalLossHandler
from pyclops.utils.constants import KB

@pytest.fixture
def chemical_loss_handler(chignolin_pdb):
    """Create a ChemicalLossHandler instance for testing using chignolin PDB."""
    return ChemicalLossHandler.from_pdb(
        pdb_path=chignolin_pdb,
        units="angstroms",
        temp=300.0,
        alpha=-3.0
    )

def test_initialization(chemical_loss_handler, chignolin_pdb):
    """Test basic initialization of ChemicalLossHandler."""
    assert chemical_loss_handler.pdb_path == chignolin_pdb
    assert chemical_loss_handler.temp == 300.0
    assert chemical_loss_handler.alpha == -3.0
    assert isinstance(chemical_loss_handler.device, torch.device)

def test_invalid_pdb_path():
    """Test that initialization fails with invalid PDB path."""
    with pytest.raises(FileNotFoundError):
        ChemicalLossHandler.from_pdb(
            pdb_path="nonexistent.pdb",
            units="angstroms"
        )

def test_invalid_units():
    """Test that initialization fails with invalid units."""
    with pytest.raises(ValueError, match="Unknown unit"):
        ChemicalLossHandler.from_pdb(
            pdb_path="test.pdb",
            units="invalid_unit"
        )

def test_units_ambiguity():
    """Test that initialization fails when both units and units_factor are provided."""
    with pytest.raises(ValueError, match="Provide either 'units' or 'units_factor'"):
        ChemicalLossHandler.from_pdb(
            pdb_path="test.pdb",
            units="angstroms",
            units_factor=1.0
        )

def test_missing_units():
    """Test that initialization fails when neither units nor units_factor is provided."""
    with pytest.raises(ValueError, match="Either 'units' or 'units_factor' must be provided"):
        ChemicalLossHandler.from_pdb(
            pdb_path="test.pdb"
        )

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

def test_mask_functionality(chignolin_pdb):
    """Test that masking residues works correctly."""
    # Create handler with masked residues
    handler = ChemicalLossHandler.from_pdb(
        pdb_path=chignolin_pdb,
        units="angstroms",
        mask={0}  # Mask first residue
    )
    
    # Verify that masked residues are not included in losses
    for loss in handler.losses:
        atom_indices = list(loss.atom_idxs.values())
        for idx in atom_indices:
            # Check that no atom from masked residue is used
            assert handler.traj.topology.atom(idx).residue.index != 0

def test_multiple_peptides(peptide_pdb_files):
    """Test ChemicalLossHandler with multiple peptide structures."""
    for name, pdb_file in peptide_pdb_files.items():
        handler = ChemicalLossHandler.from_pdb(
            pdb_path=pdb_file,
            units="angstroms"
        )
        
        # Basic validation
        assert handler.pdb_path == pdb_file
        assert handler.traj.n_atoms > 0
        
        # Test loss evaluation
        n_atoms = handler.traj.n_atoms
        positions = torch.randn(1, n_atoms, 3)
        loss = handler(positions)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == (1,)
        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any() 