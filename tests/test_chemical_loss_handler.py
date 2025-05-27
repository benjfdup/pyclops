import pytest
import torch
import mdtraj as md
from pathlib import Path
import numpy as np

from pyclops.core.chemical_loss_handler import ChemicalLossHandler
from pyclops.utils.constants import KB

# Test data directory - we'll need to create this
TEST_DATA_DIR = Path(__file__).parent / "test_data"

@pytest.fixture
def sample_pdb_path(tmp_path):
    """Create a simple test PDB file with a few residues."""
    pdb_content = """ATOM      1  N   ALA     1      27.346  24.516   5.362  1.00  0.00
ATOM      2  CA  ALA     1      26.166  25.411   5.362  1.00  0.00
ATOM      3  C   ALA     1      25.000  24.516   5.362  1.00  0.00
ATOM      4  O   ALA     1      24.000  25.000   5.362  1.00  0.00
ATOM      5  CB  ALA     1      26.166  26.000   6.800  1.00  0.00
ATOM      6  N   CYS     2      25.000  23.000   5.362  1.00  0.00
ATOM      7  CA  CYS     2      24.000  22.000   5.362  1.00  0.00
ATOM      8  C   CYS     2      23.000  22.500   5.362  1.00  0.00
ATOM      9  O   CYS     2      22.000  22.000   5.362  1.00  0.00
ATOM     10  CB  CYS     2      24.500  21.000   6.300  1.00  0.00
ATOM     11  SG  CYS     2      25.500  20.000   6.300  1.00  0.00
TER
END"""
    
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)
    return pdb_file

@pytest.fixture
def chemical_loss_handler(sample_pdb_path):
    """Create a ChemicalLossHandler instance for testing."""
    return ChemicalLossHandler.from_pdb(
        pdb_path=sample_pdb_path,
        units="angstroms",
        temp=300.0,
        alpha=-3.0
    )

def test_initialization(chemical_loss_handler, sample_pdb_path):
    """Test basic initialization of ChemicalLossHandler."""
    assert chemical_loss_handler.pdb_path == sample_pdb_path
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

def test_mask_functionality(sample_pdb_path):
    """Test that masking residues works correctly."""
    # Create handler with masked residues
    handler = ChemicalLossHandler.from_pdb(
        pdb_path=sample_pdb_path,
        units="angstroms",
        mask={0}  # Mask first residue
    )
    
    # Verify that masked residues are not included in losses
    for loss in handler.losses:
        atom_indices = list(loss.atom_idxs.values())
        for idx in atom_indices:
            # Check that no atom from masked residue is used
            assert handler.traj.topology.atom(idx).residue.index != 0 