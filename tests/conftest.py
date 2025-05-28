import pytest
import torch
import os
from pathlib import Path

# Set random seeds for reproducibility
@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    yield

# Create test data directory if it doesn't exist
@pytest.fixture(scope="session", autouse=True)
def create_test_data_dir():
    """Create test data directory if it doesn't exist."""
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    yield test_data_dir

# Set up test environment variables
@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Store original environment variables
    original_env = dict(os.environ)
    
    # Set test-specific environment variables
    os.environ["PYCLOPS_TESTING"] = "1"
    
    yield
    
    # Restore original environment variables
    os.environ.clear()
    os.environ.update(original_env)

# Device fixture for testing on different devices
@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    """Provide different devices for testing."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)

# Fixture to provide access to peptide PDB files
@pytest.fixture(scope="session")
def peptide_pdb_files():
    """Provide access to peptide PDB files in the tests/peptides directory."""
    peptides_dir = Path(__file__).parent / "peptides"
    if not peptides_dir.exists():
        raise FileNotFoundError(f"Peptides directory not found at {peptides_dir}")
    
    pdb_files = list(peptides_dir.glob("*.pdb"))
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {peptides_dir}")
    
    return {pdb_file.stem: pdb_file for pdb_file in pdb_files}

# Fixture to provide a specific peptide PDB file
@pytest.fixture
def chignolin_pdb(peptide_pdb_files):
    """Provide the chignolin PDB file."""
    if "chignolin" not in peptide_pdb_files:
        raise FileNotFoundError("chignolin.pdb not found in peptides directory")
    return peptide_pdb_files["chignolin"] 