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