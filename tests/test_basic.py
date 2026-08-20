import pytest
from pathlib import Path
import sys

# Ensure src is in path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_project_root, ProjectPaths, ProjectConfig
from src.utils.gpu_check import verify_gpu

def test_project_root_resolution():
    """Test that the project root is correctly resolved."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.name == "CI-Pipeline-Intelligent-Validator-v2"
    assert (root / "configs").exists()

def test_path_resolution():
    """Test that paths are resolved correctly from config."""
    paths = ProjectPaths()
    assert isinstance(paths.data_root, Path)
    assert isinstance(paths.raw_data_root, Path)
    assert paths.data_root.is_absolute()
    assert paths.raw_data_root.is_absolute()

def test_reproducibility_config_loading():
    """Test that the reproducibility configuration is loaded."""
    config = ProjectConfig()
    rep = config.reproducibility
    assert "seed" in rep
    assert "python_environment" in rep
    assert "device_configuration" in rep

def test_project_import_sanity():
    """Test basic module imports from src."""
    import src.utils
    import src.data
    import src.preprocessing

def test_pytorch_imports():
    """Test PyTorch can be imported."""
    try:
        import torch
    except ImportError as e:
        pytest.fail(f"PyTorch could not be imported: {e}")

def test_gpu_integration():
    """GPU Integration Test - strictly requires CUDA if present."""
    res = verify_gpu()
    
    # Fail if torch is not even importable (caught by test_pytorch_imports anyway)
    assert res["torch_version"] is not None
    
    if not res["cuda_available"]:
        pytest.skip("CUDA is not available on this machine. Skipping GPU integration test.")
    else:
        # On this machine, we WANT this to pass
        assert res["tensor_allocation_success"] is True, f"Failed tensor allocation: {res.get('error')}"
        assert res["cuda_computation_success"] is True, f"Failed CUDA computation: {res.get('error')}"
