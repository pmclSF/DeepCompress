import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

# ---------------------------------------------------------------------------
# Import hook: redirect bare module imports to src.X so relative imports work.
#
# Test files use `sys.path.insert(0, src/)` then `from model_transforms import X`.
# Source modules now use relative imports (`from .constants import X`), which
# require loading as part of the `src` package.  This meta-path finder
# intercepts bare imports of src modules and loads them as `src.<name>`,
# making both conventions compatible without changing any test files.
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

_src_dir = Path(__file__).parent.parent / 'src'


class _SrcPackageFinder:
    """Meta-path finder that loads bare src module imports as src.X."""

    _loading = set()

    def find_module(self, fullname, path=None):
        if fullname in self._loading:
            return None
        if '.' not in fullname and (_src_dir / f'{fullname}.py').exists():
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        self._loading.add(fullname)
        try:
            mod = importlib.import_module(f'src.{fullname}')
            sys.modules[fullname] = mod
            return mod
        finally:
            self._loading.discard(fullname)


sys.meta_path.insert(0, _SrcPackageFinder())


def pytest_collection_modifyitems(items):
    """Filter out tf.test.TestCase.test_session, which is a deprecated
    context manager that pytest mistakenly collects as a test."""
    items[:] = [item for item in items if not item.name == "test_session"]

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture(scope="session")
def sample_point_cloud():
    """Create a simple cube point cloud."""
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                points.append([x, y, z])
    return np.array(points, dtype=np.float32)

@pytest.fixture
def create_ply_file():
    """Create a PLY file with given points."""
    def _create_ply(filepath: Path, points: np.ndarray):
        with open(filepath, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    return _create_ply

@pytest.fixture
def create_off_file():
    """Create an OFF file with given points."""
    def _create_off(filepath: Path, points: np.ndarray):
        with open(filepath, 'w') as f:
            f.write("OFF\n")
            f.write(f"{len(points)} 0 0\n")
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
    return _create_off

@pytest.fixture(scope="session")
def tf_config():
    """Configure TensorFlow for testing."""
    # Use dynamic memory allocation
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
