import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pc_metric import validate_opt_metrics, compute_metrics

import pytest
import numpy as np

def test_validate_opt_metrics_valid():
    # Test valid metrics
    validate_opt_metrics(["d1", "d2"], with_normals=False)
    validate_opt_metrics(["d1", "d2", "n1"], with_normals=True)

def test_validate_opt_metrics_invalid():
    # Test invalid metric
    with pytest.raises(ValueError, match="Invalid metric: .*"):
        validate_opt_metrics(["invalid_metric"], with_normals=False)

def test_compute_metrics():
    # Test metric computation for simple point clouds
    pc1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pc2 = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]])
    metrics = compute_metrics(pc1, pc2)
    assert "d1" in metrics and "d2" in metrics
    assert np.isclose(metrics["d1"], 0.1732, atol=0.001)
    assert np.isclose(metrics["d2"], 0.1732, atol=0.001)

def test_compute_metrics_empty():
    # Test edge case with empty point clouds
    pc1 = np.empty((0, 3))
    pc2 = np.empty((0, 3))
    metrics = compute_metrics(pc1, pc2)
    assert metrics["d1"] == 0
    assert metrics["d2"] == 0