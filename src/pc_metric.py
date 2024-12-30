# pc_metric.py

import numpy as np
from numba import njit
from numba.typed import List

@njit
def validate_opt_metrics(metrics, with_normals=False):
    """
    Validate optimization metrics.

    Args:
        metrics (list): List of metric names.
        with_normals (bool): If True, includes normal-based metrics.

    Returns:
        None
    """
    allowed_metrics = List(["d1", "d2"])
    if with_normals:
        allowed_metrics.extend(["n1", "n2"])

    for metric in metrics:
        if metric not in allowed_metrics:
            raise ValueError(f"Invalid metric: {metric}. Allowed metrics: {allowed_metrics}")

@njit
def compute_metrics(pc1, pc2):
    """
    Compute metrics between two point clouds.

    Args:
        pc1 (np.ndarray): First point cloud as an array of shape (N, 3).
        pc2 (np.ndarray): Second point cloud as an array of shape (M, 3).

    Returns:
        dict: Dictionary containing computed metrics.
    """
    if len(pc1) == 0 or len(pc2) == 0:
        return {"d1": 0.0, "d2": 0.0}

    # Compute d1: mean distance from pc1 to pc2
    d1_sum = 0.0
    for i in range(len(pc1)):
        diff = pc1[i] - pc2[i % len(pc2)]
        d1_sum += np.sqrt(np.sum(diff ** 2))
    d1 = d1_sum / len(pc1)

    # Compute d2: mean distance from pc2 to pc1
    d2_sum = 0.0
    for i in range(len(pc2)):
        diff = pc2[i] - pc1[i % len(pc1)]
        d2_sum += np.sqrt(np.sum(diff ** 2))
    d2 = d2_sum / len(pc2)

    return {"d1": d1, "d2": d2}

# Test functions for metrics
if __name__ == "__main__":
    pc1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    pc2 = np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]])

    validate_opt_metrics(["d1", "d2"], with_normals=False)

    metrics = compute_metrics(pc1, pc2)
    print(metrics)
