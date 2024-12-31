import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict

@njit(parallel=True)
def compute_point_to_point_distances(points1: np.ndarray, 
                                     points2: np.ndarray) -> np.ndarray:
    """
    Compute point-to-point distances using parallel Numba optimization.
    """
    N = points1.shape[0]
    M = points2.shape[0]
    distances = np.empty(N, dtype=np.float32)
    
    for i in prange(N):
        min_dist = np.inf
        for j in range(M):
            dist = np.sum((points1[i] - points2[j]) ** 2)
            if dist < min_dist:
                min_dist = dist
        distances[i] = np.sqrt(min_dist)
        
    return distances

@njit(parallel=True)
def compute_point_to_normal_distances(points1: np.ndarray, 
                                      points2: np.ndarray,
                                      normals2: np.ndarray) -> np.ndarray:
    """
    Compute point-to-plane distances using parallel Numba optimization.
    """
    N = points1.shape[0]
    M = points2.shape[0]
    distances = np.empty(N, dtype=np.float32)
    
    for i in prange(N):
        min_dist = np.inf
        for j in range(M):
            vec = points1[i] - points2[j]
            dist = abs(np.sum(vec * normals2[j]))
            if dist < min_dist:
                min_dist = dist
        distances[i] = min_dist
        
    return distances

def calculate_metrics(predicted: np.ndarray, 
                      ground_truth: np.ndarray,
                      predicted_normals: Optional[np.ndarray] = None,
                      ground_truth_normals: Optional[np.ndarray] = None,
                      use_kdtree: bool = True) -> Dict[str, float]:
    """
    Calculate comprehensive metrics between predicted and ground truth point clouds.
    """
    if predicted.size == 0 or ground_truth.size == 0:
        raise ValueError("Empty point cloud provided")
    if predicted.shape[1] != 3 or ground_truth.shape[1] != 3:
        raise ValueError("Point clouds must have shape (N, 3)")
        
    metrics = {}
    if use_kdtree:
        tree_gt = cKDTree(ground_truth)
        tree_pred = cKDTree(predicted)
        d1_distances, _ = tree_gt.query(predicted, k=1)
        d2_distances, _ = tree_pred.query(ground_truth, k=1)
    else:
        d1_distances = compute_point_to_point_distances(predicted, ground_truth)
        d2_distances = compute_point_to_point_distances(ground_truth, predicted)
    
    metrics['d1'] = np.mean(d1_distances)
    metrics['d2'] = np.mean(d2_distances)
    metrics['chamfer'] = metrics['d1'] + metrics['d2']
    
    if predicted_normals is not None and ground_truth_normals is not None:
        if use_kdtree:
            _, indices_gt = tree_gt.query(predicted, k=1)
            _, indices_pred = tree_pred.query(ground_truth, k=1)
            n1_distances = np.abs(np.sum(
                (predicted - ground_truth[indices_gt]) * ground_truth_normals[indices_gt],
                axis=1
            ))
            n2_distances = np.abs(np.sum(
                (ground_truth - predicted[indices_pred]) * predicted_normals[indices_pred],
                axis=1
            ))
        else:
            n1_distances = compute_point_to_normal_distances(predicted, ground_truth, ground_truth_normals)
            n2_distances = compute_point_to_normal_distances(ground_truth, predicted, predicted_normals)
        metrics['n1'] = np.mean(n1_distances)
        metrics['n2'] = np.mean(n2_distances)
        metrics['normal_chamfer'] = metrics['n1'] + metrics['n2']
    
    return metrics

def calculate_chamfer_distance(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Wrapper to calculate only the Chamfer Distance using calculate_metrics.
    """
    metrics = calculate_metrics(predicted, target)
    return metrics["chamfer"]

def calculate_d1_metric(predicted: np.ndarray, target: np.ndarray) -> float:
    """
    Wrapper to calculate only the D1 Metric using calculate_metrics.
    """
    metrics = calculate_metrics(predicted, target)
    return metrics["d1"]

if __name__ == "__main__":
    N, M = 1000, 1200
    predicted_pc = np.random.rand(N, 3).astype(np.float32)
    ground_truth_pc = np.random.rand(M, 3).astype(np.float32)
    
    predicted_normals = np.random.randn(N, 3).astype(np.float32)
    predicted_normals /= np.linalg.norm(predicted_normals, axis=1, keepdims=True)
    
    ground_truth_normals = np.random.randn(M, 3).astype(np.float32)
    ground_truth_normals /= np.linalg.norm(ground_truth_normals, axis=1, keepdims=True)
    
    metrics_kdtree = calculate_metrics(
        predicted_pc, 
        ground_truth_pc,
        predicted_normals,
        ground_truth_normals,
        use_kdtree=True
    )
    
    print("\nMetrics using KD-tree:")
    for metric, value in metrics_kdtree.items():
        print(f"{metric}: {value:.6f}")
