import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
from typing import Tuple, Optional, Dict

@njit(parallel=True)
def compute_point_to_point_distances(points1: np.ndarray, 
                                   points2: np.ndarray) -> np.ndarray:
    """
    Compute point-to-point distances using parallel Numba optimization.
    
    Args:
        points1: First point cloud (N, 3)
        points2: Second point cloud (M, 3)
        
    Returns:
        Distances array (N,)
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
    
    Args:
        points1: First point cloud (N, 3)
        points2: Second point cloud (M, 3)
        normals2: Normals for second point cloud (M, 3)
        
    Returns:
        Distances array (N,)
    """
    N = points1.shape[0]
    M = points2.shape[0]
    distances = np.empty(N, dtype=np.float32)
    
    for i in prange(N):
        min_dist = np.inf
        for j in range(M):
            # Vector from point2 to point1
            vec = points1[i] - points2[j]
            # Point to plane distance
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
    
    Args:
        predicted: Predicted point cloud (N, 3)
        ground_truth: Ground truth point cloud (M, 3)
        predicted_normals: Optional normals for predicted points (N, 3)
        ground_truth_normals: Optional normals for ground truth points (M, 3)
        use_kdtree: Whether to use KD-tree for faster neighbor search
        
    Returns:
        Dictionary containing various metrics
    """
    # Input validation
    if predicted.size == 0 or ground_truth.size == 0:
        raise ValueError("Empty point cloud provided")
        
    if predicted.shape[1] != 3 or ground_truth.shape[1] != 3:
        raise ValueError("Point clouds must have shape (N, 3)")
        
    # Initialize results dictionary
    metrics = {}
    
    # Compute point-to-point distances
    if use_kdtree:
        # Use KD-tree for larger point clouds
        tree_gt = cKDTree(ground_truth)
        tree_pred = cKDTree(predicted)
        
        d1_distances, _ = tree_gt.query(predicted, k=1)
        d2_distances, _ = tree_pred.query(ground_truth, k=1)
    else:
        # Use Numba-optimized computation for smaller point clouds
        d1_distances = compute_point_to_point_distances(predicted, ground_truth)
        d2_distances = compute_point_to_point_distances(ground_truth, predicted)
    
    # Point-to-point metrics
    metrics['d1'] = np.mean(d1_distances)
    metrics['d2'] = np.mean(d2_distances)
    metrics['chamfer'] = metrics['d1'] + metrics['d2']
    
    # Compute normal-based metrics if normals are provided
    if predicted_normals is not None and ground_truth_normals is not None:
        if use_kdtree:
            # Find closest points first using KD-tree
            _, indices_gt = tree_gt.query(predicted, k=1)
            _, indices_pred = tree_pred.query(ground_truth, k=1)
            
            # Compute normal distances using closest point normals
            n1_distances = np.abs(np.sum(
                (predicted - ground_truth[indices_gt]) * ground_truth_normals[indices_gt],
                axis=1
            ))
            n2_distances = np.abs(np.sum(
                (ground_truth - predicted[indices_pred]) * predicted_normals[indices_pred],
                axis=1
            ))
        else:
            n1_distances = compute_point_to_normal_distances(
                predicted, ground_truth, ground_truth_normals
            )
            n2_distances = compute_point_to_normal_distances(
                ground_truth, predicted, predicted_normals
            )
        
        # Normal-based metrics
        metrics['n1'] = np.mean(n1_distances)
        metrics['n2'] = np.mean(n2_distances)
        metrics['normal_chamfer'] = metrics['n1'] + metrics['n2']
    
    return metrics

if __name__ == "__main__":
    # Example usage
    N, M = 1000, 1200
    predicted_pc = np.random.rand(N, 3).astype(np.float32)
    ground_truth_pc = np.random.rand(M, 3).astype(np.float32)
    
    # Generate random unit normals
    predicted_normals = np.random.randn(N, 3).astype(np.float32)
    predicted_normals /= np.linalg.norm(predicted_normals, axis=1, keepdims=True)
    
    ground_truth_normals = np.random.randn(M, 3).astype(np.float32)
    ground_truth_normals /= np.linalg.norm(ground_truth_normals, axis=1, keepdims=True)
    
    # Calculate metrics
    metrics_kdtree = calculate_metrics(
        predicted_pc, 
        ground_truth_pc,
        predicted_normals,
        ground_truth_normals,
        use_kdtree=True
    )
    
    metrics_numba = calculate_metrics(
        predicted_pc, 
        ground_truth_pc,
        predicted_normals,
        ground_truth_normals,
        use_kdtree=False
    )
    
    print("\nMetrics using KD-tree:")
    for metric, value in metrics_kdtree.items():
        print(f"{metric}: {value:.6f}")
        
    print("\nMetrics using Numba:")
    for metric, value in metrics_numba.items():
        print(f"{metric}: {value:.6f}")