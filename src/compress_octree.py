import tensorflow as tf
import numpy as np
from typing import Tuple, Any

# Ensure TensorFlow 2 compatibility
if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()

class OctreeCompressor:
    def __init__(self, resolution: int = 64):
        self.resolution = resolution

    def _create_voxel_grid(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Converts a point cloud into a voxel grid representation.

        Args:
            point_cloud (np.ndarray): A numpy array of shape (N, 3) representing 3D points.

        Returns:
            Tuple[np.ndarray, dict]: A 3D binary grid and grid metadata.
        """
        grid = np.zeros((self.resolution, self.resolution, self.resolution), dtype=bool)
        
        # Calculate bounds
        min_bounds = np.min(point_cloud, axis=0)
        max_bounds = np.max(point_cloud, axis=0)
        
        # Calculate the range for each dimension
        ranges = max_bounds - min_bounds
        # Avoid division by zero
        ranges = np.where(ranges == 0, 1e-6, ranges)
        
        # Scale points to [0, resolution-1]
        scaled_points = (point_cloud - min_bounds) / ranges * (self.resolution - 1)
        indices = np.clip(scaled_points, 0, self.resolution - 1).astype(int)
        
        # Mark occupied voxels
        for idx in indices:
            grid[tuple(idx)] = True
            
        grid_metadata = {
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
            'ranges': ranges
        }
        
        return grid, grid_metadata

    def compress(self, point_cloud: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Compresses a point cloud into an octree representation.

        Args:
            point_cloud (np.ndarray): A numpy array of shape (N, 3) representing 3D points.

        Returns:
            Tuple[np.ndarray, Any]: A binary grid representing the compressed octree and metadata.
        """
        grid, grid_metadata = self._create_voxel_grid(point_cloud)
        return grid, grid_metadata

    def decompress(self, grid: np.ndarray, metadata: dict) -> np.ndarray:
        """
        Decompresses an octree back into a point cloud.

        Args:
            grid (np.ndarray): A binary voxel grid representing the octree.
            metadata (dict): Metadata containing min bounds, max bounds, and ranges.

        Returns:
            np.ndarray: Decompressed point cloud.
        """
        # Get occupied voxel coordinates
        indices = np.argwhere(grid)
        
        # Convert indices back to original coordinate space
        normalized_coords = indices / (self.resolution - 1)  # Scale to [0, 1]
        decompressed_points = normalized_coords * metadata['ranges'] + metadata['min_bounds']
        
        # Ensure exact maximum bounds where needed
        max_indices = indices == (self.resolution - 1)
        for dim in range(3):
            decompressed_points[max_indices[:, dim], dim] = metadata['max_bounds'][dim]
        
        return decompressed_points

    def save_compressed(self, grid: np.ndarray, metadata: dict, filename: str):
        """
        Saves the compressed octree to a file.

        Args:
            grid (np.ndarray): Compressed octree binary grid.
            metadata (dict): Metadata to save with the grid.
            filename (str): File path to save the octree.
        """
        np.savez_compressed(filename, grid=grid, metadata=metadata)

    def load_compressed(self, filename: str) -> Tuple[np.ndarray, dict]:
        """
        Loads a compressed octree from a file.

        Args:
            filename (str): File path to load the octree.

        Returns:
            Tuple[np.ndarray, dict]: Binary grid and metadata.
        """
        data = np.load(filename, allow_pickle=True)
        return data['grid'], data['metadata'].item()

# Example usage
if __name__ == "__main__":
    # Create a random point cloud for testing
    point_cloud = np.random.rand(1000, 3) * 10  # Random point cloud

    # Initialize compressor and process point cloud
    compressor = OctreeCompressor(resolution=64)
    grid, metadata = compressor.compress(point_cloud)
    decompressed_pc = compressor.decompress(grid, metadata)

    # Print sample results and bounds comparison
    print("\nBounds Comparison:")
    print("Original min bounds:", np.min(point_cloud, axis=0))
    print("Original max bounds:", np.max(point_cloud, axis=0))
    print("Decompressed min bounds:", np.min(decompressed_pc, axis=0))
    print("Decompressed max bounds:", np.max(decompressed_pc, axis=0))

    # Verify maximum bounds preservation
    max_diff = np.abs(np.max(point_cloud, axis=0) - np.max(decompressed_pc, axis=0))
    print("\nMaximum bounds difference:", max_diff)
    assert np.all(max_diff < 0.1), "Maximum bounds not preserved within tolerance"