import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


class OctreeCompressor:
    def __init__(self, resolution: int = 64, debug_output: bool = False, output_dir: Optional[str] = None):
        self.resolution = resolution
        self.debug_output = debug_output
        self.output_dir = output_dir
        if debug_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def _create_voxel_grid(
        self, point_cloud: np.ndarray, normals: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Convert point cloud to voxel grid with enhanced metadata."""
        grid = np.zeros((self.resolution,) * 3, dtype=bool)

        # Calculate bounds with epsilon to avoid division by zero
        min_bounds = np.min(point_cloud, axis=0)
        max_bounds = np.max(point_cloud, axis=0)
        ranges = max_bounds - min_bounds
        ranges = np.where(ranges == 0, 1e-6, ranges)

        # Scale points to grid resolution
        scaled_points = (point_cloud - min_bounds) / ranges * (self.resolution - 1)
        indices = np.clip(scaled_points, 0, self.resolution - 1).astype(int)

        # Mark occupied voxels
        for idx in indices:
            grid[tuple(idx)] = True

        metadata = {
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
            'ranges': ranges,
            'has_normals': normals is not None
        }

        if normals is not None:
            metadata['normal_grid'] = self._create_normal_grid(indices, normals)

        if self.debug_output and self.output_dir:
            debug_data = {
                'grid': grid,
                'metadata': metadata,
                'scaled_points': scaled_points
            }
            self._save_debug_info('grid_creation', debug_data)

        return grid, metadata

    def _create_normal_grid(self, indices: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Create grid storing average normals for occupied voxels."""
        normal_grid = np.zeros((self.resolution, self.resolution, self.resolution, 3))
        normal_counts = np.zeros((self.resolution, self.resolution, self.resolution))

        # Accumulate normals in each voxel
        for idx, normal in zip(indices, normals):
            normal_grid[tuple(idx)] += normal
            normal_counts[tuple(idx)] += 1

        # Average normals where counts > 0 with handling for zero counts
        counts_expanded = np.expand_dims(normal_counts, -1)
        counts_expanded = np.where(counts_expanded == 0, 1, counts_expanded)  # Avoid division by zero
        normal_grid = normal_grid / counts_expanded

        # Normalize non-zero vectors
        norms = np.linalg.norm(normal_grid, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normal_grid = normal_grid / norms

        return normal_grid

    def compress(
        self, point_cloud: np.ndarray, normals: Optional[np.ndarray] = None,
        validate: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compress point cloud with optional normals and validation."""
        point_cloud = np.asarray(point_cloud)
        if normals is not None:
            normals = np.asarray(normals)
        if len(point_cloud) == 0:
            raise ValueError("Empty point cloud provided")

        if normals is not None and normals.shape != point_cloud.shape:
            raise ValueError("Normals shape must match point cloud shape")

        grid, metadata = self._create_voxel_grid(point_cloud, normals)

        if validate:
            decompressed, _ = self.decompress(grid, metadata)
            error = self._compute_error(decompressed, point_cloud)
            metadata['compression_error'] = float(error)
            logging.info(f"Compression error: {error:.6f}")

        return grid, metadata

    def decompress(
        self, grid: np.ndarray, metadata: Dict[str, Any], *,
        return_normals: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Decompress grid back to point cloud with optional normals.

        Args:
            grid: Compressed binary grid
            metadata: Compression metadata
            return_normals: Whether to return normals if available

        Returns:
            Tuple of points array and optional normals array
        """
        indices = np.argwhere(grid).astype(np.float32)

        # Scale points back to original space
        points = indices / (self.resolution - 1) * metadata['ranges'] + metadata['min_bounds']

        # Handle normals if present and requested
        normals = None
        if return_normals and metadata.get('has_normals') and 'normal_grid' in metadata:
            normals = metadata['normal_grid'][tuple(indices.astype(int).T)]

        return points, normals

    def partition_octree(
        self, point_cloud: np.ndarray, max_points_per_block: int = 1000,
        min_block_size: float = 0.1,
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """Partition point cloud into octree blocks."""
        point_cloud = np.asarray(point_cloud)
        blocks = []
        min_bounds = np.min(point_cloud, axis=0)
        max_bounds = np.max(point_cloud, axis=0)

        def partition_recursive(points: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> None:
            if len(points) <= max_points_per_block or np.min(bounds[1] - bounds[0]) <= min_block_size:
                if len(points) > 0:  # Only add non-empty blocks
                    blocks.append((points, {'bounds': bounds}))
                return

            mid = (bounds[0] + bounds[1]) / 2
            for octant in np.ndindex((2, 2, 2)):
                # Compute octant bounds
                min_corner = np.array([
                    bounds[0][i] if octant[i] == 0 else mid[i] for i in range(3)
                ])
                max_corner = np.array([
                    mid[i] if octant[i] == 0 else bounds[1][i] for i in range(3)
                ])

                # Find points in this octant with epsilon for stability
                epsilon = 1e-10
                mask = np.all(
                    (points >= min_corner - epsilon) &
                    (points <= max_corner + epsilon),
                    axis=1
                )
                if np.any(mask):
                    partition_recursive(points[mask], (min_corner, max_corner))

        partition_recursive(point_cloud, (min_bounds, max_bounds))
        return blocks

    def _compute_error(self, points1: np.ndarray, points2: np.ndarray) -> float:
        """Compute average distance between point sets."""
        tree = cKDTree(points2)
        distances, _ = tree.query(points1)
        return float(np.mean(distances))

    def _save_debug_info(self, stage: str, data: Dict[str, Any]) -> None:
        """Save debug information to files."""
        if not self.debug_output or not self.output_dir:
            return

        debug_dir = os.path.join(self.output_dir, 'debug', stage)
        os.makedirs(debug_dir, exist_ok=True)

        for name, array in data.items():
            if isinstance(array, (np.ndarray, dict)):
                np.save(os.path.join(debug_dir, f"{name}.npy"), array)

    def save_compressed(self, grid: np.ndarray, metadata: Dict[str, Any], filename: str) -> None:
        """Save compressed data with metadata."""
        import json

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        # Save grid without pickle (bool array, no object dtype)
        np.savez_compressed(filename, grid=grid)
        # Save metadata as JSON sidecar (safe, no arbitrary code execution)
        meta_path = filename + '.meta.json'
        serializable = {}
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serializable[k] = v.item()
            else:
                serializable[k] = v
        with open(meta_path, 'w') as f:
            json.dump(serializable, f)

    def load_compressed(self, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load compressed data with metadata."""
        import json

        data = np.load(filename, allow_pickle=False)
        grid = data['grid']
        meta_path = filename + '.meta.json'
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        # Convert lists back to numpy arrays for known array fields
        for key in ('min_bounds', 'max_bounds', 'ranges', 'normal_grid'):
            if key in metadata:
                metadata[key] = np.array(metadata[key])
        return grid, metadata
