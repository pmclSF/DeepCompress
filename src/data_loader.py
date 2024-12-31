import tensorflow as tf
import glob
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from ds_mesh_to_pc import read_off
from ds_pc_octree_blocks import PointCloudProcessor

class DataLoader:
    """Unified data loader for ModelNet40 and 8iVFB datasets."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.processor = PointCloudProcessor(
            block_size=config.get('block_size', 1.0),
            min_points=config.get('min_points', 100)
        )
        
    @tf.function
    def process_point_cloud(self, file_path: str) -> tf.Tensor:
        """Process a single point cloud file."""
        # Read point cloud
        vertices, _ = read_off(file_path.numpy().decode())
        points = tf.convert_to_tensor(vertices, dtype=tf.float32)
        
        # Normalize points to unit cube
        points = self._normalize_points(points)
        
        # Voxelize points
        resolution = self.config.get('resolution', 64)
        voxelized = self._voxelize_points(points, resolution)
        
        return voxelized
        
    @tf.function
    def _normalize_points(self, points: tf.Tensor) -> tf.Tensor:
        """Normalize points to unit cube."""
        center = tf.reduce_mean(points, axis=0)
        points = points - center
        scale = tf.reduce_max(tf.abs(points))
        points = points / scale
        return points
        
    @tf.function
    def _voxelize_points(self, 
                        points: tf.Tensor,
                        resolution: int) -> tf.Tensor:
        """Convert points to voxel grid."""
        # Scale points to voxel coordinates
        points = (points + 1) * (resolution - 1) / 2
        points = tf.clip_by_value(points, 0, resolution - 1)
        indices = tf.cast(tf.round(points), tf.int32)
        
        # Create voxel grid
        grid = tf.zeros((resolution, resolution, resolution), dtype=tf.float32)
        updates = tf.ones(tf.shape(indices)[0], dtype=tf.float32)
        grid = tf.tensor_scatter_nd_update(grid, indices, updates)
        
        return grid

    def load_training_data(self) -> tf.data.Dataset:
        """Load ModelNet40 training data."""
        train_path = Path(self.config['data']['modelnet40_path'])
        file_pattern = str(train_path / "**/*.off")
        files = glob.glob(file_pattern, recursive=True)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(
            lambda x: tf.py_function(
                self.process_point_cloud,
                [x],
                tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply training augmentations
        if self.config.get('augment', True):
            dataset = dataset.map(
                self._augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and prefetch
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(self.config['training']['batch_size'])
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    def load_evaluation_data(self) -> tf.data.Dataset:
        """Load 8iVFB evaluation data."""
        eval_path = Path(self.config['data']['ivfb_path'])
        file_pattern = str(eval_path / "*.ply")
        files = glob.glob(file_pattern)
        
        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.map(
            lambda x: tf.py_function(
                self.process_point_cloud,
                [x],
                tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch without shuffling for evaluation
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    @tf.function
    def _augment(self, points: tf.Tensor) -> tf.Tensor:
        """Apply data augmentation."""
        # Random rotation around z-axis
        angle = tf.random.uniform([], 0, 2 * np.pi)
        rotation = tf.stack([
            [tf.cos(angle), -tf.sin(angle), 0],
            [tf.sin(angle), tf.cos(angle), 0],
            [0, 0, 1]
        ])
        points = tf.matmul(points, rotation)
        
        # Random jittering
        if tf.random.uniform([]) < 0.5:
            jitter = tf.random.normal(tf.shape(points), mean=0.0, stddev=0.01)
            points = points + jitter
            
        return points