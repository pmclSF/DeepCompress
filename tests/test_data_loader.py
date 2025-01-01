import tensorflow as tf
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from data_loader import DataLoader

class TestDataLoader:
    @pytest.fixture
    def config(self):
        return {
            'data': {
                'modelnet40_path': 'test_data/modelnet40',
                'ivfb_path': 'test_data/8ivfb'
            },
            'resolution': 64,
            'block_size': 1.0,
            'min_points': 100,
            'augment': True
        }
        
    @pytest.fixture
    def sample_point_cloud(self):
        # Create a simple cube point cloud
        points = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    @pytest.fixture
    def data_loader(self, config):
        return DataLoader(config)

    def test_normalize_points(self, data_loader, sample_point_cloud):
        normalized = data_loader._normalize_points(
            tf.convert_to_tensor(sample_point_cloud)
        )
        
        # Check shape
        assert normalized.shape == sample_point_cloud.shape
        
        # Check normalization range
        assert tf.reduce_min(normalized) >= -1
        assert tf.reduce_max(normalized) <= 1
        
        # Check center
        center = tf.reduce_mean(normalized, axis=0)
        tf.debugging.assert_near(center, tf.zeros_like(center), atol=1e-5)

    def test_voxelize_points(self, data_loader, sample_point_cloud):
        resolution = 32
        voxelized = data_loader._voxelize_points(
            tf.convert_to_tensor(sample_point_cloud),
            resolution
        )
        
        # Check shape
        assert voxelized.shape == (resolution, resolution, resolution)
        
        # Check binary values
        unique_values = tf.unique(tf.reshape(voxelized, [-1]))[0]
        tf.debugging.assert_equal(
            tf.sort(unique_values),
            tf.convert_to_tensor([0.0, 1.0], dtype=tf.float32)
        )

    def test_process_point_cloud(self, data_loader, sample_point_cloud):
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.ply') as tmp:
            # Save sample point cloud
            with open(tmp.name, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(sample_point_cloud)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                for point in sample_point_cloud:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
            
            # Process point cloud
            processed = data_loader.process_point_cloud(
                tf.constant(tmp.name)
            )
            
            # Check output shape
            assert processed.shape == (
                data_loader.config['resolution'],
                data_loader.config['resolution'],
                data_loader.config['resolution']
            )

    def test_augment(self, data_loader, sample_point_cloud):
        augmented = data_loader._augment(
            tf.convert_to_tensor(sample_point_cloud)
        )
        
        # Check shape preservation
        assert augmented.shape == sample_point_cloud.shape
        
        # Check that points were modified
        assert not np.allclose(augmented, sample_point_cloud)
        
        # Check bounds
        assert tf.reduce_max(tf.abs(augmented)) <= 1.1  # Allow for small jitter

    def test_load_training_data(self, data_loader, tmp_path):
        # Create sample training data
        train_path = tmp_path / "modelnet40"
        train_path.mkdir()
        
        # Create sample point cloud files
        for i in range(3):
            with open(train_path / f"sample_{i}.off", 'w') as f:
                f.write("OFF\n")
                f.write("8 6 0\n")
                # Write cube vertices
                for point in self.sample_point_cloud():
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")
                # Write faces
                f.write("3 0 1 2\n" * 6)
        
        # Update config path
        data_loader.config['data']['modelnet40_path'] = str(train_path)
        
        # Load dataset
        dataset = data_loader.load_training_data()
        
        # Check dataset properties
        assert isinstance(dataset, tf.data.Dataset)
        
        # Check first batch
        batch = next(iter(dataset))
        assert batch.shape[0] == data_loader.config.get('batch_size', 32)
        assert batch.shape[1:] == (
            data_loader.config['resolution'],
            data_loader.config['resolution'],
            data_loader.config['resolution']
        )