import tensorflow as tf
import pytest
import numpy as np
from pathlib import Path
from data_loader import DataLoader

class TestDataLoader:
    @pytest.fixture
    def config(self):
        return {
            'data': {
                'modelnet40_path': 'test_data/modelnet40',
                'ivfb_path': 'test_data/8ivfb',
                'resolution': 64,
                'block_size': 1.0,
                'min_points': 100,
                'augment': True
            }
        }

    @pytest.fixture
    def data_loader(self, config):
        return DataLoader(config)

    @pytest.fixture
    def sample_point_cloud(self):
        points = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def test_normalize_points(self, data_loader, sample_point_cloud):
        normalized = data_loader._normalize_points(tf.convert_to_tensor(sample_point_cloud))
        assert normalized.shape == sample_point_cloud.shape
        assert tf.reduce_min(normalized) >= -1
        assert tf.reduce_max(normalized) <= 1
        center = tf.reduce_mean(normalized, axis=0)
        tf.debugging.assert_near(center, tf.zeros_like(center), atol=1e-5)

    def test_voxelize_points(self, data_loader, sample_point_cloud):
        resolution = 32
        voxelized = data_loader._voxelize_points(
            tf.convert_to_tensor(sample_point_cloud),
            resolution
        )
        assert voxelized.shape == (resolution, resolution, resolution)
        unique_values = tf.unique(tf.reshape(voxelized, [-1]))[0]
        tf.debugging.assert_equal(
            tf.sort(unique_values),
            tf.convert_to_tensor([0.0, 1.0], dtype=tf.float32)
        )

    def test_batch_processing(self, data_loader):
        resolution = 32
        batch_size = data_loader.config['training']['batch_size']
        dataset = data_loader.load_training_data()
        batch = next(iter(dataset))
        assert batch.shape[0] == batch_size
        assert batch.shape[1:] == (resolution,) * 3

if __name__ == '__main__':
    tf.test.main()