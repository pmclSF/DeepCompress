import sys
import tensorflow as tf
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from point_cloud_metrics import calculate_metrics, calculate_chamfer_distance, calculate_d1_metric
from test_utils import create_mock_point_cloud

class TestPointCloudMetrics(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.predicted = create_mock_point_cloud(1000)
        self.ground_truth = create_mock_point_cloud(1000)

        self.predicted_normals = tf.random.normal((1000, 3))
        self.predicted_normals = self.predicted_normals / tf.norm(
            self.predicted_normals, axis=1, keepdims=True)

        self.ground_truth_normals = tf.random.normal((1000, 3))
        self.ground_truth_normals = self.ground_truth_normals / tf.norm(
            self.ground_truth_normals, axis=1, keepdims=True)

    def test_empty_input(self):
        empty_pc = tf.zeros((0, 3), dtype=tf.float32)

        with self.assertRaisesRegex(ValueError, "Empty point cloud"):
            calculate_metrics(empty_pc, self.ground_truth)
        with self.assertRaisesRegex(ValueError, "Empty point cloud"):
            calculate_metrics(self.predicted, empty_pc)

    def test_invalid_shape(self):
        invalid_pc = tf.random.uniform((10, 2))

        with self.assertRaisesRegex(ValueError, "must have shape"):
            calculate_metrics(invalid_pc, self.ground_truth)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            calculate_metrics(self.predicted, invalid_pc)

    def test_point_metrics_basic(self):
        metrics = calculate_metrics(self.predicted, self.ground_truth)

        required_metrics = {'d1', 'd2', 'chamfer'}
        self.assertTrue(required_metrics.issubset(set(metrics.keys())))

        for metric in required_metrics:
            self.assertGreater(metrics[metric], 0)
            self.assertTrue(np.isfinite(metrics[metric]))

    def test_normal_metrics(self):
        metrics = calculate_metrics(
            self.predicted,
            self.ground_truth,
            predicted_normals=self.predicted_normals,
            ground_truth_normals=self.ground_truth_normals
        )

        normal_metrics = {'n1', 'n2', 'normal_chamfer'}
        self.assertTrue(normal_metrics.issubset(set(metrics.keys())))

        for metric in normal_metrics:
            self.assertGreater(metrics[metric], 0)
            self.assertTrue(np.isfinite(metrics[metric]))

        np.testing.assert_allclose(
            metrics['normal_chamfer'], metrics['n1'] + metrics['n2']
        )

    def test_chamfer_distance(self):
        distance = calculate_chamfer_distance(self.predicted, self.ground_truth)
        self.assertGreater(distance, 0)
        self.assertTrue(np.isfinite(distance))

        identical_distance = calculate_chamfer_distance(self.predicted, self.predicted)
        self.assertNear(identical_distance, 0, 1e-5)

    def test_d1_metric(self):
        d1 = calculate_d1_metric(self.predicted, self.ground_truth)
        self.assertGreater(d1, 0)
        self.assertTrue(np.isfinite(d1))

        identical_d1 = calculate_d1_metric(self.predicted, self.predicted)
        self.assertNear(identical_d1, 0, 1e-5)

if __name__ == '__main__':
    tf.test.main()
