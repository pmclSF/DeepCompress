from typing import Optional

# Third party
import tensorflow as tf
import pytest
import numpy as np

# Local
from pc_metric import (
    calculate_metrics,
    calculate_chamfer_distance,
    calculate_d1_metric
)
from test_utils import create_mock_point_cloud

class TestPointCloudMetrics(tf.test.TestCase):
    """Test suite for point cloud metrics computation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test cases."""
        # Create simple test point clouds
        self.predicted = create_mock_point_cloud(1000)
        self.ground_truth = create_mock_point_cloud(1000)
        
        # Create corresponding normals
        self.predicted_normals = tf.random.normal((1000, 3))
        self.predicted_normals = self.predicted_normals / tf.norm(
            self.predicted_normals,
            axis=1,
            keepdims=True
        )
        
        self.ground_truth_normals = tf.random.normal((1000, 3))
        self.ground_truth_normals = self.ground_truth_normals / tf.norm(
            self.ground_truth_normals,
            axis=1,
            keepdims=True
        )

    def test_empty_input(self):
        """Test handling of empty point clouds."""
        empty_pc = tf.zeros((0, 3), dtype=tf.float32)
        
        with self.assertRaisesRegex(ValueError, "Empty point cloud"):
            calculate_metrics(empty_pc, self.ground_truth)
        
        with self.assertRaisesRegex(ValueError, "Empty point cloud"):
            calculate_metrics(self.predicted, empty_pc)

    def test_invalid_shape(self):
        """Test handling of invalid point cloud shapes."""
        invalid_pc = tf.random.uniform((10, 2))  # Should be (N, 3)
        
        with self.assertRaisesRegex(ValueError, "must have shape"):
            calculate_metrics(invalid_pc, self.ground_truth)
        
        with self.assertRaisesRegex(ValueError, "must have shape"):
            calculate_metrics(self.predicted, invalid_pc)

    @tf.function
    def test_point_metrics_basic(self):
        """Test basic point-based metrics computation."""
        metrics = calculate_metrics(self.predicted, self.ground_truth)
        
        # Check required metrics exist
        required_metrics = {'d1', 'd2', 'chamfer'}
        self.assertSetsEqual(
            required_metrics,
            set(metrics.keys()) & required_metrics
        )
        
        # Check metric properties
        for metric in required_metrics:
            self.assertGreater(metrics[metric], 0)
            self.assertAllFinite(metrics[metric])

    @tf.function
    def test_batch_processing(self):
        """Test processing of batched point clouds."""
        # Create batched input
        batch_size = 4
        predicted_batch = tf.stack([self.predicted] * batch_size)
        ground_truth_batch = tf.stack([self.ground_truth] * batch_size)
        
        metrics_batch = calculate_metrics(predicted_batch, ground_truth_batch)
        metrics_single = calculate_metrics(self.predicted, self.ground_truth)
        
        # Compare batch results with single results
        for key in metrics_single.keys():
            self.assertAllClose(
                metrics_batch[key],
                metrics_single[key],
                rtol=1e-5
            )

    @tf.function
    def test_normal_metrics(self):
        """Test normal-based metrics computation."""
        metrics = calculate_metrics(
            self.predicted,
            self.ground_truth,
            predicted_normals=self.predicted_normals,
            ground_truth_normals=self.ground_truth_normals
        )
        
        # Check normal metrics exist
        normal_metrics = {'n1', 'n2', 'normal_chamfer'}
        self.assertSetsEqual(
            normal_metrics,
            set(metrics.keys()) & normal_metrics
        )
        
        # Check metric properties
        for metric in normal_metrics:
            self.assertGreater(metrics[metric], 0)
            self.assertAllFinite(metrics[metric])
            
        # Check normal Chamfer relationship
        self.assertAllClose(
            metrics['normal_chamfer'],
            metrics['n1'] + metrics['n2']
        )

    def test_chamfer_distance(self):
        """Test Chamfer distance computation specifically."""
        distance = calculate_chamfer_distance(self.predicted, self.ground_truth)
        
        self.assertGreater(distance, 0)
        self.assertAllFinite(distance)
        
        # Test with identical point clouds
        identical_distance = calculate_chamfer_distance(
            self.predicted,
            self.predicted
        )
        self.assertNear(identical_distance, 0, 1e-5)

    def test_d1_metric(self):
        """Test D1 metric computation specifically."""
        d1 = calculate_d1_metric(self.predicted, self.ground_truth)
        
        self.assertGreater(d1, 0)
        self.assertAllFinite(d1)
        
        # Test with identical point clouds
        identical_d1 = calculate_d1_metric(self.predicted, self.predicted)
        self.assertNear(identical_d1, 0, 1e-5)

    @pytest.mark.integration
    def test_end_to_end(self):
        """Test end-to-end metric computation workflow."""
        # Create a sequence of transformations
        transformed_pc = tf.concat([
            self.predicted + 0.1,  # Translate
            self.predicted * 1.1,  # Scale
            self.predicted @ tf.random.uniform((3, 3))  # Rotate
        ], axis=0)
        
        # Compute metrics
        metrics = calculate_metrics(
            transformed_pc,
            self.ground_truth,
            predicted_normals=self.predicted_normals,
            ground_truth_normals=self.ground_truth_normals
        )
        
        # Verify all metrics exist and are reasonable
        expected_metrics = {
            'd1', 'd2', 'chamfer',
            'n1', 'n2', 'normal_chamfer'
        }
        self.assertSetsEqual(expected_metrics, set(metrics.keys()))
        
        for metric in metrics.values():
            self.assertGreater(metric, 0)
            self.assertAllFinite(metric)

if __name__ == '__main__':
    tf.test.main()