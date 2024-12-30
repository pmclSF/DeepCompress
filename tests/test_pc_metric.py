import unittest
import numpy as np
from pc_metric import calculate_metrics, compute_point_to_point_distances, compute_point_to_normal_distances

class TestPointCloudMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create simple test point clouds
        self.predicted = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        
        self.ground_truth = np.array([
            [0.1, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.1, 0.0]
        ], dtype=np.float32)
        
        # Create corresponding normals
        self.predicted_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        self.ground_truth_normals = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def test_empty_input(self):
        """Test handling of empty point clouds."""
        empty_pc = np.array([], dtype=np.float32).reshape(0, 3)
        
        with self.assertRaises(ValueError):
            calculate_metrics(empty_pc, self.ground_truth)
        
        with self.assertRaises(ValueError):
            calculate_metrics(self.predicted, empty_pc)

    def test_invalid_shape(self):
        """Test handling of invalid point cloud shapes."""
        invalid_pc = np.random.rand(10, 2)  # Should be (N, 3)
        
        with self.assertRaises(ValueError):
            calculate_metrics(invalid_pc, self.ground_truth)
        
        with self.assertRaises(ValueError):
            calculate_metrics(self.predicted, invalid_pc)

    def test_point_metrics_kdtree(self):
        """Test point-based metrics using KD-tree."""
        metrics = calculate_metrics(
            self.predicted,
            self.ground_truth,
            use_kdtree=True
        )
        
        # Basic metric validation
        self.assertIn('d1', metrics)
        self.assertIn('d2', metrics)
        self.assertIn('chamfer', metrics)
        
        # Check that metrics are positive
        self.assertGreater(metrics['d1'], 0)
        self.assertGreater(metrics['d2'], 0)
        self.assertGreater(metrics['chamfer'], 0)
        
        # Check Chamfer distance relationship
        self.assertAlmostEqual(
            metrics['chamfer'],
            metrics['d1'] + metrics['d2']
        )

    def test_point_metrics_numba(self):
        """Test point-based metrics using Numba."""
        metrics = calculate_metrics(
            self.predicted,
            self.ground_truth,
            use_kdtree=False
        )
        
        # Compare with KD-tree results
        metrics_kdtree = calculate_metrics(
            self.predicted,
            self.ground_truth,
            use_kdtree=True
        )
        
        # Results should be close
        self.assertAlmostEqual(metrics['d1'], metrics_kdtree['d1'], places=5)
        self.assertAlmostEqual(metrics['d2'], metrics_kdtree['d2'], places=5)

    def test_normal_metrics(self):
        """Test normal-based metrics."""
        metrics = calculate_metrics(
            self.predicted,
            self.ground_truth,
            self.predicted_normals,
            self.ground_truth_normals
        )
        
        # Check normal metrics exist
        self.assertIn('n1', metrics)
        self.assertIn('n2', metrics)
        self.assertIn('normal_chamfer', metrics)
        
        # Check normal Chamfer distance relationship
        self.assertAlmostEqual(
            metrics['normal_chamfer'],
            metrics['n1'] + metrics['n2']
        )

    def test_point_to_point_distances(self):
        """Test Numba point-to-point distance computation."""
        distances = compute_point_to_point_distances(
            self.predicted,
            self.ground_truth
        )
        
        # Check shape and positivity
        self.assertEqual(distances.shape, (len(self.predicted),))
        self.assertTrue(np.all(distances >= 0))

    def test_point_to_normal_distances(self):
        """Test Numba point-to-normal distance computation."""
        distances = compute_point_to_normal_distances(
            self.predicted,
            self.ground_truth,
            self.ground_truth_normals
        )
        
        # Check shape and positivity
        self.assertEqual(distances.shape, (len(self.predicted),))
        self.assertTrue(np.all(distances >= 0))

if __name__ == '__main__':
    unittest.main()