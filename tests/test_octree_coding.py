import sys
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
import tensorflow as tf
from octree_coding import OctreeCoder, OctreeConfig

class TestOctreeCoder(unittest.TestCase):

    def setUp(self):
        """Set up test cases with a sample point cloud."""
        self.point_cloud = tf.constant([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0]
        ], dtype=tf.float32)
        self.coder = OctreeCoder(OctreeConfig(resolution=8))

    def test_encode(self):
        """Test encoding a point cloud into a binary voxel grid."""
        grid, metadata = self.coder.encode(self.point_cloud)

        self.assertEqual(grid.shape, (8, 8, 8), "Grid shape mismatch.")
        self.assertTrue(tf.reduce_any(grid).numpy(), "Grid is empty after encoding.")
        self.assertIn("min_bounds", metadata, "Metadata missing 'min_bounds'.")
        self.assertIn("max_bounds", metadata, "Metadata missing 'max_bounds'.")
        self.assertIn("scale", metadata, "Metadata missing 'scale'.")

    def test_decode(self):
        """Test decoding a binary voxel grid back into a point cloud."""
        grid, metadata = self.coder.encode(self.point_cloud)
        decoded_points = self.coder.decode(grid, metadata)

        self.assertEqual(decoded_points.shape[1], 3, "Decoded points should have 3 dimensions.")
        self.assertTrue(tf.reduce_all(decoded_points >= metadata["min_bounds"]),
                        "Decoded points are out of bounds (below).")
        self.assertTrue(tf.reduce_all(decoded_points <= metadata["max_bounds"]),
                        "Decoded points are out of bounds (above).")

    def test_round_trip(self):
        """Test round-trip encoding and decoding consistency."""
        grid, metadata = self.coder.encode(self.point_cloud)
        decoded_points = self.coder.decode(grid, metadata)

        # Ensure the decoded points match the original point cloud closely
        for point in self.point_cloud:
            distances = tf.norm(decoded_points - point, axis=1)
            self.assertLessEqual(tf.reduce_min(distances).numpy(), 1.0,
                                 f"Point {point.numpy()} not accurately represented in decoded points.")

    def test_partition_octree(self):
        """Test partitioning a point cloud into octree blocks."""
        initial_bbox = (0, 5, 0, 5, 0, 5)  # Bounding box
        level = 2
        blocks = self.coder.partition_octree(self.point_cloud, initial_bbox, level)

        self.assertGreater(len(blocks), 0, "No blocks were created.")
        for block, bbox in blocks:
            # Get min and max bounds for x, y, z
            mins = tf.constant([bbox[0], bbox[2], bbox[4]], dtype=tf.float32)  # xmin, ymin, zmin
            maxs = tf.constant([bbox[1], bbox[3], bbox[5]], dtype=tf.float32)  # xmax, ymax, zmax

            # Check if points are within bounds (with small epsilon for floating point precision)
            epsilon = 1e-10
            self.assertTrue(tf.reduce_all(block >= (mins - epsilon)),
                            f"Block points {block.numpy()} are below the bounding box {mins.numpy()}.")
            self.assertTrue(tf.reduce_all(block <= (maxs + epsilon)),
                            f"Block points {block.numpy()} are above the bounding box {maxs.numpy()}.")

if __name__ == "__main__":
    unittest.main()
