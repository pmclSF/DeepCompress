import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from compress_octree import OctreeCompressor
from test_utils import create_mock_point_cloud, setup_test_environment


class TestOctreeCompressor(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment."""
        self.test_env = setup_test_environment(tmp_path)
        self.compressor = OctreeCompressor(
            resolution=64,
            debug_output=True,
            output_dir=str(tmp_path)
        )

        # Create test point cloud with corners for boundary testing
        base_points = create_mock_point_cloud(1000).numpy()

        # Add corner points
        corners = np.array([
            [0., 0., 0.],
            [10., 0., 0.],
            [0., 10., 0.],
            [0., 0., 10.],
            [10., 10., 0.],
            [10., 0., 10.],
            [0., 10., 10.],
            [10., 10., 10.]
        ], dtype=np.float32)

        self.point_cloud = np.concatenate([base_points, corners], axis=0)

        # Create corresponding normals
        normals = np.random.randn(len(self.point_cloud), 3).astype(np.float32)
        self.normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    def test_grid_shape(self):
        """Test voxel grid shape."""
        grid, _ = self.compressor.compress(self.point_cloud)
        self.assertEqual(grid.shape, (64, 64, 64))
        self.assertEqual(grid.dtype, bool)

    def test_compress_decompress(self):
        """Test compression and decompression without normals."""
        grid, metadata = self.compressor.compress(self.point_cloud)
        decompressed_pc, _ = self.compressor.decompress(grid, metadata)

        # Test bounds preservation
        np.testing.assert_allclose(
            np.min(decompressed_pc, axis=0),
            np.min(self.point_cloud, axis=0),
            atol=0.2
        )
        np.testing.assert_allclose(
            np.max(decompressed_pc, axis=0),
            np.max(self.point_cloud, axis=0),
            atol=0.2
        )

    def test_normal_preservation(self):
        """Test compression and decompression with normals."""
        grid, metadata = self.compressor.compress(
            self.point_cloud,
            normals=self.normals
        )

        self.assertTrue(metadata['has_normals'])
        self.assertIn('normal_grid', metadata)

        decompressed_pc, decompressed_normals = self.compressor.decompress(
            grid,
            metadata,
            return_normals=True
        )

        # Check normal vectors are unit length
        norms = np.linalg.norm(decompressed_normals, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    def test_octree_partitioning(self):
        """Test octree partitioning functionality."""
        blocks = self.compressor.partition_octree(
            self.point_cloud,
            max_points_per_block=100,
            min_block_size=0.5
        )

        total_points = 0
        for points, metadata in blocks:
            # Check block bounds
            min_bound, max_bound = metadata['bounds']

            # Verify points are within bounds (with epsilon)
            self.assertTrue(np.all(points >= min_bound - 1e-9))
            self.assertTrue(np.all(points <= max_bound + 1e-9))

            # Check block constraints
            self.assertLessEqual(len(points), 100)
            block_size = np.min(max_bound - min_bound)
            self.assertGreaterEqual(block_size, 0.5)

            total_points += len(points)

        # Verify all points are accounted for
        self.assertEqual(total_points, len(self.point_cloud))

    def test_save_and_load(self):
        """Test saving and loading functionality."""
        save_path = Path(self.test_env['tmp_path']) / "test_compressed.npz"
        meta_path = Path(str(save_path) + '.meta.json')

        # Compress and save
        grid, metadata = self.compressor.compress(
            self.point_cloud,
            normals=self.normals
        )
        self.compressor.save_compressed(grid, metadata, str(save_path))

        # Verify both files exist
        self.assertTrue(save_path.exists())
        self.assertTrue(meta_path.exists())

        # Load and verify
        loaded_grid, loaded_metadata = self.compressor.load_compressed(str(save_path))

        # Check equality
        np.testing.assert_array_equal(grid, loaded_grid)

        # Check metadata
        for key in ['min_bounds', 'max_bounds', 'ranges', 'has_normals']:
            self.assertIn(key, loaded_metadata)

        # Check array fields are numpy arrays after load
        for key in ['min_bounds', 'max_bounds', 'ranges']:
            self.assertIsInstance(loaded_metadata[key], np.ndarray)

    def test_error_handling(self):
        """Test error handling."""
        # Test empty point cloud
        with self.assertRaisesRegex(ValueError, "Empty point cloud"):
            self.compressor.compress(np.zeros((0, 3), dtype=np.float32))

        # Test single point
        single_point = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        grid, metadata = self.compressor.compress(single_point)
        decompressed, _ = self.compressor.decompress(grid, metadata)
        self.assertTrue(
            np.any(np.linalg.norm(decompressed - single_point, axis=1) < 0.2)
        )

        # Test normals shape mismatch
        wrong_shape_normals = np.random.randn(10, 3).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "shape must match"):
            self.compressor.compress(self.point_cloud, normals=wrong_shape_normals)

if __name__ == "__main__":
    tf.test.main()
