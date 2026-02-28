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

    # --- NaN / Inf / degenerate value tests ---

    def test_save_load_metadata_with_nan_and_inf(self):
        """NaN and Inf scalar values in metadata are converted to None."""
        save_path = Path(self.test_env['tmp_path']) / "special_values.npz"
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[0, 0, 0] = True
        metadata = {
            'min_bounds': np.array([0.0, 0.0, 0.0]),
            'max_bounds': np.array([1.0, 1.0, 1.0]),
            'ranges': np.array([1.0, 1.0, 1.0]),
            'has_normals': False,
            'nan_value': float('nan'),
            'inf_value': float('inf'),
            'neg_inf_value': float('-inf'),
        }
        self.compressor.save_compressed(grid, metadata, str(save_path))
        _, loaded = self.compressor.load_compressed(str(save_path))
        self.assertIsNone(loaded['nan_value'])
        self.assertIsNone(loaded['inf_value'])
        self.assertIsNone(loaded['neg_inf_value'])

    def test_save_load_metadata_with_numpy_nan(self):
        """NaN from np.floating scalar is also converted to None."""
        save_path = Path(self.test_env['tmp_path']) / "np_nan.npz"
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[0, 0, 0] = True
        metadata = {
            'min_bounds': np.array([0.0, 0.0, 0.0]),
            'max_bounds': np.array([1.0, 1.0, 1.0]),
            'ranges': np.array([1.0, 1.0, 1.0]),
            'has_normals': False,
            'compression_error': np.float64('nan'),
        }
        self.compressor.save_compressed(grid, metadata, str(save_path))
        _, loaded = self.compressor.load_compressed(str(save_path))
        self.assertIsNone(loaded['compression_error'])

    def test_compress_all_points_same_voxel(self):
        """All identical points compress to single occupied voxel."""
        same_points = np.full((100, 3), 5.0, dtype=np.float32)
        grid, metadata = self.compressor.compress(same_points, validate=False)
        self.assertEqual(np.sum(grid), 1)
        np.testing.assert_allclose(metadata['ranges'], [1e-6, 1e-6, 1e-6])

    # --- Zero / empty / boundary tests ---

    def test_save_load_empty_grid(self):
        """All-False grid saves and loads correctly."""
        save_path = Path(self.test_env['tmp_path']) / "empty_grid.npz"
        grid = np.zeros((64, 64, 64), dtype=bool)
        metadata = {
            'min_bounds': np.array([0.0, 0.0, 0.0]),
            'max_bounds': np.array([1.0, 1.0, 1.0]),
            'ranges': np.array([1.0, 1.0, 1.0]),
            'has_normals': False,
        }
        self.compressor.save_compressed(grid, metadata, str(save_path))
        loaded_grid, loaded_metadata = self.compressor.load_compressed(str(save_path))
        self.assertEqual(np.sum(loaded_grid), 0)
        self.assertFalse(loaded_metadata['has_normals'])

    def test_save_load_without_normals(self):
        """Metadata without normal_grid round-trips correctly."""
        save_path = Path(self.test_env['tmp_path']) / "no_normals.npz"
        grid, metadata = self.compressor.compress(self.point_cloud, validate=False)
        self.assertFalse(metadata['has_normals'])
        self.assertNotIn('normal_grid', metadata)

        self.compressor.save_compressed(grid, metadata, str(save_path))
        loaded_grid, loaded_metadata = self.compressor.load_compressed(str(save_path))
        np.testing.assert_array_equal(grid, loaded_grid)
        self.assertFalse(loaded_metadata['has_normals'])
        self.assertNotIn('normal_grid', loaded_metadata)

    # --- Negative / error path tests ---

    def test_load_compressed_missing_metadata_file(self):
        """Missing .meta.json sidecar raises FileNotFoundError."""
        save_path = Path(self.test_env['tmp_path']) / "partial_write.npz"
        grid = np.zeros((64, 64, 64), dtype=bool)
        metadata = {
            'min_bounds': np.array([0.0, 0.0, 0.0]),
            'max_bounds': np.array([1.0, 1.0, 1.0]),
            'ranges': np.array([1.0, 1.0, 1.0]),
            'has_normals': False,
        }
        self.compressor.save_compressed(grid, metadata, str(save_path))

        # Simulate partial write: delete the sidecar
        meta_path = Path(str(save_path) + '.meta.json')
        meta_path.unlink()

        with self.assertRaises(FileNotFoundError):
            self.compressor.load_compressed(str(save_path))

    def test_load_compressed_missing_grid_file(self):
        """Missing .npz grid file raises error."""
        missing_path = Path(self.test_env['tmp_path']) / "nonexistent.npz"
        with self.assertRaises(FileNotFoundError):
            self.compressor.load_compressed(str(missing_path))

    # --- Debug output security test ---

    def test_debug_info_does_not_pickle_dicts(self):
        """Debug output skips dict values, only saves numpy arrays."""
        self.compressor.compress(self.point_cloud, validate=False)

        debug_dir = Path(self.test_env['tmp_path']) / 'debug' / 'grid_creation'
        self.assertTrue(debug_dir.exists())

        # 'metadata' (a dict) should NOT be saved as .npy
        self.assertFalse((debug_dir / 'metadata.npy').exists())

        # 'grid' and 'scaled_points' (arrays) SHOULD be saved
        self.assertTrue((debug_dir / 'grid.npy').exists())
        self.assertTrue((debug_dir / 'scaled_points.npy').exists())

        # All saved .npy files must be loadable without pickle
        for npy_file in debug_dir.glob('*.npy'):
            np.load(str(npy_file), allow_pickle=False)

    # --- Regression / format fidelity tests ---

    def test_save_load_metadata_values_roundtrip(self):
        """Numeric metadata values are preserved after JSON round-trip."""
        save_path = Path(self.test_env['tmp_path']) / "fidelity.npz"
        grid, metadata = self.compressor.compress(self.point_cloud)
        self.compressor.save_compressed(grid, metadata, str(save_path))
        _, loaded = self.compressor.load_compressed(str(save_path))

        np.testing.assert_allclose(
            loaded['min_bounds'], metadata['min_bounds'], rtol=1e-6
        )
        np.testing.assert_allclose(
            loaded['max_bounds'], metadata['max_bounds'], rtol=1e-6
        )
        np.testing.assert_allclose(
            loaded['ranges'], metadata['ranges'], rtol=1e-6
        )
        self.assertAlmostEqual(
            loaded['compression_error'], metadata['compression_error'], places=6
        )

    def test_save_load_numpy_scalar_metadata(self):
        """np.float64 and np.int32 scalars survive type conversion."""
        save_path = Path(self.test_env['tmp_path']) / "scalar_types.npz"
        grid = np.zeros((64, 64, 64), dtype=bool)
        grid[0, 0, 0] = True
        metadata = {
            'min_bounds': np.array([0.0, 0.0, 0.0]),
            'max_bounds': np.array([1.0, 1.0, 1.0]),
            'ranges': np.array([1.0, 1.0, 1.0]),
            'has_normals': False,
            'float_scalar': np.float64(3.14),
            'int_scalar': np.int32(42),
        }
        self.compressor.save_compressed(grid, metadata, str(save_path))
        _, loaded = self.compressor.load_compressed(str(save_path))
        self.assertAlmostEqual(loaded['float_scalar'], 3.14, places=10)
        self.assertEqual(loaded['int_scalar'], 42)

    def test_save_load_dtype_after_roundtrip(self):
        """Documents that float32 arrays become float64 after JSON round-trip."""
        save_path = Path(self.test_env['tmp_path']) / "dtype_test.npz"
        grid, metadata = self.compressor.compress(self.point_cloud, validate=False)
        # Original is float32 from np.min on float32 input
        self.assertEqual(metadata['min_bounds'].dtype, np.float32)

        self.compressor.save_compressed(grid, metadata, str(save_path))
        _, loaded = self.compressor.load_compressed(str(save_path))
        # After JSON round-trip, np.array() defaults to float64
        self.assertEqual(loaded['min_bounds'].dtype, np.float64)

    def test_decompress_after_save_load_matches_direct(self):
        """Decompress from loaded metadata produces same points as from original."""
        save_path = Path(self.test_env['tmp_path']) / "roundtrip_quality.npz"
        grid, metadata = self.compressor.compress(self.point_cloud, validate=False)

        # Decompress directly from original metadata
        direct_points, _ = self.compressor.decompress(grid, metadata)

        # Save, load, decompress
        self.compressor.save_compressed(grid, metadata, str(save_path))
        loaded_grid, loaded_metadata = self.compressor.load_compressed(str(save_path))
        loaded_points, _ = self.compressor.decompress(loaded_grid, loaded_metadata)

        # Points should match despite dtype change (float32 vs float64)
        np.testing.assert_allclose(
            loaded_points, direct_points.astype(np.float64), rtol=1e-5
        )

    # --- E2E test ---

    @pytest.mark.e2e
    def test_compress_save_load_decompress_quality(self):
        """Full pipeline: compress, save, load, decompress, verify quality."""
        save_path = Path(self.test_env['tmp_path']) / "e2e.npz"

        grid, metadata = self.compressor.compress(self.point_cloud)
        original_error = metadata['compression_error']
        self.compressor.save_compressed(grid, metadata, str(save_path))

        loaded_grid, loaded_metadata = self.compressor.load_compressed(str(save_path))
        decompressed, _ = self.compressor.decompress(loaded_grid, loaded_metadata)

        # Decompressed point count should be reasonable
        self.assertGreater(len(decompressed), 0)
        # Reconstruction error should match original
        self.assertAlmostEqual(
            loaded_metadata['compression_error'], original_error, places=6
        )

if __name__ == "__main__":
    tf.test.main()
