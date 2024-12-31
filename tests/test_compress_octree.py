import unittest
import numpy as np
import os
import tempfile
from compress_octree import OctreeCompressor

class TestOctreeCompressor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.compressor = OctreeCompressor(
            resolution=64,
            debug_output=True,
            output_dir=self.temp_dir
        )
        
        # Create base point cloud
        np.random.seed(42)  # For reproducibility
        base_points = np.random.rand(1000, 3) * 10
        
        # Add corner points for boundary testing
        corners = np.array([
            [0., 0., 0.],  # Origin
            [10., 0., 0.], # X-axis
            [0., 10., 0.], # Y-axis
            [0., 0., 10.], # Z-axis
            [10., 10., 0.],
            [10., 0., 10.],
            [0., 10., 10.],
            [10., 10., 10.]  # Maximum corner
        ])
        
        self.point_cloud = np.vstack([base_points, corners])
        
        # Create corresponding normals
        self.normals = np.random.rand(len(self.point_cloud), 3)
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_grid_shape(self):
        """Test if the voxel grid has correct shape."""
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
            atol=0.1
        )
        np.testing.assert_allclose(
            np.max(decompressed_pc, axis=0),
            np.max(self.point_cloud, axis=0),
            atol=0.1
        )

    def test_normal_preservation(self):
        """Test compression and decompression with normals."""
        grid, metadata = self.compressor.compress(
            self.point_cloud,
            normals=self.normals
        )
        
        # Verify metadata indicates normals presence
        self.assertTrue(metadata['has_normals'])
        self.assertIn('normal_grid', metadata)
        
        # Test decompression with normals
        decompressed_pc, decompressed_normals = self.compressor.decompress(
            grid,
            metadata,
            return_normals=True
        )
        
        # Check normal vectors are unit length
        norms = np.linalg.norm(decompressed_normals, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_corner_preservation(self):
        """Test if corner points are preserved accurately."""
        grid, metadata = self.compressor.compress(self.point_cloud)
        decompressed_pc, _ = self.compressor.decompress(grid, metadata)
        
        corners = np.array([
            [0., 0., 0.],
            [10., 10., 10.]
        ])
        
        for corner in corners:
            distances = np.linalg.norm(decompressed_pc - corner, axis=1)
            min_distance = np.min(distances)
            self.assertLess(
                min_distance, 
                0.15,
                f"Corner point {corner} not preserved. Closest point is {min_distance} units away"
            )

    def test_compression_validation(self):
        """Test compression validation feature."""
        grid, metadata = self.compressor.compress(
            self.point_cloud,
            validate=True
        )
        
        self.assertIn('compression_error', metadata)
        self.assertIsInstance(metadata['compression_error'], float)
        self.assertGreaterEqual(metadata['compression_error'], 0.0)

    def test_octree_partitioning(self):
        """Test octree partitioning functionality."""
        blocks = self.compressor.partition_octree(
            self.point_cloud,
            max_points_per_block=100,
            min_block_size=0.5
        )
        
        # Check blocks
        self.assertGreater(len(blocks), 1)  # Should create multiple blocks
        
        total_points = 0
        for points, metadata in blocks:
            # Check block bounds
            self.assertIn('bounds', metadata)
            min_bound, max_bound = metadata['bounds']
            
            # Verify points are within bounds
            self.assertTrue(np.all(points >= min_bound))
            self.assertTrue(np.all(points <= max_bound))
            
            # Check block size constraints
            self.assertLessEqual(len(points), 100)
            block_size = np.min(max_bound - min_bound)
            self.assertGreaterEqual(block_size, 0.5)
            
            total_points += len(points)
        
        # Verify all points are accounted for
        self.assertEqual(total_points, len(self.point_cloud))

    def test_save_and_load(self):
        """Test saving and loading functionality."""
        save_path = os.path.join(self.temp_dir, "test_compressed.npz")
        
        # Compress and save
        grid, metadata = self.compressor.compress(
            self.point_cloud,
            normals=self.normals
        )
        self.compressor.save_compressed(grid, metadata, save_path)
        
        # Verify files exist
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(save_path + ".debug.npz"))
        
        # Load and verify
        loaded_grid, loaded_metadata = self.compressor.load_compressed(save_path)
        
        # Check grid equality
        self.assertTrue(np.array_equal(grid, loaded_grid))
        
        # Check metadata
        for key in ['min_bounds', 'max_bounds', 'ranges', 'has_normals']:
            self.assertIn(key, loaded_metadata)
            if isinstance(metadata[key], np.ndarray):
                np.testing.assert_array_equal(metadata[key], loaded_metadata[key])
            else:
                self.assertEqual(metadata[key], loaded_metadata[key])

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test empty point cloud
        with self.assertRaises(ValueError):
            self.compressor.compress(np.array([]))
        
        # Test single point
        single_point = np.array([[5.0, 5.0, 5.0]])
        grid, metadata = self.compressor.compress(single_point)
        decompressed, _ = self.compressor.decompress(grid, metadata)
        self.assertTrue(
            np.any(np.linalg.norm(decompressed - single_point, axis=1) < 0.15)
        )
        
        # Test points with identical coordinates
        repeated_point = np.array([[1.0, 1.0, 1.0]])
        repeated_points = np.tile(repeated_point, (10, 1))
        grid, metadata = self.compressor.compress(repeated_points)
        decompressed, _ = self.compressor.decompress(grid, metadata)
        self.assertTrue(
            np.any(np.linalg.norm(decompressed - repeated_point, axis=1) < 0.15)
        )
        
        # Test normals shape mismatch
        wrong_shape_normals = np.random.rand(10, 3)
        with self.assertRaises(ValueError):
            self.compressor.compress(self.point_cloud, normals=wrong_shape_normals)

    def test_debug_output(self):
        """Test debug output functionality."""
        grid, metadata = self.compressor.compress(self.point_cloud)
        
        # Check debug directory structure
        debug_dir = os.path.join(self.temp_dir, 'debug', 'grid_creation')
        self.assertTrue(os.path.exists(debug_dir))
        
        # Check for expected debug files
        expected_files = {'grid.npy', 'metadata.npy', 'scaled_points.npy'}
        debug_files = set(os.listdir(debug_dir))
        self.assertTrue(expected_files.issubset(debug_files))

if __name__ == "__main__":
    unittest.main()