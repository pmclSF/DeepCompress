import unittest
import numpy as np
import os
from compress_octree import OctreeCompressor

class TestOctreeCompressor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.compressor = OctreeCompressor(resolution=64)
        
        # Create base point cloud
        np.random.seed(42)  # For reproducibility
        base_points = np.random.rand(1000, 3) * 10
        
        # Add corner points to ensure boundary testing
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
        
        # Combine base points and corners
        self.point_cloud = np.vstack([base_points, corners])

    def test_grid_shape(self):
        """Test if the voxel grid has correct shape."""
        grid, _ = self.compressor.compress(self.point_cloud)
        self.assertEqual(grid.shape, (64, 64, 64))
        self.assertEqual(grid.dtype, bool)

    def test_compress_decompress(self):
        """Test the compression and decompression process."""
        grid, metadata = self.compressor.compress(self.point_cloud)
        decompressed_pc = self.compressor.decompress(grid, metadata)

        # Print debugging information
        print("\nCompression Test Results:")
        print("Original min bounds:", np.min(self.point_cloud, axis=0))
        print("Original max bounds:", np.max(self.point_cloud, axis=0))
        print("Decompressed min bounds:", np.min(decompressed_pc, axis=0))
        print("Decompressed max bounds:", np.max(decompressed_pc, axis=0))

        # Test exact bounds preservation
        np.testing.assert_allclose(
            np.min(decompressed_pc, axis=0),
            np.min(self.point_cloud, axis=0),
            atol=0.1,
            err_msg="Minimum bounds not preserved"
        )
        
        np.testing.assert_allclose(
            np.max(decompressed_pc, axis=0),
            np.max(self.point_cloud, axis=0),
            atol=0.1,
            err_msg="Maximum bounds not preserved"
        )

    def test_corner_preservation(self):
        """Test if corner points are preserved accurately."""
        grid, metadata = self.compressor.compress(self.point_cloud)
        decompressed_pc = self.compressor.decompress(grid, metadata)
        
        corners = np.array([
            [0., 0., 0.],
            [10., 10., 10.]
        ])
        
        # For each corner point, check if there's a corresponding point in the decompressed cloud
        for corner in corners:
            # Find the closest point in the decompressed cloud
            distances = np.linalg.norm(decompressed_pc - corner, axis=1)
            min_distance = np.min(distances)
            self.assertLess(min_distance, 0.15, 
                          f"Corner point {corner} not preserved. Closest point is {min_distance} units away")

    def test_save_and_load(self):
        """Test saving and loading functionality."""
        test_filename = "test_compressed.npz"
        
        # Compress and save
        grid, metadata = self.compressor.compress(self.point_cloud)
        self.compressor.save_compressed(grid, metadata, test_filename)
        
        # Load and verify
        loaded_grid, loaded_metadata = self.compressor.load_compressed(test_filename)
        
        # Check grid equality
        self.assertTrue(np.array_equal(grid, loaded_grid))
        
        # Check metadata
        for key in ['min_bounds', 'max_bounds', 'ranges']:
            self.assertTrue(np.allclose(metadata[key], loaded_metadata[key]))
        
        # Clean up
        os.remove(test_filename)

    def test_edge_cases(self):
        """Test edge cases for robustness."""
        # Test single point
        single_point = np.array([[5.0, 5.0, 5.0]])
        grid, metadata = self.compressor.compress(single_point)
        decompressed = self.compressor.decompress(grid, metadata)
        self.assertTrue(
            np.any(np.linalg.norm(decompressed - single_point, axis=1) < 0.15),
            "Single point not preserved within tolerance"
        )
        
        # Test points with identical coordinates
        repeated_point = np.array([[1.0, 1.0, 1.0]])
        repeated_points = np.tile(repeated_point, (10, 1))
        grid, metadata = self.compressor.compress(repeated_points)
        decompressed = self.compressor.decompress(grid, metadata)
        self.assertTrue(
            np.any(np.linalg.norm(decompressed - repeated_point, axis=1) < 0.15),
            "Repeated points not handled correctly"
        )

if __name__ == "__main__":
    unittest.main()