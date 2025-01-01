import tensorflow as tf
import pytest
from pathlib import Path
from test_utils import create_mock_point_cloud, setup_test_environment
from compress_octree import OctreeCompressor

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
        base_points = create_mock_point_cloud(1000)
        
        # Add corner points
        corners = tf.constant([
            [0., 0., 0.],  # Origin
            [10., 0., 0.], # X-axis
            [0., 10., 0.], # Y-axis
            [0., 0., 10.], # Z-axis
            [10., 10., 0.],
            [10., 0., 10.],
            [0., 10., 10.],
            [10., 10., 10.]  # Maximum corner
        ], dtype=tf.float32)
        
        self.point_cloud = tf.concat([base_points, corners], axis=0)
        
        # Create corresponding normals
        self.normals = tf.random.normal([tf.shape(self.point_cloud)[0], 3])
        self.normals = self.normals / tf.norm(self.normals, axis=1, keepdims=True)

    def test_grid_shape(self):
        """Test voxel grid shape."""
        grid, _ = self.compressor.compress(self.point_cloud)
        self.assertEqual(grid.shape, (64, 64, 64))
        self.assertEqual(grid.dtype, tf.bool)

    @tf.function
    def test_compress_decompress(self):
        """Test compression and decompression without normals."""
        grid, metadata = self.compressor.compress(self.point_cloud)
        decompressed_pc, _ = self.compressor.decompress(grid, metadata)

        # Test bounds preservation
        self.assertAllClose(
            tf.reduce_min(decompressed_pc, axis=0),
            tf.reduce_min(self.point_cloud, axis=0),
            atol=0.1
        )
        self.assertAllClose(
            tf.reduce_max(decompressed_pc, axis=0),
            tf.reduce_max(self.point_cloud, axis=0),
            atol=0.1
        )

    @tf.function
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
        norms = tf.norm(decompressed_normals, axis=1)
        self.assertAllClose(norms, tf.ones_like(norms), atol=1e-6)

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        # Create batch
        batch_size = 4
        point_clouds = tf.stack([self.point_cloud] * batch_size)
        normals_batch = tf.stack([self.normals] * batch_size)
        
        # Test compression
        grid_batch, metadata_batch = self.compressor.compress(
            point_clouds,
            normals=normals_batch
        )
        
        self.assertEqual(grid_batch.shape[0], batch_size)
        self.assertEqual(len(metadata_batch), batch_size)
        
        # Test decompression
        decompressed_batch, normals_batch = self.compressor.decompress(
            grid_batch,
            metadata_batch,
            return_normals=True
        )
        
        self.assertEqual(decompressed_batch.shape[0], batch_size)
        self.assertEqual(normals_batch.shape[0], batch_size)

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
            
            # Verify points are within bounds
            self.assertTrue(tf.reduce_all(points >= min_bound))
            self.assertTrue(tf.reduce_all(points <= max_bound))
            
            # Check block constraints
            self.assertLessEqual(tf.shape(points)[0], 100)
            block_size = tf.reduce_min(max_bound - min_bound)
            self.assertGreaterEqual(block_size, 0.5)
            
            total_points += tf.shape(points)[0]
        
        # Verify all points are accounted for
        self.assertEqual(total_points, tf.shape(self.point_cloud)[0])

    def test_save_and_load(self):
        """Test saving and loading functionality."""
        save_path = Path(self.test_env['tmp_path']) / "test_compressed.tfrecord"
        
        # Compress and save
        grid, metadata = self.compressor.compress(
            self.point_cloud,
            normals=self.normals
        )
        self.compressor.save_compressed(grid, metadata, str(save_path))
        
        # Verify files exist
        self.assertTrue(save_path.exists())
        self.assertTrue(save_path.with_suffix('.tfrecord.debug').exists())
        
        # Load and verify
        loaded_grid, loaded_metadata = self.compressor.load_compressed(str(save_path))
        
        # Check equality
        self.assertAllEqual(grid, loaded_grid)
        
        # Check metadata
        for key in ['min_bounds', 'max_bounds', 'ranges', 'has_normals']:
            self.assertIn(key, loaded_metadata)
            if isinstance(metadata[key], tf.Tensor):
                self.assertAllClose(metadata[key], loaded_metadata[key])
            else:
                self.assertEqual(metadata[key], loaded_metadata[key])

    def test_error_handling(self):
        """Test error handling."""
        # Test empty point cloud
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "Empty point cloud"):
            self.compressor.compress(tf.zeros((0, 3), dtype=tf.float32))
        
        # Test single point
        single_point = tf.constant([[5.0, 5.0, 5.0]], dtype=tf.float32)
        grid, metadata = self.compressor.compress(single_point)
        decompressed, _ = self.compressor.decompress(grid, metadata)
        self.assertTrue(
            tf.reduce_any(tf.norm(decompressed - single_point, axis=1) < 0.15)
        )
        
        # Test normals shape mismatch
        wrong_shape_normals = tf.random.normal((10, 3))
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "Shape mismatch"):
            self.compressor.compress(self.point_cloud, normals=wrong_shape_normals)

if __name__ == "__main__":
    tf.test.main()