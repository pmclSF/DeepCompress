import tensorflow as tf
import pytest
from pathlib import Path
from test_utils import (
    create_mock_point_cloud,
    create_mock_ply_file,
    setup_test_files
)
from ds_pc_octree_blocks import (
    read_point_cloud,
    partition_point_cloud,
    save_blocks
)

class TestPointCloudOctreeBlocks(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment."""
        self.test_files = setup_test_files(tmp_path)
        self.point_cloud = create_mock_point_cloud(1000)

    @tf.function
    def test_read_point_cloud(self):
        """Test reading point cloud with TF operations."""
        points = read_point_cloud(str(self.test_files['point_cloud']))
        
        self.assertIsInstance(points, tf.Tensor)
        self.assertEqual(points.shape[1], 3)
        self.assertEqual(points.dtype, tf.float32)

    @tf.function
    def test_partition_point_cloud(self):
        """Test point cloud partitioning with TF operations."""
        # Test partitioning
        blocks = partition_point_cloud(
            self.point_cloud,
            block_size=0.5,
            min_points=50
        )
        
        # Verify blocks
        total_points = 0
        for block in blocks:
            # Check dimensionality
            self.assertEqual(block.shape[1], 3)
            
            # Check minimum points constraint
            self.assertGreaterEqual(block.shape[0], 50)
            
            # Verify points are within block bounds
            block_min = tf.reduce_min(block, axis=0)
            block_max = tf.reduce_max(block, axis=0)
            block_size = block_max - block_min
            
            # Check block size constraint
            self.assertTrue(tf.reduce_all(block_size <= 0.5))
            
            total_points += block.shape[0]
        
        # Verify all points are accounted for
        self.assertEqual(total_points, self.point_cloud.shape[0])

    def test_save_blocks(self):
        """Test saving blocks to PLY files."""
        blocks = partition_point_cloud(
            self.point_cloud,
            block_size=0.5,
            min_points=50
        )
        
        # Save blocks
        output_dir = self.test_files['blocks']
        output_dir.mkdir(exist_ok=True)
        
        save_blocks(blocks, str(output_dir), "test")
        
        # Verify saved files
        saved_files = list(output_dir.glob("test_block_*.ply"))
        self.assertEqual(len(saved_files), len(blocks))
        
        # Verify file contents
        for i, file_path in enumerate(sorted(saved_files)):
            loaded_points = read_point_cloud(str(file_path))
            self.assertAllEqual(loaded_points.shape, blocks[i].shape)
            self.assertAllClose(loaded_points, blocks[i])

    @tf.function
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        # Create batch of point clouds
        batch_size = 4
        point_clouds = tf.stack([self.point_cloud] * batch_size)
        
        # Process batch
        blocks_batch = tf.vectorized_map(
            lambda x: partition_point_cloud(x, block_size=0.5, min_points=50),
            point_clouds
        )
        
        # Verify batch results
        self.assertEqual(len(blocks_batch), batch_size)
        for blocks in blocks_batch:
            total_points = tf.reduce_sum([tf.shape(block)[0] for block in blocks])
            self.assertEqual(total_points, tf.shape(self.point_cloud)[0])

    @pytest.mark.integration
    def test_full_pipeline(self):
        """Test complete octree processing pipeline."""
        # Create test point cloud
        input_file = self.test_files['point_cloud']
        output_dir = self.test_files['blocks']
        
        # Read point cloud
        points = read_point_cloud(str(input_file))
        
        # Partition into blocks
        blocks = partition_point_cloud(points, block_size=0.5, min_points=50)
        
        # Save blocks
        save_blocks(blocks, str(output_dir), "test")
        
        # Verify results
        saved_files = list(output_dir.glob("test_block_*.ply"))
        
        # Test reconstruction
        reconstructed_points = []
        for file_path in saved_files:
            block_points = read_point_cloud(str(file_path))
            reconstructed_points.append(block_points)
        
        reconstructed = tf.concat(reconstructed_points, axis=0)
        
        # Verify point count
        self.assertEqual(reconstructed.shape[0], points.shape[0])
        
        # Verify point distribution
        original_bounds = (tf.reduce_min(points, axis=0), tf.reduce_max(points, axis=0))
        reconstructed_bounds = (tf.reduce_min(reconstructed, axis=0), tf.reduce_max(reconstructed, axis=0))
        
        self.assertAllClose(original_bounds[0], reconstructed_bounds[0], rtol=1e-5)
        self.assertAllClose(original_bounds[1], reconstructed_bounds[1], rtol=1e-5)

    def test_error_handling(self):
        """Test error handling in octree processing."""
        # Test empty point cloud
        empty_cloud = tf.zeros((0, 3), dtype=tf.float32)
        blocks = partition_point_cloud(empty_cloud, block_size=0.5, min_points=50)
        self.assertEqual(len(blocks), 0)
        
        # Test single point
        single_point = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
        blocks = partition_point_cloud(single_point, block_size=0.5, min_points=1)
        self.assertEqual(len(blocks), 1)
        
        # Test invalid block size
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "Block size must be positive"):
            partition_point_cloud(self.point_cloud, block_size=-1.0, min_points=50)
        
        # Test invalid minimum points
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "Minimum points must be positive"):
            partition_point_cloud(self.point_cloud, block_size=0.5, min_points=0)

if __name__ == "__main__":
    tf.test.main()