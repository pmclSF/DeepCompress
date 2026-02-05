import tensorflow as tf
import pytest
from pathlib import Path
from test_utils import create_mock_point_cloud, create_mock_ply_file, setup_test_files
from ds_pc_octree_blocks import read_point_cloud, partition_point_cloud, save_blocks

class TestPointCloudOctreeBlocks(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_files = setup_test_files(tmp_path)
        self.point_cloud = create_mock_point_cloud(1000)

    @tf.function
    def test_read_point_cloud(self):
        points = read_point_cloud(str(self.test_files['point_cloud']))
        self.assertIsInstance(points, tf.Tensor)
        self.assertEqual(points.shape[1], 3)
        self.assertEqual(points.dtype, tf.float32)

    @tf.function
    def test_partition_point_cloud(self):
        blocks = partition_point_cloud(self.point_cloud, block_size=0.5, min_points=50)
        total_points = 0
        for block in blocks:
            self.assertEqual(block.shape[1], 3)
            self.assertGreaterEqual(block.shape[0], 50)
            block_min = tf.reduce_min(block, axis=0)
            block_max = tf.reduce_max(block, axis=0)
            block_size = block_max - block_min
            self.assertTrue(tf.reduce_all(block_size <= 0.5))
            total_points += block.shape[0]
        self.assertEqual(total_points, self.point_cloud.shape[0])

    def test_save_blocks(self):
        blocks = partition_point_cloud(self.point_cloud, block_size=0.5, min_points=50)
        output_dir = self.test_files['blocks']
        output_dir.mkdir(exist_ok=True)
        save_blocks(blocks, str(output_dir), "test")
        saved_files = list(output_dir.glob("test_block_*.ply"))
        self.assertEqual(len(saved_files), len(blocks))
        
        for i, file_path in enumerate(sorted(saved_files)):
            loaded_points = read_point_cloud(str(file_path))
            self.assertAllEqual(loaded_points.shape, blocks[i].shape)
            self.assertAllClose(loaded_points, blocks[i])

    @tf.function
    def test_batch_processing(self):
        batch_size = 4
        point_clouds = tf.stack([self.point_cloud] * batch_size)
        blocks_batch = tf.vectorized_map(
            lambda x: partition_point_cloud(x, block_size=0.5, min_points=50),
            point_clouds
        )
        self.assertEqual(len(blocks_batch), batch_size)

    def test_error_handling(self):
        empty_cloud = tf.zeros((0, 3), dtype=tf.float32)
        blocks = partition_point_cloud(empty_cloud, block_size=0.5, min_points=50)
        self.assertEqual(len(blocks), 0)
        
        single_point = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
        blocks = partition_point_cloud(single_point, block_size=0.5, min_points=1)
        self.assertEqual(len(blocks), 1)

if __name__ == "__main__":
    tf.test.main()