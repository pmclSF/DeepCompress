import sys
import tensorflow as tf
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_utils import create_mock_point_cloud, create_mock_ply_file, setup_test_environment
from ds_pc_octree_blocks import PointCloudProcessor

class TestPointCloudOctreeBlocks(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_env = setup_test_environment(tmp_path)
        self.point_cloud = create_mock_point_cloud(1000)
        self.processor = PointCloudProcessor(block_size=0.5, min_points=50)

    def test_partition_point_cloud(self):
        # Use larger block_size and lower min_points so blocks aren't all filtered
        processor = PointCloudProcessor(block_size=2.0, min_points=1)
        blocks = processor.partition_point_cloud(self.point_cloud)
        total_points = 0
        for block in blocks:
            self.assertEqual(block.shape[1], 3)
            self.assertGreaterEqual(block.shape[0], 1)
            total_points += block.shape[0]
        self.assertEqual(total_points, self.point_cloud.shape[0])

    def test_save_blocks(self):
        blocks = self.processor.partition_point_cloud(self.point_cloud)
        output_dir = Path(self.test_env['tmp_path']) / 'blocks'
        output_dir.mkdir(exist_ok=True)
        self.processor.save_blocks(blocks, str(output_dir), "test")
        saved_files = list(output_dir.glob("test_block_*.ply"))
        self.assertEqual(len(saved_files), len(blocks))

    def test_error_handling(self):
        empty_cloud = tf.zeros((0, 3), dtype=tf.float32)
        processor_min1 = PointCloudProcessor(block_size=0.5, min_points=0)
        blocks = processor_min1.partition_point_cloud(empty_cloud)
        self.assertEqual(len(blocks), 0)

        single_point = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
        processor_min1 = PointCloudProcessor(block_size=0.5, min_points=1)
        blocks = processor_min1.partition_point_cloud(single_point)
        self.assertEqual(len(blocks), 1)

if __name__ == "__main__":
    tf.test.main()
