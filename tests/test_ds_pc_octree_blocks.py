import tensorflow as tf
import os
from ds_pc_octree_blocks import read_point_cloud, partition_point_cloud, save_blocks

def test_read_point_cloud(tmp_path):
    """Test reading a point cloud from a .ply file."""
    # Create a sample .ply file
    ply_file = tmp_path / "test.ply"
    with open(ply_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex 5\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        f.write("0.0 0.0 0.0\n")
        f.write("1.0 1.0 1.0\n")
        f.write("2.0 2.0 2.0\n")
        f.write("3.0 3.0 3.0\n")
        f.write("4.0 4.0 4.0\n")

    # Read the point cloud
    points = read_point_cloud(str(ply_file))
    assert points.shape == (5, 3), "Point cloud shape mismatch"
    assert tf.reduce_all(tf.equal(points, tf.constant([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
    ]))), "Point cloud content mismatch"

def test_partition_point_cloud():
    """Test partitioning a point cloud into blocks."""
    points = tf.constant([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
    ], dtype=tf.float32)

    # Partition the point cloud
    blocks = partition_point_cloud(points, block_size=2.0, min_points=1)

    # Verify the blocks
    assert len(blocks) == 3, "Number of blocks mismatch"
    assert tf.reduce_all(tf.equal(blocks[0], tf.constant([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))), "Block 0 mismatch"
    assert tf.reduce_all(tf.equal(blocks[1], tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]))), "Block 1 mismatch"
    assert tf.reduce_all(tf.equal(blocks[2], tf.constant([[4.0, 4.0, 4.0]]))), "Block 2 mismatch"

def test_save_blocks(tmp_path):
    """Test saving blocks to .ply files."""
    blocks = [
        tf.constant([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=tf.float32),
        tf.constant([[2.0, 2.0, 2.0]], dtype=tf.float32),
    ]

    # Save blocks
    save_blocks(blocks, str(tmp_path), "test")

    # Verify output files
    block_files = sorted(tmp_path.glob("test_block_*.ply"))
    assert len(block_files) == 2, "Number of saved blocks mismatch"

    # Verify contents of block files
    for i, block_file in enumerate(block_files):
        with open(block_file, "r") as f:
            lines = f.readlines()
            header_end = lines.index("end_header\n")
            points = [
                list(map(float, line.strip().split())) for line in lines[header_end + 1 :]
            ]
            assert len(points) == blocks[i].shape[0], f"Block {i} point count mismatch"
            assert tf.reduce_all(tf.equal(tf.constant(points, dtype=tf.float32), blocks[i])), f"Block {i} content mismatch"

if __name__ == "__main__":
    test_read_point_cloud("/tmp")
    test_partition_point_cloud()
    test_save_blocks("/tmp")
    print("All tests passed!")
