import os
import pytest
import tensorflow as tf
from tempfile import TemporaryDirectory
from ds_select_largest import count_points_in_block, prioritize_blocks

def test_count_points_in_block():
    """Test counting points in a .ply file."""
    # Create a temporary .ply file with known points
    with TemporaryDirectory() as tmp_dir:
        ply_file = os.path.join(tmp_dir, "test.ply")
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

        # Count points in the .ply file
        point_count = count_points_in_block(ply_file)
        assert point_count == 5, f"Expected 5 points, but got {point_count}"

def test_prioritize_blocks():
    """Test prioritizing blocks based on point count."""
    with TemporaryDirectory() as input_dir, TemporaryDirectory() as output_dir:
        # Create mock .ply files with different point counts
        block_1 = os.path.join(input_dir, "block_1.ply")
        block_2 = os.path.join(input_dir, "block_2.ply")
        block_3 = os.path.join(input_dir, "block_3.ply")

        # Create .ply files
        with open(block_1, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 2\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            f.write("0.0 0.0 0.0\n")
            f.write("1.0 1.0 1.0\n")

        with open(block_2, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 4\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            f.write("0.0 0.0 0.0\n")
            f.write("1.0 1.0 1.0\n")
            f.write("2.0 2.0 2.0\n")
            f.write("3.0 3.0 3.0\n")

        with open(block_3, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write("element vertex 3\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            f.write("0.0 0.0 0.0\n")
            f.write("1.0 1.0 1.0\n")
            f.write("2.0 2.0 2.0\n")

        # Prioritize blocks and select the top 2 based on point count
        prioritize_blocks(input_dir, output_dir, num_blocks=2, criteria="points")

        # Check that the output directory contains the 2 largest blocks
        output_files = os.listdir(output_dir)
        assert len(output_files) == 2, f"Expected 2 blocks, but got {len(output_files)}"
        assert "block_2.ply" in output_files, "Block 2 was not selected"
        assert "block_3.ply" in output_files, "Block 3 was not selected"
        assert "block_1.ply" not in output_files, "Block 1 should not have been selected"

if __name__ == "__main__":
    pytest.main()
