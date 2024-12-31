import os
import pytest
import tempfile
import shutil
from subprocess import run
import numpy as np

# Helper function to create mock .ply files
def create_mock_ply(file_path, num_points):
    with open(file_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for i in range(num_points):
            f.write(f"{i} {i} {i}\n")

@pytest.fixture
def setup_mock_data():
    """Fixture to set up mock data directories."""
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Create mock .ply files
        create_mock_ply(os.path.join(input_dir, "original.ply"), 2048)
        create_mock_ply(os.path.join(output_dir, "compressed.ply"), 2048)

        yield input_dir, output_dir

def test_psnr(setup_mock_data):
    input_dir, output_dir = setup_mock_data

    # Run the evaluation script on the mock data
    command = [
        "python", "ev_compare.py", os.path.join(input_dir, "original.ply"), os.path.join(output_dir, "compressed.ply")
    ]

    result = run(command, capture_output=True, text=True)

    # Check if the process was successful
    assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

    # Check if PSNR was computed
    assert "PSNR" in result.stdout, "PSNR was not calculated"

def test_compression_ratio(setup_mock_data):
    input_dir, output_dir = setup_mock_data

    # Run the evaluation script
    command = [
        "python", "ev_compare.py", os.path.join(input_dir, "original.ply"), os.path.join(output_dir, "compressed.ply")
    ]

    result = run(command, capture_output=True, text=True)

    # Check if the process was successful
    assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

    # Check if Compression Ratio was computed
    assert "Compression Ratio" in result.stdout, "Compression Ratio was not calculated"

def test_bitrate(setup_mock_data):
    input_dir, output_dir = setup_mock_data

    # Run the evaluation script
    command = [
        "python", "ev_compare.py", os.path.join(input_dir, "original.ply"), os.path.join(output_dir, "compressed.ply")
    ]

    result = run(command, capture_output=True, text=True)

    # Check if the process was successful
    assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

    # Check if Bitrate was computed
    assert "Bitrate" in result.stdout, "Bitrate was not calculated"

def test_time_and_memory_efficiency(setup_mock_data):
    input_dir, output_dir = setup_mock_data

    # Run the evaluation script
    command = [
        "python", "ev_compare.py", os.path.join(input_dir, "original.ply"), os.path.join(output_dir, "compressed.ply")
    ]

    result = run(command, capture_output=True, text=True)

    # Check if the process was successful
    assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

    # Check if Time and Memory efficiency were computed
    assert "Compression Time" in result.stdout, "Compression Time was not calculated"
    assert "Decompression Time" in result.stdout, "Decompression Time was not calculated"
    assert "Compression Memory" in result.stdout, "Compression Memory was not calculated"
    assert "Decompression Memory" in result.stdout, "Decompression Memory was not calculated"

if __name__ == "__main__":
    pytest.main()
