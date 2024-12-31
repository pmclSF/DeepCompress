import pytest
import os
import tempfile
import shutil
import json
import subprocess
from pathlib import Path
from model_opt import evaluate_compression  # Assuming this function exists from earlier

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
        create_mock_ply(os.path.join(input_dir, "original_1.ply"), 2048)
        create_mock_ply(os.path.join(input_dir, "original_2.ply"), 1024)
        create_mock_ply(os.path.join(input_dir, "original_3.ply"), 512)

        yield input_dir, output_dir

def test_run_experiment(setup_mock_data):
    """Test the full run of the experiment, including compression, decompression, and evaluation."""
    input_dir, output_dir = setup_mock_data

    # Run the compression pipeline (using subprocess to run the actual script)
    command = [
        "python", "mp_run.py", input_dir, output_dir,
        "--octree_level", "3", "--quantization_level", "8"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # Ensure that the experiment ran without errors
    assert result.returncode == 0, f"Experiment failed: {result.stderr}"

    # Verify that the experiment report was generated
    report_path = os.path.join(output_dir, "experiment_report.json")
    assert os.path.exists(report_path), f"Report file not found: {report_path}"

    # Check that the report contains valid JSON data
    with open(report_path, 'r') as f:
        report_data = json.load(f)

    # Ensure that the report contains data for each original file processed
    assert "original_1.ply" in report_data, "Missing data for original_1.ply"
    assert "original_2.ply" in report_data, "Missing data for original_2.ply"
    assert "original_3.ply" in report_data, "Missing data for original_3.ply"

    # Check if the compression metrics are present in the report
    for key, value in report_data.items():
        assert "psnr" in value, f"PSNR missing for {key}"
        assert "bd_rate" in value, f"BD-Rate missing for {key}"
        assert "bitrate" in value, f"Bitrate missing for {key}"

def test_compression_and_decompression(setup_mock_data):
    """Test the compression and decompression processes independently."""
    input_dir, output_dir = setup_mock_data

    # Simulate compression and decompression for one file
    input_file = os.path.join(input_dir, "original_1.ply")
    compressed_file = os.path.join(output_dir, "compressed_1.ply")
    decompressed_file = os.path.join(output_dir, "decompressed_1.ply")

    # Run the compression and decompression steps manually
    compressed_file_path = subprocess.run(
        ["python", "mp_run.py", input_file, compressed_file, "--octree_level", "3", "--quantization_level", "8"],
        capture_output=True, text=True
    ).stdout.strip()

    decompressed_file_path = subprocess.run(
        ["python", "mp_run.py", compressed_file, decompressed_file],
        capture_output=True, text=True
    ).stdout.strip()

    # Check if the compression and decompression were successful
    assert os.path.exists(compressed_file), f"Compression failed for {input_file}"
    assert os.path.exists(decompressed_file), f"Decompression failed for {compressed_file}"

def test_metrics_computation(setup_mock_data):
    """Test that the evaluation metrics (PSNR, BD-Rate) are computed correctly."""
    input_dir, output_dir = setup_mock_data

    # Run compression for the first file
    input_file = os.path.join(input_dir, "original_1.ply")
    compressed_file = os.path.join(output_dir, "compressed_1.ply")
    run_compression(input_file, compressed_file, octree_level=3, quantization_level=8)

    # Run decompression
    decompressed_file = os.path.join(output_dir, "decompressed_1.ply")
    run_decompression(compressed_file, decompressed_file)

    # Evaluate compression and decompression
    eval_results = evaluate_compression(input_file, decompressed_file)

    # Check that the metrics are computed
    assert "psnr" in eval_results, "PSNR not computed"
    assert "bd_rate" in eval_results, "BD-Rate not computed"
    assert "bitrate" in eval_results, "Bitrate not computed"

def test_invalid_input(setup_mock_data):
    """Test that the experiment handles invalid inputs gracefully."""
    input_dir, output_dir = setup_mock_data

    # Create an invalid file (corrupt .ply)
    invalid_file = os.path.join(input_dir, "invalid.ply")
    with open(invalid_file, "w") as f:
        f.write("Invalid data")

    # Run the experiment with the invalid file
    command = [
        "python", "mp_run.py", input_dir, output_dir,
        "--octree_level", "3", "--quantization_level", "8"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # Ensure that the experiment fails with the invalid input
    assert result.returncode != 0, f"Experiment should have failed with corrupt data: {result.stderr}"
    assert "Error" in result.stderr, "Expected error message not found"

if __name__ == "__main__":
    pytest.main()
