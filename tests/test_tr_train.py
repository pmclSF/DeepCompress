import os
import pytest
import tempfile
import tensorflow as tf
import shutil
import subprocess
from ds_select_largest import count_points_in_block  # Assuming this function is available to read point cloud

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
        create_mock_ply(os.path.join(input_dir, "file_1.ply"), 2048)
        create_mock_ply(os.path.join(input_dir, "file_2.ply"), 2048)
        create_mock_ply(os.path.join(input_dir, "file_3.ply"), 1024)

        yield input_dir, output_dir

@pytest.mark.parametrize("tune", [True, False])
def test_train_with_or_without_tuning(setup_mock_data, tune):
    input_dir, output_dir = setup_mock_data

    # Prepare the command arguments
    command = [
        "python", "tr_train.py", input_dir, output_dir,
        "--num_epochs", "2", "--batch_size", "2"
    ]
    if tune:
        command.append("--tune")  # Enable hyperparameter tuning

    # Run the training script
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Check if the process was successful
    assert result.returncode == 0, f"Training script failed: {result.stderr.decode()}"

    # Check if the model is saved
    model_dir = os.path.join(output_dir, 'best_model') if tune else os.path.join(output_dir, 'trained_model')
    assert os.path.exists(model_dir), f"Model was not saved in {model_dir}"

    # If hyperparameter tuning was enabled, check if the tuner directory is created
    if tune:
        tuner_dir = os.path.join("tuner_dir", "point_cloud_compression")
        assert os.path.exists(tuner_dir), f"Tuner directory not created in {tuner_dir}"

    # Ensure that the output directory contains the model checkpoint
    model_files = os.listdir(output_dir)
    assert any(file.endswith(".h5") for file in model_files), "No model checkpoint found in output directory"

@pytest.mark.parametrize("batch_size", [32, 64])
def test_batch_size_effect(setup_mock_data, batch_size):
    input_dir, output_dir = setup_mock_data

    # Run the training script with different batch sizes
    command = [
        "python", "tr_train.py", input_dir, output_dir,
        "--num_epochs", "2", "--batch_size", str(batch_size)
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Check if the process was successful
    assert result.returncode == 0, f"Training script failed: {result.stderr.decode()}"
    
    # Check if the model is saved
    model_dir = os.path.join(output_dir, 'trained_model')
    assert os.path.exists(model_dir), f"Model was not saved in {model_dir}"

def test_model_training_with_valid_data(setup_mock_data):
    input_dir, output_dir = setup_mock_data

    # Run the training script without tuning and with valid data
    command = [
        "python", "tr_train.py", input_dir, output_dir,
        "--num_epochs", "2", "--batch_size", "2"
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if the process was successful
    assert result.returncode == 0, f"Training script failed: {result.stderr.decode()}"

    # Check if the model is saved
    model_dir = os.path.join(output_dir, 'trained_model')
    assert os.path.exists(model_dir), f"Model was not saved in {model_dir}"

def test_invalid_data(setup_mock_data):
    """Test that the training script handles invalid data gracefully."""
    input_dir, output_dir = setup_mock_data

    # Create a mock invalid .ply file (corrupt data)
    invalid_ply_path = os.path.join(input_dir, "invalid.ply")
    with open(invalid_ply_path, "w") as f:
        f.write("Invalid data")

    # Run the training script
    command = [
        "python", "tr_train.py", input_dir, output_dir,
        "--num_epochs", "2", "--batch_size", "2"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Check if the script failed with the corrupt data
    assert result.returncode != 0, f"Training script should have failed with corrupt data: {result.stderr.decode()}"
    assert "Error" in result.stderr.decode(), "Expected error message not found"

if __name__ == "__main__":
    pytest.main()
