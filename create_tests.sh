#!/bin/bash

# Create a tests directory
mkdir -p tests

# Create a test file for model_opt.py
echo "Creating tests/test_model_opt.py..."
cat > tests/test_model_opt.py <<EOL
import pytest
import tensorflow as tf
import numpy as np
from model_opt import chamfer_distance, PointCloudAutoencoder, preprocess_point_cloud

# Test Chamfer Distance
def test_chamfer_distance():
    pc1 = tf.constant([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=tf.float32)
    pc2 = tf.constant([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=tf.float32)
    distance = chamfer_distance(pc1, pc2)
    assert distance.numpy() > 0

# Test autoencoder shape
def test_autoencoder_shape():
    autoencoder = PointCloudAutoencoder()
    input_data = tf.random.normal([1, 64, 64, 64, 1])
    output_data = autoencoder(input_data)
    assert output_data.shape == input_data.shape

# Test preprocessing function
def test_preprocess_point_cloud():
    # Placeholder test for preprocessing (requires a valid .ply file path)
    # pc_data = preprocess_point_cloud("path/to/sample.ply", voxel_size=0.1)
    # assert isinstance(pc_data, np.ndarray)
    # assert pc_data.shape[1] == 3
    pass

# Test training workflow
def test_train_autoencoder():
    dataset = tf.data.Dataset.from_tensor_slices(np.random.rand(10, 64, 64, 64, 1).astype(np.float32))
    autoencoder = PointCloudAutoencoder()
    try:
        for batch in dataset.batch(2):
            with tf.GradientTape() as tape:
                reconstructed = autoencoder(batch, training=True)
                loss = chamfer_distance(batch, reconstructed)
            gradients = tape.gradient(loss, autoencoder.trainable_variables)
            autoencoder.optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    except Exception as e:
        assert False, f"Training failed with error: {e}"
EOL

echo "Tests created successfully in tests/test_model_opt.py"

# Install pytest if not installed
if ! pip show pytest > /dev/null 2>&1; then
    echo "Installing pytest..."
    pip install pytest
else
    echo "pytest already installed."
fi

# Run the tests
echo "Running tests with pytest..."
pytest tests/test_model_opt.py
