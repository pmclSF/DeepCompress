# model_opt.py

# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Loss function for Chamfer Distance
def chamfer_distance(pc1, pc2):
    """
    Compute the Chamfer Distance between two point clouds.

    Args:
        pc1: First point cloud as a tensor of shape (N, D).
        pc2: Second point cloud as a tensor of shape (M, D).

    Returns:
        tf.Tensor: Chamfer distance between pc1 and pc2.
    """
    diff1 = tf.reduce_mean(tf.reduce_min(tf.norm(pc1[:, None] - pc2[None, :], axis=-1), axis=1))
    diff2 = tf.reduce_mean(tf.reduce_min(tf.norm(pc2[:, None] - pc1[None, :], axis=-1), axis=1))
    return diff1 + diff2

# Model definition for point cloud compression
class PointCloudAutoencoder(tf.keras.Model):
    def __init__(self):
        super(PointCloudAutoencoder, self).__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(64, 64, 64, 1)),
            layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            layers.AveragePooling3D((2, 2, 2)),
            layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            layers.AveragePooling3D((2, 2, 2)),
        ])
        # Bottleneck
        self.bottleneck = layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same')
        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same'),
            layers.UpSampling3D((2, 2, 2)),
            layers.Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same'),
            layers.UpSampling3D((2, 2, 2)),
            layers.Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same'),
        ])
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.bottleneck(x)
        return self.decoder(x)

# Preprocessing function for point clouds
def preprocess_point_cloud(point_cloud, voxel_size=0.05):
    """
    Preprocess a point cloud by voxelization using TensorFlow operations.

    Args:
        point_cloud (tf.Tensor): Tensor of shape (N, 3) representing the point cloud.
        voxel_size (float): Size of each voxel.

    Returns:
        tf.Tensor: Preprocessed point cloud with points snapped to a voxel grid.
    """
    # Quantize point cloud to voxel grid
    quantized = tf.math.round(point_cloud / voxel_size) * voxel_size

    # Remove duplicate points
    unique_quantized = tf.unique(tf.reshape(quantized, [-1, 3]))[0]

    return unique_quantized

# Training function
def train_autoencoder(autoencoder, dataset, epochs=10, batch_size=8):
    """
    Train the point cloud autoencoder.

    Args:
        autoencoder (PointCloudAutoencoder): Autoencoder model.
        dataset (tf.data.Dataset): Dataset for training.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.

    Returns:
        None
    """
    for epoch in range(epochs):
        for batch in dataset.batch(batch_size):
            with tf.GradientTape() as tape:
                reconstructed = autoencoder(batch, training=True)
                loss = chamfer_distance(batch, reconstructed)
            gradients = tape.gradient(loss, autoencoder.trainable_variables)
            autoencoder.optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))

        print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

# Explicit exports
__all__ = ['PointCloudAutoencoder', 'chamfer_distance', 'preprocess_point_cloud', 'train_autoencoder']
