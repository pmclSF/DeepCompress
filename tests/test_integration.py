import tensorflow as tf
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_utils import (
    setup_test_environment,
    create_mock_point_cloud,
    create_mock_voxel_grid,
    create_test_dataset
)
from training_pipeline import TrainingPipeline
from evaluation_pipeline import EvaluationPipeline
from data_loader import DataLoader
from model_transforms import DeepCompressModel, DeepCompressModelV2, TransformConfig

class TestIntegration(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.test_env = setup_test_environment(tmp_path)
        self.resolution = 16
        self.batch_size = 1

    def test_data_pipeline_integration(self):
        """Test DataLoader normalize + voxelize pipeline."""
        point_cloud = create_mock_point_cloud(500)
        loader = DataLoader(self.test_env['config'])

        normalized = loader._normalize_points(point_cloud)
        voxelized = loader._voxelize_points(normalized, self.resolution)

        self.assertEqual(voxelized.shape, (self.resolution,) * 3)
        # Voxel grid should have some occupied cells
        self.assertGreater(tf.reduce_sum(voxelized), 0)

    @pytest.mark.integration
    def test_training_evaluation_integration(self):
        """Test train step followed by evaluation on same data."""
        pipeline = TrainingPipeline(self.test_env['config_path'])
        eval_pipeline = EvaluationPipeline(self.test_env['config_path'])

        # Run a single train step
        voxel_grid = create_mock_voxel_grid(self.resolution, self.batch_size)
        # Remove channel dim for _train_step (it adds it back)
        batch = voxel_grid[..., 0]
        losses = pipeline._train_step(batch, training=True)

        self.assertFalse(tf.math.is_nan(losses['total_loss']))

        # Evaluate on same data
        results = eval_pipeline._evaluate_single(voxel_grid)
        self.assertIn('psnr', results)
        self.assertIn('chamfer', results)

    def test_compression_pipeline_integration(self):
        """Test V2 model compress/decompress roundtrip."""
        config = TransformConfig(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='relu',
            conv_type='standard'
        )
        model = DeepCompressModelV2(config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        # Forward pass to get expected shape
        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        # Compress/decompress roundtrip
        compressed = model.compress(input_tensor)
        self.assertIn('y', compressed)
        self.assertIn('z', compressed)

        decompressed = model.decompress(compressed)
        self.assertEqual(decompressed.shape, x_hat.shape)

        # Rate info should be positive
        self.assertGreater(rate_info['total_bits'], 0)

    @pytest.mark.e2e
    def test_complete_workflow(self):
        """End-to-end: voxelize point cloud, run model, check output."""
        # Voxelize a point cloud
        point_cloud = create_mock_point_cloud(500)
        loader = DataLoader(self.test_env['config'])
        normalized = loader._normalize_points(point_cloud)
        voxelized = loader._voxelize_points(normalized, self.resolution)

        # Add batch and channel dimensions
        input_tensor = voxelized[tf.newaxis, ..., tf.newaxis]

        # Run through V1 model
        config = TransformConfig(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='relu',
            conv_type='standard'
        )
        model = DeepCompressModel(config)
        x_hat, y, y_hat, z = model(input_tensor, training=False)

        # Output should be 1-channel occupancy in [0, 1]
        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])
        self.assertEqual(x_hat.shape[-1], 1)
        self.assertAllGreaterEqual(x_hat, 0.0)
        self.assertAllLessEqual(x_hat, 1.0)

class TestModelV2Integration(tf.test.TestCase):
    """Integration tests for DeepCompressModelV2 with advanced entropy models."""

    @pytest.fixture(autouse=True)
    def setup_v2(self):
        self.config = TransformConfig(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='relu',
            conv_type='standard'
        )
        self.batch_size = 1
        self.resolution = 16

    def test_v2_full_pipeline_hyperprior(self):
        """Full pipeline with mean-scale hyperprior."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        # Check spatial dimensions match (channels may differ)
        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])
        self.assertGreater(rate_info['total_bits'], 0)

    def test_v2_full_pipeline_context(self):
        """Full pipeline with autoregressive context."""
        model = DeepCompressModelV2(self.config, entropy_model='context')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])

    def test_v2_full_pipeline_channel(self):
        """Full pipeline with channel-wise context."""
        model = DeepCompressModelV2(
            self.config,
            entropy_model='channel',
            num_channel_groups=4
        )
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])

    def test_v2_full_pipeline_attention(self):
        """Full pipeline with attention context."""
        model = DeepCompressModelV2(
            self.config,
            entropy_model='attention',
            num_attention_layers=1
        )
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])

    def test_v2_training_step(self):
        """Training step with V2 model and hyperprior."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        optimizer = tf.keras.optimizers.Adam(1e-4)
        input_tensor = create_mock_voxel_grid(self.resolution, 2)

        with tf.GradientTape() as tape:
            x_hat, y, y_hat, z, rate_info = model(input_tensor, training=True)
            # Use mean squared output as loss (since shapes may differ)
            recon_loss = tf.reduce_mean(tf.square(x_hat))
            rate_loss = rate_info['bpp']
            loss = recon_loss + 0.01 * rate_loss

        gradients = tape.gradient(loss, model.trainable_variables)
        non_none_grads = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]

        self.assertNotEmpty(non_none_grads)
        optimizer.apply_gradients(non_none_grads)

    def test_v2_compress_decompress_roundtrip(self):
        """Compress/decompress roundtrip for V2 model."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        # Get expected shape from forward pass
        x_hat, _, _, _, _ = model(input_tensor, training=False)

        compressed = model.compress(input_tensor)
        decompressed = model.decompress(compressed)

        self.assertEqual(decompressed.shape, x_hat.shape)


class TestEntropyModelImports(tf.test.TestCase):
    """Tests that all entropy model imports work correctly."""

    def test_import_entropy_parameters(self):
        """EntropyParameters can be imported and used."""
        from entropy_parameters import EntropyParameters

        layer = EntropyParameters(latent_channels=64)
        z_hat = tf.random.normal((1, 4, 4, 4, 32))
        mean, scale = layer(z_hat)

        self.assertEqual(mean.shape[-1], 64)
        self.assertEqual(scale.shape[-1], 64)

    def test_import_conditional_gaussian(self):
        """ConditionalGaussian can be imported and used."""
        from entropy_model import ConditionalGaussian

        layer = ConditionalGaussian()
        inputs = tf.random.normal((1, 4, 4, 4, 64))
        scale = tf.ones_like(inputs)
        mean = tf.zeros_like(inputs)

        outputs, likelihood = layer(inputs, scale, mean, training=False)

        self.assertEqual(outputs.shape, inputs.shape)

    def test_import_mean_scale_hyperprior(self):
        """MeanScaleHyperprior can be imported and used."""
        from entropy_model import MeanScaleHyperprior

        model = MeanScaleHyperprior(latent_channels=64, hyper_channels=32)
        y = tf.random.normal((1, 4, 4, 4, 64))
        z_hat = tf.random.normal((1, 4, 4, 4, 32))

        y_hat, likelihood, bits = model(y, z_hat, training=False)

        self.assertEqual(y_hat.shape, y.shape)

    def test_import_context_model(self):
        """ContextualEntropyModel can be imported and used."""
        from context_model import ContextualEntropyModel

        model = ContextualEntropyModel(latent_channels=64, hyper_channels=32)
        y = tf.random.normal((1, 4, 4, 4, 64))
        z_hat = tf.random.normal((1, 4, 4, 4, 32))

        y_hat, likelihood, bits = model(y, z_hat, training=False)

        self.assertEqual(y_hat.shape, y.shape)

    def test_import_channel_context(self):
        """ChannelContextEntropyModel can be imported and used."""
        from channel_context import ChannelContextEntropyModel

        model = ChannelContextEntropyModel(
            latent_channels=64,
            hyper_channels=32,
            num_groups=4
        )
        y = tf.random.normal((1, 4, 4, 4, 64))
        z_hat = tf.random.normal((1, 4, 4, 4, 32))

        y_hat, likelihood, bits = model(y, z_hat, training=False)

        self.assertEqual(y_hat.shape, y.shape)

    def test_import_attention_context(self):
        """AttentionEntropyModel can be imported and used."""
        from attention_context import AttentionEntropyModel

        model = AttentionEntropyModel(
            latent_channels=64,
            hyper_channels=32,
            num_attention_layers=1
        )
        y = tf.random.normal((1, 4, 4, 4, 64))
        z_hat = tf.random.normal((1, 4, 4, 4, 32))

        y_hat, likelihood, bits = model(y, z_hat, training=False)

        self.assertEqual(y_hat.shape, y.shape)


class TestBackwardCompatibility(tf.test.TestCase):
    """Tests ensuring backward compatibility with original model."""

    def test_v1_model_still_works(self):
        """Original DeepCompressModel should work unchanged."""
        config = TransformConfig(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='cenic_gdn',
            conv_type='separable'
        )
        model = DeepCompressModel(config)
        input_tensor = create_mock_voxel_grid(16, 1)

        x_hat, y, y_hat, z = model(input_tensor, training=False)

        # Check spatial dimensions (channels may differ due to transform architecture)
        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])

    def test_v2_gaussian_backward_compatible(self):
        """V2 with gaussian entropy model matches original behavior."""
        config = TransformConfig(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='cenic_gdn',
            conv_type='separable'
        )
        model = DeepCompressModelV2(config, entropy_model='gaussian')
        input_tensor = create_mock_voxel_grid(16, 1)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])


if __name__ == '__main__':
    tf.test.main()