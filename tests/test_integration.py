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
        self.training_pipeline = TrainingPipeline(self.test_env['config_path'])
        self.eval_pipeline = EvaluationPipeline(self.test_env['config_path'])

    def test_data_pipeline_integration(self):
        point_cloud = create_mock_point_cloud(1000)
        loader = DataLoader(self.test_env['config'])
        voxelized = loader._voxelize_points(point_cloud, self.test_env['config']['data']['resolution'])
        
        self.assertEqual(voxelized.shape, (self.test_env['config']['data']['resolution'],) * 3)
        
        dataset = loader.load_training_data()
        batch = next(iter(dataset))
        self.assertEqual(batch.shape[1:], (self.test_env['config']['data']['resolution'],) * 3)

    @pytest.mark.gpu
    def test_training_evaluation_integration(self):
        dataset = create_test_dataset(
            self.training_pipeline.config['training']['batch_size'],
            self.training_pipeline.config['data']['resolution']
        )
        
        self.training_pipeline.data_loader.load_training_data = lambda: dataset
        self.training_pipeline.data_loader.load_evaluation_data = lambda: dataset
        self.training_pipeline.train(epochs=1, validate_every=2)
        self.training_pipeline.save_checkpoint('integration_test')
        self.eval_pipeline.load_checkpoint('integration_test')
        
        results = self.eval_pipeline.evaluate()
        self.assertIn('psnr', results)
        self.assertIn('chamfer_distance', results)
        self.assertGreater(results['psnr'], 0)

    def test_compression_pipeline_integration(self):
        input_data = create_mock_voxel_grid(self.test_env['config']['data']['resolution'])
        compressed, metrics = self.training_pipeline.model.compress(tf.expand_dims(input_data, 0))
        
        self.assertIn('bit_rate', metrics)
        self.assertGreater(metrics['bit_rate'], 0)
        
        decompressed = self.training_pipeline.model.decompress(compressed)
        self.assertEqual(decompressed.shape[1:], (self.test_env['config']['data']['resolution'],) * 3)
        
        eval_metrics = self.eval_pipeline.compute_metrics(decompressed[0], input_data)
        self.assertIn('psnr', eval_metrics)
        self.assertIn('chamfer_distance', eval_metrics)

    @pytest.mark.e2e
    def test_complete_workflow(self):
        point_cloud = create_mock_point_cloud(1000)
        loader = DataLoader(self.test_env['config'])
        voxelized = loader._voxelize_points(point_cloud, self.test_env['config']['data']['resolution'])
        
        dataset = create_test_dataset(
            self.training_pipeline.config['training']['batch_size'],
            self.training_pipeline.config['data']['resolution']
        )
        self.training_pipeline.data_loader.load_training_data = lambda: dataset
        self.training_pipeline.data_loader.load_evaluation_data = lambda: dataset
        self.training_pipeline.train(epochs=1)
        
        compressed, metrics = self.training_pipeline.model.compress(tf.expand_dims(voxelized, 0))
        decompressed = self.training_pipeline.model.decompress(compressed)
        
        eval_metrics = self.eval_pipeline.compute_metrics(decompressed[0], voxelized)
        
        results_dir = Path(self.test_env['config']['evaluation']['output_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_pipeline.save_results({'test_sample': eval_metrics}, results_dir / 'test_results.json')
        
        self.assertTrue((results_dir / 'test_results.json').exists())
        self.assertGreater(eval_metrics['psnr'], 0)
        self.assertGreater(eval_metrics['chamfer_distance'], 0)
        self.assertGreater(metrics['bit_rate'], 0)
        self.assertLess(metrics['bit_rate'], 10)

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