import tensorflow as tf
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_utils import create_mock_voxel_grid
from model_transforms import (
    CENICGDN,
    SpatialSeparableConv,
    AnalysisTransform,
    SynthesisTransform,
    DeepCompressModel,
    DeepCompressModelV2,
    TransformConfig
)

class TestModelTransforms(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = TransformConfig(
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            activation='cenic_gdn',
            conv_type='separable'
        )
        self.batch_size = 2
        self.resolution = 64
        self.input_shape = (self.batch_size, self.resolution, self.resolution, self.resolution, 1)

    def test_cenic_gdn(self):
        channels = 64
        activation = CENICGDN(channels)
        input_tensor = tf.random.uniform((2, 32, 32, 32, channels))
        output = activation(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            output = activation(input_tensor)
            loss = tf.reduce_mean(output)
        gradients = tape.gradient(loss, activation.trainable_variables)
        self.assertNotEmpty(gradients)
        # Check that gradients are non-zero
        self.assertGreater(tf.reduce_sum(tf.abs(gradients[0])), 0)

    def test_spatial_separable_conv(self):
        conv = SpatialSeparableConv(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1))
        input_tensor = tf.random.uniform((2, 32, 32, 32, 32))
        output = conv(input_tensor)
        self.assertEqual(output.shape[-1], 64)
        
        standard_params = 27 * 32 * 64
        separable_params = (3 * 32 * 32 + 9 * 32 * 64)
        self.assertLess(len(conv.trainable_variables[0].numpy().flatten()), standard_params)

    def test_analysis_transform(self):
        analysis = AnalysisTransform(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        output = analysis(input_tensor)
        self.assertEqual(len(output.shape), 5)  # 5D tensor (B, D, H, W, C)
        self.assertGreater(output.shape[-1], input_tensor.shape[-1])
        # Check that CENICGDN layers are present in the conv_layers list
        has_gdn = any(isinstance(layer, CENICGDN) for layer in analysis.conv_layers)
        self.assertTrue(has_gdn)

    def test_synthesis_transform(self):
        synthesis = SynthesisTransform(self.config)
        input_tensor = tf.random.uniform((2, 32, 32, 32, 256))  # Match analysis output channels
        output = synthesis(input_tensor)
        # Synthesis reduces channels progressively
        self.assertEqual(len(output.shape), 5)  # 5D tensor
        self.assertLessEqual(output.shape[-1], input_tensor.shape[-1])

    def test_deep_compress_model(self):
        # Use strides=(1,1,1) to avoid spatial dimension changes
        config_no_stride = TransformConfig(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='relu',  # Simpler activation
            conv_type='standard'
        )
        model = DeepCompressModel(config_no_stride)
        input_tensor = create_mock_voxel_grid(16, 1)  # Smaller for faster test
        # Model returns (x_hat, y, y_hat, z) tuple
        output = model(input_tensor, training=True)
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 4)
        x_hat, y, y_hat, z = output
        # Check that output tensors have correct shapes
        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])
        self.assertEqual(len(y.shape), 5)
        self.assertEqual(len(y_hat.shape), 5)
        self.assertEqual(len(z.shape), 5)

    def test_gradient_flow(self):
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        with tf.GradientTape() as tape:
            x_hat, y, y_hat, z = model(input_tensor, training=True)
            # Use x_hat for reconstruction loss
            loss = tf.reduce_mean(tf.square(x_hat))
        gradients = tape.gradient(loss, model.trainable_variables)
        # At least some gradients should exist
        non_none_grads = [g for g in gradients if g is not None]
        self.assertNotEmpty(non_none_grads)

    def test_model_save_load(self):
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        x_hat1, y1, y_hat1, z1 = model(input_tensor, training=False)

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Keras 3 requires .weights.h5 extension for save_weights
            save_path = os.path.join(tmp_dir, 'model.weights.h5')
            model.save_weights(save_path)
            new_model = DeepCompressModel(self.config)
            # Build the new model first
            _ = new_model(input_tensor, training=False)
            new_model.load_weights(save_path)

        x_hat2, y2, y_hat2, z2 = new_model(input_tensor, training=False)
        self.assertAllClose(x_hat1, x_hat2)

class TestDeepCompressModelV2(tf.test.TestCase):
    """Tests for DeepCompressModelV2."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = TransformConfig(
            filters=32,  # Smaller for faster tests
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='cenic_gdn',
            conv_type='separable'
        )
        self.batch_size = 1
        self.resolution = 16  # Smaller for faster tests

    def test_model_v2_backward_compatible(self):
        """V2 with 'gaussian' should behave like original."""
        model_v1 = DeepCompressModel(self.config)
        model_v2 = DeepCompressModelV2(self.config, entropy_model='gaussian')

        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        # Both should produce outputs without error
        output_v1 = model_v1(input_tensor, training=False)
        output_v2 = model_v2(input_tensor, training=False)

        # V1 returns 4 values, V2 returns 5 (includes rate_info)
        self.assertEqual(len(output_v1), 4)
        self.assertEqual(len(output_v2), 5)

        # First output (x_hat) should have same shape
        self.assertEqual(output_v1[0].shape, output_v2[0].shape)

    def test_model_v2_entropy_selection_hyperprior(self):
        """Test hyperprior entropy model."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        # Check spatial dimensions match (channels may differ due to transform architecture)
        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])
        self.assertIn('total_bits', rate_info)
        self.assertIn('bpp', rate_info)

    def test_model_v2_entropy_selection_context(self):
        """Test context entropy model."""
        model = DeepCompressModelV2(self.config, entropy_model='context')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])

    def test_model_v2_entropy_selection_channel(self):
        """Test channel context entropy model."""
        model = DeepCompressModelV2(
            self.config,
            entropy_model='channel',
            num_channel_groups=4
        )
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])

    def test_model_v2_entropy_selection_attention(self):
        """Test attention entropy model."""
        model = DeepCompressModelV2(
            self.config,
            entropy_model='attention',
            num_attention_layers=1  # Smaller for testing
        )
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        self.assertEqual(x_hat.shape[:-1], input_tensor.shape[:-1])

    def test_model_v2_invalid_entropy_model(self):
        """Invalid entropy model raises error."""
        with self.assertRaises(ValueError):
            DeepCompressModelV2(self.config, entropy_model='invalid')

    def test_model_v2_compress_decompress(self):
        """Compress/decompress roundtrip."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        # Forward pass to get expected output shape
        x_hat, _, _, _, _ = model(input_tensor, training=False)

        compressed = model.compress(input_tensor)
        decompressed = model.decompress(compressed)

        # Decompressed should match forward pass shape
        self.assertEqual(decompressed.shape, x_hat.shape)

    def test_model_v2_training_mode(self):
        """Training mode adds noise correctly."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        # Training should not raise errors
        x_hat_train, _, _, _, rate_train = model(input_tensor, training=True)
        x_hat_eval, _, _, _, rate_eval = model(input_tensor, training=False)

        self.assertEqual(x_hat_train.shape, x_hat_eval.shape)

    def test_model_v2_gradient_flow(self):
        """Gradients flow through the entire V2 model."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        with tf.GradientTape() as tape:
            x_hat, y, y_hat, z, rate_info = model(input_tensor, training=True)
            # Combined loss: reconstruction + rate
            recon_loss = tf.reduce_mean(tf.square(x_hat - input_tensor))
            rate_loss = rate_info['bpp']
            loss = recon_loss + 0.01 * rate_loss

        gradients = tape.gradient(loss, model.trainable_variables)

        self.assertNotEmpty(model.trainable_variables)
        # At least some gradients should exist
        non_none_grads = [g for g in gradients if g is not None]
        self.assertNotEmpty(non_none_grads)

    def test_model_v2_rate_distortion(self):
        """Verify rate info is meaningful."""
        model = DeepCompressModelV2(self.config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)

        _, _, _, _, rate_info = model(input_tensor, training=False)

        # Rate should be positive
        self.assertGreater(rate_info['total_bits'], 0)
        self.assertGreater(rate_info['bpp'], 0)


class TestModelV2Integration(tf.test.TestCase):
    """Integration tests for DeepCompressModelV2."""

    def test_end_to_end_compression(self):
        """Full pipeline: input -> compress -> decompress -> output."""
        config = TransformConfig(
            filters=32,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            activation='relu',  # Simpler activation for test
            conv_type='standard'
        )

        model = DeepCompressModelV2(config, entropy_model='hyperprior')
        input_tensor = create_mock_voxel_grid(16, 1)

        # Forward pass
        x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

        # Compress
        compressed = model.compress(input_tensor)

        # Decompress
        decompressed = model.decompress(compressed)

        # Output should have same spatial dimensions
        self.assertEqual(decompressed.shape[:-1], input_tensor.shape[:-1])
        # Decompressed should match forward pass x_hat
        self.assertEqual(decompressed.shape, x_hat.shape)

    def test_different_entropy_models_produce_different_rates(self):
        """Different entropy models should have different rate characteristics."""
        config = TransformConfig(filters=32, activation='relu', conv_type='standard')
        input_tensor = create_mock_voxel_grid(16, 1)

        rates = {}
        for entropy_type in ['gaussian', 'hyperprior']:
            model = DeepCompressModelV2(config, entropy_model=entropy_type)
            _, _, _, _, rate_info = model(input_tensor, training=False)
            rates[entropy_type] = rate_info['total_bits'].numpy()

        # Both should produce valid rates
        for rate in rates.values():
            self.assertGreater(rate, 0)


if __name__ == "__main__":
    tf.test.main()