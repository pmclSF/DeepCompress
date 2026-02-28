"""
Tests for end-to-end compress/decompress roundtrip consistency.

Validates that DeepCompressModel and DeepCompressModelV2 produce correct
output shapes, bounded values, and deterministic inference across entropy
model configurations.
"""

import sys
from pathlib import Path

import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_transforms import DeepCompressModel, DeepCompressModelV2, TransformConfig
from test_utils import create_mock_voxel_grid

# Standard small config for all roundtrip tests
_SMALL_CONFIG = TransformConfig(
    filters=32,
    kernel_size=(3, 3, 3),
    strides=(1, 1, 1),
    activation='relu',
    conv_type='standard'
)

_RESOLUTION = 16
_BATCH_SIZE = 1


class TestDeepCompressModelV1Roundtrip(tf.test.TestCase):
    """Roundtrip tests for V1 model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.model = DeepCompressModel(_SMALL_CONFIG)
        self.input_tensor = create_mock_voxel_grid(_RESOLUTION, _BATCH_SIZE)

    def test_output_shape_matches_input(self):
        """x_hat should have same shape as input."""
        x_hat, y, z_hat, z_noisy = self.model(self.input_tensor, training=False)
        self.assertEqual(x_hat.shape, self.input_tensor.shape)

    def test_output_bounded_zero_one(self):
        """x_hat should be in [0, 1] (sigmoid output)."""
        x_hat, _, _, _ = self.model(self.input_tensor, training=False)
        self.assertAllGreaterEqual(x_hat, 0.0)
        self.assertAllLessEqual(x_hat, 1.0)

    def test_inference_deterministic(self):
        """Inference should be deterministic (tf.round, no noise)."""
        out1 = self.model(self.input_tensor, training=False)
        out2 = self.model(self.input_tensor, training=False)

        self.assertAllClose(out1[0], out2[0])  # x_hat
        self.assertAllClose(out1[1], out2[1])  # y
        self.assertAllClose(out1[2], out2[2])  # z_hat
        self.assertAllClose(out1[3], out2[3])  # z_noisy (rounded)

    def test_training_stochastic(self):
        """Training should be stochastic (uniform noise)."""
        out1 = self.model(self.input_tensor, training=True)
        out2 = self.model(self.input_tensor, training=True)

        # y_hat values should differ due to noise (check z_noisy)
        diff = tf.reduce_sum(tf.abs(out1[3] - out2[3]))
        self.assertGreater(float(diff), 0.0)

    def test_latent_has_channels(self):
        """Latent y should have more channels than input."""
        _, y, _, _ = self.model(self.input_tensor, training=False)
        self.assertGreater(y.shape[-1], self.input_tensor.shape[-1])

    def test_returns_four_values(self):
        """V1 model should return exactly 4 values."""
        outputs = self.model(self.input_tensor, training=False)
        self.assertEqual(len(outputs), 4)


class TestDeepCompressModelV2Gaussian(tf.test.TestCase):
    """Roundtrip tests for V2 model with gaussian entropy model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.input_tensor = create_mock_voxel_grid(_RESOLUTION, _BATCH_SIZE)
        self.model = DeepCompressModelV2(
            _SMALL_CONFIG, entropy_model='gaussian',
            num_channel_groups=4, num_attention_layers=1
        )

    def test_output_shape(self):
        """x_hat should match input shape."""
        x_hat, y, y_hat, z, rate_info = self.model(self.input_tensor, training=False)
        self.assertEqual(x_hat.shape, self.input_tensor.shape)

    def test_output_bounded(self):
        """x_hat should be in [0, 1]."""
        x_hat, _, _, _, _ = self.model(self.input_tensor, training=False)
        self.assertAllGreaterEqual(x_hat, 0.0)
        self.assertAllLessEqual(x_hat, 1.0)

    def test_returns_five_values(self):
        """V2 model should return exactly 5 values."""
        outputs = self.model(self.input_tensor, training=False)
        self.assertEqual(len(outputs), 5)

    def test_rate_info_keys(self):
        """rate_info should contain required keys."""
        _, _, _, _, rate_info = self.model(self.input_tensor, training=False)
        for key in ['likelihood', 'total_bits', 'y_bits', 'z_bits', 'bpp']:
            self.assertIn(key, rate_info, msg=f"Missing key: {key}")

    def test_total_bits_positive(self):
        """Total bits should be positive."""
        _, _, _, _, rate_info = self.model(self.input_tensor, training=False)
        self.assertGreater(float(rate_info['total_bits']), 0.0)

    def test_bpp_positive(self):
        """Bits per point should be positive."""
        _, _, _, _, rate_info = self.model(self.input_tensor, training=False)
        self.assertGreater(float(rate_info['bpp']), 0.0)

    def test_inference_deterministic(self):
        """Inference should be deterministic."""
        out1 = self.model(self.input_tensor, training=False)
        out2 = self.model(self.input_tensor, training=False)
        self.assertAllClose(out1[0], out2[0])


class TestDeepCompressModelV2Hyperprior(tf.test.TestCase):
    """Roundtrip tests for V2 model with hyperprior entropy model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.input_tensor = create_mock_voxel_grid(_RESOLUTION, _BATCH_SIZE)
        self.model = DeepCompressModelV2(
            _SMALL_CONFIG, entropy_model='hyperprior',
            num_channel_groups=4, num_attention_layers=1
        )

    def test_output_shape(self):
        """x_hat should match input shape."""
        x_hat, y, y_hat, z, rate_info = self.model(self.input_tensor, training=False)
        self.assertEqual(x_hat.shape, self.input_tensor.shape)

    def test_output_bounded(self):
        """x_hat should be in [0, 1]."""
        x_hat, _, _, _, _ = self.model(self.input_tensor, training=False)
        self.assertAllGreaterEqual(x_hat, 0.0)
        self.assertAllLessEqual(x_hat, 1.0)

    def test_returns_five_values(self):
        """V2 model should return exactly 5 values."""
        outputs = self.model(self.input_tensor, training=False)
        self.assertEqual(len(outputs), 5)

    def test_rate_info_keys(self):
        """rate_info should contain required keys."""
        _, _, _, _, rate_info = self.model(self.input_tensor, training=False)
        for key in ['likelihood', 'total_bits', 'y_bits', 'z_bits', 'bpp']:
            self.assertIn(key, rate_info, msg=f"Missing key: {key}")

    def test_total_bits_positive(self):
        """Total bits should be positive."""
        _, _, _, _, rate_info = self.model(self.input_tensor, training=False)
        self.assertGreater(float(rate_info['total_bits']), 0.0)

    def test_inference_deterministic(self):
        """Inference should be deterministic."""
        out1 = self.model(self.input_tensor, training=False)
        out2 = self.model(self.input_tensor, training=False)
        self.assertAllClose(out1[0], out2[0])


class TestV2CompressDecompressGaussian(tf.test.TestCase):
    """Tests for V2 compress/decompress path with gaussian model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.input_tensor = create_mock_voxel_grid(_RESOLUTION, _BATCH_SIZE)
        self.model = DeepCompressModelV2(_SMALL_CONFIG, entropy_model='gaussian')
        _ = self.model(self.input_tensor, training=False)  # build

    def test_compress_returns_dict(self):
        """compress() should return a dict with 'y', 'z', 'side_info'."""
        compressed = self.model.compress(self.input_tensor)
        self.assertIn('y', compressed)
        self.assertIn('z', compressed)
        self.assertIn('side_info', compressed)

    def test_decompress_shape(self):
        """decompress() output should match input shape."""
        compressed = self.model.compress(self.input_tensor)
        x_hat = self.model.decompress(compressed)
        self.assertEqual(x_hat.shape, self.input_tensor.shape)

    def test_decompress_bounded(self):
        """Decompressed output should be in [0, 1]."""
        compressed = self.model.compress(self.input_tensor)
        x_hat = self.model.decompress(compressed)
        self.assertAllGreaterEqual(x_hat, 0.0)
        self.assertAllLessEqual(x_hat, 1.0)

    def test_compress_decompress_deterministic(self):
        """Compress + decompress should be deterministic."""
        compressed1 = self.model.compress(self.input_tensor)
        x_hat1 = self.model.decompress(compressed1)
        compressed2 = self.model.compress(self.input_tensor)
        x_hat2 = self.model.decompress(compressed2)
        self.assertAllClose(x_hat1, x_hat2)


class TestV2CompressDecompressHyperprior(tf.test.TestCase):
    """Tests for V2 compress/decompress path with hyperprior model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.input_tensor = create_mock_voxel_grid(_RESOLUTION, _BATCH_SIZE)
        self.model = DeepCompressModelV2(_SMALL_CONFIG, entropy_model='hyperprior')
        _ = self.model(self.input_tensor, training=False)  # build

    def test_compress_returns_dict(self):
        """compress() should return a dict with 'y', 'z', 'side_info'."""
        compressed = self.model.compress(self.input_tensor)
        self.assertIn('y', compressed)
        self.assertIn('z', compressed)
        self.assertIn('side_info', compressed)

    def test_decompress_shape(self):
        """decompress() output should match input shape."""
        compressed = self.model.compress(self.input_tensor)
        x_hat = self.model.decompress(compressed)
        self.assertEqual(x_hat.shape, self.input_tensor.shape)

    def test_decompress_bounded(self):
        """Decompressed output should be in [0, 1]."""
        compressed = self.model.compress(self.input_tensor)
        x_hat = self.model.decompress(compressed)
        self.assertAllGreaterEqual(x_hat, 0.0)
        self.assertAllLessEqual(x_hat, 1.0)

    def test_compress_decompress_deterministic(self):
        """Compress + decompress should be deterministic."""
        compressed1 = self.model.compress(self.input_tensor)
        x_hat1 = self.model.decompress(compressed1)
        compressed2 = self.model.compress(self.input_tensor)
        x_hat2 = self.model.decompress(compressed2)
        self.assertAllClose(x_hat1, x_hat2)


class TestGradientFlow(tf.test.TestCase):
    """Tests that gradients flow through the model during training."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.input_tensor = create_mock_voxel_grid(_RESOLUTION, _BATCH_SIZE)

    def test_v1_gradients_flow(self):
        """V1 model should produce non-zero gradients."""
        model = DeepCompressModel(_SMALL_CONFIG)

        with tf.GradientTape() as tape:
            x_hat, y, z_hat, z_noisy = model(self.input_tensor, training=True)
            loss = tf.reduce_mean(tf.square(self.input_tensor - x_hat))

        grads = tape.gradient(loss, model.trainable_variables)
        non_none = [g for g in grads if g is not None]

        self.assertNotEmpty(non_none, "No gradients computed")
        total_grad_norm = sum(float(tf.reduce_sum(tf.abs(g))) for g in non_none)
        self.assertGreater(total_grad_norm, 0.0)

    def test_v2_gaussian_gradients_flow(self):
        """V2 gaussian model should produce non-zero gradients."""
        model = DeepCompressModelV2(_SMALL_CONFIG, entropy_model='gaussian')

        with tf.GradientTape() as tape:
            x_hat, y, y_hat, z, rate_info = model(self.input_tensor, training=True)
            loss = tf.reduce_mean(tf.square(self.input_tensor - x_hat))

        grads = tape.gradient(loss, model.trainable_variables)
        non_none = [g for g in grads if g is not None]

        self.assertNotEmpty(non_none, "No gradients computed")

    def test_v2_hyperprior_gradients_flow(self):
        """V2 hyperprior model should produce non-zero gradients."""
        model = DeepCompressModelV2(_SMALL_CONFIG, entropy_model='hyperprior')

        with tf.GradientTape() as tape:
            x_hat, y, y_hat, z, rate_info = model(self.input_tensor, training=True)
            distortion = tf.reduce_mean(tf.square(self.input_tensor - x_hat))
            rate = rate_info['total_bits']
            loss = distortion + 0.01 * rate

        grads = tape.gradient(loss, model.trainable_variables)
        non_none = [g for g in grads if g is not None]

        self.assertNotEmpty(non_none, "No gradients computed")


class TestInvalidEntropyModel(tf.test.TestCase):
    """Tests for invalid entropy model selection."""

    def test_invalid_entropy_model_raises(self):
        """Invalid entropy model string should raise ValueError."""
        with self.assertRaises(ValueError):
            DeepCompressModelV2(_SMALL_CONFIG, entropy_model='invalid')

    def test_valid_entropy_models_accepted(self):
        """All valid entropy model strings should be accepted."""
        for name in DeepCompressModelV2.ENTROPY_MODELS:
            model = DeepCompressModelV2(_SMALL_CONFIG, entropy_model=name)
            self.assertEqual(model.entropy_model_type, name)


if __name__ == '__main__':
    tf.test.main()
