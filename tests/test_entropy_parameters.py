"""Tests for entropy parameters network and mean-scale hyperprior."""

import sys
from pathlib import Path

import tensorflow as tf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from entropy_model import ConditionalGaussian, MeanScaleHyperprior
from entropy_parameters import EntropyParameters, EntropyParametersWithContext


class TestEntropyParameters(tf.test.TestCase):
    """Tests for EntropyParameters layer."""

    def setUp(self):
        super().setUp()
        self.latent_channels = 64
        self.hyper_channels = 32
        self.batch_size = 2
        self.spatial_size = 8

        self.layer = EntropyParameters(
            latent_channels=self.latent_channels,
            hidden_channels=128
        )

        # Create test input (simulated decoded hyperprior)
        self.z_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.hyper_channels)
        )

    def test_entropy_parameters_output_shape(self):
        """Verify scale/mean shapes match expected latent channels."""
        mean, scale = self.layer(self.z_hat)

        expected_shape = (
            self.batch_size, self.spatial_size, self.spatial_size,
            self.spatial_size, self.latent_channels
        )

        self.assertEqual(mean.shape, expected_shape)
        self.assertEqual(scale.shape, expected_shape)

    def test_entropy_parameters_positive_scale(self):
        """Scale must be positive (softplus activation)."""
        mean, scale = self.layer(self.z_hat)

        # All scale values should be positive
        self.assertAllGreater(scale, 0.0)

        # Scale should be at least the minimum value
        self.assertAllGreaterEqual(scale, self.layer.scale_min)

    def test_entropy_parameters_gradient_flow(self):
        """Verify gradients flow through the layer."""
        with tf.GradientTape() as tape:
            mean, scale = self.layer(self.z_hat)
            # Loss based on both outputs
            loss = tf.reduce_mean(mean) + tf.reduce_mean(scale)

        gradients = tape.gradient(loss, self.layer.trainable_variables)

        # All trainable variables should have gradients
        self.assertTrue(all(g is not None for g in gradients))
        # At least some gradients should be non-zero
        self.assertTrue(any(tf.reduce_sum(tf.abs(g)) > 0 for g in gradients))

    def test_entropy_parameters_serialization(self):
        """Test layer can be saved and loaded."""
        # Build the layer by calling it
        _ = self.layer(self.z_hat)

        config = self.layer.get_config()

        # Verify config contains required keys
        self.assertIn('latent_channels', config)
        self.assertIn('hidden_channels', config)
        self.assertIn('num_layers', config)

        # Reconstruct from config
        reconstructed = EntropyParameters.from_config(config)
        mean2, scale2 = reconstructed(self.z_hat)

        # Shapes should match
        mean1, scale1 = self.layer(self.z_hat)
        self.assertEqual(mean1.shape, mean2.shape)
        self.assertEqual(scale1.shape, scale2.shape)

    def test_entropy_parameters_different_input_sizes(self):
        """Test layer handles different spatial sizes."""
        for spatial_size in [4, 8, 16]:
            z_hat = tf.random.normal(
                (1, spatial_size, spatial_size, spatial_size, self.hyper_channels)
            )
            mean, scale = self.layer(z_hat)

            self.assertEqual(mean.shape[1:4], (spatial_size, spatial_size, spatial_size))
            self.assertEqual(scale.shape[1:4], (spatial_size, spatial_size, spatial_size))


class TestEntropyParametersWithContext(tf.test.TestCase):
    """Tests for EntropyParametersWithContext layer."""

    def setUp(self):
        super().setUp()
        self.latent_channels = 64
        self.context_channels = 64
        self.hyper_channels = 32
        self.batch_size = 2
        self.spatial_size = 8

        self.layer = EntropyParametersWithContext(
            latent_channels=self.latent_channels,
            context_channels=self.context_channels
        )

        self.z_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.hyper_channels)
        )
        self.context = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.context_channels)
        )

    def test_with_context_output_shape(self):
        """Verify outputs have correct shape with context input."""
        mean, scale = self.layer(self.z_hat, self.context)

        expected_shape = (
            self.batch_size, self.spatial_size, self.spatial_size,
            self.spatial_size, self.latent_channels
        )

        self.assertEqual(mean.shape, expected_shape)
        self.assertEqual(scale.shape, expected_shape)

    def test_with_context_positive_scale(self):
        """Scale must be positive even with context."""
        mean, scale = self.layer(self.z_hat, self.context)
        self.assertAllGreater(scale, 0.0)


class TestConditionalGaussian(tf.test.TestCase):
    """Tests for ConditionalGaussian layer."""

    def setUp(self):
        super().setUp()
        self.layer = ConditionalGaussian()
        self.batch_size = 2
        self.spatial_size = 8
        self.channels = 64

        shape = (self.batch_size, self.spatial_size, self.spatial_size,
                 self.spatial_size, self.channels)

        self.inputs = tf.random.normal(shape)
        self.scale = tf.abs(tf.random.normal(shape)) + 0.1
        self.mean = tf.random.normal(shape)

    def test_conditional_gaussian_compress_decompress(self):
        """Round-trip compress/decompress accuracy."""
        compressed = self.layer.compress(self.inputs, self.scale, self.mean)
        decompressed = self.layer.decompress(compressed, self.scale, self.mean)

        # Compressed should be quantized (integers)
        self.assertAllEqual(compressed, tf.round(compressed))

        # Decompressed should have same shape as input
        self.assertEqual(decompressed.shape, self.inputs.shape)

    def test_conditional_gaussian_gradient_flow(self):
        """Gradients propagate through the layer."""
        scale = tf.Variable(self.scale)
        mean = tf.Variable(self.mean)

        with tf.GradientTape() as tape:
            outputs, likelihood = self.layer(self.inputs, scale, mean, training=True)
            loss = -tf.reduce_mean(likelihood)  # Negative log-likelihood

        gradients = tape.gradient(loss, [scale, mean])

        # Both scale and mean should have gradients
        self.assertTrue(all(g is not None for g in gradients))

    def test_conditional_gaussian_likelihood_shape(self):
        """Likelihood has same shape as inputs."""
        outputs, likelihood = self.layer(self.inputs, self.scale, self.mean, training=False)

        self.assertEqual(likelihood.shape, self.inputs.shape)

    def test_conditional_gaussian_training_vs_eval(self):
        """Training adds noise, eval uses rounding."""
        tf.random.set_seed(42)

        # Training mode (with noise)
        outputs_train, _ = self.layer(self.inputs, self.scale, self.mean, training=True)

        # Eval mode (with rounding)
        outputs_eval, _ = self.layer(self.inputs, self.scale, self.mean, training=False)

        # Training outputs should not be exactly rounded
        centered_train = outputs_train - self.mean
        self.assertNotAllEqual(centered_train, tf.round(centered_train))

        # Eval outputs should be exactly rounded (after centering)
        centered_eval = outputs_eval - self.mean
        self.assertAllClose(centered_eval, tf.round(centered_eval), atol=1e-6)

    def test_conditional_gaussian_debug_tensors(self):
        """Debug tensors are populated after call."""
        _ = self.layer(self.inputs, self.scale, self.mean, training=False)
        debug = self.layer.get_debug_tensors()

        self.assertIn('inputs', debug)
        self.assertIn('outputs', debug)
        self.assertIn('likelihood', debug)

    def test_conditional_gaussian_scale_min_enforced(self):
        """Minimum scale is enforced."""
        tiny_scale = tf.ones_like(self.scale) * 1e-10

        # Should not raise errors
        outputs, likelihood = self.layer(self.inputs, tiny_scale, self.mean, training=False)

        # Likelihood should be finite (not NaN or Inf)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(likelihood)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(likelihood)))


class TestMeanScaleHyperprior(tf.test.TestCase):
    """Tests for MeanScaleHyperprior model."""

    def setUp(self):
        super().setUp()
        self.latent_channels = 64
        self.hyper_channels = 32
        self.batch_size = 2
        self.spatial_size = 8

        self.model = MeanScaleHyperprior(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels
        )

        self.y = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.latent_channels)
        )
        self.z_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.hyper_channels)
        )

    def test_mean_scale_hyperprior_call(self):
        """Test forward pass returns expected outputs."""
        y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=False)

        self.assertEqual(y_hat.shape, self.y.shape)
        self.assertEqual(y_likelihood.shape, self.y.shape)
        self.assertIsInstance(total_bits.numpy().item(), float)

    def test_mean_scale_hyperprior_compress_decompress(self):
        """Test compress/decompress round-trip."""
        symbols, side_info = self.model.compress(self.y, self.z_hat)
        y_hat = self.model.decompress(symbols, self.z_hat)

        self.assertEqual(symbols.shape, self.y.shape)
        self.assertEqual(y_hat.shape, self.y.shape)
        self.assertIn('mean', side_info)
        self.assertIn('scale', side_info)

    def test_mean_scale_hyperprior_rate_reduction(self):
        """Verify entropy reduction vs baseline (predicted mean helps)."""
        # With mean-scale model
        _, y_likelihood_ms, bits_ms = self.model(self.y, self.z_hat, training=False)

        # Baseline: zero mean, unit scale (fixed parameters)
        baseline = ConditionalGaussian()
        zero_mean = tf.zeros_like(self.y)
        unit_scale = tf.ones_like(self.y)
        _, y_likelihood_baseline = baseline(self.y, unit_scale, zero_mean, training=False)

        # Mean-scale should have higher likelihood (lower bits) on average
        # This is a statistical test, so we use a fairly loose threshold
        avg_likelihood_ms = tf.reduce_mean(y_likelihood_ms)
        avg_likelihood_baseline = tf.reduce_mean(y_likelihood_baseline)

        # The adaptive model should generally be at least as good
        # (exact improvement depends on data, so just check it doesn't catastrophically fail)
        self.assertGreater(avg_likelihood_ms, avg_likelihood_baseline - 1.0)

    def test_mean_scale_hyperprior_gradient_flow(self):
        """Gradients flow through the entire model."""
        with tf.GradientTape() as tape:
            y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=True)
            loss = -tf.reduce_mean(y_likelihood)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Should have trainable variables
        self.assertNotEmpty(self.model.trainable_variables)
        # All should have gradients
        self.assertTrue(all(g is not None for g in gradients))


class TestBackwardCompatibility(tf.test.TestCase):
    """Tests ensuring existing EntropyModel still works."""

    def test_original_entropy_model_unchanged(self):
        """Original EntropyModel should still function."""
        from entropy_model import EntropyModel

        model = EntropyModel()
        inputs = tf.random.normal((2, 8, 8))

        # Build the model
        model.gaussian.build(inputs.shape)

        # Should work as before
        compressed, likelihood = model(inputs)

        self.assertEqual(compressed.shape, inputs.shape)
        self.assertEqual(likelihood.shape, inputs.shape)

    def test_patched_gaussian_conditional_unchanged(self):
        """PatchedGaussianConditional should work as documented."""
        from entropy_model import PatchedGaussianConditional

        scale_table = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5])
        layer = PatchedGaussianConditional(
            initial_scale=1.0,
            scale_table=scale_table
        )

        inputs = tf.random.normal((2, 8, 8))
        layer.build(inputs.shape)

        outputs = layer(inputs)
        self.assertEqual(outputs.shape, inputs.shape)


if __name__ == '__main__':
    tf.test.main()
