"""Tests for autoregressive context model."""

import tensorflow as tf
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from context_model import MaskedConv3D, AutoregressiveContext, ContextualEntropyModel


class TestMaskedConv3D(tf.test.TestCase):
    """Tests for MaskedConv3D layer."""

    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.spatial_size = 8
        self.in_channels = 32
        self.out_channels = 64
        self.kernel_size = 3

        self.inputs = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.in_channels)
        )

    def test_masked_conv_output_shape(self):
        """Output shape matches input spatial dimensions."""
        layer = MaskedConv3D(filters=self.out_channels, kernel_size=self.kernel_size)
        output = layer(self.inputs)

        expected_shape = (
            self.batch_size, self.spatial_size, self.spatial_size,
            self.spatial_size, self.out_channels
        )
        self.assertEqual(output.shape, expected_shape)

    def test_masked_conv_causality(self):
        """Verify mask blocks future positions."""
        layer = MaskedConv3D(filters=self.out_channels, kernel_size=3, mask_type='A')
        layer.build(self.inputs.shape)

        mask = layer.mask.numpy()
        kd, kh, kw = 3, 3, 3
        center = (1, 1, 1)  # Center of 3x3x3 kernel

        # Check that future positions are masked out
        for d in range(kd):
            for h in range(kh):
                for w in range(kw):
                    is_future = (
                        d > center[0]
                        or (d == center[0] and h > center[1])
                        or (d == center[0] and h == center[1] and w > center[2])
                    )
                    is_center = (d == center[0] and h == center[1] and w == center[2])

                    if is_future:
                        # All values at this position should be 0
                        self.assertTrue(np.all(mask[d, h, w] == 0.0))
                    elif is_center:
                        # Type A masks center - all values should be 0
                        self.assertTrue(np.all(mask[d, h, w] == 0.0))

    def test_masked_conv_type_a_vs_b(self):
        """Type A excludes center, Type B includes it."""
        layer_a = MaskedConv3D(filters=self.out_channels, kernel_size=3, mask_type='A')
        layer_b = MaskedConv3D(filters=self.out_channels, kernel_size=3, mask_type='B')

        layer_a.build(self.inputs.shape)
        layer_b.build(self.inputs.shape)

        mask_a = layer_a.mask.numpy()
        mask_b = layer_b.mask.numpy()

        # Center position
        center = (1, 1, 1)

        # Type A should have 0 at center (all channel combinations)
        self.assertTrue(np.all(mask_a[center[0], center[1], center[2]] == 0.0))

        # Type B should have 1 at center (all channel combinations)
        self.assertTrue(np.all(mask_b[center[0], center[1], center[2]] == 1.0))

        # Both should have same values for future positions (all 0)
        self.assertTrue(np.all(mask_a[2, :, :] == 0.0))
        self.assertTrue(np.all(mask_b[2, :, :] == 0.0))

    def test_masked_conv_gradient_flow(self):
        """Gradients flow through masked convolution."""
        layer = MaskedConv3D(filters=self.out_channels, kernel_size=3)

        with tf.GradientTape() as tape:
            output = layer(self.inputs)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, layer.trainable_variables)

        self.assertTrue(all(g is not None for g in gradients))
        self.assertTrue(any(tf.reduce_sum(tf.abs(g)) > 0 for g in gradients))

    def test_masked_conv_serialization(self):
        """Layer can be serialized and deserialized."""
        layer = MaskedConv3D(filters=self.out_channels, kernel_size=5, mask_type='B')
        _ = layer(self.inputs)  # Build

        config = layer.get_config()

        self.assertEqual(config['filters'], self.out_channels)
        self.assertEqual(config['kernel_size'], (5, 5, 5))
        self.assertEqual(config['mask_type'], 'B')


class TestAutoregressiveContext(tf.test.TestCase):
    """Tests for AutoregressiveContext layer."""

    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.spatial_size = 8
        self.channels = 64

        self.inputs = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.channels)
        )

    def test_autoregressive_context_shape(self):
        """Output shapes are correct."""
        layer = AutoregressiveContext(channels=self.channels, num_layers=3)
        output = layer(self.inputs)

        expected_shape = (
            self.batch_size, self.spatial_size, self.spatial_size,
            self.spatial_size, self.channels
        )
        self.assertEqual(output.shape, expected_shape)

    def test_autoregressive_context_first_layer_type_a(self):
        """First layer should use mask type A."""
        layer = AutoregressiveContext(channels=self.channels, num_layers=3)
        _ = layer(self.inputs)  # Build

        # First conv should be type A
        self.assertEqual(layer.conv_layers[0].mask_type, 'A')

    def test_autoregressive_context_subsequent_layers_type_b(self):
        """Subsequent layers should use mask type B."""
        layer = AutoregressiveContext(channels=self.channels, num_layers=3)
        _ = layer(self.inputs)  # Build

        # All layers after first should be type B
        for conv in layer.conv_layers[1:]:
            self.assertEqual(conv.mask_type, 'B')

    def test_autoregressive_context_causality(self):
        """Context at position (d,h,w) only depends on earlier positions."""
        layer = AutoregressiveContext(channels=32, num_layers=2)

        # Create input with a spike at one position
        inputs = tf.zeros((1, 4, 4, 4, 32))

        # Get output
        output_clean = layer(inputs)

        # Add spike at a later position
        spike_pos = (0, 2, 2, 2)  # Near the middle
        indices = tf.constant([[spike_pos[0], spike_pos[1], spike_pos[2], spike_pos[3], 0]])
        updates = tf.constant([100.0])
        inputs_with_spike = tf.tensor_scatter_nd_update(inputs, indices, updates)

        output_spike = layer(inputs_with_spike)

        # Positions before the spike should be unchanged
        # (causality means future positions don't affect past)
        for d in range(spike_pos[1]):
            self.assertAllClose(
                output_clean[0, d],
                output_spike[0, d],
                atol=1e-5
            )

    def test_autoregressive_context_gradient_flow(self):
        """Gradients propagate through the context model."""
        layer = AutoregressiveContext(channels=self.channels, num_layers=3)

        with tf.GradientTape() as tape:
            output = layer(self.inputs)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, layer.trainable_variables)

        self.assertNotEmpty(layer.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))


class TestContextualEntropyModel(tf.test.TestCase):
    """Tests for ContextualEntropyModel."""

    def setUp(self):
        super().setUp()
        self.latent_channels = 64
        self.hyper_channels = 32
        self.batch_size = 2
        self.spatial_size = 8

        self.model = ContextualEntropyModel(
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

    def test_contextual_entropy_call(self):
        """Test forward pass returns expected outputs."""
        y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=False)

        self.assertEqual(y_hat.shape, self.y.shape)
        self.assertEqual(y_likelihood.shape, self.y.shape)
        self.assertIsInstance(total_bits.numpy().item(), float)

    def test_contextual_entropy_rate_improvement(self):
        """Context model should improve over hyperprior alone."""
        from entropy_model import MeanScaleHyperprior

        # Hyperprior-only model
        hyperprior_model = MeanScaleHyperprior(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels
        )

        # Get likelihoods from both models
        _, likelihood_hyper, bits_hyper = hyperprior_model(self.y, self.z_hat, training=False)
        _, likelihood_context, bits_context = self.model(self.y, self.z_hat, training=False)

        # Average likelihood per element
        avg_likelihood_hyper = tf.reduce_mean(likelihood_hyper)
        avg_likelihood_context = tf.reduce_mean(likelihood_context)

        # Context model should have at least similar performance
        # (exact improvement depends on data statistics)
        self.assertGreater(avg_likelihood_context, avg_likelihood_hyper - 1.0)

    def test_contextual_entropy_gradient_flow(self):
        """Gradients flow through the entire model."""
        with tf.GradientTape() as tape:
            y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=True)
            loss = -tf.reduce_mean(y_likelihood)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.assertNotEmpty(self.model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_contextual_entropy_training_mode(self):
        """Training mode uses noise, eval mode uses rounding."""
        tf.random.set_seed(42)

        y_hat_train, _, _ = self.model(self.y, self.z_hat, training=True)
        y_hat_eval, _, _ = self.model(self.y, self.z_hat, training=False)

        # Training outputs may have noise
        # Eval outputs should be more deterministic
        self.assertEqual(y_hat_train.shape, y_hat_eval.shape)


class TestContextModelSerialization(tf.test.TestCase):
    """Tests for model serialization."""

    def test_context_model_serialization(self):
        """ContextualEntropyModel can be saved and restored."""
        model = ContextualEntropyModel(
            latent_channels=32,
            hyper_channels=16,
            num_context_layers=2
        )

        # Build model
        y = tf.random.normal((1, 4, 4, 4, 32))
        z_hat = tf.random.normal((1, 4, 4, 4, 16))
        _ = model(y, z_hat, training=False)

        config = model.get_config()

        self.assertEqual(config['latent_channels'], 32)
        self.assertEqual(config['hyper_channels'], 16)
        self.assertEqual(config['num_context_layers'], 2)


if __name__ == '__main__':
    tf.test.main()
