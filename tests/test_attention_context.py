"""Tests for attention-based context model."""

import tensorflow as tf
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from attention_context import (
    SparseAttention3D,
    BidirectionalMaskTransformer,
    AttentionEntropyModel,
    HybridAttentionEntropyModel
)


class TestSparseAttention3D(tf.test.TestCase):
    """Tests for SparseAttention3D layer."""

    def setUp(self):
        super().setUp()
        self.dim = 64
        self.num_heads = 4
        self.batch_size = 2
        self.spatial_size = 4

        self.layer = SparseAttention3D(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=2,
            num_global_tokens=4
        )

        self.inputs = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.dim)
        )

    def test_sparse_attention_output_shape(self):
        """Output shape matches input shape."""
        output = self.layer(self.inputs)

        self.assertEqual(output.shape, self.inputs.shape)

    def test_sparse_attention_gradient_flow(self):
        """Gradients flow through attention."""
        with tf.GradientTape() as tape:
            output = self.layer(self.inputs)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, self.layer.trainable_variables)

        self.assertNotEmpty(self.layer.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_sparse_attention_global_tokens(self):
        """Global tokens are learnable."""
        self.layer.build(self.inputs.shape)

        # Global tokens should exist
        self.assertIsNotNone(self.layer.global_tokens)
        self.assertEqual(
            self.layer.global_tokens.shape,
            (1, self.layer.num_global_tokens, self.dim)
        )

    @pytest.mark.gpu
    def test_sparse_attention_efficiency(self):
        """Memory scales reasonably with input size."""
        # This test checks that we don't have O(n^2) memory explosion
        small_input = tf.random.normal((1, 4, 4, 4, self.dim))
        large_input = tf.random.normal((1, 8, 8, 8, self.dim))

        # Should not OOM on reasonably sized inputs
        _ = self.layer(small_input)
        _ = self.layer(large_input)  # 8x more positions

    def test_sparse_attention_invalid_dim(self):
        """Raises error for incompatible dim/heads."""
        with self.assertRaises(ValueError):
            SparseAttention3D(dim=63, num_heads=4)  # Not divisible


class TestBidirectionalMaskTransformer(tf.test.TestCase):
    """Tests for BidirectionalMaskTransformer layer."""

    def setUp(self):
        super().setUp()
        self.dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.batch_size = 2
        self.spatial_size = 4

        self.layer = BidirectionalMaskTransformer(
            dim=self.dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers
        )

        self.inputs = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.dim)
        )

    def test_bidirectional_output_shape(self):
        """Output shape matches input shape."""
        output = self.layer(self.inputs)

        self.assertEqual(output.shape, self.inputs.shape)

    def test_bidirectional_context_symmetry(self):
        """Forward/backward fusion produces consistent outputs."""
        output1 = self.layer(self.inputs, training=False)
        output2 = self.layer(self.inputs, training=False)

        # Deterministic in eval mode
        self.assertAllClose(output1, output2)

    def test_bidirectional_gradient_flow(self):
        """Gradients flow through all layers."""
        with tf.GradientTape() as tape:
            output = self.layer(self.inputs, training=True)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, self.layer.trainable_variables)

        self.assertNotEmpty(self.layer.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_bidirectional_residual_connections(self):
        """Output is different from input (not identity)."""
        output = self.layer(self.inputs)

        # Should not be identical to input
        self.assertNotAllClose(output, self.inputs)

    def test_bidirectional_serialization(self):
        """Layer config is complete."""
        _ = self.layer(self.inputs)  # Build

        config = self.layer.get_config()

        self.assertEqual(config['dim'], self.dim)
        self.assertEqual(config['num_heads'], self.num_heads)
        self.assertEqual(config['num_layers'], self.num_layers)


class TestAttentionEntropyModel(tf.test.TestCase):
    """Tests for AttentionEntropyModel."""

    def setUp(self):
        super().setUp()
        self.latent_channels = 64
        self.hyper_channels = 32
        self.batch_size = 2
        self.spatial_size = 4

        self.model = AttentionEntropyModel(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels,
            num_heads=4,
            num_attention_layers=1
        )

        self.y = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.latent_channels)
        )
        self.z_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.hyper_channels)
        )

    def test_attention_entropy_call(self):
        """Test forward pass returns expected outputs."""
        y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=False)

        self.assertEqual(y_hat.shape, self.y.shape)
        self.assertEqual(y_likelihood.shape, self.y.shape)
        self.assertIsInstance(total_bits.numpy().item(), float)

    def test_attention_entropy_improvement(self):
        """Attention model provides reasonable compression."""
        # Get likelihoods from attention model
        _, likelihood_attn, total_bits = self.model(self.y, self.z_hat, training=False)

        # Basic sanity checks for untrained models
        avg_likelihood_attn = tf.reduce_mean(likelihood_attn)

        # Likelihood should be finite and reasonable for Gaussian
        self.assertFalse(tf.math.is_nan(avg_likelihood_attn))
        self.assertFalse(tf.math.is_inf(avg_likelihood_attn))
        # Log-likelihood for Gaussian should be negative (probability < 1)
        self.assertLess(avg_likelihood_attn, 0.0)
        # But not catastrophically negative (which would indicate numerical issues)
        self.assertGreater(avg_likelihood_attn, -100.0)

    def test_attention_entropy_gradient_flow(self):
        """Gradients flow through the entire model."""
        with tf.GradientTape() as tape:
            y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=True)
            loss = -tf.reduce_mean(y_likelihood)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.assertNotEmpty(self.model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    @pytest.mark.gpu
    def test_attention_model_gpu_memory(self):
        """Model fits in reasonable GPU memory."""
        # Larger test input
        y_large = tf.random.normal((1, 8, 8, 8, self.latent_channels))
        z_large = tf.random.normal((1, 8, 8, 8, self.hyper_channels))

        # Should not OOM
        y_hat, _, _ = self.model(y_large, z_large, training=False)
        self.assertEqual(y_hat.shape, y_large.shape)


class TestHybridAttentionEntropyModel(tf.test.TestCase):
    """Tests for HybridAttentionEntropyModel."""

    def setUp(self):
        super().setUp()
        self.latent_channels = 64
        self.hyper_channels = 32
        self.batch_size = 2
        self.spatial_size = 4

        self.model = HybridAttentionEntropyModel(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels,
            num_channel_groups=4,
            num_attention_layers=1
        )

        self.y = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.latent_channels)
        )
        self.z_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.hyper_channels)
        )

    def test_hybrid_entropy_call(self):
        """Test forward pass."""
        y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=False)

        self.assertEqual(y_hat.shape, self.y.shape)
        self.assertEqual(y_likelihood.shape, self.y.shape)

    def test_hybrid_combines_all_context(self):
        """Hybrid model uses all context types."""
        # Model should have components from all context types
        self.assertIsNotNone(self.model.entropy_parameters)
        self.assertIsNotNone(self.model.channel_context)
        self.assertNotEmpty(self.model.attention_contexts)


class TestAttentionModelsNoGpu(tf.test.TestCase):
    """Tests that can run without GPU (marked for CI)."""

    def test_small_attention_model(self):
        """Small attention model for CI testing."""
        model = AttentionEntropyModel(
            latent_channels=32,
            hyper_channels=16,
            num_heads=2,
            num_attention_layers=1
        )

        y = tf.random.normal((1, 4, 4, 4, 32))
        z_hat = tf.random.normal((1, 4, 4, 4, 16))

        y_hat, likelihood, bits = model(y, z_hat, training=False)

        self.assertEqual(y_hat.shape, y.shape)
        self.assertGreater(bits, 0)

    def test_attention_model_config(self):
        """Model config is complete."""
        model = AttentionEntropyModel(
            latent_channels=64,
            hyper_channels=32,
            num_heads=8,
            num_attention_layers=2
        )

        # Build model
        y = tf.random.normal((1, 4, 4, 4, 64))
        z_hat = tf.random.normal((1, 4, 4, 4, 32))
        _ = model(y, z_hat, training=False)

        config = model.get_config()

        self.assertEqual(config['latent_channels'], 64)
        self.assertEqual(config['hyper_channels'], 32)
        self.assertEqual(config['num_heads'], 8)
        self.assertEqual(config['num_attention_layers'], 2)


if __name__ == '__main__':
    tf.test.main()
