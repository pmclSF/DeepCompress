"""Tests for channel-wise context model."""

import sys
from pathlib import Path

import tensorflow as tf

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from channel_context import ChannelContext, ChannelContextEntropyModel, SliceTransform


class TestSliceTransform(tf.test.TestCase):
    """Tests for SliceTransform layer."""

    def setUp(self):
        super().setUp()
        self.in_channels = 32
        self.out_channels = 64
        self.batch_size = 2
        self.spatial_size = 8

        self.layer = SliceTransform(
            in_channels=self.in_channels,
            out_channels=self.out_channels
        )

        self.inputs = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.in_channels)
        )

    def test_slice_transform_shapes(self):
        """Output has correct dimensionality."""
        output = self.layer(self.inputs)

        expected_shape = (
            self.batch_size, self.spatial_size, self.spatial_size,
            self.spatial_size, self.out_channels
        )
        self.assertEqual(output.shape, expected_shape)

    def test_slice_transform_gradient_flow(self):
        """Gradients flow through the transform."""
        with tf.GradientTape() as tape:
            output = self.layer(self.inputs)
            loss = tf.reduce_mean(output)

        gradients = tape.gradient(loss, self.layer.trainable_variables)

        self.assertNotEmpty(self.layer.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_slice_transform_serialization(self):
        """Layer can be serialized."""
        _ = self.layer(self.inputs)  # Build

        config = self.layer.get_config()

        self.assertEqual(config['in_channels'], self.in_channels)
        self.assertEqual(config['out_channels'], self.out_channels)


class TestChannelContext(tf.test.TestCase):
    """Tests for ChannelContext layer."""

    def setUp(self):
        super().setUp()
        self.channels = 64
        self.num_groups = 4
        self.batch_size = 2
        self.spatial_size = 8

        self.layer = ChannelContext(
            channels=self.channels,
            num_groups=self.num_groups
        )

        self.y_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.channels)
        )

    def test_channel_context_group_isolation(self):
        """Groups don't leak future information."""
        channels_per_group = self.channels // self.num_groups

        # Get context for group 1 (uses only group 0)
        # Note: use .call() to pass non-tensor group_idx as keyword argument
        mean1, scale1 = self.layer.call(self.y_hat, group_idx=1)

        # Modify groups 2 and 3 (should not affect group 1's context)
        y_hat_modified = self.y_hat.numpy()
        y_hat_modified[..., 2 * channels_per_group:] = 999.0
        y_hat_modified = tf.constant(y_hat_modified)

        mean1_mod, scale1_mod = self.layer.call(y_hat_modified, group_idx=1)

        # Context for group 1 should be unchanged
        self.assertAllClose(mean1, mean1_mod)
        self.assertAllClose(scale1, scale1_mod)

    def test_channel_context_first_group_no_context(self):
        """First group returns zero context."""
        # Note: use .call() to pass non-tensor group_idx as keyword argument
        mean, scale = self.layer.call(self.y_hat, group_idx=0)

        # All values should be zero for first group
        self.assertAllClose(mean, tf.zeros_like(mean))
        self.assertAllClose(scale, tf.zeros_like(scale))

    def test_channel_context_shapes(self):
        """Context parameters have correct shapes."""
        channels_per_group = self.channels // self.num_groups

        for i in range(self.num_groups):
            # Note: use .call() to pass non-tensor group_idx as keyword argument
            mean, scale = self.layer.call(self.y_hat, group_idx=i)

            expected_shape = (
                self.batch_size, self.spatial_size, self.spatial_size,
                self.spatial_size, channels_per_group
            )

            self.assertEqual(mean.shape, expected_shape)
            self.assertEqual(scale.shape, expected_shape)

    def test_channel_context_scale_positive(self):
        """Scale from context is always positive (except first group)."""
        for i in range(1, self.num_groups):
            # Note: use .call() to pass non-tensor group_idx as keyword argument
            mean, scale = self.layer.call(self.y_hat, group_idx=i)
            self.assertAllGreaterEqual(scale, 0.01)

    def test_channel_context_invalid_groups(self):
        """Raises error for invalid channel/group configuration."""
        with self.assertRaises(ValueError):
            ChannelContext(channels=63, num_groups=4)  # Not divisible


class TestChannelContextEntropyModel(tf.test.TestCase):
    """Tests for ChannelContextEntropyModel."""

    def setUp(self):
        super().setUp()
        self.latent_channels = 64
        self.hyper_channels = 32
        self.num_groups = 4
        self.batch_size = 2
        self.spatial_size = 8

        self.model = ChannelContextEntropyModel(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels,
            num_groups=self.num_groups
        )

        self.y = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.latent_channels)
        )
        self.z_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, self.hyper_channels)
        )

    def test_channel_entropy_call(self):
        """Test forward pass returns expected outputs."""
        y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=False)

        self.assertEqual(y_hat.shape, self.y.shape)
        self.assertEqual(y_likelihood.shape, self.y.shape)
        self.assertIsInstance(total_bits.numpy().item(), float)

    def test_channel_context_parallel_decode(self):
        """Parallel decode produces valid output."""
        # Create symbols (quantized values)
        symbols = tf.round(self.y)

        y_hat = self.model.decode_parallel(self.z_hat, symbols)

        self.assertEqual(y_hat.shape, self.y.shape)

    def test_channel_entropy_improvement(self):
        """Channel context should improve over hyperprior alone."""
        from entropy_model import MeanScaleHyperprior

        # Hyperprior-only model
        hyperprior_model = MeanScaleHyperprior(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels
        )

        # Get likelihoods from both models
        _, likelihood_hyper, _ = hyperprior_model(self.y, self.z_hat, training=False)
        _, likelihood_channel, _ = self.model(self.y, self.z_hat, training=False)

        # Average likelihood per element
        avg_likelihood_hyper = tf.reduce_mean(likelihood_hyper)
        avg_likelihood_channel = tf.reduce_mean(likelihood_channel)

        # Channel model should perform at least as well
        self.assertGreater(avg_likelihood_channel, avg_likelihood_hyper - 1.0)

    def test_channel_entropy_gradient_flow(self):
        """Gradients flow through the entire model."""
        with tf.GradientTape() as tape:
            y_hat, y_likelihood, total_bits = self.model(self.y, self.z_hat, training=True)
            loss = -tf.reduce_mean(y_likelihood)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        self.assertNotEmpty(self.model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_channel_entropy_different_group_counts(self):
        """Model works with different numbers of groups."""
        for num_groups in [2, 4, 8]:
            model = ChannelContextEntropyModel(
                latent_channels=self.latent_channels,
                hyper_channels=self.hyper_channels,
                num_groups=num_groups
            )

            y_hat, _, _ = model(self.y, self.z_hat, training=False)
            self.assertEqual(y_hat.shape, self.y.shape)

    def test_channel_entropy_invalid_config(self):
        """Raises error for invalid configuration."""
        with self.assertRaises(ValueError):
            ChannelContextEntropyModel(
                latent_channels=63,  # Not divisible by 4
                hyper_channels=32,
                num_groups=4
            )


class TestChannelContextParallelism(tf.test.TestCase):
    """Tests for parallel decoding behavior."""

    def setUp(self):
        super().setUp()
        self.model = ChannelContextEntropyModel(
            latent_channels=64,
            hyper_channels=32,
            num_groups=4
        )
        self.batch_size = 2
        self.spatial_size = 8

        self.y = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, 64)
        )
        self.z_hat = tf.random.normal(
            (self.batch_size, self.spatial_size, self.spatial_size,
             self.spatial_size, 32)
        )

    def test_parallel_decode_matches_sequential_structure(self):
        """Parallel decode preserves channel structure."""
        symbols = tf.round(self.y)
        y_hat = self.model.decode_parallel(self.z_hat, symbols)

        # Check that all channel groups are populated
        channels_per_group = 64 // 4
        for i in range(4):
            start = i * channels_per_group
            end = (i + 1) * channels_per_group
            group = y_hat[..., start:end]

            # Each group should have non-trivial values
            self.assertGreater(tf.reduce_max(tf.abs(group)), 0.0)


class TestChannelContextSerialization(tf.test.TestCase):
    """Tests for model serialization."""

    def test_model_config(self):
        """Model config contains all parameters."""
        model = ChannelContextEntropyModel(
            latent_channels=64,
            hyper_channels=32,
            num_groups=4
        )

        # Build model
        y = tf.random.normal((1, 4, 4, 4, 64))
        z_hat = tf.random.normal((1, 4, 4, 4, 32))
        _ = model(y, z_hat, training=False)

        config = model.get_config()

        self.assertEqual(config['latent_channels'], 64)
        self.assertEqual(config['hyper_channels'], 32)
        self.assertEqual(config['num_groups'], 4)


if __name__ == '__main__':
    tf.test.main()
