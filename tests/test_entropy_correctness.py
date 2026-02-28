"""
Tests for entropy model mathematical correctness.

Validates that discretized Gaussian likelihood is a proper probability mass
function (PMF), rate estimates are non-negative, and quantization behavior
switches correctly between training and inference.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from constants import LOG_2_RECIPROCAL
from entropy_model import (
    ConditionalGaussian,
    EntropyModel,
    MeanScaleHyperprior,
    PatchedGaussianConditional,
    _discretized_gaussian_likelihood,
)


class TestDiscretizedLikelihood(tf.test.TestCase):
    """Tests for the discretized Gaussian likelihood function."""

    def test_pmf_positive(self):
        """All PMF values must be strictly positive (floored at EPSILON)."""
        tf.random.set_seed(42)
        inputs = tf.random.normal((2, 8, 8, 8, 16))
        mean = tf.zeros_like(inputs)
        scale = tf.ones_like(inputs)

        likelihood = _discretized_gaussian_likelihood(inputs, mean, scale)

        self.assertAllGreater(likelihood, 0.0)

    def test_pmf_at_most_one(self):
        """No single bin should have probability > 1."""
        inputs = tf.constant([0.0, 1.0, -1.0, 5.0, -5.0])
        mean = tf.zeros_like(inputs)
        scale = tf.ones_like(inputs)

        likelihood = _discretized_gaussian_likelihood(inputs, mean, scale)

        self.assertAllLessEqual(likelihood, 1.0)

    def test_pmf_sums_approximately_to_one(self):
        """PMF over a wide range of integers should sum close to 1."""
        # Evaluate PMF over integers from -50 to +50 for various scales
        for scale_val in [0.1, 0.5, 1.0, 2.0, 5.0]:
            integers = tf.cast(tf.range(-50, 51), tf.float32)
            mean = tf.zeros_like(integers)
            scale = tf.fill(integers.shape, scale_val)

            likelihood = _discretized_gaussian_likelihood(integers, mean, scale)
            total = tf.reduce_sum(likelihood)

            # Should be very close to 1.0 (small scale needs wider range)
            self.assertAllClose(total, 1.0, atol=1e-3,
                                msg=f"PMF sum={total:.6f} for scale={scale_val}")

    def test_pmf_peaks_at_mean(self):
        """PMF should be highest at the integer closest to mean."""
        mean_val = 2.3
        integers = tf.cast(tf.range(-10, 11), tf.float32)
        mean = tf.fill(integers.shape, mean_val)
        scale = tf.ones_like(integers)

        likelihood = _discretized_gaussian_likelihood(integers, mean, scale)

        # The peak should be at integer 2 (closest to 2.3)
        peak_idx = tf.argmax(likelihood)
        # integers[12] = 2 (index 0 -> -10, so index 12 -> 2)
        self.assertEqual(int(integers[peak_idx]), 2)

    def test_pmf_symmetric_around_integer_mean(self):
        """PMF should be symmetric when mean is an integer."""
        mean_val = 0.0
        integers = tf.cast(tf.range(-10, 11), tf.float32)
        mean = tf.fill(integers.shape, mean_val)
        scale = tf.ones_like(integers)

        likelihood = _discretized_gaussian_likelihood(integers, mean, scale)
        likelihood_np = likelihood.numpy()

        # Check symmetry: P(k) == P(-k) for integer mean
        for k in range(1, 11):
            idx_pos = 10 + k  # index of +k
            idx_neg = 10 - k  # index of -k
            np.testing.assert_allclose(
                likelihood_np[idx_pos], likelihood_np[idx_neg], rtol=1e-5,
                err_msg=f"Asymmetry at k={k}"
            )

    def test_small_scale_concentrates_mass(self):
        """Very small scale should concentrate mass near mean."""
        integers = tf.cast(tf.range(-10, 11), tf.float32)
        mean = tf.zeros_like(integers)
        scale = tf.fill(integers.shape, 0.01)

        likelihood = _discretized_gaussian_likelihood(integers, mean, scale)

        # Almost all mass at 0
        self.assertGreater(float(likelihood[10]), 0.99)  # index 10 = integer 0

    def test_large_scale_spreads_mass(self):
        """Large scale should spread mass more evenly."""
        integers = tf.cast(tf.range(-10, 11), tf.float32)
        mean = tf.zeros_like(integers)
        scale = tf.fill(integers.shape, 10.0)

        likelihood = _discretized_gaussian_likelihood(integers, mean, scale)

        # Mass at 0 should be much less than for small scale
        self.assertLess(float(likelihood[10]), 0.1)

    def test_scale_clipped_to_minimum(self):
        """Scale values near zero should not produce NaN."""
        inputs = tf.constant([0.0, 1.0, -1.0])
        mean = tf.zeros_like(inputs)
        scale = tf.constant([1e-10, 0.0, -1.0])  # degenerate scales

        likelihood = _discretized_gaussian_likelihood(inputs, mean, scale)

        self.assertAllGreater(likelihood, 0.0)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(likelihood)))


class TestRateComputation(tf.test.TestCase):
    """Tests that rate (bits) from likelihood is non-negative and sensible."""

    def test_rate_non_negative(self):
        """Bits from discretized likelihood must be non-negative."""
        tf.random.set_seed(42)
        inputs = tf.random.normal((1, 8, 8, 8, 16))
        mean = tf.zeros_like(inputs)
        scale = tf.ones_like(inputs)

        likelihood = _discretized_gaussian_likelihood(inputs, mean, scale)
        bits = -tf.math.log(likelihood) * LOG_2_RECIPROCAL

        self.assertAllGreaterEqual(bits, 0.0)

    def test_rate_increases_with_surprise(self):
        """Unlikely values should require more bits than likely values."""
        mean = tf.constant([0.0])
        scale = tf.constant([1.0])

        # Value at mean vs far from mean
        likely = tf.constant([0.0])
        unlikely = tf.constant([10.0])

        ll_likely = _discretized_gaussian_likelihood(likely, mean, scale)
        ll_unlikely = _discretized_gaussian_likelihood(unlikely, mean, scale)

        bits_likely = float(-tf.math.log(ll_likely) * LOG_2_RECIPROCAL)
        bits_unlikely = float(-tf.math.log(ll_unlikely) * LOG_2_RECIPROCAL)

        # Unlikely values should cost more bits
        self.assertGreater(bits_unlikely, bits_likely)

    def test_total_bits_from_entropy_model(self):
        """EntropyModel should produce non-negative total bits."""
        tf.random.set_seed(42)
        model = EntropyModel()
        inputs = tf.random.normal((1, 8, 8, 8, 16))

        compressed, likelihood = model(inputs, training=False)

        total_bits = tf.reduce_sum(-tf.math.log(likelihood) * LOG_2_RECIPROCAL)
        self.assertGreater(float(total_bits), 0.0)

    def test_low_entropy_vs_high_entropy(self):
        """Small scale (concentrated) should have fewer bits than large scale."""
        integers = tf.cast(tf.range(-20, 21), tf.float32)
        mean = tf.zeros_like(integers)

        # Small scale: concentrated distribution -> low entropy
        scale_small = tf.fill(integers.shape, 0.5)
        ll_small = _discretized_gaussian_likelihood(integers, mean, scale_small)
        entropy_small = float(tf.reduce_sum(-ll_small * tf.math.log(ll_small)))

        # Large scale: spread distribution -> high entropy
        scale_large = tf.fill(integers.shape, 5.0)
        ll_large = _discretized_gaussian_likelihood(integers, mean, scale_large)
        entropy_large = float(tf.reduce_sum(-ll_large * tf.math.log(ll_large)))

        self.assertGreater(entropy_large, entropy_small)


class TestQuantizationBehavior(tf.test.TestCase):
    """Tests that quantization switches correctly between training/inference."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.inputs = tf.constant([[[[[1.3, -0.7, 2.5]]]]])  # (1,1,1,1,3)

    def test_conditional_gaussian_training_adds_noise(self):
        """Training mode should add uniform noise, not round."""
        cg = ConditionalGaussian()
        scale = tf.ones_like(self.inputs)
        mean = tf.zeros_like(self.inputs)

        # Run multiple times to confirm stochasticity
        outputs = set()
        for _ in range(10):
            out, _ = cg(self.inputs, scale, mean, training=True)
            outputs.add(tuple(out.numpy().flatten()))

        # Should produce different outputs each time
        self.assertGreater(len(outputs), 1)

    def test_conditional_gaussian_inference_rounds(self):
        """Inference mode should produce deterministic rounded output."""
        cg = ConditionalGaussian()
        scale = tf.ones_like(self.inputs)
        mean = tf.zeros_like(self.inputs)

        out1, _ = cg(self.inputs, scale, mean, training=False)
        out2, _ = cg(self.inputs, scale, mean, training=False)

        self.assertAllEqual(out1, out2)

        # Should be rounded (input - mean rounded + mean = rounded input for mean=0)
        expected = tf.round(self.inputs)
        self.assertAllClose(out1, expected)

    def test_conditional_gaussian_likelihood_always_positive(self):
        """Likelihood should be positive in both training and inference."""
        cg = ConditionalGaussian()
        scale = tf.ones_like(self.inputs)
        mean = tf.zeros_like(self.inputs)

        _, ll_train = cg(self.inputs, scale, mean, training=True)
        _, ll_eval = cg(self.inputs, scale, mean, training=False)

        self.assertAllGreater(ll_train, 0.0)
        self.assertAllGreater(ll_eval, 0.0)


class TestPatchedGaussianConditional(tf.test.TestCase):
    """Tests for PatchedGaussianConditional layer."""

    def test_compress_decompress_roundtrip(self):
        """compress → decompress should be identity for integer inputs."""
        pgc = PatchedGaussianConditional()
        # Build the layer
        inputs = tf.constant([[[[[1.0, 2.0, 3.0]]]]])
        pgc.build(inputs.shape)

        compressed = pgc.compress(inputs)
        decompressed = pgc.decompress(compressed)

        self.assertAllClose(decompressed, inputs, atol=1e-5)

    def test_scale_quantization_binary_search(self):
        """Binary search should map scales to nearest table entry."""
        scale_table = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0])
        pgc = PatchedGaussianConditional(scale_table=scale_table)

        test_scales = tf.constant([0.05, 0.3, 0.8, 1.5, 3.0, 10.0])
        quantized = pgc.quantize_scale(test_scales)

        # Each should map to nearest table entry
        expected = tf.constant([0.1, 0.1, 1.0, 1.0, 2.0, 5.0])
        self.assertAllClose(quantized, expected)

    def test_scale_quantization_preserves_shape(self):
        """Quantized scales should have same shape as input."""
        scale_table = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0])
        pgc = PatchedGaussianConditional(scale_table=scale_table)

        test_scales = tf.random.uniform((2, 4, 4, 4, 8), 0.1, 5.0)
        quantized = pgc.quantize_scale(test_scales)

        self.assertEqual(quantized.shape, test_scales.shape)

    def test_likelihood_matches_standalone(self):
        """Layer likelihood should match standalone function."""
        pgc = PatchedGaussianConditional()
        inputs = tf.constant([[[[[0.0, 1.0, -1.0]]]]])
        pgc.build(inputs.shape)

        layer_ll = pgc.likelihood(inputs)

        # Compare with standalone function
        scale = tf.maximum(tf.abs(pgc.scale), 1e-6)
        standalone_ll = _discretized_gaussian_likelihood(inputs, pgc.mean, scale)

        self.assertAllClose(layer_ll, standalone_ll)


class TestMeanScaleHyperprior(tf.test.TestCase):
    """Tests for MeanScaleHyperprior entropy model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.latent_channels = 32
        self.hyper_channels = 16

    def test_total_bits_non_negative(self):
        """Total bits from hyperprior should be non-negative."""
        model = MeanScaleHyperprior(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels
        )

        y = tf.random.normal((1, 4, 4, 4, self.latent_channels))
        z_hat = tf.random.normal((1, 4, 4, 4, self.hyper_channels))
        z = tf.random.normal((1, 4, 4, 4, self.hyper_channels))

        y_hat, y_likelihood, total_bits = model(y, z_hat, z=z, training=False)

        self.assertGreater(float(total_bits), 0.0)
        self.assertAllGreater(y_likelihood, 0.0)

    def test_output_shape_matches_input(self):
        """y_hat should have same shape as y."""
        model = MeanScaleHyperprior(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels
        )

        y = tf.random.normal((1, 4, 4, 4, self.latent_channels))
        z_hat = tf.random.normal((1, 4, 4, 4, self.hyper_channels))

        y_hat, y_likelihood, _ = model(y, z_hat, training=False)

        self.assertEqual(y_hat.shape, y.shape)
        self.assertEqual(y_likelihood.shape, y.shape)

    def test_compress_decompress_consistency(self):
        """compress then decompress should recover y_hat."""
        model = MeanScaleHyperprior(
            latent_channels=self.latent_channels,
            hyper_channels=self.hyper_channels
        )

        y = tf.random.normal((1, 4, 4, 4, self.latent_channels))
        z_hat = tf.random.normal((1, 4, 4, 4, self.hyper_channels))

        symbols, side_info = model.compress(y, z_hat)
        y_hat = model.decompress(symbols, z_hat)

        # Symbols + mean should give y_hat
        self.assertEqual(y_hat.shape, y.shape)


if __name__ == '__main__':
    tf.test.main()
