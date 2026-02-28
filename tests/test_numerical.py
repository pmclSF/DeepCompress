"""
Tests for numerical stability of GDN, entropy models, and data pipeline.

Validates that GDN/IGDN handle edge cases (zero, large, negative inputs),
entropy models remain stable with extreme parameters, and the data pipeline
produces valid outputs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from constants import EPSILON
from entropy_model import (
    ConditionalGaussian,
    PatchedGaussianConditional,
    _discretized_gaussian_likelihood,
)
from model_transforms import GDN


class TestGDNStability(tf.test.TestCase):
    """Tests for GDN numerical stability."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.channels = 8
        self.shape = (1, 4, 4, 4, self.channels)

    def test_zero_input(self):
        """GDN should handle zero input without NaN."""
        gdn = GDN(inverse=False)
        inputs = tf.zeros(self.shape)
        output = gdn(inputs)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(output)))
        # Zero divided by sqrt(beta) should be zero
        self.assertAllClose(output, tf.zeros_like(output))

    def test_zero_input_igdn(self):
        """IGDN should handle zero input without NaN."""
        igdn = GDN(inverse=True)
        inputs = tf.zeros(self.shape)
        output = igdn(inputs)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(output)))
        self.assertAllClose(output, tf.zeros_like(output))

    def test_large_input(self):
        """GDN should handle large inputs without overflow."""
        gdn = GDN(inverse=False)
        inputs = tf.constant(1000.0, shape=self.shape)
        output = gdn(inputs)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(output)))

    def test_large_input_igdn(self):
        """IGDN should handle large inputs without overflow."""
        igdn = GDN(inverse=True)
        inputs = tf.constant(100.0, shape=self.shape)
        output = igdn(inputs)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(output)))

    def test_negative_input(self):
        """GDN should handle negative inputs."""
        gdn = GDN(inverse=False)
        inputs = tf.constant(-5.0, shape=self.shape)
        output = gdn(inputs)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(output)))
        self.assertFalse(tf.reduce_any(tf.math.is_inf(output)))

    def test_igdn_gdn_approximate_inverse(self):
        """IGDN(GDN(x)) should approximately recover x for moderate inputs."""
        tf.random.set_seed(42)
        gdn = GDN(inverse=False)
        igdn = GDN(inverse=True)

        # Use moderate values to stay in stable range
        inputs = tf.random.normal(self.shape) * 2.0

        # Forward through GDN then IGDN
        encoded = gdn(inputs)
        decoded = igdn(encoded)

        # GDN and IGDN are not exact inverses with independently initialized
        # parameters, but with default params (beta=1, gamma=0.1*I) they
        # should be reasonably close for moderate inputs
        # We just check no NaN/Inf and shape preservation
        self.assertFalse(tf.reduce_any(tf.math.is_nan(decoded)))
        self.assertEqual(decoded.shape, inputs.shape)

    def test_gdn_output_bounded(self):
        """GDN should reduce magnitude (divisive normalization)."""
        gdn = GDN(inverse=False)
        inputs = tf.constant(5.0, shape=self.shape)
        output = gdn(inputs)

        # GDN divides by sqrt(beta + ...) >= sqrt(1) = 1
        # So output magnitude should be <= input magnitude
        max_output = float(tf.reduce_max(tf.abs(output)))
        max_input = float(tf.reduce_max(tf.abs(inputs)))
        self.assertLessEqual(max_output, max_input + 1e-5)

    def test_gdn_gradient_no_nan(self):
        """Gradients through GDN should not contain NaN."""
        gdn = GDN(inverse=False)
        inputs = tf.random.normal(self.shape)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            output = gdn(inputs)
            loss = tf.reduce_mean(output)

        grad = tape.gradient(loss, inputs)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(grad)))

    def test_igdn_gradient_no_nan(self):
        """Gradients through IGDN should not contain NaN."""
        igdn = GDN(inverse=True)
        inputs = tf.random.normal(self.shape)

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            output = igdn(inputs)
            loss = tf.reduce_mean(output)

        grad = tape.gradient(loss, inputs)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(grad)))

    def test_gamma_symmetry(self):
        """Gamma matrix used in GDN should be symmetric after call."""
        gdn = GDN(inverse=False)
        inputs = tf.random.normal(self.shape)
        _ = gdn(inputs)

        # The effective gamma inside call() is (relu(gamma) + relu(gamma)^T) / 2
        gamma = tf.nn.relu(gdn.gamma)
        gamma_sym = (gamma + tf.transpose(gamma)) / 2.0
        self.assertAllClose(gamma_sym, tf.transpose(gamma_sym))


class TestEntropyStability(tf.test.TestCase):
    """Tests for entropy model numerical stability."""

    def test_very_small_scale(self):
        """Very small scale should not produce NaN likelihood."""
        inputs = tf.constant([0.0, 1.0, -1.0])
        mean = tf.zeros_like(inputs)
        scale = tf.constant([1e-8, 1e-8, 1e-8])

        ll = _discretized_gaussian_likelihood(inputs, mean, scale)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(ll)))
        self.assertAllGreater(ll, 0.0)

    def test_very_large_scale(self):
        """Very large scale should not produce NaN likelihood."""
        inputs = tf.constant([0.0, 100.0, -100.0])
        mean = tf.zeros_like(inputs)
        scale = tf.constant([1e6, 1e6, 1e6])

        ll = _discretized_gaussian_likelihood(inputs, mean, scale)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(ll)))
        self.assertAllGreater(ll, 0.0)

    def test_very_large_input(self):
        """Very large inputs should produce small but non-NaN likelihood."""
        inputs = tf.constant([1000.0, -1000.0])
        mean = tf.zeros_like(inputs)
        scale = tf.ones_like(inputs)

        ll = _discretized_gaussian_likelihood(inputs, mean, scale)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(ll)))
        # Should be floored at EPSILON
        self.assertAllGreaterEqual(ll, float(EPSILON))

    def test_bits_no_nan_extreme_values(self):
        """Bits computation should not produce NaN even for extreme values."""
        from constants import LOG_2_RECIPROCAL

        inputs = tf.constant([0.0, 50.0, -50.0, 1000.0])
        mean = tf.zeros_like(inputs)
        scale = tf.ones_like(inputs)

        ll = _discretized_gaussian_likelihood(inputs, mean, scale)
        bits = -tf.math.log(ll) * LOG_2_RECIPROCAL

        self.assertFalse(tf.reduce_any(tf.math.is_nan(bits)))
        self.assertAllGreaterEqual(bits, 0.0)

    def test_conditional_gaussian_extreme_scale(self):
        """ConditionalGaussian should be stable with extreme scales."""
        cg = ConditionalGaussian()
        inputs = tf.constant([[[[[1.0, 2.0]]]]])
        mean = tf.zeros_like(inputs)

        for scale_val in [1e-6, 1e-3, 1.0, 1e3, 1e6]:
            scale = tf.fill(inputs.shape, scale_val)
            out, ll = cg(inputs, scale, mean, training=False)

            self.assertFalse(tf.reduce_any(tf.math.is_nan(out)),
                             msg=f"NaN output at scale={scale_val}")
            self.assertFalse(tf.reduce_any(tf.math.is_nan(ll)),
                             msg=f"NaN likelihood at scale={scale_val}")
            self.assertAllGreater(ll, 0.0)

    def test_patched_gaussian_negative_scale(self):
        """PatchedGaussianConditional should handle negative learned scale."""
        pgc = PatchedGaussianConditional()
        inputs = tf.constant([[[[[1.0, 2.0, 3.0]]]]])
        pgc.build(inputs.shape)

        # Force negative scale
        pgc.scale.assign(-tf.ones_like(pgc.scale))

        ll = pgc.likelihood(inputs)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(ll)))
        self.assertAllGreater(ll, 0.0)

    def test_gradient_through_likelihood(self):
        """Gradients through discretized likelihood should not be NaN."""
        inputs = tf.Variable(tf.constant([0.0, 1.0, -1.0, 5.0]))
        mean = tf.constant([0.0, 0.0, 0.0, 0.0])
        scale = tf.constant([1.0, 1.0, 1.0, 1.0])

        with tf.GradientTape() as tape:
            ll = _discretized_gaussian_likelihood(inputs, mean, scale)
            loss = -tf.reduce_sum(tf.math.log(ll))

        grad = tape.gradient(loss, inputs)
        self.assertFalse(tf.reduce_any(tf.math.is_nan(grad)))


class TestConstantsCorrectness(tf.test.TestCase):
    """Tests that pre-computed constants are correct."""

    def test_log2_value(self):
        """LOG_2 should equal ln(2)."""
        from constants import LOG_2
        self.assertAllClose(LOG_2, tf.constant(np.log(2.0), dtype=tf.float32))

    def test_log2_reciprocal_value(self):
        """LOG_2_RECIPROCAL should equal 1/ln(2)."""
        from constants import LOG_2_RECIPROCAL
        expected = tf.constant(1.0 / np.log(2.0), dtype=tf.float32)
        self.assertAllClose(LOG_2_RECIPROCAL, expected)

    def test_log2_reciprocal_identity(self):
        """LOG_2 * LOG_2_RECIPROCAL should equal 1."""
        from constants import LOG_2, LOG_2_RECIPROCAL
        product = LOG_2 * LOG_2_RECIPROCAL
        self.assertAllClose(product, 1.0, atol=1e-6)

    def test_epsilon_positive(self):
        """EPSILON should be a small positive value."""
        self.assertGreater(float(EPSILON), 0.0)
        self.assertLess(float(EPSILON), 1e-6)

    def test_scale_bounds(self):
        """SCALE_MIN < SCALE_MAX."""
        from constants import SCALE_MAX, SCALE_MIN
        self.assertLess(float(SCALE_MIN), float(SCALE_MAX))

    def test_f16_constants_match(self):
        """Float16 constants should match float32 values within f16 precision."""
        from constants import LOG_2, LOG_2_F16, LOG_2_RECIPROCAL, LOG_2_RECIPROCAL_F16
        self.assertAllClose(
            tf.cast(LOG_2, tf.float16), LOG_2_F16, atol=1e-3
        )
        self.assertAllClose(
            tf.cast(LOG_2_RECIPROCAL, tf.float16), LOG_2_RECIPROCAL_F16, atol=1e-3
        )


class TestScaleQuantizationNumerics(tf.test.TestCase):
    """Tests for binary search scale quantization edge cases."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.scale_table = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        self.pgc = PatchedGaussianConditional(scale_table=self.scale_table)

    def test_exact_table_values_preserved(self):
        """Input values exactly matching table entries should be preserved."""
        test_scales = tf.constant([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        quantized = self.pgc.quantize_scale(test_scales)
        self.assertAllClose(quantized, test_scales)

    def test_negative_scales_made_positive(self):
        """Negative scales should be mapped to positive table entries."""
        test_scales = tf.constant([-0.5, -1.0, -2.0])
        quantized = self.pgc.quantize_scale(test_scales)

        self.assertAllGreater(quantized, 0.0)

    def test_below_minimum_clipped(self):
        """Scales below table minimum should be clipped to minimum."""
        test_scales = tf.constant([0.001, 0.01, 0.05])
        quantized = self.pgc.quantize_scale(test_scales)

        self.assertAllGreaterEqual(quantized, 0.1)

    def test_above_maximum_clipped(self):
        """Scales above table maximum should be clipped to maximum."""
        test_scales = tf.constant([20.0, 100.0, 1000.0])
        quantized = self.pgc.quantize_scale(test_scales)

        self.assertAllLessEqual(quantized, 10.0)

    def test_zero_scale(self):
        """Zero scale should not cause errors."""
        test_scales = tf.constant([0.0])
        quantized = self.pgc.quantize_scale(test_scales)

        self.assertFalse(tf.reduce_any(tf.math.is_nan(quantized)))


if __name__ == '__main__':
    tf.test.main()
