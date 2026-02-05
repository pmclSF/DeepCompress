"""
Performance regression tests for DeepCompress optimizations.

These tests verify that:
1. Optimizations don't break functionality (correctness)
2. Optimizations provide measurable improvements (performance)
3. Memory usage is within expected bounds

Run with: pytest tests/test_performance.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import tensorflow as tf
import numpy as np
import time


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope='module')
def tf_setup():
    """Configure TensorFlow for testing."""
    tf.random.set_seed(42)
    np.random.seed(42)
    yield


@pytest.fixture
def sample_latent():
    """Create a sample latent tensor for testing."""
    return tf.random.normal((1, 8, 8, 8, 32), dtype=tf.float32)


@pytest.fixture
def sample_scale_table():
    """Create a standard scale table."""
    return tf.constant(
        [0.01 * (2 ** (i / 4)) for i in range(64)],
        dtype=tf.float32
    )


# =============================================================================
# Constants Tests
# =============================================================================

class TestConstants:
    """Test pre-computed constants."""

    def test_log2_constant_accuracy(self, tf_setup):
        """Verify LOG_2 constant matches tf.math.log(2.0)."""
        from constants import LOG_2
        expected = tf.math.log(2.0)
        np.testing.assert_allclose(LOG_2.numpy(), expected.numpy(), rtol=1e-6)

    def test_log2_reciprocal_accuracy(self, tf_setup):
        """Verify LOG_2_RECIPROCAL is correct."""
        from constants import LOG_2_RECIPROCAL
        expected = 1.0 / np.log(2.0)
        np.testing.assert_allclose(LOG_2_RECIPROCAL.numpy(), expected, rtol=1e-6)

    def test_bits_calculation_equivalence(self, tf_setup):
        """Verify bits calculation with constant matches original."""
        from constants import LOG_2_RECIPROCAL

        log_likelihood = tf.random.uniform((100,), -10.0, 0.0)

        # Original method
        bits_original = -log_likelihood / tf.math.log(2.0)

        # Optimized method
        bits_optimized = -log_likelihood * LOG_2_RECIPROCAL

        np.testing.assert_allclose(
            bits_original.numpy(),
            bits_optimized.numpy(),
            rtol=1e-5
        )


# =============================================================================
# Scale Quantization Tests
# =============================================================================

class TestScaleQuantization:
    """Test binary search scale quantization."""

    def test_quantization_correctness(self, tf_setup, sample_scale_table):
        """Verify binary search produces correct quantization."""
        from entropy_model import PatchedGaussianConditional

        layer = PatchedGaussianConditional(scale_table=sample_scale_table)

        # Test various scale values
        test_scales = tf.constant([0.015, 0.05, 0.1, 0.5, 1.0])
        quantized = layer.quantize_scale(test_scales)

        # Each quantized value should be in the scale table
        for q in quantized.numpy():
            assert q in sample_scale_table.numpy(), f"{q} not in scale table"

    def test_quantization_nearest_neighbor(self, tf_setup, sample_scale_table):
        """Verify quantization picks nearest neighbor."""
        from entropy_model import PatchedGaussianConditional

        layer = PatchedGaussianConditional(scale_table=sample_scale_table)

        # Test a value that's exactly between two table entries
        table_np = sample_scale_table.numpy()
        midpoint = (table_np[10] + table_np[11]) / 2

        # Slightly below midpoint should go to lower value (ensure float32 dtype)
        below = tf.constant([midpoint - 0.0001], dtype=tf.float32)
        q_below = layer.quantize_scale(below)
        assert q_below.numpy()[0] == table_np[10]

        # Slightly above midpoint should go to higher value
        above = tf.constant([midpoint + 0.0001], dtype=tf.float32)
        q_above = layer.quantize_scale(above)
        assert q_above.numpy()[0] == table_np[11]

    def test_quantization_clipping(self, tf_setup, sample_scale_table):
        """Verify out-of-range values are clipped."""
        from entropy_model import PatchedGaussianConditional

        layer = PatchedGaussianConditional(scale_table=sample_scale_table)
        table_np = sample_scale_table.numpy()

        # Very small value should map to minimum (ensure float32 dtype)
        small = tf.constant([0.001], dtype=tf.float32)
        q_small = layer.quantize_scale(small)
        assert q_small.numpy()[0] == table_np[0]

        # Value larger than max should map to maximum (551.09 is max)
        large = tf.constant([1000.0], dtype=tf.float32)
        q_large = layer.quantize_scale(large)
        assert q_large.numpy()[0] == table_np[-1]

    def test_quantization_batch_consistency(self, tf_setup, sample_scale_table):
        """Verify batch quantization matches element-wise."""
        from entropy_model import PatchedGaussianConditional

        layer = PatchedGaussianConditional(scale_table=sample_scale_table)

        # Create batch input
        batch = tf.random.uniform((4, 8, 8, 8, 16), 0.01, 1.0)
        batch_quantized = layer.quantize_scale(batch)

        # Quantize first element individually
        single = batch[0:1, 0:1, 0:1, 0:1, 0:1]
        single_quantized = layer.quantize_scale(single)

        np.testing.assert_allclose(
            batch_quantized[0, 0, 0, 0, 0].numpy(),
            single_quantized[0, 0, 0, 0, 0].numpy()
        )


# =============================================================================
# Vectorized Mask Tests
# =============================================================================

class TestVectorizedMask:
    """Test vectorized mask creation."""

    def test_mask_shape(self, tf_setup):
        """Verify mask has correct shape."""
        from context_model import MaskedConv3D

        layer = MaskedConv3D(filters=64, kernel_size=5, mask_type='A')
        layer.build((None, 8, 8, 8, 32))

        assert layer.mask.shape == (5, 5, 5, 32, 64)

    def test_mask_type_a_excludes_center(self, tf_setup):
        """Verify mask type A excludes center position."""
        from context_model import MaskedConv3D

        layer = MaskedConv3D(filters=4, kernel_size=3, mask_type='A')
        layer.build((None, 8, 8, 8, 2))

        mask = layer.mask.numpy()
        center = mask[1, 1, 1, :, :]  # Center of 3x3x3 kernel
        assert np.all(center == 0), "Mask type A should exclude center"

    def test_mask_type_b_includes_center(self, tf_setup):
        """Verify mask type B includes center position."""
        from context_model import MaskedConv3D

        layer = MaskedConv3D(filters=4, kernel_size=3, mask_type='B')
        layer.build((None, 8, 8, 8, 2))

        mask = layer.mask.numpy()
        center = mask[1, 1, 1, :, :]  # Center of 3x3x3 kernel
        assert np.all(center == 1), "Mask type B should include center"

    def test_mask_causal_structure(self, tf_setup):
        """Verify mask follows causal structure."""
        from context_model import MaskedConv3D

        layer = MaskedConv3D(filters=4, kernel_size=3, mask_type='A')
        layer.build((None, 8, 8, 8, 2))

        mask = layer.mask.numpy()

        # Future positions should be zero
        # Position (2, 1, 1) is after center (1, 1, 1) in raster order
        assert np.all(mask[2, 1, 1, :, :] == 0), "Future d positions should be masked"
        assert np.all(mask[1, 2, 1, :, :] == 0), "Future h positions should be masked"
        assert np.all(mask[1, 1, 2, :, :] == 0), "Future w positions should be masked"

        # Past positions should be one
        assert np.all(mask[0, 1, 1, :, :] == 1), "Past d positions should not be masked"
        assert np.all(mask[1, 0, 1, :, :] == 1), "Past h positions should not be masked"
        assert np.all(mask[1, 1, 0, :, :] == 1), "Past w positions should not be masked"


# =============================================================================
# Windowed Attention Tests
# =============================================================================

class TestWindowedAttention:
    """Test windowed attention implementation."""

    def test_output_shape(self, tf_setup):
        """Verify windowed attention preserves shape."""
        from attention_context import WindowedAttention3D

        layer = WindowedAttention3D(dim=32, num_heads=4, window_size=4)
        x = tf.random.normal((2, 16, 16, 16, 32))
        out = layer(x)

        assert out.shape == x.shape

    def test_window_partition_unpartition(self, tf_setup):
        """Verify window partition/unpartition are inverses."""
        from attention_context import WindowedAttention3D

        layer = WindowedAttention3D(dim=32, num_heads=4, window_size=4)
        x = tf.random.normal((2, 16, 16, 16, 32))

        # Build layer
        _ = layer(x)

        # Test partition/unpartition
        windows, shape_info = layer._window_partition(x)
        reconstructed = layer._window_unpartition(windows, shape_info)

        np.testing.assert_allclose(x.numpy(), reconstructed.numpy(), rtol=1e-5)

    def test_padding_handled_correctly(self, tf_setup):
        """Verify non-divisible dimensions are padded correctly."""
        from attention_context import WindowedAttention3D

        layer = WindowedAttention3D(dim=32, num_heads=4, window_size=4)

        # Input size not divisible by window_size
        x = tf.random.normal((1, 10, 10, 10, 32))
        out = layer(x)

        assert out.shape == x.shape


# =============================================================================
# Precision Config Tests
# =============================================================================

class TestPrecisionConfig:
    """Test mixed precision configuration."""

    def test_configure_float32(self, tf_setup):
        """Verify float32 configuration works."""
        from precision_config import PrecisionManager

        PrecisionManager.configure('float32')
        assert PrecisionManager.get_compute_dtype() == tf.float32
        PrecisionManager.restore_default()

    def test_wrap_optimizer_float32(self, tf_setup):
        """Verify optimizer wrapping in float32 mode."""
        from precision_config import PrecisionManager

        PrecisionManager.configure('float32')
        optimizer = tf.keras.optimizers.Adam()
        wrapped = PrecisionManager.wrap_optimizer(optimizer)

        # Should return same optimizer (no wrapping needed)
        assert wrapped is optimizer
        PrecisionManager.restore_default()

    def test_is_mixed_precision(self, tf_setup):
        """Test mixed precision detection."""
        from precision_config import PrecisionManager

        PrecisionManager.configure('float32')
        assert not PrecisionManager.is_mixed_precision()
        PrecisionManager.restore_default()


# =============================================================================
# Integration Tests
# =============================================================================

class TestOptimizationIntegration:
    """Integration tests for optimized components."""

    def test_entropy_model_with_optimized_scale(self, tf_setup, sample_scale_table):
        """Test scale quantization with optimized binary search."""
        from entropy_model import PatchedGaussianConditional

        layer = PatchedGaussianConditional(scale_table=sample_scale_table)

        # Test quantize_scale directly (doesn't require building the layer)
        test_scales = tf.random.uniform((2, 4, 4, 4, 8), 0.01, 1.0, dtype=tf.float32)
        quantized = layer.quantize_scale(test_scales)

        # Output shape should match input
        assert quantized.shape == test_scales.shape

        # All values should be from scale table
        table_values = set(sample_scale_table.numpy().tolist())
        for v in quantized.numpy().flatten()[:10]:  # Check first 10 values
            assert v in table_values or np.isclose(v, list(table_values), rtol=1e-5).any()

    def test_context_model_with_vectorized_mask(self, tf_setup):
        """Test context model with vectorized mask creation."""
        from context_model import AutoregressiveContext

        layer = AutoregressiveContext(channels=32, num_layers=2, kernel_size=3)
        x = tf.random.normal((1, 8, 8, 8, 32), dtype=tf.float32)

        output = layer(x)
        assert output.shape == x.shape

    def test_channel_context_optimized_decoding(self, tf_setup):
        """Test channel context with optimized decoding path."""
        from channel_context import ChannelContext

        layer = ChannelContext(channels=32, num_groups=4)
        x = tf.random.normal((1, 8, 8, 8, 32), dtype=tf.float32)

        # Test all groups - must call the method directly, not via __call__
        # because Keras __call__ has strict signature requirements
        for i in range(4):
            mean, scale = layer.call(x, group_idx=i)
            assert mean.shape[-1] == 8  # channels_per_group
            assert scale.shape[-1] == 8


# =============================================================================
# Performance Regression Tests
# =============================================================================

@pytest.mark.slow
class TestPerformanceRegression:
    """
    Performance regression tests.

    These tests ensure optimizations provide expected speedups.
    Mark with @pytest.mark.slow as they take longer to run.
    """

    def test_vectorized_mask_faster_than_loops(self, tf_setup):
        """Verify vectorized mask creation is faster."""
        import numpy as np

        kernel_size = (5, 5, 5)
        in_channels = 64
        filters = 128
        iterations = 50

        # Loop-based (original)
        def create_mask_loops():
            kd, kh, kw = kernel_size
            center_d, center_h, center_w = kd // 2, kh // 2, kw // 2
            mask = np.ones((kd, kh, kw, in_channels, filters), dtype=np.float32)
            for d in range(kd):
                for h in range(kh):
                    for w in range(kw):
                        if d > center_d:
                            mask[d, h, w, :, :] = 0
                        elif d == center_d:
                            if h > center_h:
                                mask[d, h, w, :, :] = 0
                            elif h == center_h:
                                if w > center_w:
                                    mask[d, h, w, :, :] = 0
            return mask

        # Vectorized (optimized)
        def create_mask_vectorized():
            kd, kh, kw = kernel_size
            center_d, center_h, center_w = kd // 2, kh // 2, kw // 2
            d_coords = np.arange(kd)[:, None, None]
            h_coords = np.arange(kh)[None, :, None]
            w_coords = np.arange(kw)[None, None, :]
            is_future = (
                (d_coords > center_d) |
                ((d_coords == center_d) & (h_coords > center_h)) |
                ((d_coords == center_d) & (h_coords == center_h) & (w_coords > center_w))
            )
            mask = np.where(is_future, 0.0, 1.0).astype(np.float32)
            return np.broadcast_to(
                mask[:, :, :, None, None],
                (kd, kh, kw, in_channels, filters)
            ).copy()

        # Time both
        start = time.perf_counter()
        for _ in range(iterations):
            _ = create_mask_loops()
        loop_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(iterations):
            _ = create_mask_vectorized()
        vectorized_time = time.perf_counter() - start

        speedup = loop_time / vectorized_time
        print(f"\nMask creation speedup: {speedup:.1f}x")

        # Expect at least 1.2x speedup (actual speedup varies by environment)
        # Note: 10-100x speedup is typical for production-size arrays, but
        # test arrays are small and NumPy loops are well-optimized
        assert speedup > 1.2, f"Expected >1.2x speedup, got {speedup:.1f}x"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
