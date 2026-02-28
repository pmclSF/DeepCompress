"""
Tests for masked convolution causality and autoregressive ordering.

Validates that MaskedConv3D enforces correct causal masks in raster-scan
order (depth, height, width), type A excludes center, type B includes it,
and the AutoregressiveContext model maintains causality.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from context_model import AutoregressiveContext, MaskedConv3D


class TestMaskedConv3DCausality(tf.test.TestCase):
    """Tests for MaskedConv3D mask correctness."""

    def test_mask_type_a_excludes_center(self):
        """Type A mask should be 0 at the center position."""
        conv = MaskedConv3D(filters=4, kernel_size=3, mask_type='A')
        conv.build((None, 8, 8, 8, 1))

        mask = conv.mask.numpy()
        # Center of 3x3x3 kernel is (1, 1, 1)
        center_vals = mask[1, 1, 1, :, :]
        np.testing.assert_array_equal(
            center_vals, 0.0,
            err_msg="Type A mask should exclude center position"
        )

    def test_mask_type_b_includes_center(self):
        """Type B mask should be 1 at the center position."""
        conv = MaskedConv3D(filters=4, kernel_size=3, mask_type='B')
        conv.build((None, 8, 8, 8, 1))

        mask = conv.mask.numpy()
        center_vals = mask[1, 1, 1, :, :]
        np.testing.assert_array_equal(
            center_vals, 1.0,
            err_msg="Type B mask should include center position"
        )

    def test_future_positions_masked(self):
        """All future positions in raster-scan order should be 0."""
        conv = MaskedConv3D(filters=4, kernel_size=5, mask_type='A')
        conv.build((None, 8, 8, 8, 1))

        mask = conv.mask.numpy()
        kd, kh, kw = 5, 5, 5
        center_d, center_h, center_w = 2, 2, 2

        for d in range(kd):
            for h in range(kh):
                for w in range(kw):
                    is_future = (
                        (d > center_d) or
                        (d == center_d and h > center_h) or
                        (d == center_d and h == center_h and w > center_w)
                    )
                    is_center = (d == center_d and h == center_h and w == center_w)

                    if is_future or is_center:
                        np.testing.assert_array_equal(
                            mask[d, h, w, :, :], 0.0,
                            err_msg=f"Position ({d},{h},{w}) should be masked"
                        )
                    else:
                        np.testing.assert_array_equal(
                            mask[d, h, w, :, :], 1.0,
                            err_msg=f"Position ({d},{h},{w}) should be unmasked"
                        )

    def test_past_positions_unmasked_type_b(self):
        """All past + center positions in raster-scan order should be 1 for type B."""
        conv = MaskedConv3D(filters=4, kernel_size=5, mask_type='B')
        conv.build((None, 8, 8, 8, 1))

        mask = conv.mask.numpy()
        kd, kh, kw = 5, 5, 5
        center_d, center_h, center_w = 2, 2, 2

        for d in range(kd):
            for h in range(kh):
                for w in range(kw):
                    is_future = (
                        (d > center_d) or
                        (d == center_d and h > center_h) or
                        (d == center_d and h == center_h and w > center_w)
                    )

                    if is_future:
                        np.testing.assert_array_equal(
                            mask[d, h, w, :, :], 0.0,
                            err_msg=f"Position ({d},{h},{w}) should be masked (future)"
                        )
                    else:
                        np.testing.assert_array_equal(
                            mask[d, h, w, :, :], 1.0,
                            err_msg=f"Position ({d},{h},{w}) should be unmasked (past/center)"
                        )

    def test_mask_shape_matches_kernel(self):
        """Mask shape should match (kd, kh, kw, in_channels, filters)."""
        in_channels = 8
        filters = 16
        conv = MaskedConv3D(filters=filters, kernel_size=3, mask_type='A')
        conv.build((None, 8, 8, 8, in_channels))

        self.assertEqual(conv.mask.shape, (3, 3, 3, in_channels, filters))

    def test_mask_broadcast_across_channels(self):
        """Mask should be the same across all input/output channel pairs."""
        conv = MaskedConv3D(filters=8, kernel_size=3, mask_type='A')
        conv.build((None, 8, 8, 8, 4))

        mask = conv.mask.numpy()
        # All channel slices should be identical
        reference = mask[:, :, :, 0, 0]
        for ic in range(4):
            for oc in range(8):
                np.testing.assert_array_equal(
                    mask[:, :, :, ic, oc], reference,
                    err_msg=f"Channel ({ic},{oc}) mask differs from reference"
                )

    def test_invalid_mask_type_raises(self):
        """Invalid mask type should raise ValueError."""
        with self.assertRaises(ValueError):
            MaskedConv3D(filters=4, kernel_size=3, mask_type='C')

    def test_output_shape_same_padding(self):
        """Output should have same spatial dims with 'same' padding."""
        conv = MaskedConv3D(filters=8, kernel_size=3, mask_type='A', padding='same')
        inputs = tf.random.normal((1, 8, 8, 8, 4))
        output = conv(inputs)

        self.assertEqual(output.shape, (1, 8, 8, 8, 8))

    def test_kernel_size_1_type_a_all_zero(self):
        """Kernel size 1 with type A should have all-zero mask (no past)."""
        conv = MaskedConv3D(filters=4, kernel_size=1, mask_type='A')
        conv.build((None, 8, 8, 8, 2))

        mask = conv.mask.numpy()
        np.testing.assert_array_equal(mask, 0.0)

    def test_kernel_size_1_type_b_all_one(self):
        """Kernel size 1 with type B should have all-one mask (center only)."""
        conv = MaskedConv3D(filters=4, kernel_size=1, mask_type='B')
        conv.build((None, 8, 8, 8, 2))

        mask = conv.mask.numpy()
        np.testing.assert_array_equal(mask, 1.0)


class TestCausalOutputDependence(tf.test.TestCase):
    """Tests that masked conv output at position (d,h,w) depends only on past."""

    def test_type_a_output_independent_of_current_position(self):
        """With type A, changing center input should not affect center output."""
        tf.random.set_seed(42)
        conv = MaskedConv3D(filters=1, kernel_size=3, mask_type='A')

        # Create two inputs that differ only at center position (4,4,4)
        input1 = tf.random.normal((1, 8, 8, 8, 1))
        input2 = tf.identity(input1)
        # Modify center position
        input2_np = input2.numpy()
        input2_np[0, 4, 4, 4, 0] = 999.0
        input2 = tf.constant(input2_np)

        out1 = conv(input1)
        out2 = conv(input2)

        # Output at (4,4,4) should be the same (center is masked for type A)
        self.assertAllClose(
            out1[0, 4, 4, 4, :], out2[0, 4, 4, 4, :],
            atol=1e-5,
            msg="Type A output should not depend on current position"
        )

    def test_type_b_output_depends_on_current_position(self):
        """With type B, changing center input should affect center output."""
        tf.random.set_seed(42)
        conv = MaskedConv3D(filters=1, kernel_size=3, mask_type='B')

        input1 = tf.random.normal((1, 8, 8, 8, 1))
        input2_np = input1.numpy().copy()
        input2_np[0, 4, 4, 4, 0] = 999.0
        input2 = tf.constant(input2_np)

        out1 = conv(input1)
        out2 = conv(input2)

        # Output at (4,4,4) should differ (center is unmasked for type B)
        diff = tf.abs(out1[0, 4, 4, 4, :] - out2[0, 4, 4, 4, :])
        self.assertGreater(float(tf.reduce_max(diff)), 0.01)

    def test_future_change_does_not_affect_past_output(self):
        """Changing a future position should not affect any past output."""
        tf.random.set_seed(42)
        conv = MaskedConv3D(filters=1, kernel_size=3, mask_type='A')

        input1 = tf.random.normal((1, 8, 8, 8, 1))
        input2_np = input1.numpy().copy()
        # Modify a "future" position (7,7,7)
        input2_np[0, 7, 7, 7, 0] = 999.0
        input2 = tf.constant(input2_np)

        out1 = conv(input1)
        out2 = conv(input2)

        # All positions before (7,7,7) should be identical
        # Check positions 0..6 in depth (all are strictly before depth=7)
        self.assertAllClose(
            out1[0, :7, :, :, :], out2[0, :7, :, :, :],
            atol=1e-5,
            msg="Past outputs should not change when future input changes"
        )


class TestAutoregressiveContext(tf.test.TestCase):
    """Tests for the AutoregressiveContext model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tf.random.set_seed(42)
        self.channels = 16
        self.resolution = 8

    def test_output_shape(self):
        """Output should have shape (B, D, H, W, channels)."""
        ctx = AutoregressiveContext(channels=self.channels, num_layers=3)
        inputs = tf.random.normal((1, self.resolution, self.resolution, self.resolution, 4))
        output = ctx(inputs)

        self.assertEqual(output.shape, (1, self.resolution, self.resolution, self.resolution, self.channels))

    def test_first_layer_is_type_a(self):
        """First conv layer should use mask type A."""
        ctx = AutoregressiveContext(channels=self.channels, num_layers=3)
        self.assertEqual(ctx.conv_layers[0].mask_type, 'A')

    def test_subsequent_layers_are_type_b(self):
        """All subsequent conv layers should use mask type B."""
        ctx = AutoregressiveContext(channels=self.channels, num_layers=3)
        for conv in ctx.conv_layers[1:]:
            self.assertEqual(conv.mask_type, 'B')

    def test_causal_output(self):
        """Changing future input should not affect past outputs."""
        tf.random.set_seed(42)
        ctx = AutoregressiveContext(channels=self.channels, num_layers=2, kernel_size=3)

        input1 = tf.random.normal((1, self.resolution, self.resolution, self.resolution, 4))
        input2_np = input1.numpy().copy()
        # Modify last depth slice (future)
        input2_np[0, -1, :, :, :] = 999.0
        input2 = tf.constant(input2_np)

        out1 = ctx(input1)
        out2 = ctx(input2)

        # Outputs at depth 0 should be identical (far from modified depth)
        # With kernel_size=3 and 2 layers, receptive field is at most 4
        self.assertAllClose(
            out1[0, 0, :, :, :], out2[0, 0, :, :, :],
            atol=1e-5,
            msg="Early depth outputs should not depend on last depth slice"
        )


class TestMaskCountProperties(tf.test.TestCase):
    """Tests for statistical properties of the mask."""

    def test_type_a_has_fewer_ones_than_type_b(self):
        """Type A (excludes center) should have fewer 1s than type B."""
        conv_a = MaskedConv3D(filters=1, kernel_size=3, mask_type='A')
        conv_b = MaskedConv3D(filters=1, kernel_size=3, mask_type='B')
        conv_a.build((None, 8, 8, 8, 1))
        conv_b.build((None, 8, 8, 8, 1))

        ones_a = np.sum(conv_a.mask.numpy())
        ones_b = np.sum(conv_b.mask.numpy())

        self.assertLess(ones_a, ones_b)

    def test_type_a_count_3x3x3(self):
        """3x3x3 type A should have 13 unmasked positions (half minus center)."""
        conv = MaskedConv3D(filters=1, kernel_size=3, mask_type='A')
        conv.build((None, 8, 8, 8, 1))

        mask = conv.mask.numpy()[:, :, :, 0, 0]
        # In 3x3x3=27 positions, 13 are past, 1 is center, 13 are future
        # Type A: 13 past = unmasked
        self.assertEqual(int(np.sum(mask)), 13)

    def test_type_b_count_3x3x3(self):
        """3x3x3 type B should have 14 unmasked positions (half + center)."""
        conv = MaskedConv3D(filters=1, kernel_size=3, mask_type='B')
        conv.build((None, 8, 8, 8, 1))

        mask = conv.mask.numpy()[:, :, :, 0, 0]
        # 13 past + 1 center = 14
        self.assertEqual(int(np.sum(mask)), 14)


if __name__ == '__main__':
    tf.test.main()
