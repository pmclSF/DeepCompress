import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import tensorflow as tf

from entropy_model import EntropyModel, PatchedGaussianConditional



class TestEntropyModel(tf.test.TestCase):
    def setUp(self):
        tf.random.set_seed(42)
        self.scale_table = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5])
        self.input_shape = (5, 5)
        self.inputs = tf.random.uniform(self.input_shape, -2.0, 2.0)

        self.layer = PatchedGaussianConditional(
            scale_table=self.scale_table
        )
        # Build the layer so scale/mean weights are created
        self.layer.build((None, *self.input_shape))

    def test_initialization(self):
        self.assertTrue(hasattr(self.layer.scale, 'numpy'))
        self.assertTrue(hasattr(self.layer.mean, 'numpy'))
        self.assertTrue(hasattr(self.layer.scale_table, 'numpy'))
        self.assertEqual(self.layer.scale.shape, self.input_shape)
        self.assertEqual(self.layer.mean.shape, self.input_shape)
        self.assertAllClose(self.layer.scale_table, self.scale_table)

    def test_quantize_scale(self):
        test_scales = tf.constant([0.15, 0.25, 0.35])
        quantized = self.layer.quantize_scale(test_scales)
        self.assertEqual(quantized.shape, test_scales.shape)
        for value in quantized.numpy():
            self.assertIn(value, self.scale_table.numpy())

    def test_compression_cycle(self):
        compressed = self.layer.compress(self.inputs)
        self.assertEqual(compressed.shape, self.inputs.shape)
        self.assertAllEqual(compressed, tf.round(compressed))

        decompressed = self.layer.decompress(compressed)
        self.assertEqual(decompressed.shape, self.inputs.shape)
        self.assertAllClose(decompressed, self.layer(self.inputs), rtol=1e-5, atol=1e-5)

    def test_debug_tensors(self):
        _ = self.layer(self.inputs)
        debug_tensors = self.layer.get_debug_tensors()

        required_keys = {'inputs', 'outputs', 'compress_inputs', 'compress_outputs',
                        'decompress_inputs', 'decompress_outputs'}
        self.assertTrue(required_keys.issubset(set(debug_tensors.keys())))

        for key in required_keys:
            self.assertEqual(debug_tensors[key].shape, self.inputs.shape)

    def test_get_config(self):
        config = self.layer.get_config()
        required_keys = {'initial_scale', 'scale_table', 'tail_mass'}
        self.assertTrue(required_keys.issubset(set(config.keys())))

        reconstructed = PatchedGaussianConditional(
            initial_scale=config['initial_scale'],
            scale_table=config['scale_table'],
            tail_mass=config['tail_mass']
        )
        self.assertEqual(reconstructed.initial_scale, self.layer.initial_scale)
        self.assertEqual(reconstructed.tail_mass, self.layer.tail_mass)
        self.assertAllClose(reconstructed.scale_table, self.layer.scale_table)

    def test_entropy_model_forward(self):
        """Test EntropyModel wrapping PatchedGaussianConditional."""
        model = EntropyModel()
        compressed, likelihood = model(self.inputs, training=False)
        self.assertEqual(compressed.shape, self.inputs.shape)
        self.assertEqual(likelihood.shape, self.inputs.shape)


if __name__ == '__main__':
    tf.test.main()
