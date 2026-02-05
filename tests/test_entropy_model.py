import tensorflow as tf
from entropy_model import PatchedGaussianConditional, EntropyModel

class TestEntropyModel(tf.test.TestCase):
    def setUp(self):
        self.scale = tf.random.uniform((5, 5), 0.1, 1.0)
        self.mean = tf.random.uniform((5, 5), -1.0, 1.0)
        self.scale_table = tf.constant([0.1, 0.2, 0.3, 0.4, 0.5])
        self.inputs = tf.random.uniform((5, 5), -2.0, 2.0)
        
        self.layer = PatchedGaussianConditional(
            scale=self.scale,
            mean=self.mean,
            scale_table=self.scale_table
        )

    def test_initialization(self):
        self.assertIsInstance(self.layer.scale, tf.Variable)
        self.assertIsInstance(self.layer.mean, tf.Variable)
        self.assertIsInstance(self.layer.scale_table, tf.Variable)
        self.assertAllClose(self.layer.scale, self.scale)
        self.assertAllClose(self.layer.mean, self.mean)
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
        self.assertSetEqual(set(debug_tensors.keys()) - {'compress_scale', 'decompress_scale'}, required_keys)
        
        for key in required_keys:
            self.assertEqual(debug_tensors[key].shape, self.inputs.shape)

    def test_get_config(self):
        config = self.layer.get_config()
        required_keys = {'scale', 'mean', 'scale_table', 'tail_mass'}
        self.assertSetEqual(set(config.keys()) & required_keys, required_keys)
        
        reconstructed = PatchedGaussianConditional(**config)
        self.assertAllClose(reconstructed.scale, self.layer.scale)
        self.assertAllClose(reconstructed.mean, self.layer.mean)

if __name__ == '__main__':
    tf.test.main()