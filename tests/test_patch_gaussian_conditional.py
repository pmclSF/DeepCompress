import sys
import os
import tensorflow as tf
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from patch_gaussian_conditional import PatchedGaussianConditional

class TestPatchedGaussianConditional(tf.test.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        # Use TF's random number generation
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
        """Test layer initialization."""
        # Check variable creation
        self.assertIsInstance(self.layer.scale, tf.Variable)
        self.assertIsInstance(self.layer.mean, tf.Variable)
        self.assertIsInstance(self.layer.scale_table, tf.Variable)
        
        # Check values
        self.assertAllClose(self.layer.scale, self.scale)
        self.assertAllClose(self.layer.mean, self.mean)
        self.assertAllClose(self.layer.scale_table, self.scale_table)
        
    def test_quantize_scale(self):
        """Test scale quantization."""
        test_scales = tf.constant([0.15, 0.25, 0.35])
        quantized = self.layer.quantize_scale(test_scales)
        
        # Check shape
        self.assertEqual(quantized.shape, test_scales.shape)
        
        # Check values are from scale table
        for value in quantized.numpy():
            self.assertIn(
                value,
                self.scale_table.numpy(),
                msg=f"Quantized value {value} not in scale table"
            )
            
    def test_compression_cycle(self):
        """Test compress-decompress cycle."""
        # Compress
        compressed = self.layer.compress(self.inputs)
        self.assertEqual(compressed.shape, self.inputs.shape)
        
        # Check quantization
        self.assertAllEqual(compressed, tf.round(compressed))
        
        # Decompress
        decompressed = self.layer.decompress(compressed)
        self.assertEqual(decompressed.shape, self.inputs.shape)
        
        # Check cycle consistency
        self.assertAllClose(
            decompressed,
            self.layer(self.inputs),
            rtol=1e-5,
            atol=1e-5
        )
        
    def test_debug_tensors(self):
        """Test debug tensor collection."""
        # Run forward pass
        _ = self.layer(self.inputs)
        debug_tensors = self.layer.get_debug_tensors()
        
        # Check required keys exist
        required_keys = {
            'inputs', 'outputs',
            'compress_inputs', 'compress_outputs',
            'decompress_inputs', 'decompress_outputs'
        }
        self.assertSetEqual(
            set(debug_tensors.keys()) - {'compress_scale', 'decompress_scale'},
            required_keys
        )
        
        # Check tensor shapes
        for key in required_keys:
            self.assertEqual(
                debug_tensors[key].shape,
                self.inputs.shape,
                msg=f"Shape mismatch for {key}"
            )
            
    def test_get_config(self):
        """Test config serialization."""
        config = self.layer.get_config()
        
        # Check required keys
        required_keys = {'scale', 'mean', 'scale_table', 'tail_mass'}
        self.assertSetEqual(
            set(config.keys()) & required_keys,
            required_keys
        )
        
        # Test reconstruction
        reconstructed = PatchedGaussianConditional(**config)
        self.assertAllClose(
            reconstructed.scale,
            self.layer.scale
        )
        self.assertAllClose(
            reconstructed.mean,
            self.layer.mean
        )

if __name__ == '__main__':
    tf.test.main()