import tensorflow as tf
import pytest
from test_utils import create_mock_voxel_grid
from model_transforms import (
    CENICGDN,
    SpatialSeparableConv,
    AnalysisTransform,
    SynthesisTransform,
    DeepCompressModel,
    TransformConfig
)

class TestModelTransforms(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test components."""
        self.config = TransformConfig(
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(2, 2, 2),
            activation='cenic_gdn',
            conv_type='separable'
        )
        self.batch_size = 2
        self.resolution = 64
        self.input_shape = (self.batch_size, self.resolution, self.resolution, self.resolution, 1)

    def test_cenic_gdn(self):
        """Test CENIC-GDN activation."""
        channels = 64
        activation = CENICGDN(channels)
        input_tensor = tf.random.uniform((2, 32, 32, 32, channels))
        
        # Test forward pass
        output = activation(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Test normalization properties
        self.assertAllInRange(output, -10, 10)
        
        # Test training
        with tf.GradientTape() as tape:
            output = activation(input_tensor)
            loss = tf.reduce_mean(output)
        
        gradients = tape.gradient(loss, activation.trainable_variables)
        self.assertNotEmpty(gradients)
        self.assertAllNotEqual(gradients[0], 0)

    def test_spatial_separable_conv(self):
        """Test spatially separable convolutions."""
        conv = SpatialSeparableConv(
            filters=64,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1)
        )
        input_tensor = tf.random.uniform((2, 32, 32, 32, 32))
        
        # Test forward pass
        output = conv(input_tensor)
        self.assertEqual(output.shape[-1], 64)
        
        # Test parameter efficiency
        standard_params = 27 * 32 * 64  # 3x3x3 kernel
        separable_params = (3 * 32 * 32 + 9 * 32 * 64)  # 1D + 2D conv
        self.assertLess(
            len(conv.trainable_variables[0].numpy().flatten()),
            standard_params
        )

    def test_analysis_transform(self):
        """Test analysis transform."""
        analysis = AnalysisTransform(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        
        # Test forward pass
        output = analysis(input_tensor)
        self.assertIsNotNone(output)
        
        # Test progressive channel expansion
        self.assertGreater(output.shape[-1], input_tensor.shape[-1])
        
        # Test activation function
        self.assertIsInstance(
            analysis.get_layer('cenic_gdn_0'),
            CENICGDN
        )

    def test_synthesis_transform(self):
        """Test synthesis transform."""
        synthesis = SynthesisTransform(self.config)
        input_tensor = tf.random.uniform((2, 32, 32, 32, 64))
        
        # Test forward pass
        output = synthesis(input_tensor)
        self.assertEqual(output.shape[-1], 1)
        
        # Test progressive channel reduction
        self.assertLess(output.shape[-1], input_tensor.shape[-1])

    def test_deep_compress_model(self):
        """Test complete DeepCompressModel."""
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        
        # Test forward pass
        output = model(input_tensor, training=True)
        self.assertEqual(output.shape, input_tensor.shape)
        
        # Test compression
        compressed, metrics = model.compress(input_tensor)
        self.assertIn('bit_rate', metrics)
        
        # Test decompression
        decompressed = model.decompress(compressed)
        self.assertEqual(decompressed.shape, input_tensor.shape)

    @tf.function
    def test_gradient_flow(self):
        """Test gradient flow through all components."""
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        
        with tf.GradientTape() as tape:
            output = model(input_tensor, training=True)
            loss = tf.reduce_mean(tf.square(output - input_tensor))
            
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_model_save_load(self):
        """Test model saving and loading."""
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        
        # Get initial outputs
        initial_output = model(input_tensor, training=False)
        
        # Save and reload model
        with tf.compat.v2.io.TempDirectory() as tmp_dir:
            save_path = tmp_dir + '/model'
            model.save_weights(save_path)
            
            # Create new model and load weights
            new_model = DeepCompressModel(self.config)
            new_model.load_weights(save_path)
        
        # Compare outputs
        new_output = new_model(input_tensor, training=False)
        self.assertAllClose(initial_output, new_output)

    @pytest.mark.integration
    def test_transform_integration(self):
        """Test integration between transforms."""
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        
        # Test analysis transform
        analyzed = model.analysis_transform(input_tensor)
        
        # Test entropy model
        quantized = model.entropy_model.quantize(analyzed)
        
        # Test synthesis transform
        reconstructed = model.synthesis_transform(quantized)
        
        # Verify shapes
        self.assertEqual(reconstructed.shape, input_tensor.shape)
        self.assertGreater(analyzed.shape[-1], input_tensor.shape[-1])
        self.assertEqual(quantized.shape, analyzed.shape)

if __name__ == "__main__":
    tf.test.main()