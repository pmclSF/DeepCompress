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
        channels = 64
        activation = CENICGDN(channels)
        input_tensor = tf.random.uniform((2, 32, 32, 32, channels))
        output = activation(input_tensor)
        self.assertEqual(output.shape, input_tensor.shape)
        self.assertAllInRange(output, -10, 10)
        
        with tf.GradientTape() as tape:
            output = activation(input_tensor)
            loss = tf.reduce_mean(output)
        gradients = tape.gradient(loss, activation.trainable_variables)
        self.assertNotEmpty(gradients)
        self.assertAllNotEqual(gradients[0], 0)

    def test_spatial_separable_conv(self):
        conv = SpatialSeparableConv(filters=64, kernel_size=(3, 3, 3), strides=(1, 1, 1))
        input_tensor = tf.random.uniform((2, 32, 32, 32, 32))
        output = conv(input_tensor)
        self.assertEqual(output.shape[-1], 64)
        
        standard_params = 27 * 32 * 64
        separable_params = (3 * 32 * 32 + 9 * 32 * 64)
        self.assertLess(len(conv.trainable_variables[0].numpy().flatten()), standard_params)

    def test_analysis_transform(self):
        analysis = AnalysisTransform(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        output = analysis(input_tensor)
        self.assertIsNotNone(output)
        self.assertGreater(output.shape[-1], input_tensor.shape[-1])
        self.assertIsInstance(analysis.get_layer('cenic_gdn_0'), CENICGDN)

    def test_synthesis_transform(self):
        synthesis = SynthesisTransform(self.config)
        input_tensor = tf.random.uniform((2, 32, 32, 32, 64))
        output = synthesis(input_tensor)
        self.assertEqual(output.shape[-1], 1)
        self.assertLess(output.shape[-1], input_tensor.shape[-1])

    def test_deep_compress_model(self):
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        output = model(input_tensor, training=True)
        self.assertEqual(output.shape, input_tensor.shape)
        
        compressed, metrics = model.compress(input_tensor)
        self.assertIn('bit_rate', metrics)
        
        decompressed = model.decompress(compressed)
        self.assertEqual(decompressed.shape, input_tensor.shape)

    @tf.function
    def test_gradient_flow(self):
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        
        with tf.GradientTape() as tape:
            output = model(input_tensor, training=True)
            loss = tf.reduce_mean(tf.square(output - input_tensor))
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_model_save_load(self):
        model = DeepCompressModel(self.config)
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        initial_output = model(input_tensor, training=False)
        
        with tf.compat.v2.io.TempDirectory() as tmp_dir:
            save_path = tmp_dir + '/model'
            model.save_weights(save_path)
            new_model = DeepCompressModel(self.config)
            new_model.load_weights(save_path)
        
        new_output = new_model(input_tensor, training=False)
        self.assertAllClose(initial_output, new_output)

if __name__ == "__main__":
    tf.test.main()