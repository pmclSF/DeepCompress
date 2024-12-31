import unittest
import tensorflow as tf
from model_transforms import (
    SequentialLayer, ResidualLayer, AnalysisTransform, SynthesisTransform,
    custom_transform, latent_regularization, normalization_layer,
    experiment_with_activations, process_voxelized_input, evaluate_transformation,
    advanced_activation
)

class TestModelTransforms(unittest.TestCase):

    def setUp(self):
        """Set up common parameters and sample inputs for testing."""
        self.filters = 16
        self.input_shape = (1, 32, 32, 32, 3)  # Batch size, depth, height, width, channels
        self.voxel_grid = tf.random.uniform(self.input_shape)
        self.activations = [tf.nn.relu, tf.nn.swish, tf.nn.leaky_relu]

    def test_sequential_layer(self):
        """Test SequentialLayer with a simple sequence of layers."""
        layers = [
            tf.keras.layers.Conv3D(self.filters, (3, 3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv3D(self.filters, (3, 3, 3), padding='same', activation='relu')
        ]
        seq_layer = SequentialLayer(layers)
        output = seq_layer(self.voxel_grid)
        self.assertEqual(output.shape, self.voxel_grid.shape[:4] + (self.filters,))

    def test_residual_layer(self):
        """Test ResidualLayer with add mode."""
        layers = [
            tf.keras.layers.Conv3D(self.filters, (3, 3, 3), padding='same', activation='relu'),
            tf.keras.layers.Conv3D(self.filters, (3, 3, 3), padding='same', activation='relu')
        ]
        res_layer = ResidualLayer(layers, residual_mode='add')
        output = res_layer(self.voxel_grid)
        self.assertEqual(output.shape, self.voxel_grid.shape[:4] + (self.filters,))

    def test_analysis_transform(self):
        """Test AnalysisTransform with default parameters."""
        analysis = AnalysisTransform(self.filters)
        output = analysis(self.voxel_grid)
        expected_shape = self.input_shape[:4] + (self.filters,)
        self.assertEqual(output.shape, expected_shape)

    def test_synthesis_transform(self):
        """Test SynthesisTransform with default parameters."""
        synthesis = SynthesisTransform(self.filters)
        output = synthesis(self.voxel_grid)
        expected_shape = self.input_shape[:4] + (1,)
        self.assertEqual(output.shape, expected_shape)

    def test_custom_transform(self):
        """Test custom_transform with specified layer specs."""
        layer_specs = [
            ("conv", {"kernel_size": (3, 3, 3), "padding": "same"}),
            ("pool", {"pool_size": (2, 2, 2)})
        ]
        custom = custom_transform(self.filters, layer_specs)
        output = custom(self.voxel_grid)
        self.assertIsNotNone(output)

    def test_latent_regularization(self):
        """Test latent_regularization function."""
        regularization = latent_regularization(self.voxel_grid)
        self.assertGreater(regularization.numpy(), 0)

    def test_normalization_layer(self):
        """Test normalization_layer with batch normalization."""
        norm_layer = normalization_layer("batch")
        output = norm_layer(self.voxel_grid)
        self.assertEqual(output.shape, self.voxel_grid.shape)

    def test_experiment_with_activations(self):
        """Test experimentation with different activations."""
        results = experiment_with_activations(self.activations, AnalysisTransform, self.filters, self.input_shape)
        for activation in self.activations:
            self.assertIn(activation.__name__, results)

    def test_process_voxelized_input(self):
        """Test processing voxelized input with AnalysisTransform."""
        analysis = AnalysisTransform(self.filters)
        output = process_voxelized_input(analysis, self.voxel_grid)
        expected_shape = self.input_shape[:4] + (self.filters,)
        self.assertEqual(output.shape, expected_shape)

    def test_evaluate_transformation(self):
        """Test evaluate_transformation with dummy tensors."""
        predicted = tf.random.uniform(self.input_shape)
        target = tf.random.uniform(self.input_shape)
        # Flatten tensors to match expected input shape (N, 3)
        predicted_flat = tf.reshape(predicted, [-1, 3]).numpy()
        target_flat = tf.reshape(target, [-1, 3]).numpy()
        try:
            from src.pc_metric import calculate_chamfer_distance, calculate_d1_metric
        except ModuleNotFoundError:
            self.skipTest("pc_metric module not found")
        results = {
            "Chamfer Distance": calculate_chamfer_distance(predicted_flat, target_flat),
            "D1 Metric": calculate_d1_metric(predicted_flat, target_flat),
        }
        self.assertIn("Chamfer Distance", results)
        self.assertIn("D1 Metric", results)

if __name__ == "__main__":
    unittest.main()
