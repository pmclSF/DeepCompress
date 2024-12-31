import unittest
import numpy as np
import tensorflow as tf
from model_opt import ModelOptimizer

class TestModelOpt(unittest.TestCase):

    def setUp(self):
        """Set up a simple model and synthetic dataset for testing."""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.inputs = np.random.rand(100, 10).astype(np.float32)
        self.targets = np.random.rand(100, 1).astype(np.float32)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = ModelOptimizer(learning_rate=0.001)

    def test_optimize_model(self):
        """Test the optimize_model method for training."""
        initial_weights = [layer.get_weights() for layer in self.model.layers]

        trained_model, loss_history = self.optimizer.optimize_model(
            self.model, self.inputs, self.targets, self.loss_fn, epochs=5, batch_size=10
        )

        self.assertEqual(len(loss_history), 5, "Loss history length does not match epochs.")
        self.assertNotEqual(initial_weights, [layer.get_weights() for layer in trained_model.layers],
                            "Model weights did not update after training.")

    def test_evaluate_model(self):
        """Test the evaluate_model method for loss computation."""
        evaluation_loss = self.optimizer.evaluate_model(self.model, self.inputs, self.targets, self.loss_fn)
        self.assertTrue(evaluation_loss >= 0, "Evaluation loss should be non-negative.")

if __name__ == "__main__":
    unittest.main()