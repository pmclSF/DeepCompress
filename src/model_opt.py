import numpy as np
import tensorflow as tf
from typing import Any, Tuple

class ModelOptimizer:
    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize the ModelOptimizer class.

        Args:
            learning_rate (float): Learning rate for the optimizer.
        """
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def optimize_model(self, model: tf.keras.Model, inputs: np.ndarray, targets: np.ndarray, 
                       loss_fn: Any, epochs: int = 10, batch_size: int = 32) -> Tuple[tf.keras.Model, list]:
        """
        Optimize a Keras model using a given loss function and dataset.

        Args:
            model (tf.keras.Model): The Keras model to optimize.
            inputs (np.ndarray): Input data.
            targets (np.ndarray): Target labels.
            loss_fn (callable): Loss function for optimization.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            Tuple[tf.keras.Model, list]: Trained model and list of loss values for each epoch.
        """
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(batch_size)
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_inputs, batch_targets in dataset:
                with tf.GradientTape() as tape:
                    predictions = model(batch_inputs, training=True)
                    loss = loss_fn(batch_targets, predictions)

                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                epoch_loss += loss.numpy()

            loss_history.append(epoch_loss / len(dataset))
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_history[-1]:.4f}")

        return model, loss_history

    def evaluate_model(self, model: tf.keras.Model, inputs: np.ndarray, targets: np.ndarray, loss_fn: Any) -> float:
        """
        Evaluate the model using a given dataset and loss function.

        Args:
            model (tf.keras.Model): The Keras model to evaluate.
            inputs (np.ndarray): Input data.
            targets (np.ndarray): Target labels.
            loss_fn (callable): Loss function for evaluation.

        Returns:
            float: Evaluation loss.
        """
        predictions = model(inputs, training=False)
        loss = loss_fn(targets, predictions)
        return loss.numpy()

# Example usage
if __name__ == "__main__":
    # Sample model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Synthetic dataset
    inputs = np.random.rand(1000, 10).astype(np.float32)
    targets = np.random.rand(1000, 1).astype(np.float32)

    # Loss function
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Initialize optimizer
    optimizer = ModelOptimizer(learning_rate=0.001)

    # Train model
    model, loss_history = optimizer.optimize_model(model, inputs, targets, loss_fn, epochs=10, batch_size=32)

    # Evaluate model
    evaluation_loss = optimizer.evaluate_model(model, inputs, targets, loss_fn)
    print(f"Evaluation Loss: {evaluation_loss:.4f}")
