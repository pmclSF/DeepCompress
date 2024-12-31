import tensorflow as tf
import os
import argparse
import glob
import numpy as np
import keras_tuner as kt
from ds_mesh_to_pc import read_off  # Assuming this function is available to read point cloud

# Define the model (e.g., a simple autoencoder-like model for point cloud compression)
def create_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(2048, 3)))  # Input shape: number of points (2048) with 3 coordinates
    
    # Add tunable layers (using Keras Tuner to search for optimal number of layers and units)
    for i in range(hp.Int('num_layers', 1, 5)):  # Search for the optimal number of layers (1 to 5)
        model.add(tf.keras.layers.Dense(hp.Int(f'layer_{i}_units', min_value=64, max_value=1024, step=64), 
                                        activation='relu'))
    
    # Output layer (reconstructing point cloud)
    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))  # Output: 3 coordinates
    
    # Compile the model with a learning rate from the hyperparameter search
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')),
                  loss='mean_squared_error')
    return model

# Function to load and preprocess the dataset (point clouds)
def load_and_preprocess_data(input_dir, batch_size):
    file_paths = glob.glob(os.path.join(input_dir, "*.ply"))
    
    def parse_ply_file(file_path):
        """Parse the .ply file and return the point cloud data."""
        vertices, _ = read_off(file_path)  # Assuming the read_off function returns vertices and faces
        return vertices

    def data_generator():
        """Generator function to yield batches of point clouds."""
        for file_path in file_paths:
            vertices = parse_ply_file(file_path)
            # Preprocess the vertices (normalize, sample points, etc.)
            vertices = vertices[:2048, :]  # Assuming we are using the first 2048 points
            yield vertices

    # Create the tf.data.Dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tf.TensorSpec(shape=(2048, 3), dtype=tf.float32)
    )

    # Shuffle, batch, and prefetch the dataset for efficiency
    dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset

# Hyperparameter tuning function
def tune_hyperparameters(input_dir, output_dir, num_epochs=10):
    tuner = kt.RandomSearch(
        create_model,
        objective='val_loss',
        max_trials=10,  # Number of trials for hyperparameter search
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='point_cloud_compression',
        overwrite=True
    )

    # Load and preprocess the data
    dataset = load_and_preprocess_data(input_dir, batch_size=32)

    # Perform hyperparameter search
    tuner.search(dataset, epochs=num_epochs, validation_data=dataset)

    # Get the best model and hyperparameters
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("Best Hyperparameters:", best_hps.values)

    # Save the best model
    best_model.save(os.path.join(output_dir, 'best_model'))

# Main function to handle argument parsing and start training
def main():
    parser = argparse.ArgumentParser(description="Train a point cloud compression model with hyperparameter tuning.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing point cloud .ply files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the trained model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs (default: 10).")
    parser.add_argument("--tune", action='store_true', help="Enable hyperparameter tuning.")
    args = parser.parse_args()

    if args.tune:
        tune_hyperparameters(args.input_dir, args.output_dir, num_epochs=args.num_epochs)
    else:
        # Create model (for standard training without tuning)
        model = create_model(hp=None)

        # Load and preprocess the data
        dataset = load_and_preprocess_data(args.input_dir, args.batch_size)

        # Train the model
        model.fit(dataset, epochs=args.num_epochs)
        model.save(os.path.join(args.output_dir, 'trained_model'))

    print("Training complete!")

if __name__ == "__main__":
    main()
