import argparse
import glob
import os

import keras_tuner as kt
import tensorflow as tf

from .ds_mesh_to_pc import read_off


def create_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(2048, 3)))

    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(tf.keras.layers.Dense(
            hp.Int(f'layer_{i}_units', min_value=64, max_value=1024, step=64),
            activation='relu'
        ))

    model.add(tf.keras.layers.Dense(3, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
        ),
        loss='mean_squared_error'
    )
    return model

def load_and_preprocess_data(input_dir, batch_size):
    file_paths = glob.glob(os.path.join(input_dir, "*.ply"))

    def parse_ply_file(file_path):
        mesh_data = read_off(file_path)
        return mesh_data.vertices

    def data_generator():
        for file_path in file_paths:
            vertices = parse_ply_file(file_path)
            vertices = vertices[:2048, :]
            yield vertices

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=tf.TensorSpec(shape=(2048, 3), dtype=tf.float32)
    )

    dataset = dataset.shuffle(buffer_size=len(file_paths))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def tune_hyperparameters(input_dir, output_dir, num_epochs=10):
    tuner = kt.RandomSearch(
        create_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='point_cloud_compression',
        overwrite=True
    )

    dataset = load_and_preprocess_data(input_dir, batch_size=32)
    tuner.search(dataset, epochs=num_epochs, validation_data=dataset)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best Hyperparameters:", best_hps.values)
    best_model.save_weights(os.path.join(output_dir, 'best_model.weights.h5'))

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
        # Build a default model without hyperparameter tuning
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(2048, 3)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        dataset = load_and_preprocess_data(args.input_dir, args.batch_size)
        model.fit(dataset, epochs=args.num_epochs)
        model.save_weights(os.path.join(args.output_dir, 'trained_model.weights.h5'))

if __name__ == "__main__":
    main()
