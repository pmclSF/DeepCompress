import logging
from typing import Dict

import numpy as np
import tensorflow as tf


class TrainingPipeline:
    def compute_focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        alpha = self.config['training']['focal_loss']['alpha']
        gamma = self.config['training']['focal_loss']['gamma']

        y_true = tf.cast(y_true > 0, tf.float32)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(y_true == 1, alpha_factor, 1 - alpha_factor)
        focal_weight = alpha_factor * tf.pow(1 - pt, gamma)

        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(focal_weight * bce)

    def train(self, validate_every: int = 100):
        train_dataset = self.data_loader.load_training_data()
        val_dataset = self.data_loader.load_evaluation_data()

        step = 0
        best_val_loss = float('inf')

        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")

            for batch in train_dataset:
                step += 1
                losses = self._train_step(batch)

                with self.summary_writer.as_default():
                    for name, value in losses.items():
                        tf.summary.scalar(f'train/{name}', value, step=step)

                if step % validate_every == 0:
                    val_losses = self._validate(val_dataset)

                    with self.summary_writer.as_default():
                        for name, value in val_losses.items():
                            tf.summary.scalar(f'val/{name}', value, step=step)

                    if val_losses['total_loss'] < best_val_loss:
                        best_val_loss = val_losses['total_loss']
                        self.save_checkpoint('best_model')

            self.save_checkpoint(f'epoch_{epoch+1}')

    def _validate(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        val_losses = []
        for batch in val_dataset:
            losses = self._train_step(batch, training=False)
            val_losses.append({k: v.numpy() for k, v in losses.items()})

        avg_losses = {}
        for metric in val_losses[0].keys():
            avg_losses[metric] = float(tf.reduce_mean([x[metric] for x in val_losses]))

        return avg_losses

    def save_checkpoint(self, name: str):
        checkpoint_path = self.checkpoint_dir / name
        self.model.save_weights(str(checkpoint_path / 'model.h5'))
        self.entropy_model.save_weights(str(checkpoint_path / 'entropy.h5'))

        for opt_name, optimizer in self.optimizers.items():
            np.save(str(checkpoint_path / f'{opt_name}_optimizer.npy'), optimizer.get_weights())

        self.logger.info(f"Saved checkpoint: {name}")

    def load_checkpoint(self, name: str):
        checkpoint_path = self.checkpoint_dir / name
        self.model.load_weights(str(checkpoint_path / 'model.h5'))
        self.entropy_model.load_weights(str(checkpoint_path / 'entropy.h5'))

        for opt_name, optimizer in self.optimizers.items():
            optimizer_weights = np.load(str(checkpoint_path / f'{opt_name}_optimizer.npy'), allow_pickle=True)
            optimizer.set_weights(optimizer_weights)

        self.logger.info(f"Loaded checkpoint: {name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train DeepCompress model")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    pipeline = TrainingPipeline(args.config)

    if args.resume:
        pipeline.load_checkpoint(args.resume)

    pipeline.train()

if __name__ == "__main__":
    main()
