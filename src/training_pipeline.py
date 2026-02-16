import logging
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf


class TrainingPipeline:
    def __init__(self, config_path: str):
        import yaml

        from data_loader import DataLoader
        from entropy_model import EntropyModel
        from model_transforms import DeepCompressModel, TransformConfig

        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.logger = logging.getLogger(__name__)

        # Initialize data loader
        self.data_loader = DataLoader(self.config)

        # Initialize model
        model_config = TransformConfig(
            filters=self.config['model'].get('filters', 64),
            activation=self.config['model'].get('activation', 'cenic_gdn'),
            conv_type=self.config['model'].get('conv_type', 'separable'),
        )
        self.model = DeepCompressModel(model_config)
        self.entropy_model = EntropyModel()

        # Initialize optimizers
        lrs = self.config['training']['learning_rates']
        self.optimizers = {
            'reconstruction': tf.keras.optimizers.Adam(learning_rate=lrs['reconstruction']),
            'entropy': tf.keras.optimizers.Adam(learning_rate=lrs['entropy']),
        }

        # Checkpoint directory
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Summary writer
        log_dir = self.checkpoint_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.summary_writer = tf.summary.create_file_writer(str(log_dir))

    def _train_step(self, batch: tf.Tensor, training: bool = True) -> Dict[str, tf.Tensor]:
        """Run a single training step."""
        with tf.GradientTape(persistent=True) as tape:
            inputs = batch[..., tf.newaxis] if len(batch.shape) == 4 else batch
            x_hat, y, y_hat, z = self.model(inputs, training=training)

            # Compute focal loss on reconstruction
            focal_loss = self.compute_focal_loss(
                batch[..., tf.newaxis] if len(batch.shape) == 4 else batch,
                x_hat,
            )

            # Compute entropy loss
            # EntropyModel returns log-probabilities, so use them directly
            _, log_likelihood = self.entropy_model(y, training=training)
            entropy_loss = -tf.reduce_mean(log_likelihood)

            total_loss = focal_loss + entropy_loss

        if training:
            # Update reconstruction model
            model_grads = tape.gradient(focal_loss, self.model.trainable_variables)
            self.optimizers['reconstruction'].apply_gradients(
                zip(model_grads, self.model.trainable_variables)
            )

            # Update entropy model
            entropy_grads = tape.gradient(entropy_loss, self.entropy_model.trainable_variables)
            if entropy_grads and any(g is not None for g in entropy_grads):
                self.optimizers['entropy'].apply_gradients(
                    zip(entropy_grads, self.entropy_model.trainable_variables)
                )

        del tape

        return {
            'focal_loss': focal_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss,
        }

    def compute_focal_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        alpha = self.config['training']['focal_loss']['alpha']
        gamma = self.config['training']['focal_loss']['gamma']

        y_true = tf.cast(y_true > 0, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(y_true == 1, alpha_factor, 1 - alpha_factor)
        focal_weight = alpha_factor * tf.pow(1 - pt, gamma)

        # Element-wise binary cross-entropy (avoids Keras last-axis reduction)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
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
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(str(checkpoint_path / 'model.weights.h5'))
        self.entropy_model.save_weights(str(checkpoint_path / 'entropy.weights.h5'))

        for opt_name, optimizer in self.optimizers.items():
            if optimizer.variables:
                opt_dir = checkpoint_path / f'{opt_name}_optimizer'
                opt_dir.mkdir(parents=True, exist_ok=True)
                for i, v in enumerate(optimizer.variables):
                    np.save(str(opt_dir / f'{i}.npy'), v.numpy())

        self.logger.info(f"Saved checkpoint: {name}")

    def load_checkpoint(self, name: str):
        checkpoint_path = (self.checkpoint_dir / name).resolve()
        try:
            checkpoint_path.relative_to(self.checkpoint_dir.resolve())
        except ValueError:
            raise ValueError(f"Checkpoint path escapes checkpoint directory: {name}")
        self.model.load_weights(str(checkpoint_path / 'model.weights.h5'))
        self.entropy_model.load_weights(str(checkpoint_path / 'entropy.weights.h5'))

        for opt_name, optimizer in self.optimizers.items():
            opt_dir = checkpoint_path / f'{opt_name}_optimizer'
            if opt_dir.exists() and optimizer.variables:
                for i, var in enumerate(optimizer.variables):
                    path = opt_dir / f'{i}.npy'
                    if path.exists():
                        var.assign(np.load(str(path), allow_pickle=False))

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
