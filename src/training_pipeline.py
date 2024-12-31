import tensorflow as tf
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

class TrainingPipeline:
    # ... [previous methods remain the same] ...

    @tf.function
    def compute_focal_loss(self,
                          y_true: tf.Tensor,
                          y_pred: tf.Tensor) -> tf.Tensor:
        """Compute focal loss with alpha=0.75 and gamma=2.0."""
        alpha = self.config['training']['focal_loss']['alpha']
        gamma = self.config['training']['focal_loss']['gamma']
        
        # Convert to binary values
        y_true = tf.cast(y_true > 0, tf.float32)
        
        # Compute focal weights
        pt = tf.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(y_true == 1, alpha_factor, 1 - alpha_factor)
        focal_weight = alpha_factor * tf.pow(1 - pt, gamma)
        
        # Compute binary cross entropy
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        return tf.reduce_mean(focal_weight * bce)

    def train(self, validate_every: int = 100):
        """Train the model."""
        # Get training dataset
        train_dataset = self.data_loader.load_training_data()
        val_dataset = self.data_loader.load_evaluation_data()
        
        # Training loop
        step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            
            # Training steps
            for batch in train_dataset:
                step += 1
                losses = self._train_step(batch)
                
                # Log training metrics
                with self.summary_writer.as_default():
                    for name, value in losses.items():
                        tf.summary.scalar(f'train/{name}', value, step=step)
                
                # Validation
                if step % validate_every == 0:
                    val_losses = self._validate(val_dataset)
                    
                    # Log validation metrics
                    with self.summary_writer.as_default():
                        for name, value in val_losses.items():
                            tf.summary.scalar(f'val/{name}', value, step=step)
                    
                    # Save if best model
                    if val_losses['total_loss'] < best_val_loss:
                        best_val_loss = val_losses['total_loss']
                        self.save_checkpoint('best_model')
                        
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch+1}')
            
    def _validate(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Run validation."""
        val_losses = []
        for batch in val_dataset:
            losses = self._train_step(batch, training=False)
            val_losses.append({k: v.numpy() for k, v in losses.items()})
        
        # Average validation losses
        avg_losses = {}
        for metric in val_losses[0].keys():
            avg_losses[metric] = float(
                tf.reduce_mean([x[metric] for x in val_losses])
            )
            
        return avg_losses

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / name
        
        # Save model weights
        self.model.save_weights(str(checkpoint_path / 'model.h5'))
        self.entropy_model.save_weights(str(checkpoint_path / 'entropy.h5'))
        
        # Save optimizer states
        for opt_name, optimizer in self.optimizers.items():
            np.save(
                str(checkpoint_path / f'{opt_name}_optimizer.npy'),
                optimizer.get_weights()
            )
            
        self.logger.info(f"Saved checkpoint: {name}")

    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / name
        
        # Load model weights
        self.model.load_weights(str(checkpoint_path / 'model.h5'))
        self.entropy_model.load_weights(str(checkpoint_path / 'entropy.h5'))
        
        # Load optimizer states
        for opt_name, optimizer in self.optimizers.items():
            optimizer_weights = np.load(
                str(checkpoint_path / f'{opt_name}_optimizer.npy'),
                allow_pickle=True
            )
            optimizer.set_weights(optimizer_weights)
            
        self.logger.info(f"Loaded checkpoint: {name}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train DeepCompress model")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and train
    pipeline = TrainingPipeline(args.config)
    
    if args.resume:
        pipeline.load_checkpoint(args.resume)
        
    pipeline.train()

if __name__ == "__main__":
    main()