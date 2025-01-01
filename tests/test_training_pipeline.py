import tensorflow as tf
import pytest
import numpy as np
from pathlib import Path
import yaml
from training_pipeline import TrainingPipeline

class TestTrainingPipeline:
    @pytest.fixture
    def config_path(self, tmp_path):
        config = {
            'data': {
                'modelnet40_path': str(tmp_path / 'modelnet40'),
                'ivfb_path': str(tmp_path / '8ivfb'),
                'resolution': 64,
                'block_size': 1.0,
                'min_points': 100,
                'augment': True
            },
            'model': {
                'filters': 64,
                'activation': 'cenic_gdn',
                'conv_type': 'separable'
            },
            'training': {
                'batch_size': 2,
                'epochs': 2,
                'learning_rates': {
                    'reconstruction': 1e-4,
                    'entropy': 1e-3
                },
                'focal_loss': {
                    'alpha': 0.75,
                    'gamma': 2.0
                },
                'checkpoint_dir': str(tmp_path / 'checkpoints')
            }
        }
        
        config_file = tmp_path / 'config.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        return str(config_file)

    @pytest.fixture
    def pipeline(self, config_path):
        return TrainingPipeline(config_path)

    def test_initialization(self, pipeline):
        assert pipeline.model is not None
        assert pipeline.entropy_model is not None
        assert len(pipeline.optimizers) == 2
        assert 'reconstruction' in pipeline.optimizers
        assert 'entropy' in pipeline.optimizers

    def test_compute_focal_loss(self, pipeline):
        batch_size = 4
        resolution = 32
        
        y_true = tf.cast(tf.random.uniform((batch_size, resolution, resolution, resolution)) > 0.5, tf.float32)
        y_pred = tf.random.uniform((batch_size, resolution, resolution, resolution))
        
        loss = pipeline.compute_focal_loss(y_true, y_pred)
        assert loss.shape == ()
        assert loss >= 0
        assert not tf.math.is_nan(loss)

    @pytest.mark.parametrize("training", [True, False])
    def test_train_step(self, pipeline, training):
        batch_size = 2
        resolution = 32
        point_cloud = tf.cast(tf.random.uniform((batch_size, resolution, resolution, resolution)) > 0.5, tf.float32)
        
        losses = pipeline._train_step(point_cloud, training=training)
        
        assert 'focal_loss' in losses
        assert 'entropy_loss' in losses
        assert 'total_loss' in losses
        
        for loss_name, loss_value in losses.items():
            assert not tf.math.is_nan(loss_value)
            assert loss_value >= 0

    def test_save_load_checkpoint(self, pipeline, tmp_path):
        checkpoint_name = 'test_checkpoint'
        pipeline.save_checkpoint(checkpoint_name)
        
        checkpoint_dir = Path(pipeline.checkpoint_dir) / checkpoint_name
        assert (checkpoint_dir / 'model.h5').exists()
        assert (checkpoint_dir / 'entropy.h5').exists()
        
        new_pipeline = TrainingPipeline(pipeline.config_path)
        new_pipeline.load_checkpoint(checkpoint_name)
        
        for w1, w2 in zip(pipeline.model.weights, new_pipeline.model.weights):
            tf.debugging.assert_equal(w1, w2)

    @pytest.mark.integration
    def test_training_loop(self, pipeline, tmp_path):
        batch_size = 2
        resolution = 32
        
        def create_sample_batch():
            return tf.cast(tf.random.uniform((batch_size, resolution, resolution, resolution)) > 0.5, tf.float32)
            
        dataset = tf.data.Dataset.from_tensors(create_sample_batch()).repeat(3)
        
        pipeline.data_loader.load_training_data = lambda: dataset
        pipeline.data_loader.load_evaluation_data = lambda: dataset
        
        pipeline.config['training']['epochs'] = 2
        pipeline.train(validate_every=2)
        
        checkpoint_dir = Path(pipeline.checkpoint_dir)
        assert len(list(checkpoint_dir.glob('epoch_*'))) > 0
        assert (checkpoint_dir / 'best_model').exists()