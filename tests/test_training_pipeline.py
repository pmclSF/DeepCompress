import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
                'filters': 32,
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
        batch_size = 1
        resolution = 16
        point_cloud = tf.cast(tf.random.uniform((batch_size, resolution, resolution, resolution)) > 0.5, tf.float32)

        losses = pipeline._train_step(point_cloud, training=training)

        assert 'focal_loss' in losses
        assert 'entropy_loss' in losses
        assert 'total_loss' in losses

        for loss_name, loss_value in losses.items():
            assert not tf.math.is_nan(loss_value)
            assert loss_value >= 0

    def test_save_load_checkpoint(self, pipeline, tmp_path):
        # Build the model by running a forward pass
        dummy = tf.zeros((1, 16, 16, 16, 1))
        pipeline.model(dummy, training=False)
        y = pipeline.model.analysis(dummy)
        pipeline.entropy_model(y, training=False)

        checkpoint_name = 'test_checkpoint'
        pipeline.save_checkpoint(checkpoint_name)

        checkpoint_dir = Path(pipeline.checkpoint_dir) / checkpoint_name
        assert (checkpoint_dir / 'model.weights.h5').exists()
        assert (checkpoint_dir / 'entropy.weights.h5').exists()
        # Optimizer variables saved as individual .npy files in subdirectories
        for opt_name in pipeline.optimizers:
            opt_dir = checkpoint_dir / f'{opt_name}_optimizer'
            if pipeline.optimizers[opt_name].variables:
                assert opt_dir.exists()

        new_pipeline = TrainingPipeline(pipeline.config_path)
        # Build the new model before loading weights
        new_pipeline.model(dummy, training=False)
        y2 = new_pipeline.model.analysis(dummy)
        new_pipeline.entropy_model(y2, training=False)
        new_pipeline.load_checkpoint(checkpoint_name)

        for w1, w2 in zip(pipeline.model.weights, new_pipeline.model.weights):
            tf.debugging.assert_equal(w1, w2)

    @pytest.mark.integration
    def test_training_loop(self, pipeline, tmp_path):
        batch_size = 1
        resolution = 16

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

    # --- Security / path validation tests ---

    def test_load_checkpoint_rejects_path_traversal(self, pipeline):
        """Path traversal via ../ is rejected."""
        with pytest.raises(ValueError, match="escapes"):
            pipeline.load_checkpoint('../../etc/passwd')

    def test_load_checkpoint_rejects_absolute_path(self, pipeline):
        """Absolute path outside checkpoint dir is rejected."""
        with pytest.raises(ValueError, match="escapes"):
            pipeline.load_checkpoint('/tmp/evil_checkpoint')

    def test_load_checkpoint_prefix_collision(self, pipeline, tmp_path):
        """Sibling directory with prefix-matching name is rejected."""
        # checkpoint_dir is tmp_path / 'checkpoints'
        # Create a sibling with a name that is a prefix match
        evil_dir = tmp_path / 'checkpoints_evil'
        evil_dir.mkdir()

        # '../checkpoints_evil' resolves outside checkpoint_dir but
        # starts with the same string prefix — must still be rejected
        with pytest.raises(ValueError, match="escapes"):
            pipeline.load_checkpoint('../checkpoints_evil')

    # --- NaN / degenerate value tests ---

    def test_checkpoint_nan_in_optimizer_variable(self, pipeline):
        """NaN in optimizer variables is preserved through save/load."""
        dummy = tf.zeros((1, 16, 16, 16, 1))
        pipeline.model(dummy, training=False)
        y = pipeline.model.analysis(dummy)
        pipeline.entropy_model(y, training=False)

        # Train to populate momentum/variance variables
        batch = tf.zeros((1, 16, 16, 16))
        pipeline._train_step(batch, training=True)

        opt = pipeline.optimizers['reconstruction']
        # Find a float variable (skip int64 iteration counter)
        float_vars = [(i, v) for i, v in enumerate(opt.variables)
                      if v.dtype == tf.float32]
        assert len(float_vars) > 0
        idx, target_var = float_vars[0]

        nan_value = np.full_like(target_var.numpy(), float('nan'))
        target_var.assign(nan_value)

        pipeline.save_checkpoint('nan_test')

        # Load into fresh pipeline
        new_pipeline = TrainingPipeline(pipeline.config_path)
        new_pipeline.model(dummy, training=False)
        y2 = new_pipeline.model.analysis(dummy)
        new_pipeline.entropy_model(y2, training=False)
        new_pipeline._train_step(batch, training=True)
        new_pipeline.load_checkpoint('nan_test')

        loaded_var = new_pipeline.optimizers['reconstruction'].variables[idx]
        assert np.all(np.isnan(loaded_var.numpy()))

    # --- Zero / empty / boundary tests ---

    def test_save_checkpoint_before_training(self, pipeline):
        """Checkpoint saved before training loads without error."""
        dummy = tf.zeros((1, 16, 16, 16, 1))
        pipeline.model(dummy, training=False)
        y = pipeline.model.analysis(dummy)
        pipeline.entropy_model(y, training=False)

        # No training step — optimizer has only internal state (iteration counter)
        pipeline.save_checkpoint('untrained')

        checkpoint_dir = Path(pipeline.checkpoint_dir) / 'untrained'
        assert (checkpoint_dir / 'model.weights.h5').exists()
        assert (checkpoint_dir / 'entropy.weights.h5').exists()

        # Loading the untrained checkpoint should not crash
        new_pipeline = TrainingPipeline(pipeline.config_path)
        new_pipeline.model(dummy, training=False)
        y2 = new_pipeline.model.analysis(dummy)
        new_pipeline.entropy_model(y2, training=False)
        new_pipeline.load_checkpoint('untrained')

    # --- Negative / error path tests ---

    def test_load_checkpoint_missing_weights_file(self, pipeline):
        """Missing model weights file raises error on load."""
        dummy = tf.zeros((1, 16, 16, 16, 1))
        pipeline.model(dummy, training=False)
        y = pipeline.model.analysis(dummy)
        pipeline.entropy_model(y, training=False)

        pipeline.save_checkpoint('incomplete')

        # Delete the model weights file
        weights_path = Path(pipeline.checkpoint_dir) / 'incomplete' / 'model.weights.h5'
        weights_path.unlink()

        new_pipeline = TrainingPipeline(pipeline.config_path)
        new_pipeline.model(dummy, training=False)
        y2 = new_pipeline.model.analysis(dummy)
        new_pipeline.entropy_model(y2, training=False)

        with pytest.raises(Exception):
            new_pipeline.load_checkpoint('incomplete')

    def test_checkpoint_partial_optimizer_files(self, pipeline):
        """Missing optimizer .npy files are silently skipped."""
        dummy = tf.zeros((1, 16, 16, 16, 1))
        pipeline.model(dummy, training=False)
        y = pipeline.model.analysis(dummy)
        pipeline.entropy_model(y, training=False)

        batch = tf.zeros((1, 16, 16, 16))
        pipeline._train_step(batch, training=True)
        pipeline.save_checkpoint('partial_test')

        # Delete the last .npy file from an optimizer dir
        opt_dir = Path(pipeline.checkpoint_dir) / 'partial_test' / 'reconstruction_optimizer'
        if opt_dir.exists():
            npy_files = sorted(opt_dir.glob('*.npy'))
            if len(npy_files) > 1:
                npy_files[-1].unlink()

        # Loading should succeed — missing files silently skipped
        new_pipeline = TrainingPipeline(pipeline.config_path)
        new_pipeline.model(dummy, training=False)
        y2 = new_pipeline.model.analysis(dummy)
        new_pipeline.entropy_model(y2, training=False)
        new_pipeline._train_step(batch, training=True)
        new_pipeline.load_checkpoint('partial_test')

    # --- Regression tests ---

    def test_load_old_format_pickle_file_ignored(self, pipeline):
        """Old-style pickle .npy file at checkpoint level is safely ignored."""
        dummy = tf.zeros((1, 16, 16, 16, 1))
        pipeline.model(dummy, training=False)
        y = pipeline.model.analysis(dummy)
        pipeline.entropy_model(y, training=False)

        pipeline.save_checkpoint('format_test')

        # Place an old-format pickle file alongside new-format directories
        checkpoint_dir = Path(pipeline.checkpoint_dir) / 'format_test'
        old_file = checkpoint_dir / 'stale_optimizer.npy'
        np.save(str(old_file), np.array([np.zeros(5)], dtype=object),
                allow_pickle=True)

        # Loading should succeed, ignoring the old file
        new_pipeline = TrainingPipeline(pipeline.config_path)
        new_pipeline.model(dummy, training=False)
        y2 = new_pipeline.model.analysis(dummy)
        new_pipeline.entropy_model(y2, training=False)
        new_pipeline.load_checkpoint('format_test')

    # --- Integration test ---

    def test_checkpoint_optimizer_state_values_survive_roundtrip(self, pipeline):
        """Optimizer variable values are numerically equal after save/load."""
        dummy = tf.zeros((1, 16, 16, 16, 1))
        pipeline.model(dummy, training=False)
        y = pipeline.model.analysis(dummy)
        pipeline.entropy_model(y, training=False)

        batch = tf.zeros((1, 16, 16, 16))
        for _ in range(3):
            pipeline._train_step(batch, training=True)

        opt = pipeline.optimizers['reconstruction']
        original_values = [v.numpy().copy() for v in opt.variables]

        pipeline.save_checkpoint('opt_fidelity')

        new_pipeline = TrainingPipeline(pipeline.config_path)
        new_pipeline.model(dummy, training=False)
        y2 = new_pipeline.model.analysis(dummy)
        new_pipeline.entropy_model(y2, training=False)
        new_pipeline._train_step(batch, training=True)
        new_pipeline.load_checkpoint('opt_fidelity')

        new_opt = new_pipeline.optimizers['reconstruction']
        for orig, loaded in zip(original_values,
                                [v.numpy() for v in new_opt.variables]):
            np.testing.assert_array_equal(orig, loaded)
