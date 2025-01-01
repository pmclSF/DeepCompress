import tensorflow as tf
import pytest
from pathlib import Path
from test_utils import (
    setup_test_environment,
    create_mock_point_cloud,
    create_mock_voxel_grid,
    create_test_dataset
)
from training_pipeline import TrainingPipeline
from evaluation_pipeline import EvaluationPipeline
from data_loader import DataLoader

class TestIntegration(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up test environment."""
        self.test_env = setup_test_environment(tmp_path)
        self.training_pipeline = TrainingPipeline(self.test_env['config_path'])
        self.eval_pipeline = EvaluationPipeline(self.test_env['config_path'])
        
    def test_data_pipeline_integration(self):
        """Test data loading and preprocessing pipeline."""
        # Test point cloud to voxel conversion
        point_cloud = create_mock_point_cloud(1000)
        loader = DataLoader(self.test_env['config'])
        
        # Test complete preprocessing pipeline
        voxelized = loader._voxelize_points(
            point_cloud,
            self.test_env['config']['data']['resolution']
        )
        
        self.assertEqual(
            voxelized.shape,
            (self.test_env['config']['data']['resolution'],) * 3
        )
        
        # Test dataset creation
        dataset = loader.load_training_data()
        batch = next(iter(dataset))
        self.assertEqual(
            batch.shape[1:],
            (self.test_env['config']['data']['resolution'],) * 3
        )

    @pytest.mark.gpu
    def test_training_evaluation_integration(self):
        """Test training and evaluation pipeline integration."""
        # Create small dataset
        dataset = create_test_dataset(
            self.training_pipeline.config['training']['batch_size'],
            self.training_pipeline.config['data']['resolution']
        )
        
        # Mock data loaders
        self.training_pipeline.data_loader.load_training_data = lambda: dataset
        self.training_pipeline.data_loader.load_evaluation_data = lambda: dataset
        
        # Train for a few steps
        self.training_pipeline.train(epochs=1, validate_every=2)
        
        # Save checkpoint
        self.training_pipeline.save_checkpoint('integration_test')
        
        # Load checkpoint in evaluation pipeline
        self.eval_pipeline.load_checkpoint('integration_test')
        
        # Run evaluation
        results = self.eval_pipeline.evaluate()
        
        # Verify results
        self.assertIn('psnr', results)
        self.assertIn('chamfer_distance', results)
        self.assertGreater(results['psnr'], 0)

    def test_compression_pipeline_integration(self):
        """Test complete compression pipeline."""
        # Create test input
        input_data = create_mock_voxel_grid(
            self.test_env['config']['data']['resolution']
        )
        
        # Compress
        compressed, metrics = self.training_pipeline.model.compress(
            tf.expand_dims(input_data, 0)
        )
        
        # Verify compression metrics
        self.assertIn('bit_rate', metrics)
        self.assertGreater(metrics['bit_rate'], 0)
        
        # Decompress
        decompressed = self.training_pipeline.model.decompress(compressed)
        
        # Verify shape preservation
        self.assertEqual(
            decompressed.shape[1:],
            (self.test_env['config']['data']['resolution'],) * 3
        )
        
        # Calculate metrics
        eval_metrics = self.eval_pipeline.compute_metrics(
            decompressed[0],
            input_data
        )
        
        # Verify evaluation metrics
        self.assertIn('psnr', eval_metrics)
        self.assertIn('chamfer_distance', eval_metrics)

    @pytest.mark.e2e
    def test_complete_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Create and preprocess data
        point_cloud = create_mock_point_cloud(1000)
        loader = DataLoader(self.test_env['config'])
        voxelized = loader._voxelize_points(
            point_cloud,
            self.test_env['config']['data']['resolution']
        )
        
        # 2. Train model
        dataset = create_test_dataset(
            self.training_pipeline.config['training']['batch_size'],
            self.training_pipeline.config['data']['resolution']
        )
        self.training_pipeline.data_loader.load_training_data = lambda: dataset
        self.training_pipeline.data_loader.load_evaluation_data = lambda: dataset
        self.training_pipeline.train(epochs=1)
        
        # 3. Compress and decompress
        compressed, metrics = self.training_pipeline.model.compress(
            tf.expand_dims(voxelized, 0)
        )
        decompressed = self.training_pipeline.model.decompress(compressed)
        
        # 4. Evaluate results
        eval_metrics = self.eval_pipeline.compute_metrics(
            decompressed[0],
            voxelized
        )
        
        # 5. Save results and generate report
        results_dir = Path(self.test_env['config']['evaluation']['output_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        
        self.eval_pipeline.save_results(
            {'test_sample': eval_metrics},
            results_dir / 'test_results.json'
        )
        
        # Verify results exist and contain expected metrics
        self.assertTrue((results_dir / 'test_results.json').exists())
        self.assertGreater(eval_metrics['psnr'], 0)
        self.assertGreater(eval_metrics['chamfer_distance'], 0)
        
        # Verify bit rate is within expected range
        self.assertGreater(metrics['bit_rate'], 0)
        self.assertLess(metrics['bit_rate'], 10)  # Reasonable upper bound for test data

    def test_model_component_integration(self):
        """Test integration between model components."""
        # Test input
        input_data = create_mock_voxel_grid(
            self.test_env['config']['data']['resolution']
        )
        input_batch = tf.expand_dims(input_data, 0)
        
        # 1. Test Analysis Transform
        analysis_output = self.training_pipeline.model.analysis_transform(input_batch)
        self.assertIsNotNone(analysis_output)
        
        # 2. Test Entropy Model
        entropy_code = self.training_pipeline.entropy_model.encode(analysis_output)
        entropy_decoded = self.training_pipeline.entropy_model.decode(entropy_code)
        self.assertAllClose(analysis_output, entropy_decoded, rtol=0.1)
        
        # 3. Test Synthesis Transform
        synthesis_output = self.training_pipeline.model.synthesis_transform(entropy_decoded)
        self.assertEqual(synthesis_output.shape, input_batch.shape)
        
        # 4. Verify gradients flow through all components
        with tf.GradientTape() as tape:
            output = self.training_pipeline.model(input_batch, training=True)
            loss = tf.reduce_mean(tf.square(output - input_batch))
            
        gradients = tape.gradient(loss, self.training_pipeline.model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))

    def test_dataset_model_integration(self):
        """Test integration between dataset pipeline and model."""
        # Load a batch from the dataset
        dataset = self.training_pipeline.data_loader.load_training_data()
        batch = next(iter(dataset))
        
        # Process through model
        output = self.training_pipeline.model(batch, training=False)
        
        # Verify output properties
        self.assertEqual(output.shape, batch.shape)
        self.assertTrue(tf.reduce_all(output >= 0))
        self.assertTrue(tf.reduce_all(output <= 1))
        
        # Test data augmentation integration
        augmented_batch = self.training_pipeline.data_loader._augment(batch)
        augmented_output = self.training_pipeline.model(augmented_batch, training=False)
        
        # Verify augmentation affects output
        self.assertNotAllClose(output, augmented_output)

if __name__ == '__main__':
    tf.test.main()