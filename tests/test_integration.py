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
        self.test_env = setup_test_environment(tmp_path)
        self.training_pipeline = TrainingPipeline(self.test_env['config_path'])
        self.eval_pipeline = EvaluationPipeline(self.test_env['config_path'])

    def test_data_pipeline_integration(self):
        point_cloud = create_mock_point_cloud(1000)
        loader = DataLoader(self.test_env['config'])
        voxelized = loader._voxelize_points(point_cloud, self.test_env['config']['data']['resolution'])
        
        self.assertEqual(voxelized.shape, (self.test_env['config']['data']['resolution'],) * 3)
        
        dataset = loader.load_training_data()
        batch = next(iter(dataset))
        self.assertEqual(batch.shape[1:], (self.test_env['config']['data']['resolution'],) * 3)

    @pytest.mark.gpu
    def test_training_evaluation_integration(self):
        dataset = create_test_dataset(
            self.training_pipeline.config['training']['batch_size'],
            self.training_pipeline.config['data']['resolution']
        )
        
        self.training_pipeline.data_loader.load_training_data = lambda: dataset
        self.training_pipeline.data_loader.load_evaluation_data = lambda: dataset
        self.training_pipeline.train(epochs=1, validate_every=2)
        self.training_pipeline.save_checkpoint('integration_test')
        self.eval_pipeline.load_checkpoint('integration_test')
        
        results = self.eval_pipeline.evaluate()
        self.assertIn('psnr', results)
        self.assertIn('chamfer_distance', results)
        self.assertGreater(results['psnr'], 0)

    def test_compression_pipeline_integration(self):
        input_data = create_mock_voxel_grid(self.test_env['config']['data']['resolution'])
        compressed, metrics = self.training_pipeline.model.compress(tf.expand_dims(input_data, 0))
        
        self.assertIn('bit_rate', metrics)
        self.assertGreater(metrics['bit_rate'], 0)
        
        decompressed = self.training_pipeline.model.decompress(compressed)
        self.assertEqual(decompressed.shape[1:], (self.test_env['config']['data']['resolution'],) * 3)
        
        eval_metrics = self.eval_pipeline.compute_metrics(decompressed[0], input_data)
        self.assertIn('psnr', eval_metrics)
        self.assertIn('chamfer_distance', eval_metrics)

    @pytest.mark.e2e
    def test_complete_workflow(self):
        point_cloud = create_mock_point_cloud(1000)
        loader = DataLoader(self.test_env['config'])
        voxelized = loader._voxelize_points(point_cloud, self.test_env['config']['data']['resolution'])
        
        dataset = create_test_dataset(
            self.training_pipeline.config['training']['batch_size'],
            self.training_pipeline.config['data']['resolution']
        )
        self.training_pipeline.data_loader.load_training_data = lambda: dataset
        self.training_pipeline.data_loader.load_evaluation_data = lambda: dataset
        self.training_pipeline.train(epochs=1)
        
        compressed, metrics = self.training_pipeline.model.compress(tf.expand_dims(voxelized, 0))
        decompressed = self.training_pipeline.model.decompress(compressed)
        
        eval_metrics = self.eval_pipeline.compute_metrics(decompressed[0], voxelized)
        
        results_dir = Path(self.test_env['config']['evaluation']['output_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_pipeline.save_results({'test_sample': eval_metrics}, results_dir / 'test_results.json')
        
        self.assertTrue((results_dir / 'test_results.json').exists())
        self.assertGreater(eval_metrics['psnr'], 0)
        self.assertGreater(eval_metrics['chamfer_distance'], 0)
        self.assertGreater(metrics['bit_rate'], 0)
        self.assertLess(metrics['bit_rate'], 10)

if __name__ == '__main__':
    tf.test.main()