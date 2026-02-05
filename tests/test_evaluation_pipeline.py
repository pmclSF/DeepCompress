import tensorflow as tf
import pytest
import numpy as np
from pathlib import Path
import yaml
from evaluation_pipeline import EvaluationPipeline, EvaluationResult

class TestEvaluationPipeline:
    @pytest.fixture
    def config_path(self, tmp_path):
        config = {
            'data': {
                'modelnet40_path': str(tmp_path / 'modelnet40'),
                'ivfb_path': str(tmp_path / '8ivfb')
            },
            'model': {
                'filters': 64,
                'activation': 'cenic_gdn',
                'conv_type': 'separable'
            },
            'evaluation': {
                'metrics': ['psnr', 'chamfer', 'bd_rate'],
                'output_dir': str(tmp_path / 'results'),
                'visualize': True
            }
        }
        
        config_file = tmp_path / 'config.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        return str(config_file)

    @pytest.fixture
    def pipeline(self, config_path):
        return EvaluationPipeline(config_path)

    @pytest.fixture
    def create_sample_ply(self, tmp_path):
        def _create_ply(filename):
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write("element vertex 8\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                for x in [-1, 1]:
                    for y in [-1, 1]:
                        for z in [-1, 1]:
                            f.write(f"{x} {y} {z}\n")
        return _create_ply

    def test_initialization(self, pipeline):
        assert pipeline.model is not None
        assert pipeline.metrics is not None
        assert pipeline.data_loader is not None

    def test_evaluate_single(self, pipeline):
        point_cloud = tf.random.uniform((1000, 3), -1, 1)
        results = pipeline._evaluate_single(point_cloud)
        
        for metric in ['psnr', 'chamfer', 'bd_rate']:
            assert metric in results
            assert isinstance(results[metric], float)
            assert not np.isnan(results[metric])
            assert results[metric] >= 0

    def test_evaluate_full_pipeline(self, pipeline, tmp_path, create_sample_ply):
        test_dir = tmp_path / '8ivfb'
        test_dir.mkdir(parents=True)
        
        num_samples = 3
        for i in range(num_samples):
            create_sample_ply(test_dir / f'test_{i}.ply')
        
        pipeline.config['data']['ivfb_path'] = str(test_dir)
        results = pipeline.evaluate()
        
        assert isinstance(results, dict)
        assert len(results) == num_samples
        
        for filename, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert hasattr(result, 'psnr')
            assert hasattr(result, 'chamfer_distance')
            assert hasattr(result, 'bd_rate')
            assert all(not np.isnan(getattr(result, metric)) 
                      for metric in ['psnr', 'chamfer_distance', 'bd_rate'])

    def test_generate_report(self, pipeline, tmp_path):
        results = {
            'test_1.ply': EvaluationResult(
                psnr=35.5,
                chamfer_distance=0.001,
                bd_rate=0.95,
                file_size=1000,
                compression_time=0.1,
                decompression_time=0.05
            ),
            'test_2.ply': EvaluationResult(
                psnr=34.8,
                chamfer_distance=0.002,
                bd_rate=0.93,
                file_size=1200,
                compression_time=0.12,
                decompression_time=0.06
            )
        }
        
        pipeline.generate_report(results)
        report_path = Path(pipeline.config['evaluation']['output_dir']) / "evaluation_report.json"
        assert report_path.exists()
        
        with open(report_path) as f:
            report_data = json.load(f)
            
        assert 'model_performance' in report_data
        assert 'aggregate_metrics' in report_data
        assert len(report_data['model_performance']) == len(results)

if __name__ == '__main__':
    tf.test.main()