import json
import sys
from pathlib import Path

import pytest
import tensorflow as tf
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

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
                'filters': 32,
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

    def test_initialization(self, pipeline):
        assert pipeline.model is not None
        assert pipeline.metrics is not None
        assert pipeline.data_loader is not None

    def test_evaluate_single(self, pipeline):
        # Model expects a 5D voxel grid (B, D, H, W, C)
        voxel_grid = tf.cast(
            tf.random.uniform((1, 16, 16, 16, 1)) > 0.5, tf.float32
        )
        results = pipeline._evaluate_single(voxel_grid)

        for metric in ['psnr', 'chamfer']:
            assert metric in results

    def test_evaluate_multiple_inputs(self, pipeline):
        """Test evaluation on multiple voxel grids produces consistent results."""
        grids = [
            tf.cast(tf.random.uniform((1, 16, 16, 16, 1)) > 0.5, tf.float32)
            for _ in range(3)
        ]

        all_results = []
        for grid in grids:
            results = pipeline._evaluate_single(grid)
            assert 'psnr' in results
            assert 'chamfer' in results
            all_results.append(results)

        assert len(all_results) == 3

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

    def test_load_model_no_checkpoint_configured(self, config_path):
        """Pipeline initializes when config has no checkpoint_path."""
        pipeline = EvaluationPipeline(config_path)
        assert pipeline.model is not None
        assert pipeline.config.get('checkpoint_path') is None

    def test_load_model_empty_string_checkpoint(self, tmp_path):
        """Empty string checkpoint_path is treated as no checkpoint."""
        config = {
            'data': {
                'modelnet40_path': str(tmp_path / 'modelnet40'),
                'ivfb_path': str(tmp_path / '8ivfb')
            },
            'model': {
                'filters': 32,
                'activation': 'cenic_gdn',
                'conv_type': 'separable'
            },
            'evaluation': {
                'metrics': ['psnr'],
                'output_dir': str(tmp_path / 'results'),
                'visualize': True
            },
            'checkpoint_path': ''
        }
        config_file = tmp_path / 'config_empty_ckpt.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        pipeline = EvaluationPipeline(str(config_file))
        assert pipeline.model is not None

    def test_load_model_missing_checkpoint_raises(self, tmp_path):
        """Non-existent checkpoint_path raises FileNotFoundError."""
        config = {
            'data': {
                'modelnet40_path': str(tmp_path / 'modelnet40'),
                'ivfb_path': str(tmp_path / '8ivfb')
            },
            'model': {
                'filters': 32,
                'activation': 'cenic_gdn',
                'conv_type': 'separable'
            },
            'evaluation': {
                'metrics': ['psnr'],
                'output_dir': str(tmp_path / 'results'),
                'visualize': True
            },
            'checkpoint_path': str(tmp_path / 'nonexistent' / 'model.weights.h5')
        }
        config_file = tmp_path / 'config_missing_ckpt.yml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            EvaluationPipeline(str(config_file))

if __name__ == '__main__':
    tf.test.main()
