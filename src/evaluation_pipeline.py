import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import tensorflow as tf

from data_loader import DataLoader
from ev_compare import PointCloudMetrics
from model_transforms import DeepCompressModel, TransformConfig
from mp_report import ExperimentReporter


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    psnr: float
    chamfer_distance: float
    bd_rate: float
    file_size: int
    compression_time: float
    decompression_time: float

class EvaluationPipeline:
    """Pipeline for evaluating DeepCompress model."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.metrics = PointCloudMetrics()
        self.model = self._load_model()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML."""
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_model(self) -> DeepCompressModel:
        """Load model from checkpoint."""
        model_config = TransformConfig(
            filters=self.config['model'].get('filters', 64),
            activation=self.config['model'].get('activation', 'cenic_gdn'),
            conv_type=self.config['model'].get('conv_type', 'separable')
        )

        model = DeepCompressModel(model_config)

        # Load weights if checkpoint provided
        checkpoint_path = self.config.get('checkpoint_path')
        if checkpoint_path:
            model.load_weights(checkpoint_path)

        return model

    @tf.function
    def _evaluate_single(self,
                        point_cloud: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Evaluate model on single point cloud."""
        # Forward pass
        compressed, metrics = self.model.compress(point_cloud)
        decompressed = self.model.decompress(compressed)

        # Compute metrics
        results = {}
        results['psnr'] = self.metrics.compute_psnr(point_cloud, decompressed)
        results['chamfer'] = self.metrics.compute_chamfer(point_cloud, decompressed)

        # Add compression metrics
        results.update(metrics)

        return results

    def evaluate(self) -> Dict[str, List[EvaluationResult]]:
        """Run evaluation on test dataset."""
        results = {}
        dataset = self.data_loader.load_evaluation_data()

        for point_cloud in dataset:
            # Get filename from dataset
            filename = point_cloud.filename.numpy().decode('utf-8')
            self.logger.info(f"Evaluating {filename}")

            try:
                # Time compression and decompression
                start_time = tf.timestamp()
                metrics = self._evaluate_single(point_cloud)
                end_time = tf.timestamp()

                # Create result object
                result = EvaluationResult(
                    psnr=float(metrics['psnr']),
                    chamfer_distance=float(metrics['chamfer']),
                    bd_rate=float(metrics.get('bd_rate', 0.0)),
                    file_size=int(metrics['compressed_size']),
                    compression_time=float(end_time - start_time),
                    decompression_time=float(metrics.get('decompress_time', 0.0))
                )

                results[filename] = result

            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}")
                continue

        return results

    def generate_report(self, results: Dict[str, List[EvaluationResult]]):
        """Generate evaluation report."""
        reporter = ExperimentReporter(results)

        # Generate and save report
        output_dir = Path(self.config['evaluation']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        report_path = output_dir / "evaluation_report.json"
        reporter.save_report(str(report_path))

        self.logger.info(f"Evaluation report saved to {report_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DeepCompress model")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run evaluation
    pipeline = EvaluationPipeline(args.config)
    results = pipeline.evaluate()
    pipeline.generate_report(results)

if __name__ == "__main__":
    main()
