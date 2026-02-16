import tensorflow as tf
import json
import os
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
import logging

@dataclass
class ExperimentMetrics:
    """Container for experiment metrics."""
    psnr: float
    bd_rate: float
    bitrate: float
    compression_ratio: float
    compression_time: float
    decompression_time: float

class ExperimentReporter:
    """Reporter for compression experiments."""
    
    def __init__(self, experiment_results: Dict[str, Any]):
        self.experiment_results = experiment_results
        self.summary = self._initialize_summary()
        self.logger = logging.getLogger(__name__)
        
    def _initialize_summary(self) -> Dict[str, Any]:
        """Initialize summary metrics."""
        return {
            'experiment_metadata': {
                'octree_levels': self.experiment_results.get('octree_levels', 'N/A'),
                'quantization_levels': self.experiment_results.get('quantization_levels', 'N/A'),
                'num_files': len(self.experiment_results),
                'timestamp': self.experiment_results.get('timestamp', 'N/A')
            }
        }
        
    @tf.function
    def compute_aggregate_metrics(self) -> Dict[str, tf.Tensor]:
        """Compute aggregate metrics using TensorFlow operations."""
        metrics = []
        
        for file_name, results in self.experiment_results.items():
            if file_name in ['timestamp', 'octree_levels', 'quantization_levels']:
                continue
                
            if all(key in results for key in ['psnr', 'bd_rate', 'bitrate']):
                metrics.append([
                    results['psnr'],
                    results['bd_rate'],
                    results['bitrate']
                ])
        
        if not metrics:
            return {}
            
        metrics_tensor = tf.convert_to_tensor(metrics, dtype=tf.float32)
        
        return {
            'avg_psnr': tf.reduce_mean(metrics_tensor[:, 0]),
            'avg_bd_rate': tf.reduce_mean(metrics_tensor[:, 1]),
            'avg_bitrate': tf.reduce_mean(metrics_tensor[:, 2])
        }
        
    def _compute_best_metrics(self) -> Dict[str, Any]:
        """Compute best metrics across all experiments."""
        best_metrics = {
            'psnr': float('-inf'),
            'bd_rate': float('inf'),
            'bitrate': float('inf'),
            'compression_ratio': float('inf'),
            'compression_time': float('inf'),
            'decompression_time': float('inf')
        }
        
        best_models = {
            'psnr': None,
            'bd_rate': None,
            'bitrate': None,
            'compression_ratio': None,
            'compression_time': None,
            'decompression_time': None
        }
        
        for file_name, results in self.experiment_results.items():
            if file_name in ['timestamp', 'octree_levels', 'quantization_levels']:
                continue
                
            # Update best metrics
            for metric in best_metrics.keys():
                if metric in results:
                    value = results[metric]
                    if metric == 'psnr':  # Higher is better
                        if value > best_metrics[metric]:
                            best_metrics[metric] = value
                            best_models[metric] = file_name
                    else:  # Lower is better
                        if value < best_metrics[metric]:
                            best_metrics[metric] = value
                            best_models[metric] = file_name
        
        return {
            'metrics': best_metrics,
            'models': best_models
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report of experiment results."""
        # Compute aggregate metrics
        aggregate_metrics = self.compute_aggregate_metrics()

        # Get best performance metrics
        best_perf = self._compute_best_metrics()

        # Build best_performance with flat keys (best_psnr, best_bd_rate, etc.)
        best_performance = {}
        for metric, model_name in best_perf['models'].items():
            best_performance[f'best_{metric}'] = model_name

        # Find overall best model (highest PSNR)
        best_model = {}
        if best_perf['models'].get('psnr'):
            best_model_name = best_perf['models']['psnr']
            for file_name, results in self.experiment_results.items():
                if file_name == best_model_name and isinstance(results, dict):
                    best_model = {'file': file_name, **results}
                    break

        # Compile model performance data
        model_performance = []
        for file_name, results in self.experiment_results.items():
            if file_name in ['timestamp', 'octree_levels', 'quantization_levels']:
                continue

            model_data = {
                'file': file_name,
                'metrics': {
                    metric: results.get(metric, None)
                    for metric in [
                        'psnr', 'bd_rate', 'bitrate',
                        'compression_ratio', 'compression_time',
                        'decompression_time'
                    ]
                }
            }
            model_performance.append(model_data)

        # Build aggregate statistics
        aggregate_statistics = {
            k: float(v.numpy()) if isinstance(v, tf.Tensor) else v
            for k, v in aggregate_metrics.items()
        }
        aggregate_statistics['best_model'] = best_model

        report = {
            'experiment_metadata': self.summary['experiment_metadata'],
            'aggregate_metrics': {
                k: float(v.numpy()) if isinstance(v, tf.Tensor) else v
                for k, v in aggregate_metrics.items()
            },
            'aggregate_statistics': aggregate_statistics,
            'best_performance': best_performance,
            'model_performance': model_performance
        }

        return report

    def save_report(self, output_file: str):
        """Save the generated report to a file."""
        report = self.generate_report()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        self.logger.info(f"Report saved to {output_file}")

def load_experiment_results(input_file: str) -> Dict[str, Any]:
    """Load experiment results from a JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate experiment report")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to experiment results JSON file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the generated report"
    )
    args = parser.parse_args()
    
    try:
        # Load results and generate report
        results = load_experiment_results(args.input_file)
        reporter = ExperimentReporter(results)
        reporter.save_report(args.output_file)
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()