import argparse
import logging
import os
from dataclasses import dataclass
from typing import Dict

import tensorflow as tf


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    max_distance: float = 1.0
    num_points: int = 2048
    use_normals: bool = True

class PointCloudMetrics(tf.keras.metrics.Metric):
    """Custom metric class for point cloud evaluation."""

    def __init__(self, name='point_cloud_metrics', **kwargs):
        super().__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')
        self.chamfer = self.add_weight(name='chamfer', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    @tf.function
    def compute_psnr(self, original: tf.Tensor, compressed: tf.Tensor) -> tf.Tensor:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = tf.reduce_mean(tf.square(original - compressed))
        max_val = tf.reduce_max(tf.abs(original))
        return 10 * tf.math.log(max_val**2 / mse) / tf.math.log(10.0)

    @tf.function
    def compute_chamfer(self,
                       original: tf.Tensor,
                       compressed: tf.Tensor) -> tf.Tensor:
        """Compute Chamfer distance."""
        # Compute pairwise distances
        original_expanded = tf.expand_dims(original, 1)
        compressed_expanded = tf.expand_dims(compressed, 0)

        distances = tf.reduce_sum(
            tf.square(original_expanded - compressed_expanded),
            axis=-1
        )

        # Compute minimum distances in both directions
        d1 = tf.reduce_min(distances, axis=1)
        d2 = tf.reduce_min(distances, axis=0)

        return tf.reduce_mean(d1) + tf.reduce_mean(d2)

    @tf.function
    def update_state(self, original: tf.Tensor, compressed: tf.Tensor):
        """Update metric states."""
        psnr = self.compute_psnr(original, compressed)
        chamfer = self.compute_chamfer(original, compressed)

        self.psnr.assign_add(psnr)
        self.chamfer.assign_add(chamfer)
        self.count.assign_add(1)

    def result(self) -> Dict[str, tf.Tensor]:
        """Return computed metrics."""
        return {
            'psnr': self.psnr / self.count,
            'chamfer': self.chamfer / self.count
        }

    def reset_state(self):
        """Reset metric states."""
        self.psnr.assign(0)
        self.chamfer.assign(0)
        self.count.assign(0)

class CompressionEvaluator:
    """Evaluator for point cloud compression."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = PointCloudMetrics()

    @tf.function
    def load_point_cloud(self, file_path: str) -> tf.Tensor:
        """Load point cloud from file."""
        raw_data = tf.io.read_file(file_path)
        lines = tf.strings.split(raw_data, '\n')

        # Skip header
        data_lines = lines[tf.where(
            tf.strings.regex_full_match(lines, r'[\d\.\-\+eE\s]+')
        )[:, 0]]

        # Parse points
        points = tf.strings.to_number(
            tf.strings.split(data_lines),
            out_type=tf.float32
        )

        return points[:, :3]  # Return only XYZ coordinates

    def evaluate_compression(self,
                           original_path: str,
                           compressed_path: str) -> Dict[str, float]:
        """Evaluate compression quality."""
        # Load point clouds
        original = self.load_point_cloud(original_path)
        compressed = self.load_point_cloud(compressed_path)

        # Update metrics
        self.metrics.update_state(original, compressed)

        # Get results
        results = self.metrics.result()

        # Add additional metrics
        results['file_size_ratio'] = (
            os.path.getsize(compressed_path) /
            os.path.getsize(original_path)
        )

        return {k: float(v) for k, v in results.items()}

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate point cloud compression."
    )
    parser.add_argument(
        "original_dir",
        type=str,
        help="Directory containing original point clouds"
    )
    parser.add_argument(
        "compressed_dir",
        type=str,
        help="Directory containing compressed point clouds"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation results"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize evaluator
    evaluator = CompressionEvaluator(EvaluationConfig())

    # Find matching files
    original_files = set(os.listdir(args.original_dir))
    compressed_files = set(os.listdir(args.compressed_dir))
    common_files = original_files & compressed_files

    results = {}
    for filename in common_files:
        logger.info(f"Evaluating {filename}")

        original_path = os.path.join(args.original_dir, filename)
        compressed_path = os.path.join(args.compressed_dir, filename)

        results[filename] = evaluator.evaluate_compression(
            original_path,
            compressed_path
        )

    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
