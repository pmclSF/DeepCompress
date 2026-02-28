"""
Quick benchmark for testing DeepCompress compression performance.

This script tests the model's compression capabilities without requiring
a trained checkpoint or external dataset. It uses synthetic voxel grids
and measures:
- Compression ratio (bits per voxel)
- Reconstruction quality (MSE, PSNR)
- Encoding/decoding speed
- Memory usage

Usage:
    python -m src.quick_benchmark
    python -m src.quick_benchmark --resolution 64 --batch_size 2
"""

import argparse
import os

# Add src to path
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))

from .model_transforms import DeepCompressModel, DeepCompressModelV2, TransformConfig


@dataclass
class CompressionMetrics:
    """Metrics from compression test."""
    # Quality metrics
    mse: float
    psnr: float

    # Compression metrics
    input_elements: int
    latent_elements: int
    estimated_bits: float
    bits_per_voxel: float
    compression_ratio: float

    # Speed metrics
    encode_time_ms: float
    decode_time_ms: float
    total_time_ms: float

    # Memory (if available)
    peak_memory_mb: Optional[float] = None

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "Compression Benchmark Results",
            "=" * 60,
            "",
            "Quality Metrics:",
            f"  MSE:                    {self.mse:.6f}",
            f"  PSNR:                   {self.psnr:.2f} dB",
            "",
            "Compression Metrics:",
            f"  Input elements:         {self.input_elements:,}",
            f"  Latent elements:        {self.latent_elements:,}",
            f"  Estimated bits:         {self.estimated_bits:,.0f}",
            f"  Bits per voxel:         {self.bits_per_voxel:.3f}",
            f"  Compression ratio:      {self.compression_ratio:.1f}x",
            "",
            "Speed Metrics:",
            f"  Encode time:            {self.encode_time_ms:.1f} ms",
            f"  Decode time:            {self.decode_time_ms:.1f} ms",
            f"  Total time:             {self.total_time_ms:.1f} ms",
        ]

        if self.peak_memory_mb is not None:
            lines.append(f"  Peak memory:            {self.peak_memory_mb:.1f} MB")

        lines.append("=" * 60)
        return "\n".join(lines)


def create_synthetic_voxel_grid(
    batch_size: int,
    resolution: int,
    density: float = 0.1,
    seed: int = 42
) -> tf.Tensor:
    """
    Create synthetic voxel grid for testing.

    Args:
        batch_size: Number of samples in batch.
        resolution: Spatial resolution (resolution^3 voxels).
        density: Fraction of voxels that are occupied (0-1).
        seed: Random seed for reproducibility.

    Returns:
        Binary voxel grid tensor of shape (B, D, H, W, 1).
    """
    np.random.seed(seed)

    # Create sparse binary occupancy grid
    shape = (batch_size, resolution, resolution, resolution, 1)
    grid = np.random.random(shape) < density

    # Add some structure (spherical objects)
    for b in range(batch_size):
        # Add 2-5 random spheres
        num_spheres = np.random.randint(2, 6)
        for _ in range(num_spheres):
            # Random center and radius
            cx = np.random.randint(resolution // 4, 3 * resolution // 4)
            cy = np.random.randint(resolution // 4, 3 * resolution // 4)
            cz = np.random.randint(resolution // 4, 3 * resolution // 4)
            radius = np.random.randint(resolution // 8, resolution // 4)

            # Create sphere
            for x in range(max(0, cx - radius), min(resolution, cx + radius)):
                for y in range(max(0, cy - radius), min(resolution, cy + radius)):
                    for z in range(max(0, cz - radius), min(resolution, cz + radius)):
                        if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2:
                            grid[b, x, y, z, 0] = True

    return tf.constant(grid, dtype=tf.float32)


def compute_psnr(original: tf.Tensor, reconstructed: tf.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = tf.reduce_mean(tf.square(original - reconstructed))
    if mse == 0:
        return float('inf')
    # For binary data, max value is 1.0
    psnr = 20 * tf.math.log(1.0 / tf.sqrt(mse)) / tf.math.log(10.0)
    return float(psnr)


def benchmark_model(
    model: DeepCompressModel,
    input_tensor: tf.Tensor,
    warmup_runs: int = 2,
    timed_runs: int = 5
) -> CompressionMetrics:
    """
    Benchmark compression performance of a model.

    Args:
        model: DeepCompress model to benchmark.
        input_tensor: Input voxel grid.
        warmup_runs: Number of warmup runs (not timed).
        timed_runs: Number of timed runs to average.

    Returns:
        CompressionMetrics with all measurements.
    """
    # Warmup runs
    for _ in range(warmup_runs):
        _ = model(input_tensor, training=False)

    # Timed encode runs
    encode_times = []
    decode_times = []

    for _ in range(timed_runs):
        if isinstance(model, DeepCompressModelV2):
            # V2: measure encode and decode separately
            start = time.perf_counter()
            compressed = model.compress(input_tensor)
            encode_time = time.perf_counter() - start

            start = time.perf_counter()
            _ = model.decompress(compressed)
            decode_time = time.perf_counter() - start
        else:
            # V1: full forward pass (no separate encode/decode)
            start = time.perf_counter()
            _ = model(input_tensor, training=False)
            encode_time = time.perf_counter() - start
            decode_time = 0

        encode_times.append(encode_time)
        decode_times.append(decode_time)

    # Average times
    avg_encode_ms = np.mean(encode_times) * 1000
    avg_decode_ms = np.mean(decode_times) * 1000

    # Get final outputs for metrics
    # V1 returns (x_hat, y, z_hat, z_noisy)
    # V2 returns (x_hat, y, y_hat, z, rate_info)
    outputs = model(input_tensor, training=False)
    if len(outputs) == 4:
        x_hat, y, z_hat, z_noisy = outputs
        rate_info = None
    else:
        x_hat, y, y_hat, z, rate_info = outputs

    # Compute quality metrics
    mse = float(tf.reduce_mean(tf.square(input_tensor - x_hat)))
    psnr = compute_psnr(input_tensor, x_hat)

    # Compute compression metrics
    input_elements = int(np.prod(input_tensor.shape))
    latent_elements = int(np.prod(y.shape))

    # Estimate bits from latent representation
    if rate_info is not None and 'total_bits' in rate_info:
        # Use actual bits from entropy model
        estimated_bits = float(rate_info['total_bits'])
    else:
        # Approximate using Shannon entropy of quantized latent
        y_quantized = tf.round(y)
        y_flat = y_quantized.numpy().flatten()
        _, counts = np.unique(y_flat, return_counts=True)
        probs = counts / counts.sum()
        entropy_per_symbol = -np.sum(probs * np.log2(probs))
        estimated_bits = latent_elements * entropy_per_symbol

    bits_per_voxel = estimated_bits / input_elements

    # Compression ratio (assuming 32-bit float input)
    original_bits = input_elements * 32
    compression_ratio = original_bits / max(estimated_bits, 1)

    return CompressionMetrics(
        mse=mse,
        psnr=psnr,
        input_elements=input_elements,
        latent_elements=latent_elements,
        estimated_bits=estimated_bits,
        bits_per_voxel=bits_per_voxel,
        compression_ratio=compression_ratio,
        encode_time_ms=avg_encode_ms,
        decode_time_ms=avg_decode_ms,
        total_time_ms=avg_encode_ms + avg_decode_ms,
    )


def run_benchmark(
    resolution: int = 32,
    batch_size: int = 1,
    model_version: str = 'v1',
    filters: int = 32,
    entropy_model: str = 'hyperprior'
) -> CompressionMetrics:
    """
    Run compression benchmark.

    Args:
        resolution: Voxel grid resolution.
        batch_size: Batch size.
        model_version: 'v1' or 'v2'.
        filters: Number of filters in model.
        entropy_model: Entropy model type for v2.

    Returns:
        CompressionMetrics with results.
    """
    print("\nBenchmark Configuration:")
    print(f"  Resolution:     {resolution}x{resolution}x{resolution}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Model version:  {model_version}")
    print(f"  Filters:        {filters}")
    if model_version == 'v2':
        print(f"  Entropy model:  {entropy_model}")
    print()

    # Create config
    config = TransformConfig(
        filters=filters,
        kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        activation='relu',  # Use relu for faster testing
        conv_type='standard'
    )

    # Create model
    print("Creating model...")
    if model_version == 'v2':
        model = DeepCompressModelV2(config, entropy_model=entropy_model)
    else:
        model = DeepCompressModel(config)

    # Create synthetic data
    print("Creating synthetic data...")
    input_tensor = create_synthetic_voxel_grid(batch_size, resolution)
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Occupied voxels: {int(tf.reduce_sum(input_tensor))} / {int(np.prod(input_tensor.shape[1:4]))}")

    # Build model
    print("Building model...")
    _ = model(input_tensor, training=False)

    # Count parameters
    total_params = sum(np.prod(v.shape) for v in model.trainable_variables)
    print(f"  Total parameters: {total_params:,}")

    # Run benchmark
    print("\nRunning benchmark...")
    metrics = benchmark_model(model, input_tensor)

    return metrics


def compare_models(resolution: int = 32, batch_size: int = 1):
    """Compare different model configurations."""
    print("\n" + "=" * 70)
    print("Model Comparison Benchmark")
    print("=" * 70)

    configs = [
        {'model_version': 'v1', 'filters': 32},
        {'model_version': 'v2', 'filters': 32, 'entropy_model': 'hyperprior'},
        {'model_version': 'v2', 'filters': 32, 'entropy_model': 'channel'},
    ]

    results = []
    for cfg in configs:
        name = f"{cfg['model_version']}"
        if 'entropy_model' in cfg:
            name += f"-{cfg['entropy_model']}"

        print(f"\n--- Testing {name} ---")
        try:
            metrics = run_benchmark(
                resolution=resolution,
                batch_size=batch_size,
                **cfg
            )
            results.append((name, metrics))
            print(metrics)
        except Exception as e:
            print(f"Error: {e}")
            results.append((name, None))

    # Summary table
    print("\n" + "=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"{'Model':<20} {'PSNR (dB)':<12} {'BPV':<10} {'Time (ms)':<12} {'Ratio':<10}")
    print("-" * 70)
    for name, metrics in results:
        if metrics:
            print(f"{name:<20} {metrics.psnr:<12.2f} {metrics.bits_per_voxel:<10.3f} "
                  f"{metrics.total_time_ms:<12.1f} {metrics.compression_ratio:<10.1f}x")
        else:
            print(f"{name:<20} {'ERROR':<12}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Quick DeepCompress benchmark")
    parser.add_argument('--resolution', type=int, default=32,
                        help='Voxel grid resolution (default: 32)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--model', type=str, default='v1',
                        choices=['v1', 'v2'], help='Model version')
    parser.add_argument('--filters', type=int, default=32,
                        help='Number of filters (default: 32)')
    parser.add_argument('--entropy', type=str, default='hyperprior',
                        choices=['hyperprior', 'channel', 'context'],
                        help='Entropy model type for v2')
    parser.add_argument('--compare', action='store_true',
                        help='Compare multiple model configurations')

    args = parser.parse_args()

    if args.compare:
        compare_models(args.resolution, args.batch_size)
    else:
        metrics = run_benchmark(
            resolution=args.resolution,
            batch_size=args.batch_size,
            model_version=args.model,
            filters=args.filters,
            entropy_model=args.entropy
        )
        print(metrics)


if __name__ == '__main__':
    main()
