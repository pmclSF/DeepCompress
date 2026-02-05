"""
Performance Benchmarking Utilities for DeepCompress.

This module provides utilities for measuring and comparing performance
of different model configurations and optimizations.

Usage:
    python -m src.benchmarks

Or programmatically:
    from benchmarks import Benchmark, MemoryProfiler, compare_implementations

    # Time a function
    with Benchmark("my_operation"):
        result = expensive_function()

    # Profile memory
    with MemoryProfiler() as mem:
        result = memory_intensive_function()
    print(f"Peak memory: {mem.peak_mb:.1f} MB")
"""

import tensorflow as tf
import numpy as np
import time
import functools
from typing import Callable, Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import sys


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    elapsed_seconds: float
    iterations: int
    memory_mb: Optional[float] = None
    throughput: Optional[float] = None  # items/second
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ms_per_iteration(self) -> float:
        return (self.elapsed_seconds / self.iterations) * 1000

    def __str__(self) -> str:
        s = f"{self.name}: {self.ms_per_iteration:.2f} ms/iter"
        if self.memory_mb is not None:
            s += f", {self.memory_mb:.1f} MB"
        if self.throughput is not None:
            s += f", {self.throughput:.1f} items/s"
        return s


class Benchmark:
    """
    Context manager for timing code blocks.

    Usage:
        with Benchmark("operation_name") as b:
            result = expensive_function()
        print(b.result)
    """

    def __init__(self, name: str, iterations: int = 1):
        self.name = name
        self.iterations = iterations
        self.result: Optional[BenchmarkResult] = None
        self._start_time: Optional[float] = None

    def __enter__(self) -> 'Benchmark':
        # Sync GPU operations before timing
        if tf.config.list_physical_devices('GPU'):
            tf.debugging.set_log_device_placement(False)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed = time.perf_counter() - self._start_time
        self.result = BenchmarkResult(
            name=self.name,
            elapsed_seconds=elapsed,
            iterations=self.iterations
        )


class MemoryProfiler:
    """
    Context manager for profiling memory usage.

    Tracks peak memory allocation during a code block.

    Usage:
        with MemoryProfiler() as mem:
            large_tensor = tf.zeros((10000, 10000))
        print(f"Peak: {mem.peak_mb} MB")
    """

    def __init__(self):
        self.peak_mb: float = 0.0
        self.allocated_mb: float = 0.0
        self._initial_memory: int = 0

    def __enter__(self) -> 'MemoryProfiler':
        # Reset memory stats
        tf.config.experimental.reset_memory_stats('GPU:0') if tf.config.list_physical_devices('GPU') else None
        self._initial_memory = self._get_current_memory()
        return self

    def __exit__(self, *args) -> None:
        final_memory = self._get_current_memory()
        self.allocated_mb = (final_memory - self._initial_memory) / (1024 * 1024)

        # Get peak memory if available
        if tf.config.list_physical_devices('GPU'):
            try:
                stats = tf.config.experimental.get_memory_info('GPU:0')
                self.peak_mb = stats.get('peak', 0) / (1024 * 1024)
            except Exception:
                self.peak_mb = self.allocated_mb

    def _get_current_memory(self) -> int:
        """Get current memory usage in bytes."""
        if tf.config.list_physical_devices('GPU'):
            try:
                stats = tf.config.experimental.get_memory_info('GPU:0')
                return stats.get('current', 0)
            except Exception:
                pass
        return 0


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 3,
    iterations: int = 10,
    name: Optional[str] = None
) -> BenchmarkResult:
    """
    Benchmark a function with warmup and multiple iterations.

    Args:
        func: Function to benchmark.
        args: Positional arguments for the function.
        kwargs: Keyword arguments for the function.
        warmup: Number of warmup iterations (not timed).
        iterations: Number of timed iterations.
        name: Name for the benchmark (defaults to function name).

    Returns:
        BenchmarkResult with timing information.
    """
    kwargs = kwargs or {}
    name = name or func.__name__

    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Ensure GPU sync
    if tf.config.list_physical_devices('GPU'):
        tf.test.gpu_device_name()

    # Timed iterations
    start = time.perf_counter()
    for _ in range(iterations):
        _ = func(*args, **kwargs)

    # Sync GPU before stopping timer
    if tf.config.list_physical_devices('GPU'):
        tf.test.gpu_device_name()

    elapsed = time.perf_counter() - start

    return BenchmarkResult(
        name=name,
        elapsed_seconds=elapsed,
        iterations=iterations
    )


def compare_implementations(
    implementations: Dict[str, Callable],
    args: tuple = (),
    kwargs: dict = None,
    warmup: int = 3,
    iterations: int = 10
) -> Dict[str, BenchmarkResult]:
    """
    Compare multiple implementations of the same functionality.

    Args:
        implementations: Dict mapping names to functions.
        args: Positional arguments for all functions.
        kwargs: Keyword arguments for all functions.
        warmup: Number of warmup iterations.
        iterations: Number of timed iterations.

    Returns:
        Dict mapping names to BenchmarkResults.
    """
    results = {}
    for name, func in implementations.items():
        results[name] = benchmark_function(
            func, args, kwargs, warmup, iterations, name
        )
    return results


def print_comparison(results: Dict[str, BenchmarkResult]) -> None:
    """Print a formatted comparison of benchmark results."""
    if not results:
        return

    print("\n" + "=" * 60)
    print("Benchmark Results")
    print("=" * 60)

    # Find baseline (first result)
    baseline_name = list(results.keys())[0]
    baseline_time = results[baseline_name].ms_per_iteration

    for name, result in results.items():
        speedup = baseline_time / result.ms_per_iteration
        speedup_str = f"({speedup:.2f}x)" if name != baseline_name else "(baseline)"
        print(f"  {name:30s}: {result.ms_per_iteration:8.2f} ms {speedup_str}")

    print("=" * 60 + "\n")


@contextmanager
def gpu_memory_limit(limit_mb: int):
    """
    Context manager to temporarily limit GPU memory.

    Useful for testing memory efficiency of different implementations.

    Args:
        limit_mb: Memory limit in megabytes.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        yield
        return

    # Note: Memory limit can only be set before any GPU operations
    # This is mainly useful for documentation/testing guidance
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=limit_mb
                )]
            )
    except RuntimeError:
        pass  # Virtual devices must be set before GPUs are initialized

    yield


def create_test_input(
    batch_size: int = 1,
    depth: int = 32,
    height: int = 32,
    width: int = 32,
    channels: int = 64,
    dtype: tf.DType = tf.float32
) -> tf.Tensor:
    """Create a test input tensor for benchmarking."""
    return tf.random.normal(
        (batch_size, depth, height, width, channels),
        dtype=dtype
    )


# =============================================================================
# DeepCompress-specific benchmarks
# =============================================================================

def benchmark_scale_quantization():
    """Benchmark scale quantization implementations."""
    from entropy_model import PatchedGaussianConditional

    # Create scale table
    scale_table = tf.constant(
        [0.01 * (2 ** (i / 4)) for i in range(64)],
        dtype=tf.float32
    )

    # Create test scales
    test_scales = tf.random.uniform((1, 32, 32, 32, 64), 0.01, 1.0)

    # Original broadcasting implementation (for comparison)
    def broadcast_quantize(scale, table):
        scale = tf.abs(scale)
        scale = tf.clip_by_value(scale, table[0], table[-1])
        scale_exp = tf.expand_dims(scale, -1)
        table_exp = tf.expand_dims(table, 0)
        distances = tf.abs(scale_exp - table_exp)
        indices = tf.argmin(distances, axis=-1)
        return tf.gather(table, indices)

    # Binary search implementation
    midpoints = (scale_table[:-1].numpy() + scale_table[1:].numpy()) / 2
    midpoints_tf = tf.constant(midpoints, dtype=tf.float32)

    def binary_search_quantize(scale, table, midpoints):
        scale = tf.abs(scale)
        scale = tf.clip_by_value(scale, table[0], table[-1])
        scale_flat = tf.reshape(scale, [-1])
        indices = tf.searchsorted(midpoints, scale_flat, side='left')
        indices = tf.minimum(indices, tf.shape(table)[0] - 1)
        quantized_flat = tf.gather(table, indices)
        return tf.reshape(quantized_flat, tf.shape(scale))

    # Compare
    results = compare_implementations({
        'broadcast': lambda: broadcast_quantize(test_scales, scale_table),
        'binary_search': lambda: binary_search_quantize(test_scales, scale_table, midpoints_tf)
    })

    print_comparison(results)
    return results


def benchmark_masked_conv():
    """Benchmark mask creation implementations."""
    from context_model import MaskedConv3D
    import numpy as np

    # Original loop-based implementation
    def create_mask_loops(kernel_size, mask_type, in_channels, filters):
        kd, kh, kw = kernel_size
        center_d, center_h, center_w = kd // 2, kh // 2, kw // 2
        mask = np.ones((kd, kh, kw, in_channels, filters), dtype=np.float32)

        for d in range(kd):
            for h in range(kh):
                for w in range(kw):
                    if d > center_d:
                        mask[d, h, w, :, :] = 0
                    elif d == center_d:
                        if h > center_h:
                            mask[d, h, w, :, :] = 0
                        elif h == center_h:
                            if w > center_w:
                                mask[d, h, w, :, :] = 0
                            elif w == center_w and mask_type == 'A':
                                mask[d, h, w, :, :] = 0
        return mask

    # Vectorized implementation
    def create_mask_vectorized(kernel_size, mask_type, in_channels, filters):
        kd, kh, kw = kernel_size
        center_d, center_h, center_w = kd // 2, kh // 2, kw // 2

        d_coords = np.arange(kd)[:, None, None]
        h_coords = np.arange(kh)[None, :, None]
        w_coords = np.arange(kw)[None, None, :]

        is_future = (
            (d_coords > center_d) |
            ((d_coords == center_d) & (h_coords > center_h)) |
            ((d_coords == center_d) & (h_coords == center_h) & (w_coords > center_w))
        )

        if mask_type == 'A':
            is_center = (
                (d_coords == center_d) &
                (h_coords == center_h) &
                (w_coords == center_w)
            )
            is_future = is_future | is_center

        mask = np.where(is_future, 0.0, 1.0).astype(np.float32)
        mask = np.broadcast_to(mask[:, :, :, None, None], (kd, kh, kw, in_channels, filters))
        return mask.copy()

    # Test parameters
    kernel_size = (5, 5, 5)
    in_channels = 64
    filters = 128

    results = compare_implementations({
        'loops': lambda: create_mask_loops(kernel_size, 'A', in_channels, filters),
        'vectorized': lambda: create_mask_vectorized(kernel_size, 'A', in_channels, filters)
    }, iterations=100)

    print_comparison(results)
    return results


def benchmark_attention():
    """Benchmark attention implementations."""
    from attention_context import SparseAttention3D, WindowedAttention3D

    dim = 64
    input_shape = (1, 16, 16, 16, dim)  # Smaller for testing

    sparse_attn = SparseAttention3D(dim=dim, num_heads=4)
    windowed_attn = WindowedAttention3D(dim=dim, num_heads=4, window_size=4)

    test_input = tf.random.normal(input_shape)

    # Build layers
    _ = sparse_attn(test_input)
    _ = windowed_attn(test_input)

    results = compare_implementations({
        'sparse_attention': lambda: sparse_attn(test_input, training=False),
        'windowed_attention': lambda: windowed_attn(test_input, training=False)
    })

    print_comparison(results)
    return results


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "=" * 70)
    print("DeepCompress Performance Benchmarks")
    print("=" * 70)

    print("\n1. Scale Quantization Benchmark")
    print("-" * 40)
    try:
        benchmark_scale_quantization()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n2. Masked Convolution Mask Creation Benchmark")
    print("-" * 40)
    try:
        benchmark_masked_conv()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n3. Attention Implementation Benchmark")
    print("-" * 40)
    try:
        benchmark_attention()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n" + "=" * 70)
    print("Benchmarks complete")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    run_all_benchmarks()
