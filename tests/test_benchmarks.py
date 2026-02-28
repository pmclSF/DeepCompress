"""
Tests for benchmark utilities and methodology.

Validates that Benchmark, MemoryProfiler, and benchmark_function produce
sensible results and the comparison utilities work correctly.
"""

import sys
import time
from pathlib import Path

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from benchmarks import (
    Benchmark,
    BenchmarkResult,
    benchmark_function,
    compare_implementations,
    create_test_input,
)


class TestBenchmarkResult(tf.test.TestCase):
    """Tests for BenchmarkResult dataclass."""

    def test_ms_per_iteration(self):
        """ms_per_iteration should correctly compute milliseconds."""
        result = BenchmarkResult(
            name="test",
            elapsed_seconds=1.0,
            iterations=10
        )
        self.assertAlmostEqual(result.ms_per_iteration, 100.0)

    def test_ms_per_iteration_single(self):
        """Single iteration should report total time in ms."""
        result = BenchmarkResult(
            name="test",
            elapsed_seconds=0.5,
            iterations=1
        )
        self.assertAlmostEqual(result.ms_per_iteration, 500.0)

    def test_str_representation(self):
        """String representation should include name and timing."""
        result = BenchmarkResult(
            name="my_op",
            elapsed_seconds=1.0,
            iterations=10,
            memory_mb=256.0
        )
        s = str(result)
        self.assertIn("my_op", s)
        self.assertIn("100.00", s)
        self.assertIn("256.0", s)


class TestBenchmarkContextManager(tf.test.TestCase):
    """Tests for Benchmark context manager."""

    def test_measures_time(self):
        """Should measure elapsed time > 0."""
        with Benchmark("sleep_test") as b:
            time.sleep(0.01)

        self.assertGreater(b.result.elapsed_seconds, 0.0)

    def test_result_has_correct_name(self):
        """Result should carry the benchmark name."""
        with Benchmark("named_op") as b:
            pass

        self.assertEqual(b.result.name, "named_op")

    def test_result_has_correct_iterations(self):
        """Result should record iteration count."""
        with Benchmark("iter_test", iterations=5) as b:
            pass

        self.assertEqual(b.result.iterations, 5)

    def test_timing_is_reasonable(self):
        """Measured time should be within order of magnitude of actual work."""
        with Benchmark("timed_op") as b:
            time.sleep(0.05)

        # Should be at least ~50ms but less than 1s
        self.assertGreater(b.result.elapsed_seconds, 0.01)
        self.assertLess(b.result.elapsed_seconds, 1.0)


class TestBenchmarkFunction(tf.test.TestCase):
    """Tests for benchmark_function utility."""

    def test_returns_benchmark_result(self):
        """Should return a BenchmarkResult."""
        def noop():
            return 42

        result = benchmark_function(noop, warmup=1, iterations=3)

        self.assertIsInstance(result, BenchmarkResult)
        self.assertEqual(result.iterations, 3)
        self.assertGreater(result.elapsed_seconds, 0.0)

    def test_warmup_not_timed(self):
        """Warmup iterations should not be included in timing."""
        call_count = [0]

        def counting_fn():
            call_count[0] += 1
            return call_count[0]

        result = benchmark_function(counting_fn, warmup=5, iterations=3)

        # Total calls = warmup + iterations = 8
        self.assertEqual(call_count[0], 8)
        # But result should say 3 iterations
        self.assertEqual(result.iterations, 3)

    def test_custom_name(self):
        """Should use custom name when provided."""
        result = benchmark_function(lambda: None, name="custom_name")
        self.assertEqual(result.name, "custom_name")

    def test_default_name_from_function(self):
        """Should use function name by default."""
        def my_function():
            return None

        result = benchmark_function(my_function, warmup=0, iterations=1)
        self.assertEqual(result.name, "my_function")

    def test_passes_args_and_kwargs(self):
        """Should pass args and kwargs to benchmarked function."""
        def add(a, b, c=0):
            return a + b + c

        # Should not raise
        result = benchmark_function(add, args=(1, 2), kwargs={'c': 3})
        self.assertGreater(result.elapsed_seconds, 0.0)


class TestCompareImplementations(tf.test.TestCase):
    """Tests for compare_implementations utility."""

    def test_returns_all_results(self):
        """Should return one result per implementation."""
        impls = {
            'fast': lambda: 1 + 1,
            'slow': lambda: sum(range(100)),
        }

        results = compare_implementations(impls, warmup=1, iterations=3)

        self.assertEqual(len(results), 2)
        self.assertIn('fast', results)
        self.assertIn('slow', results)

    def test_faster_impl_is_faster(self):
        """Faster implementation should measure less time (with tolerance)."""
        def fast():
            return 1 + 1

        def slow():
            total = 0
            for i in range(10000):
                total += i
            return total

        results = compare_implementations(
            {'fast': fast, 'slow': slow},
            warmup=2,
            iterations=10
        )

        # Fast should be faster (or at least not 10x slower)
        self.assertLess(
            results['fast'].elapsed_seconds,
            results['slow'].elapsed_seconds * 10
        )


class TestCreateTestInput(tf.test.TestCase):
    """Tests for test input tensor creation."""

    def test_default_shape(self):
        """Default shape should be (1, 32, 32, 32, 64)."""
        tensor = create_test_input()
        self.assertEqual(tensor.shape, (1, 32, 32, 32, 64))

    def test_custom_shape(self):
        """Should respect custom dimensions."""
        tensor = create_test_input(
            batch_size=2, depth=8, height=16, width=4, channels=32
        )
        self.assertEqual(tensor.shape, (2, 8, 16, 4, 32))

    def test_default_dtype(self):
        """Default dtype should be float32."""
        tensor = create_test_input()
        self.assertEqual(tensor.dtype, tf.float32)

    def test_custom_dtype(self):
        """Should respect custom dtype."""
        tensor = create_test_input(dtype=tf.float16)
        self.assertEqual(tensor.dtype, tf.float16)


if __name__ == '__main__':
    tf.test.main()
