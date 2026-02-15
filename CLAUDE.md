# CLAUDE.md — DeepCompress

> This file is for AI coding agents working on this codebase. It defines how to write, test, and modify code here without introducing the mistakes that LLMs typically make.

## What This Project Is

DeepCompress is a TensorFlow 2 library for learned point cloud compression. It compresses 3D voxelized occupancy grids using neural analysis/synthesis transforms with configurable entropy models. **It is not a PyTorch project.**

Python 3.10+ · TensorFlow ~=2.15 · tensorflow-probability ~=0.23 · MIT License

## Before You Write Any Code

1. **Read the existing code first.** Before modifying a file, read it entirely. Before adding a function, check if a similar one exists. Before creating a test utility, check `tests/test_utils.py` — it likely already has what you need.

2. **Run the tests.** Before and after every change: `pytest tests/ -v`. Do not submit changes that break existing tests. If a test was already skipped/broken before your change, leave it that way — do not fix unrelated tests as part of your change.

3. **Make minimal changes.** Fix what was asked. Do not refactor adjacent code, rename variables for "clarity," add type hints to unrelated functions, or restructure files. Every unnecessary change is a potential regression.

---

## Agent Workflow Rules

### Change Verification Loop

Every code change must follow this cycle:

```
1. Read the file(s) you plan to modify
2. Make the smallest change that addresses the task
3. Run pytest — at minimum the test file for the module you changed
4. If tests fail, fix your change (not the tests) unless the tests are genuinely wrong
5. Run full pytest to check for regressions
```

### Do Not

- Edit multiple source files at once without testing between edits
- "Improve" code you weren't asked to touch
- Add logging, comments, or docstrings unless specifically requested
- Create new abstractions (base classes, mixins, utility modules) without being asked
- Remove or weaken existing error handling, assertions, or numerical guards
- Replace working code with "cleaner" alternatives that change behavior

---

## Framework Rules: This Is TensorFlow

This is the single most common LLM mistake on this codebase. Every line below matters.

- **All layers subclass `tf.keras.layers.Layer`**. Models subclass `tf.keras.Model`. There is no `nn.Module`, no `.forward()`, no `torch.Tensor`.
- **Trainable state goes in `build()`.** Use `self.add_weight()` for learned parameters. Non-trainable configuration state (like `scale_table` in `PatchedGaussianConditional`) may be set in `__init__()` as a `tf.Variable(..., trainable=False)` — this is an established pattern in the codebase.
- **Do not create `tf.Variable` or `tf.constant` inside `call()`.** This leaks memory under `@tf.function` tracing. All persistent state must be initialized in `__init__()` or `build()`.
- **Use `get_config()`** for serialization on Keras layers that participate in model graphs. Standalone utility classes (like `OctreeCoder`) do not need it.
- **`@tf.function` compatibility.** Code in `call()` must work in graph mode. No Python-level `if tensor_value > 0:` — use `tf.cond`. No creating new ops that depend on Python-side tensor evaluation.
- **Never import torch, torch.nn, or any PyTorch module.** Do not suggest PyTorch alternatives.

### The `training` flag

The main model `call()` methods (`DeepCompressModel.call()`, `DeepCompressModelV2.call()`) accept and use `training=None`. This is required because quantization behavior switches between noise injection (training) and hard rounding (inference):

```python
# This pattern appears in DeepCompressModel.call(), ConditionalGaussian._add_noise(), etc.
if training:
    y = y + tf.random.uniform(tf.shape(y), -0.5, 0.5)  # Gradient-friendly noise
else:
    y = tf.round(y)  # Hard quantization for inference
```

**Rules:**
- Model-level `call()` methods and any layer that branches on training mode **must** accept `training=None` and pass it through to sub-layers that need it.
- Leaf layers that do not use `training` internally (e.g., `CENICGDN`, `SpatialSeparableConv`, `MaskedConv3D`, `SliceTransform`) currently omit it from their signatures. This is the established convention — do not add `training` to these unless they gain training-dependent behavior.
- **Never remove the training conditional** from methods that have it. Never replace noise injection with unconditional `tf.round()`.
- When adding new layers: include `training=None` if the layer has any training-dependent behavior. Omit it for pure computation layers.

## Tensor Shape Convention

All model tensors are 5D: `(batch, depth, height, width, channels)` — channels-last.

- Convolutions are `Conv3D`, never `Conv2D`. Kernels are 3-tuples: `(3, 3, 3)`.
- Channel axis is axis 4 (see `CENICGDN.call()` which does `tf.tensordot(norm, self.gamma, [[4], [0]])`).
- Input voxel grids have 1 channel: shape `(B, D, H, W, 1)`.
- Do not flatten spatial dimensions to use 2D ops. The 3D structure is load-bearing.

## Constants and Numerical Stability

**Use pre-computed constants from `src/constants.py`:**
```python
from constants import LOG_2_RECIPROCAL
bits = -log_likelihood * LOG_2_RECIPROCAL  # CORRECT
bits = -log_likelihood / tf.math.log(2.0)  # WRONG: creates new op node every call
```

**Do not remove numerical guards.** The codebase uses:
- `tf.maximum(scale, self.scale_min)` — prevents division by zero in Gaussian likelihood
- `EPSILON = 1e-9` — prevents log(0)
- `tf.clip_by_value` on scales — prevents NaN in entropy computation
- `tf.abs(scale)` before quantization — ensures positive scale

These look like they could be "simplified away." They cannot. Removing them causes NaN gradients during training.

## Entropy Model Contracts

The six entropy models (`gaussian`, `hyperprior`, `channel`, `context`, `attention`, `hybrid`) are not interchangeable at runtime. Each is selected as a string at model construction and creates different submodules. Key contracts:

- `DeepCompressModel.call()` returns 4 values: `(x_hat, y, y_hat, z)`
- `DeepCompressModelV2.call()` returns 5 values: `(x_hat, y, y_hat, z, rate_info)`
- `rate_info` is a dict with keys: `likelihood`, `total_bits`, `bpp`
- Don't mix V1 and V2 return signatures
- Entropy model submodules use deferred imports inside `_create_entropy_model()` to avoid circular dependencies. Keep this pattern.

## Binary Search Scale Quantization

`PatchedGaussianConditional.quantize_scale()` uses `tf.searchsorted` on pre-computed midpoints. This is an intentional optimization (O(log T) vs O(T)).

- Do not replace with linear scan or full-table broadcasting
- The scale table and midpoints are created once in `__init__`/`build()`, not per call
- `_precompute_midpoints()` must be called after any scale table change

## Masked Convolutions Are Causal

`context_model.py` has `MaskedConv3D` with type A (excludes center) and type B (includes center) masks. These enforce autoregressive ordering in raster-scan order over (D, H, W).

- Masks are created via vectorized NumPy operations in `build()`, stored as weights
- Do not convert to Python loops — the vectorized version is 10-100x faster
- Do not modify the raster-scan ordering — it will silently produce incorrect likelihoods

---

## Testing Standards

### Use the Existing Test Infrastructure

The test suite uses `tf.test.TestCase` mixed with `pytest` fixtures. This is intentional and established — do not convert to pure pytest.

**Existing utilities in `tests/test_utils.py` — use these instead of creating your own:**
- `create_mock_voxel_grid(resolution, batch_size)` — binary occupancy grid
- `create_mock_point_cloud(num_points)` — random 3D points
- `create_test_dataset(batch_size, resolution, num_batches)` — tf.data.Dataset
- `create_test_config(tmp_path)` — complete YAML config dict
- `setup_test_environment(tmp_path)` — full test env with files, configs, dirs
- `MockCallback` — for testing training loops

**Existing fixtures in `tests/conftest.py`:**
- `sample_point_cloud` (session-scoped) — 8-point cube
- `create_ply_file`, `create_off_file` — file factories
- `tf_config` (session-scoped) — GPU memory growth + seed setting

### How to Write Tests

**Test structure:** Tests are organized by module. `test_foo.py` tests `src/foo.py`. Follow the existing pattern:

```python
import tensorflow as tf
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from test_utils import create_mock_voxel_grid
from module_under_test import ClassUnderTest

class TestClassUnderTest(tf.test.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        # Use small tensors: resolution 16, batch_size 1, filters 32
        self.config = TransformConfig(filters=32, ...)
        self.resolution = 16
        self.batch_size = 1

    def test_specific_behavior(self):
        # Arrange
        input_tensor = create_mock_voxel_grid(self.resolution, self.batch_size)
        model = ClassUnderTest(self.config)

        # Act
        output = model(input_tensor, training=False)

        # Assert — use tf.test.TestCase assertions
        self.assertEqual(output.shape[:-1], input_tensor.shape[:-1])
        self.assertGreater(rate_info['total_bits'], 0)
```

**Test sizing:** Use small tensors. Resolution 16, batch size 1, filters 32 is the standard for unit tests. Tests using resolution 64 or larger should be marked `@pytest.mark.slow`.

**Random seeds:** Set `tf.random.set_seed(42)` at the start of any test that creates random tensors and makes value-dependent assertions. The session-level seed in `conftest.py` provides baseline reproducibility, but per-test seeds are more robust for value-dependent checks.

**Pytest markers** (defined in `pytest.ini`):
- `@pytest.mark.slow` — performance/timing tests
- `@pytest.mark.gpu` — requires GPU
- `@pytest.mark.e2e` — end-to-end pipeline tests
- `@pytest.mark.integration` — integration tests

### What Makes a Good Assertion

**Good assertions test meaningful properties:**
```python
# Shape is correct and specific
self.assertEqual(output.shape, (1, 16, 16, 16, 128))

# Values are in expected range
self.assertGreater(rate_info['total_bits'], 0)

# Roundtrip consistency
self.assertEqual(decompressed.shape, x_hat.shape)

# Gradients actually flow and are non-zero
non_none_grads = [g for g in gradients if g is not None]
self.assertNotEmpty(non_none_grads)
self.assertGreater(tf.reduce_sum(tf.abs(gradients[0])), 0)

# Structural correctness (mask causality)
assert np.all(mask[2, 1, 1, :, :] == 0), "Future positions should be masked"

# Numerical correctness with appropriate tolerances
np.testing.assert_allclose(bits_original.numpy(), bits_optimized.numpy(), rtol=1e-5)
self.assertAllClose(x_hat1, x_hat2)

# Output values are in valid range for binary occupancy
self.assertAllGreaterEqual(output, 0.0)
self.assertAllLessEqual(output, 1.0)
```

**Bad assertions — do not write these:**
```python
self.assertIsNotNone(output)           # Almost always true, tests nothing useful
self.assertTrue(isinstance(x, tf.Tensor))  # If it wasn't a tensor, the test already crashed
assert output.shape is not None        # Tautological
self.assertEqual(len(output), len(output))  # Obviously true
assert model is not None               # Construction already succeeded
```

**Note:** The existing test suite has some `assertIsNotNone` calls (e.g., in `test_model_transforms.py`). Do not add more of these. When modifying existing tests, replace them with meaningful assertions (shape checks, value range checks, dtype checks).

### What Makes a Bad Test

**Do not write tests that:**

1. **Re-implement the function and check equality.** If your test duplicates the source logic to produce an expected value, it will always pass — even if both are wrong.

2. **Mock the thing being tested.** Mock external dependencies, not the code under test. If you're testing `AnalysisTransform`, don't mock `Conv3D` — test with real convolutions on small tensors.

3. **Only test the happy path.** Also test: empty batch (batch_size=0 if applicable), single-element batch, mismatched shapes, invalid entropy model string, out-of-range scale values.

4. **Assert on random values without seeding.** If your test creates random tensors and asserts on specific values, set `tf.random.set_seed(42)` first.

5. **Have side effects between test methods.** Each test method must be independent. Don't rely on execution order. Use fixtures, not class-level mutation.

### Performance Tests

Performance tests in `test_performance.py` assert timing bounds. When writing new ones:

```python
@pytest.mark.slow
def test_optimized_is_faster(self):
    # Time both approaches with enough iterations
    start = time.perf_counter()
    for _ in range(iterations):
        _ = slow_approach()
    slow_time = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(iterations):
        _ = fast_approach()
    fast_time = time.perf_counter() - start

    speedup = slow_time / fast_time
    # Use conservative thresholds — CI machines are slower than dev machines
    assert speedup > 1.2, f"Expected >1.2x speedup, got {speedup:.1f}x"
```

Set conservative speedup thresholds (1.2x, not 10x) — CI environments vary. Always run multiple iterations to amortize startup.

---

## Code Quality Standards

### Two Contexts: Library Code vs CLI Tools

The codebase has two distinct contexts with different conventions:

**Library code** (`model_transforms.py`, `entropy_model.py`, `entropy_parameters.py`, `context_model.py`, `channel_context.py`, `attention_context.py`, `constants.py`, `precision_config.py`):
- No `print()` — these are imported modules, not user-facing scripts
- No `logging` — not used in library code
- Strict numerical discipline and `@tf.function` compatibility

**CLI tools and pipeline scripts** (`benchmarks.py`, `quick_benchmark.py`, `cli_train.py`, `ds_*.py`, `ev_*.py`, `mp_*.py`, `training_pipeline.py`, `evaluation_pipeline.py`, `compress_octree.py`, `parallel_process.py`, `experiment.py`):
- `print()` is fine — these are user-facing programs that output to console
- `logging` is used and appropriate — several pipeline files use the `logging` module
- Standard Python error handling conventions apply

When adding new code, follow the convention of the file you're editing.

### Error Handling

- **Do not add bare `except:` or `except Exception: pass` blocks** that silently swallow errors. TensorFlow errors (shape mismatches, OOM, NaN gradients) must propagate. There are a few existing `except Exception` blocks in `benchmarks.py` and `evaluation_pipeline.py` — do not add more.
- **Do not add try/except around tensor operations** in library code unless handling a specific, documented failure mode.
- Use `tf.debugging.assert_*` for development-time checks that can be disabled in production.

### Linting & Imports

- **Linter:** `ruff` (configured in `pyproject.toml`). No flake8, no pylint.
- Selected rule sets: **F** (Pyflakes), **I** (isort), **E/W** (pycodestyle). Do not add other rule sets without discussion.
- `sys.path.insert(0, ...)` is used in test files to find `src/`. Follow this pattern, don't fight it.
- Entropy model submodules use deferred imports inside `_create_entropy_model()` to avoid circular dependencies. Keep it this way.
- Import from `constants.py`, not inline math: `from constants import LOG_2_RECIPROCAL`.
- When adding new source modules, add the module name to `known-first-party` in `pyproject.toml` so isort classifies it correctly.

### Typing

- The codebase uses `typing` annotations (`Optional`, `Dict`, `Tuple`, etc.) in function signatures.
- Follow existing style. Don't add type annotations to existing functions that don't have them unless asked.

### Things Not to Add Unless Asked

- New dependencies in `requirements.txt`
- New base classes, ABCs, or mixin layers
- README updates to match code changes (the README describes some aspirational features)

---

## Known Issues

These are pre-existing issues in the codebase. Do not fix them unless specifically asked — they are documented here so you don't waste time rediscovering them:

- `tests/test_point_cloud_metricss.py` — filename has a double 's' typo
- `tests/test_integration.py::TestIntegration` — entire class is `@pytest.mark.skip` due to API mismatch with `TrainingPipeline()`
- `pytest.ini` is missing the `slow` marker definition (tests use `@pytest.mark.slow` but it's not registered, causing a warning)
- Some test methods in `test_model_transforms.py` use `assertIsNotNone` where a shape or value assertion would be more meaningful
- Some test files (`test_data_loader`, `test_training_pipeline`, etc.) fail collection on machines with network timeouts — pre-existing, not caused by lint migration

---

## Quick Reference

```bash
# Lint (must pass before commit)
ruff check src/ tests/

# Auto-fix import order and whitespace
ruff check --fix src/ tests/

# Run all tests
pytest tests/ -v

# Run one test file
pytest tests/test_entropy_model.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Run only GPU tests
pytest tests/ -v -m gpu

# Quick smoke test (no data needed)
python -m src.quick_benchmark --compare
```

### Repository Layout

```
src/
  # Library code (strict quality rules)
  model_transforms.py      # AnalysisTransform, SynthesisTransform, DeepCompressModel(V2)
  entropy_model.py         # PatchedGaussianConditional, EntropyModel, MeanScaleHyperprior
  entropy_parameters.py    # Hyperprior μ/σ prediction network
  context_model.py         # MaskedConv3D, AutoregressiveContext
  channel_context.py       # ChannelContext, ChannelContextEntropyModel
  attention_context.py     # WindowedAttention3D, AttentionEntropyModel
  constants.py             # Pre-computed LOG_2, LOG_2_RECIPROCAL, EPSILON, etc.
  precision_config.py      # PrecisionManager for mixed float16

  # CLI tools and pipeline scripts (standard Python conventions)
  training_pipeline.py     # TrainingPipeline class (uses logging)
  evaluation_pipeline.py   # EvaluationPipeline class (uses logging)
  quick_benchmark.py       # Synthetic smoke test (uses print)
  benchmarks.py            # Performance benchmarks (uses print)
  cli_train.py             # Training CLI entry point
  ds_*.py                  # Data pipeline tools (mesh→pointcloud→octree→blocks)
  ev_*.py, mp_*.py         # Evaluation and MPEG comparison scripts
  compress_octree.py       # Compression entry point (uses logging)
  octree_coding.py         # Octree codec utility class

tests/
  conftest.py              # Session-scoped fixtures (tf_config, sample_point_cloud, file factories)
  test_utils.py            # Shared test utilities — USE THESE
  test_*.py                # One test file per source module
```
