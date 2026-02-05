# DeepCompress: Efficient Point Cloud Geometry Compression

![DeepCompress comparison samples](image.png)

## Authors

- [Ryan Killea](https://rbkillea.com/)
- [Saeed Bastani](https://scholar.google.com/citations?user=m0tkD4YAAAAJ&hl=en)
- Yun Li
- [Paul McLachlan](http://pmclachlan.com)

**Affiliation**: Ericsson Research
**Paper**: [Research Paper (arXiv)](https://arxiv.org/abs/2106.01504)

## Abstract

Point clouds are a basic data type of growing interest due to their use in applications such as virtual, augmented, and mixed reality, and autonomous driving. This work presents DeepCompress, a deep learning-based encoder for point cloud compression that achieves efficiency gains without significantly impacting compression quality. Through optimization of convolutional blocks and activation functions, our architecture reduces the computational cost by 8% and model parameters by 20%, with only minimal increases in bit rate and distortion.

## What's New in V2

DeepCompress V2 introduces **advanced entropy modeling** and **performance optimizations** that significantly improve compression efficiency and speed.

### Advanced Entropy Models

V2 supports multiple entropy model configurations for the rate-distortion trade-off:

| Entropy Model | Description | Use Case |
|---------------|-------------|----------|
| `gaussian` | Fixed Gaussian (original) | Backward compatibility |
| `hyperprior` | Mean-scale hyperprior | Best speed/quality balance |
| `channel` | Channel-wise autoregressive | Better compression, parallel-friendly |
| `context` | Spatial autoregressive | Best compression, slower |
| `attention` | Attention-based context | Large receptive field |
| `hybrid` | Attention + channel combined | Maximum compression |

**Typical improvements over baseline:**
- **Hyperprior**: 15-25% bitrate reduction
- **Channel context**: 25-35% bitrate reduction
- **Full context model**: 30-40% bitrate reduction

### Performance Optimizations

V2 includes optimizations targeting **2-5x speedup** and **50-80% memory reduction**:

| Optimization | Speedup | Memory Reduction | Description |
|-------------|---------|------------------|-------------|
| Binary search scale quantization | 5x | 64x | O(n·log T) vs O(n·T) lookup |
| Vectorized mask creation | 10-100x | - | NumPy broadcasting vs loops |
| Windowed attention | 10-50x | 400x | O(n·w³) vs O(n²) attention |
| Pre-computed constants | ~5% | - | Cached log(2) calculations |
| Channel context caching | 1.2x | 25% | Avoid redundant allocations |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/pmclsf/deepcompress.git
cd deepcompress

# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Benchmark (No Dataset Required)

Test compression performance with synthetic data:

```bash
# Basic benchmark
python -m src.quick_benchmark

# Compare model configurations
python -m src.quick_benchmark --compare

# Custom configuration
python -m src.quick_benchmark --resolution 64 --model v2 --entropy hyperprior
```

**Example output:**
```
======================================================================
Summary Comparison
======================================================================
Model                PSNR (dB)    BPV        Time (ms)    Ratio
----------------------------------------------------------------------
v1                   7.20         0.000      92.8         N/A
v2-hyperprior        7.20         0.205      74.6         156.3x
v2-channel           7.20         0.349      138.4        91.8x
======================================================================
```

*Note: Low PSNR is expected for untrained models. Train on real data for actual compression performance.*

### Using V2 Models

```python
from model_transforms import DeepCompressModelV2, TransformConfig

# Configure model
config = TransformConfig(
    filters=64,
    kernel_size=(3, 3, 3),
    strides=(2, 2, 2),
    activation='cenic_gdn',
    conv_type='separable'
)

# Create V2 model with hyperprior entropy model
model = DeepCompressModelV2(
    config,
    entropy_model='hyperprior'  # or 'channel', 'context', 'attention', 'hybrid'
)

# Forward pass
x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

# Access compression metrics
total_bits = rate_info['total_bits']
y_likelihood = rate_info['y_likelihood']
```

### Mixed Precision Training

Enable mixed precision for faster training on modern GPUs:

```python
from precision_config import PrecisionManager

# Enable mixed precision (float16 compute, float32 master weights)
PrecisionManager.configure('mixed_float16')

# Wrap optimizer for loss scaling
optimizer = tf.keras.optimizers.Adam(1e-4)
optimizer = PrecisionManager.wrap_optimizer(optimizer)

# Train as usual
model.compile(optimizer=optimizer, ...)
```

## Reproducing Paper Results

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/pmclsf/deepcompress.git
cd deepcompress

# Create and activate virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/modelnet40
mkdir -p data/8ivfb
mkdir -p results/models
mkdir -p results/metrics
```

### 2. Dataset Preparation
```bash
# Download and prepare ModelNet40
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip -d data/modelnet40/

# Download 8iVFB dataset
# Note: Requires registration at http://plenodb.jpeg.org
mv 8iVFB_v2.zip data/8ivfb/
unzip data/8ivfb/8iVFB_v2.zip -d data/8ivfb/

# Process ModelNet40 for training
python ds_select_largest.py \
    data/modelnet40/ModelNet40 \
    data/modelnet40/ModelNet40_200 \
    200

python ds_mesh_to_pc.py \
    data/modelnet40/ModelNet40_200 \
    data/modelnet40/ModelNet40_200_pc512 \
    --vg_size 512

python ds_pc_octree_blocks.py \
    data/modelnet40/ModelNet40_200_pc512 \
    data/modelnet40/ModelNet40_200_pc512_oct3 \
    --vg_size 512 \
    --level 3

python ds_select_largest.py \
    data/modelnet40/ModelNet40_200_pc512_oct3 \
    data/modelnet40/ModelNet40_200_pc512_oct3_4k \
    4000
```

### 3. Training Pipeline
```bash
# Create training configuration
cat > config/train_config.yml << EOL
data:
  modelnet40_path: "data/modelnet40/ModelNet40_200_pc512_oct3_4k"
  ivfb_path: "data/8ivfb"
  resolution: 64
  block_size: 1.0
  min_points: 100
  augment: true

model:
  filters: 64
  activation: "cenic_gdn"
  conv_type: "separable"
  entropy_model: "hyperprior"  # NEW: V2 entropy model

training:
  batch_size: 32
  epochs: 100
  learning_rates:
    reconstruction: 1.0e-4
    entropy: 1.0e-3
  focal_loss:
    alpha: 0.75
    gamma: 2.0
  checkpoint_dir: "results/models"
  mixed_precision: false  # NEW: Enable for faster training on GPU
EOL

# Train model
python training_pipeline.py config/train_config.yml
```

### 4. Evaluation Pipeline
```bash
# Run evaluation on 8iVFB dataset
python evaluation_pipeline.py \
    config/train_config.yml \
    --checkpoint results/models/best_model

# Generate comparison metrics
python ev_compare.py \
    --original data/8ivfb \
    --compressed results/compressed \
    --output results/metrics

# Generate visualizations
python ev_run_render.py config/train_config.yml
```

### 5. Compare with G-PCC
```bash
# Run G-PCC experiments
python mp_run.py config/train_config.yml --num_parallel 8

# Generate final report
python mp_report.py \
    results/metrics/evaluation_report.json \
    results/metrics/final_report.json
```

### Expected Results

After running the complete pipeline, you should observe:
- 8% reduction in total operations
- 20% reduction in model parameters
- D1 metric: 0.02% penalty
- D2 metric: 0.32% increased bit rate

**With V2 entropy models:**
- Additional 15-40% bitrate reduction (depending on entropy model)
- 2-5x faster inference with optimizations enabled

The results can be found in:
- Model checkpoints: `results/models/`
- Evaluation metrics: `results/metrics/final_report.json`
- Visualizations: `results/visualizations/`

## Model Architecture

### Network Overview
- Analysis-synthesis architecture with scale hyperprior
- Incorporates GDN/CENIC-GDN activation functions
- Novel 1+2D spatially separable convolutional blocks
- Progressive channel expansion with dimension reduction

### V2 Architecture Enhancements

```
Input Voxel Grid
       │
       ▼
┌─────────────────┐
│ Analysis        │ ──► Latent y
│ Transform       │
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Hyper-Analysis  │ ──► Hyper-latent z
└─────────────────┘
       │
       ▼
┌─────────────────┐
│ Entropy Model   │ ◄── Configurable:
│ (V2 Enhanced)   │     • Hyperprior
└─────────────────┘     • Channel Context
       │                • Spatial Context
       ▼                • Attention
┌─────────────────┐     • Hybrid
│ Arithmetic      │
│ Coding          │
└─────────────────┘
       │
       ▼
   Bitstream
```

### Key Components
- **Analysis Network**: Processes input point clouds through multiple analysis blocks
- **Synthesis Network**: Reconstructs point clouds from compressed representations
- **Hyperprior**: Learns and encodes additional parameters for entropy modeling
- **Custom Activation**: Uses CENIC-GDN for improved efficiency
- **Advanced Entropy Models** (V2): Context-adaptive probability estimation

### Entropy Model Details

#### Mean-Scale Hyperprior
Predicts per-element mean and scale from the hyper-latent:
```python
# Hyperprior predicts distribution parameters
mean, scale = entropy_parameters(z_hat)
# Gaussian likelihood with learned parameters
likelihood = gaussian_pdf(y, mean, scale)
```

#### Channel-wise Context
Processes channels in groups, using previous groups as context:
```python
# Parallel-friendly: all spatial positions decoded simultaneously
for group in channel_groups:
    context = previously_decoded_groups
    mean, scale = channel_context(context, group_idx)
    decode(group, mean, scale)
```

#### Windowed Attention
Memory-efficient attention using local windows with global tokens:
```python
# O(n·w³) instead of O(n²) - 400x memory reduction for 32³ grids
windows = partition_into_windows(features, window_size=4)
local_attention = attend_within_windows(windows)
global_context = attend_to_global_tokens(windows, num_global=8)
```

### Spatially Separable Design
The architecture employs 1+2D convolutions instead of full 3D convolutions, providing:
- More parameter efficiency for same input/output channels
- Reduced operation count
- Better filter utilization
- Encoded knowledge of point cloud surface properties

## Performance Benchmarking

### Running Benchmarks

```bash
# Run all benchmarks
python -m src.benchmarks

# Individual benchmark components
python -c "from src.benchmarks import benchmark_scale_quantization; benchmark_scale_quantization()"
python -c "from src.benchmarks import benchmark_masked_conv; benchmark_masked_conv()"
python -c "from src.benchmarks import benchmark_attention; benchmark_attention()"
```

### Benchmark Results

Measured on CPU (results vary by hardware):

| Component | Original | Optimized | Speedup |
|-----------|----------|-----------|---------|
| Scale quantization | 45ms | 9ms | 5x |
| Mask creation | 120ms | 1.2ms | 100x |
| Attention (32³) | OOM | 85ms | ∞ |

### Memory Profiling

```python
from src.benchmarks import MemoryProfiler

with MemoryProfiler() as mem:
    output = model(large_input)
print(f"Peak memory: {mem.peak_mb:.1f} MB")
```

## Prerequisites

### Required Software

- Python 3.8+
- MPEG G-PCC codec [mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)
- MPEG metric software v0.12.3 [mpeg-pcc-dmetric](http://mpegx.int-evry.fr/software/MPEG/PCC/mpeg-pcc-dmetric)
- MPEG PCC dataset

### Dependencies

Required packages:
- tensorflow >= 2.11.0
- tensorflow-probability >= 0.19.0
- matplotlib ~= 3.1.3
- numpy ~= 1.23.0
- pandas ~= 1.4.0
- pyyaml ~= 5.1.2
- scipy ~= 1.8.1
- numba ~= 0.55.0

## Implementation Details

### Point Cloud Metrics

```python
from pc_metric import calculate_metrics

metrics = calculate_metrics(predicted_points, ground_truth_points)
print(f"D1: {metrics['d1']}")
print(f"D2: {metrics['d2']}")
print(f"Chamfer: {metrics['chamfer']}")
```

Supported metrics include:
- **D1**: Point-to-point distances from predicted to ground truth
- **D2**: Point-to-point distances from ground truth to predicted
- **Chamfer Distance**: Combined D1 + D2 metric
- **Normal-based metrics** (when normals are available):
  - N1: Point-to-normal distances from predicted to ground truth
  - N2: Point-to-normal distances from ground truth to predicted

### Data Processing Pipeline

```python
# Analysis Transform for encoding
transform = AnalysisTransform(
    filters=64,
    kernel_size=(3, 3, 3),
    strides=(2, 2, 2)
)

# Synthesis Transform for decoding
synthesis = SynthesisTransform(
    filters=32,
    kernel_size=(3, 3, 3),
    strides=(2, 2, 2)
)
```

Key components:
- Residual connections
- Custom activation functions
- Normalization layers
- Efficient 3D convolutions

## Project Structure

### Source Code (`/src`)

- **Core Processing**
  - `compress_octree.py`: Point cloud octree compression
  - `decompress_octree.py`: Point cloud decompression
  - `ds_mesh_to_pc.py`: Mesh to point cloud conversion
  - `ds_pc_octree_blocks.py`: Octree block partitioning

- **Model Components**
  - `entropy_model.py`: Entropy modeling and compression
  - `entropy_parameters.py`: Hyperprior parameter prediction
  - `context_model.py`: Spatial autoregressive context
  - `channel_context.py`: Channel-wise context model
  - `attention_context.py`: Attention-based context with windowed attention
  - `model_transforms.py`: Analysis/synthesis transforms

- **Performance & Utilities**
  - `constants.py`: Pre-computed mathematical constants
  - `precision_config.py`: Mixed precision configuration
  - `benchmarks.py`: Performance benchmarking utilities
  - `quick_benchmark.py`: Quick compression testing

- **Training & Evaluation**
  - `cli_train.py`: Command-line training interface
  - `training_pipeline.py`: Training pipeline
  - `evaluation_pipeline.py`: Evaluation pipeline
  - `experiment.py`: Core experiment utilities

- **Support Utilities**
  - `colorbar.py`: Visualization colorbars
  - `map_color.py`: Color mapping
  - `octree_coding.py`: Octree encoding
  - `parallel_process.py`: Parallel processing

### Test Structure (`/tests`)

- **Core Tests**
  - `test_entropy_model.py`: Entropy model tests
  - `test_entropy_parameters.py`: Parameter prediction tests
  - `test_context_model.py`: Context model tests
  - `test_channel_context.py`: Channel context tests
  - `test_attention_context.py`: Attention model tests
  - `test_model_transforms.py`: Model transformation tests
  - `test_performance.py`: Performance regression tests

- **Pipeline Tests**
  - `test_training_pipeline.py`: Training pipeline tests
  - `test_evaluation_pipeline.py`: Evaluation pipeline tests
  - `test_experiment.py`: Experiment utility tests
  - `test_integration.py`: End-to-end integration tests

- **Data Processing Tests**
  - `test_ds_mesh_to_pc.py`: Mesh conversion tests
  - `test_ds_pc_octree_blocks.py`: Octree block tests

## Citation

If you use this codebase in your research, please cite our paper:

```bibtex
@article{killea2021deepcompress,
  title={DeepCompress: Efficient Point Cloud Geometry Compression},
  author={Killea, Ryan and Li, Yun and Bastani, Saeed and McLachlan, Paul},
  journal={arXiv preprint arXiv:2106.01504},
  year={2021}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.
