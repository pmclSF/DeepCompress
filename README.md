# DeepCompress: Efficient Point Cloud Geometry Compression

![DeepCompress comparison samples](image.png)

## Authors

- [Ryan Killea](https://rbkillea.com/)
- [Saeed Bastani](https://scholar.google.com/citations?user=m0tkD4YAAAAJ&hl=en)
- Yun Li
- [Paul McLachlan](http://pmclachlan.com)

**Affiliation**: Ericsson Research
**Paper**: [Research Paper (arXiv)](https://arxiv.org/abs/2106.01504)

---

## What is Point Cloud Compression?

### The Problem

A **point cloud** is a collection of 3D points that represent the shape of an object or environment. Think of it like a 3D scan of the world—each point has an X, Y, and Z coordinate, and together they form a detailed 3D model.

Point clouds are used in:
- **Self-driving cars**: LIDAR sensors generate millions of 3D points to understand the environment
- **Virtual/Augmented Reality**: Creating realistic 3D environments
- **3D mapping**: Surveying buildings, cities, and landscapes
- **Medical imaging**: 3D body scans and organ models

**The challenge**: Point clouds are *huge*. A single LIDAR scan can contain millions of points, and streaming or storing this data requires enormous bandwidth and storage. For example:
- A 10-second LIDAR capture might be 500MB uncompressed
- Streaming this in real-time would require 400 Mbps bandwidth

### The Solution

DeepCompress uses **deep learning** to compress point clouds efficiently—similar to how JPEG compresses images or MP3 compresses audio. The key insight is that point clouds have patterns and structure that a neural network can learn to represent more efficiently.

**How it works (simplified)**:
1. **Encode**: The neural network analyzes the point cloud and creates a compact "summary" (called a latent representation)
2. **Compress**: This summary is converted to a small file using entropy coding (like ZIP, but smarter)
3. **Decompress**: The summary is expanded back
4. **Decode**: The neural network reconstructs the original point cloud

The result: **10-100x smaller files** with minimal quality loss.

---

## What's New in V2

DeepCompress V2 introduces two major improvements:

### 1. Smarter Compression (Advanced Entropy Models)

**What is entropy modeling?**

When compressing data, we need to predict "how surprising" each value is. Common values can be stored with fewer bits; rare values need more bits. This is called **entropy coding**.

*Analogy*: In English text, the letter 'E' is very common, so we could represent it with a short code (like '1'). The letter 'Z' is rare, so it gets a longer code (like '10110'). This is how Morse code works, and it's the foundation of all compression.

V2 offers multiple ways to predict these probabilities:

| Entropy Model | How It Works | Best For |
|---------------|--------------|----------|
| `gaussian` | Assumes all values follow a simple bell curve | Fast, basic compression |
| `hyperprior` | Learns a custom probability for each location | Good balance of speed and compression |
| `channel` | Uses already-decoded parts to predict the rest | Better compression, still fast |
| `context` | Looks at neighboring values for prediction | Best compression, slower |
| `attention` | Considers long-range patterns across the entire cloud | Complex shapes with repeating patterns |
| `hybrid` | Combines multiple approaches | Maximum compression quality |

**Expected improvements over baseline**:
- **Hyperprior**: 15-25% smaller files
- **Channel context**: 25-35% smaller files
- **Full context**: 30-40% smaller files

### 2. Faster Processing (Performance Optimizations)

V2 includes engineering optimizations that make the code run faster and use less memory:

| What We Optimized | What It Does | Improvement |
|-------------------|--------------|-------------|
| **Binary search for scale lookup** | Finding the right compression parameter is now O(log n) instead of O(n) | 5x faster, 64x less memory |
| **Vectorized mask creation** | Creating neural network masks uses efficient array operations | 10-100x faster |
| **Windowed attention** | Instead of comparing every point to every other point, we only compare nearby points | 10-50x faster, 400x less memory |
| **Pre-computed constants** | Mathematical constants like log(2) are calculated once, not every time | ~5% faster |
| **Smarter memory allocation** | Avoid creating unnecessary temporary data | 25% less memory |

**Why does this matter?**
- Real-time compression becomes possible
- Can run on less powerful hardware
- Larger point clouds can be processed without running out of memory

---

## Quick Start

### Step 1: Installation

First, set up your Python environment:

```bash
# Download the code
git clone https://github.com/pmclsf/deepcompress.git
cd deepcompress

# Create an isolated Python environment (keeps dependencies separate)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**What this does**: Downloads DeepCompress and installs the necessary Python libraries (TensorFlow for neural networks, NumPy for math, etc.).

### Step 2: Quick Test (No Dataset Needed)

Want to see it work without downloading any data? Run our synthetic benchmark:

```bash
python -m src.quick_benchmark --compare
```

**What this does**: Creates artificial 3D shapes (spheres, random points) and tests how well different model configurations compress them. You'll see output like:

```
======================================================================
Summary Comparison
======================================================================
Model                PSNR (dB)    BPV        Time (ms)    Ratio
----------------------------------------------------------------------
v1                   7.20         N/A        92.8         N/A
v2-hyperprior        7.20         0.205      74.6         156.3x
v2-channel           7.20         0.349      138.4        91.8x
======================================================================
```

**Reading the results**:
- **PSNR (dB)**: Quality metric—higher is better. Low values here are expected because the model isn't trained yet.
- **BPV (Bits Per Voxel)**: How many bits needed per 3D point—lower is better compression.
- **Time (ms)**: Processing speed in milliseconds—lower is faster.
- **Ratio**: Compression ratio—higher means smaller files.

---

## Using V2 Models in Your Code

### Basic Example

```python
from model_transforms import DeepCompressModelV2, TransformConfig

# Step 1: Configure the model architecture
config = TransformConfig(
    filters=64,              # Number of neural network channels (more = better quality, slower)
    kernel_size=(3, 3, 3),   # Size of 3D convolution filters
    strides=(2, 2, 2),       # How much to downsample at each layer
    activation='cenic_gdn',  # Special activation function for compression
    conv_type='separable'    # Efficient convolution type
)

# Step 2: Create the model with your chosen entropy model
model = DeepCompressModelV2(
    config,
    entropy_model='hyperprior'  # Options: 'gaussian', 'hyperprior', 'channel', 'context', 'attention', 'hybrid'
)

# Step 3: Compress a point cloud
# input_tensor should be a 5D tensor: (batch, depth, height, width, channels)
x_hat, y, y_hat, z, rate_info = model(input_tensor, training=False)

# x_hat: The reconstructed point cloud
# rate_info['total_bits']: How many bits the compressed version would take
```

### Enabling Faster Training with Mixed Precision

Modern GPUs can compute faster using 16-bit numbers instead of 32-bit. This is called **mixed precision**:

```python
from precision_config import PrecisionManager

# Enable mixed precision (uses float16 for speed, float32 for accuracy where needed)
PrecisionManager.configure('mixed_float16')

# Wrap your optimizer to handle the precision scaling
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
optimizer = PrecisionManager.wrap_optimizer(optimizer)

# Now train as usual—it will automatically be faster on compatible GPUs
model.compile(optimizer=optimizer, loss=your_loss_function)
model.fit(training_data, epochs=100)
```

**When to use this**: If you have an NVIDIA GPU with Tensor Cores (RTX series, V100, A100, etc.), mixed precision can give you 1.5-2x speedup with minimal quality impact.

---

## Full Training Pipeline

If you want to train your own model from scratch on real data, follow these steps:

### Step 1: Environment Setup

```bash
# Clone and enter the repository
git clone https://github.com/pmclsf/deepcompress.git
cd deepcompress

# Create isolated Python environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create folders for data and results
mkdir -p data/modelnet40    # Training data will go here
mkdir -p data/8ivfb         # Evaluation data will go here
mkdir -p results/models     # Trained models saved here
mkdir -p results/metrics    # Evaluation results saved here
```

### Step 2: Dataset Preparation

We use two datasets:
- **ModelNet40**: 3D CAD models for training (chairs, tables, airplanes, etc.)
- **8iVFB**: High-quality point cloud sequences for evaluation

```bash
# Download ModelNet40 (3D object dataset from Princeton)
wget http://modelnet.cs.princeton.edu/ModelNet40.zip
unzip ModelNet40.zip -d data/modelnet40/
```

**What is ModelNet40?** A collection of 12,311 3D CAD models across 40 categories (airplane, bathtub, bed, bench, etc.). We use these to teach the neural network what 3D shapes look like.

Now we need to convert these 3D models into the format DeepCompress uses:

```bash
# Step 2a: Select the 200 largest models from each category
# (Larger models have more detail and are better for training)
python ds_select_largest.py \
    data/modelnet40/ModelNet40 \
    data/modelnet40/ModelNet40_200 \
    200
```

**What this does**: Goes through each category and picks the 200 models with the most vertices. Small models don't have enough detail to train on effectively.

```bash
# Step 2b: Convert 3D meshes to point clouds
# A mesh is triangles; a point cloud is just points
python ds_mesh_to_pc.py \
    data/modelnet40/ModelNet40_200 \
    data/modelnet40/ModelNet40_200_pc512 \
    --vg_size 512
```

**What this does**: Samples points from the surface of each 3D model and places them in a 512×512×512 voxel grid. Think of it like converting a smooth surface into LEGO blocks.

```bash
# Step 2c: Split into octree blocks
# Large point clouds are divided into smaller chunks for processing
python ds_pc_octree_blocks.py \
    data/modelnet40/ModelNet40_200_pc512 \
    data/modelnet40/ModelNet40_200_pc512_oct3 \
    --vg_size 512 \
    --level 3
```

**What this does**: Divides each point cloud into 8³ = 512 smaller blocks using an octree (a tree where each node has 8 children). This makes training more efficient because:
- Each block fits in GPU memory
- The network sees more variety (different parts of different objects)
- Blocks can be processed in parallel

```bash
# Step 2d: Select the 4000 most detailed blocks
python ds_select_largest.py \
    data/modelnet40/ModelNet40_200_pc512_oct3 \
    data/modelnet40/ModelNet40_200_pc512_oct3_4k \
    4000
```

**What this does**: Not all blocks are useful—some are empty or nearly empty. We keep only the 4000 blocks with the most points, ensuring we train on meaningful data.

### Step 3: Training

Create a configuration file that defines all training parameters:

```bash
cat > config/train_config.yml << EOL
# Data settings
data:
  modelnet40_path: "data/modelnet40/ModelNet40_200_pc512_oct3_4k"
  ivfb_path: "data/8ivfb"
  resolution: 64          # Size of input blocks (64×64×64 voxels)
  block_size: 1.0         # Physical size of each block
  min_points: 100         # Ignore blocks with fewer points
  augment: true           # Apply random rotations/flips for variety

# Model architecture
model:
  filters: 64             # Neural network width (more = more capacity)
  activation: "cenic_gdn" # Activation function optimized for compression
  conv_type: "separable"  # Efficient 1+2D convolutions instead of full 3D
  entropy_model: "hyperprior"  # Which entropy model to use

# Training settings
training:
  batch_size: 32          # How many blocks to process at once
  epochs: 100             # How many times to go through all data
  learning_rates:
    reconstruction: 1.0e-4  # Learning rate for quality
    entropy: 1.0e-3         # Learning rate for compression
  focal_loss:
    alpha: 0.75           # Weight for hard examples
    gamma: 2.0            # Focus on difficult cases
  checkpoint_dir: "results/models"
  mixed_precision: false  # Set to true for faster GPU training
EOL
```

**Understanding the parameters**:
- **batch_size**: Larger batches are more stable but need more GPU memory
- **epochs**: More epochs = more training, but eventually you overfit
- **learning_rate**: How big of steps to take when learning. Too high = unstable, too low = slow
- **focal_loss**: Helps the network focus on the hard parts of the point cloud (edges, fine details)

Now start training:

```bash
python training_pipeline.py config/train_config.yml
```

**What happens during training**:
1. The model loads batches of point cloud blocks
2. It tries to compress and reconstruct each block
3. It measures two things: reconstruction quality and compressed size
4. It adjusts its weights to improve both metrics
5. Every epoch, it saves a checkpoint so you can resume if interrupted

Training typically takes:
- **CPU only**: Several days
- **Single GPU**: 12-24 hours
- **Multiple GPUs**: A few hours

### Step 4: Evaluation

After training, test how well your model performs on new data:

```bash
# Run evaluation on the 8iVFB dataset
python evaluation_pipeline.py \
    config/train_config.yml \
    --checkpoint results/models/best_model
```

**What this measures**:
- **PSNR (Peak Signal-to-Noise Ratio)**: How similar the reconstruction is to the original (higher = better)
- **Chamfer Distance**: Average distance between original and reconstructed points (lower = better)
- **Bits per point**: How many bits needed per 3D point (lower = better compression)
- **Compression/decompression time**: How fast is it?

```bash
# Generate comparison metrics against other methods
python ev_compare.py \
    --original data/8ivfb \
    --compressed results/compressed \
    --output results/metrics
```

```bash
# Create visualizations of the results
python ev_run_render.py config/train_config.yml
```

**What this creates**: Side-by-side images showing original vs. reconstructed point clouds, color-coded by error.

### Step 5: Compare with Industry Standard (G-PCC)

G-PCC is the industry-standard point cloud codec from MPEG. Compare your results:

```bash
# Generate a final comparison report
python mp_report.py \
    results/metrics/evaluation_report.json \
    results/metrics/final_report.json
```

**What you'll see**: A table comparing DeepCompress vs. G-PCC on metrics like:
- BD-Rate: Percentage bitrate savings at the same quality
- BD-PSNR: Quality improvement at the same bitrate

### Expected Results

After completing the full pipeline:

| Metric | DeepCompress V1 | DeepCompress V2 (Hyperprior) |
|--------|-----------------|------------------------------|
| BD-Rate vs G-PCC | -8% | -20% to -30% |
| Model Parameters | 1.0M | 1.2M |
| Inference Speed | Baseline | 2-3x faster |
| Memory Usage | Baseline | 50% lower |

---

## Understanding the Architecture

### How Neural Compression Works

Traditional compression (like ZIP) looks for repeated patterns in data. Neural compression goes further—it *learns* what patterns exist in a specific type of data.

```
                    ENCODER                              DECODER

Original         ┌─────────────┐                     ┌─────────────┐
Point Cloud  ──► │  Analysis   │ ──► Latent y ──►   │  Synthesis  │ ──► Reconstructed
(Large)          │  Transform  │     (Small)        │  Transform  │     Point Cloud
                 └─────────────┘                     └─────────────┘
                       │                                   ▲
                       ▼                                   │
                 ┌─────────────┐                     ┌─────────────┐
                 │   Hyper     │ ──► z (Tiny) ──►   │   Hyper     │
                 │  Encoder    │                    │  Decoder    │
                 └─────────────┘                     └─────────────┘
                                                           │
                                                           ▼
                                                    ┌─────────────┐
                                                    │  Entropy    │
                                                    │   Model     │
                                                    └─────────────┘
                                                           │
                                                           ▼
                                                      Bitstream
                                                    (Compressed File)
```

**The key insight**: The "latent" representation (y) is much smaller than the original, but contains enough information to reconstruct it. The "hyper" path (z) helps the entropy model know what probabilities to use.

### Why Different Entropy Models Matter

The entropy model is crucial because it determines how efficiently we can convert the latent representation into bits.

**Gaussian (baseline)**: Assumes every value follows the same bell curve. Simple but not accurate.

**Hyperprior**: Learns a custom mean and variance for each position. Like having a different bell curve for each value.

**Channel Context**: Processes channels in order, using earlier channels to predict later ones. Like reading a book—earlier words help predict later words.

**Spatial Context**: Uses neighboring positions to predict each value. Like filling in a crossword puzzle—the letters around you give hints.

**Attention**: Looks at the entire point cloud to find relevant patterns. Like having a photographic memory of similar shapes you've seen before.

### V2 Architecture Diagram

```
Input Voxel Grid (e.g., 64×64×64×1)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ ANALYSIS TRANSFORM                                          │
│ ┌─────────┐    ┌─────────┐    ┌─────────┐                  │
│ │ Conv3D  │───►│ Conv3D  │───►│ Conv3D  │───► Latent y     │
│ │ + GDN   │    │ + GDN   │    │ + GDN   │    (8×8×8×192)   │
│ └─────────┘    └─────────┘    └─────────┘                  │
│   64→128         128→192        192→192                     │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ HYPER ENCODER                                               │
│ Latent y ──► Conv3D ──► Conv3D ──► Hyper-latent z          │
│                                    (4×4×4×128)              │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ ENTROPY MODEL (V2 - Configurable)                           │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐│
│ │ Hyperprior: z ──► mean, scale for each position         ││
│ │                                                         ││
│ │ Channel: Process channels 1,2,3... using previous ones  ││
│ │          as context                                     ││
│ │                                                         ││
│ │ Attention: Use windowed self-attention to find          ││
│ │            long-range dependencies                      ││
│ └─────────────────────────────────────────────────────────┘│
│                          │                                  │
│                          ▼                                  │
│                    Probability                              │
│                    Distribution                             │
│                          │                                  │
│                          ▼                                  │
│               Arithmetic Coding ──► Bitstream               │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Benchmarking

### Running Benchmarks

Test the performance optimizations:

```bash
# Run all benchmarks
python -m src.benchmarks
```

This will output timing comparisons like:

```
============================================================
Benchmark Results
============================================================
  broadcast_quantize          :    45.23 ms (baseline)
  binary_search_quantize      :     9.05 ms (5.00x)
============================================================
```

### What Each Benchmark Tests

| Benchmark | What It Measures | Why It Matters |
|-----------|------------------|----------------|
| `benchmark_scale_quantization` | Speed of finding optimal quantization levels | Called millions of times during compression |
| `benchmark_masked_conv` | Speed of creating causal masks | Done once per layer, but slow if not optimized |
| `benchmark_attention` | Memory and speed of attention mechanism | Attention is O(n²) by default—we make it O(n) |

### Memory Profiling

Check how much GPU memory your model uses:

```python
from src.benchmarks import MemoryProfiler

with MemoryProfiler() as mem:
    output = model(large_input)

print(f"Peak memory: {mem.peak_mb:.1f} MB")
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Programming language |
| TensorFlow | ~=2.15 | Neural network framework |
| TensorFlow Probability | ~=0.23 | Probability distributions for entropy modeling |
| MPEG G-PCC | Latest | Industry-standard codec for comparison |
| MPEG PCC Metrics | v0.12.3 | Standard evaluation metrics |

### Python Dependencies

Install these with `pip install -r requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | ~=2.15 | Neural network operations |
| tensorflow-probability | ~=0.23 | Probability distributions for entropy modeling |
| numpy | ~=1.26 | Numerical computations |
| matplotlib | ~=3.8 | Visualization |
| pandas | ~=2.1 | Data analysis |
| pyyaml | ~=6.0 | Configuration file parsing |
| scipy | ~=1.11 | Scientific computing |
| tqdm | ~=4.66 | Progress bars |
| numba | ~=0.58 | JIT compilation for speed |
| keras-tuner | ~=1.4 | Hyperparameter tuning (for cli_train.py) |
| pytest | ~=8.0 | Test framework |
| ruff | >=0.4 | Linter (configured in pyproject.toml) |

---

## Project Structure

```
deepcompress/
├── src/                            # Source code
│   ├── Model Components
│   │   ├── model_transforms.py     # Main encoder/decoder (V1 + V2) architecture
│   │   ├── entropy_model.py        # Gaussian conditional, hyperprior entropy models
│   │   ├── entropy_parameters.py   # Hyperprior mean/scale prediction network
│   │   ├── context_model.py        # MaskedConv3D, autoregressive spatial context
│   │   ├── channel_context.py      # Channel-wise context model
│   │   └── attention_context.py    # Windowed attention context model
│   │
│   ├── Performance
│   │   ├── constants.py            # Pre-computed math constants (LOG_2, EPSILON)
│   │   ├── precision_config.py     # Mixed precision (float16) settings
│   │   ├── benchmarks.py           # Performance measurement
│   │   └── quick_benchmark.py      # Quick synthetic smoke test
│   │
│   ├── Data Processing
│   │   ├── data_loader.py          # Unified data loader (ModelNet40 / 8iVFB)
│   │   ├── ds_mesh_to_pc.py        # Convert .off meshes to point clouds
│   │   ├── ds_pc_octree_blocks.py  # Split point clouds into octree blocks
│   │   ├── ds_select_largest.py    # Select N largest blocks by point count
│   │   ├── octree_coding.py        # Octree encode/decode for voxel grids
│   │   ├── compress_octree.py      # Compression entry point
│   │   └── map_color.py            # Transfer colors between point clouds
│   │
│   ├── Training & Evaluation
│   │   ├── training_pipeline.py    # End-to-end training loop
│   │   ├── evaluation_pipeline.py  # Model evaluation pipeline
│   │   ├── cli_train.py            # Training CLI with hyperparameter tuning
│   │   └── experiment.py           # Experiment runner
│   │
│   └── Evaluation & Comparison
│       ├── ev_compare.py           # Point cloud quality metrics (PSNR, Chamfer)
│       ├── ev_run_render.py        # Visualization / rendering
│       ├── point_cloud_metrics.py  # D1/D2 point-to-point metrics
│       ├── mp_report.py            # MPEG G-PCC comparison reports
│       ├── colorbar.py             # Colorbar visualization utility
│       └── parallel_process.py     # Parallel processing utility
│
├── tests/                          # Automated tests (pytest + tf.test.TestCase)
│   ├── conftest.py                 # Session-scoped fixtures (tf_config, file factories)
│   ├── test_utils.py               # Shared test utilities (mock grids, configs)
│   ├── test_model_transforms.py    # V1 + V2 model tests
│   ├── test_entropy_model.py       # Entropy model tests
│   ├── test_context_model.py       # Context model tests
│   ├── test_channel_context.py     # Channel context tests
│   ├── test_attention_context.py   # Attention context tests
│   ├── test_performance.py         # Performance regression + optimization tests
│   ├── test_training_pipeline.py   # Training loop tests
│   ├── test_evaluation_pipeline.py # Evaluation pipeline tests
│   ├── test_data_loader.py         # Data loading tests
│   ├── test_compress_octree.py     # Compression pipeline tests
│   ├── test_octree_coding.py       # Octree codec tests
│   └── ...                         # + 10 more module-level test files
│
├── data/                           # Datasets (not in git)
├── results/                        # Output files (not in git)
├── CLAUDE.md                       # AI agent coding standards
├── pyproject.toml                  # Ruff linter configuration
├── pytest.ini                      # Pytest configuration and markers
├── setup.py                        # Package setup
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## Troubleshooting

### Common Issues

**"Out of memory" errors**
- Reduce `batch_size` in config
- Use `resolution: 32` instead of 64
- Enable mixed precision training
- Use `entropy_model: 'hyperprior'` (most memory-efficient)

**Training is slow**
- Enable mixed precision: `mixed_precision: true`
- Use a GPU (CPU training is 10-50x slower)
- Reduce model size: `filters: 32`

**Poor reconstruction quality**
- Train for more epochs
- Increase model size: `filters: 128`
- Try a better entropy model: `entropy_model: 'channel'`

**Compression ratio is worse than expected**
- Ensure the model is fully trained
- Use an advanced entropy model
- Check that input data is similar to training data

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{killea2021deepcompress,
  title={DeepCompress: Efficient Point Cloud Geometry Compression},
  author={Killea, Ryan and Li, Yun and Bastani, Saeed and McLachlan, Paul},
  journal={arXiv preprint arXiv:2106.01504},
  year={2021}
}
```

---

## License

This project is licensed under the terms specified in the LICENSE file.

---

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/pmclsf/deepcompress/issues)
- **Paper**: [arXiv:2106.01504](https://arxiv.org/abs/2106.01504)
- **Questions**: Open a GitHub issue with the "question" label
