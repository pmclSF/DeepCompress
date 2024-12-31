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

## Experimental Data

We provide comprehensive experimental data including:

- Trained models in the `models/` directory
- Bitrates and objective metric values (D1/D2) for all models and point clouds in `results/data.csv`
- Compressed and decompressed point clouds for all models (c1 to c6, G-PCC trisoup, G-PCC octree)

[Download the experimental data](https://drive.google.com/file/d/18uHmr0ZpgFLeL9Y5TUFTsQkRfz4XpQdJ/view?usp=sharing)

## Prerequisites

### Required Software

- Python 3.6.9+
- MPEG G-PCC codec [mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13)
- MPEG metric software v0.12.3 [mpeg-pcc-dmetric](http://mpegx.int-evry.fr/software/MPEG/PCC/mpeg-pcc-dmetric)
- MPEG PCC dataset

### Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt
```

Required packages:
- matplotlib ~= 3.1.3
- pyntcloud ~= 0.1.2
- numpy ~= 1.23.0
- pandas ~= 1.4.0
- tqdm ~= 4.64.0
- tensorflow ~= 2.11.0
- pyyaml ~= 5.1.2
- pytest ~= 7.1.0
- scipy ~= 1.8.1
- numba ~= 0.55.0

### Environment Configuration

Create a `ev_experiment.yml` file with the following structure:

```yaml
# Environment paths
MPEG_TMC13_DIR: "/path/to/mpeg-pcc-tmc13"
PCERROR: "/path/to/mpeg-pcc-dmetric/test/pc_error_d"
MPEG_DATASET_DIR: "/path/to/mpeg_pcc"
TRAIN_DATASET_PATH: "/path/to/ModelNet40_200_pc512_oct3_4k/**/*.ply"
TRAIN_RESOLUTION: 64
EXPERIMENT_DIR: "/path/to/experiments"

# Training parameters
batch_size: 32
alpha: 0.9
gamma: 2.0
train_mode: "independent"
fixed_threshold: True

# Model configurations
model_configs:
  - id: 'c4-ws'
    config: 'c3p'
    lambdas: [3.0e-4, 1.0e-4, 5.0e-5, 2.0e-5, 1.0e-5]
    alpha: 0.75
    train_mode: 'warm_seq'
    label: 'c6'
  # Additional configurations...
```

**Note**: For parallel MPEG experiments, storing the `EXPERIMENT_DIR` on an SSD is highly recommended.

### Notes

- Linux distribution (Ubuntu recommended) is preferred
- CTCs available at [wg11.sc29.org](http://wg11.sc29.org) under "All Meetings > Latest Meeting > Output documents"

## Configuration

Edit `ev_experiment.yml` with your environment settings:

```yaml
MPEG_TMC13_DIR: "/path/to/gpcc"
PCERROR: "/path/to/dmetric"
MPEG_DATASET_DIR: "/path/to/dataset"
TRAIN_DATASET_PATH: "/path/to/training"
TRAIN_RESOLUTION: "resolution_value"
EXPERIMENT_DIR: "/path/to/results"
```

## Dataset Preparation

1. Download ModelNet40 dataset from [modelnet.cs.princeton.edu](http://modelnet.cs.princeton.edu)

2. Generate training dataset:

   ```bash
   # Select largest point clouds
   python ds_select_largest.py ~/data/datasets/ModelNet40 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200 200

   # Convert meshes to point clouds
   python ds_mesh_to_pc.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512 --vg_size 512

   # Partition into octree blocks
   python ds_pc_octree_blocks.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3 --vg_size 512 --level 3

   # Select largest blocks
   python ds_select_largest.py ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3 ~/data/datasets/pcc_geo_cnn_v2/ModelNet40_200_pc512_oct3_4k 4000
   ```

## Running Experiments

### G-PCC Experiments

```bash
python mp_run.py ev_experiment.yml
```

**Note**: For HDD installations, use `--num_parallel 1` to avoid performance issues.

### Training and Evaluation

Run complete pipeline:

```bash
python tr_train.py ev_experiment.yml && \
python ev_run_experiment.py ev_experiment.yml --num_parallel 8 && \
python ev_run_compare.py ev_experiment.yml
```

**Note**: Training typically takes 4 days on an Nvidia GeForce GTX 1080 Ti. Adjust `--num_parallel` based on available GPU memory.

### Hyperparameter Tuning

1. Start tuning:
   ```bash
   python tr_train.py ev_experiment.yml --tune --num_epochs 10
   ```

2. Hyperparameter search space:
   ```python
   def create_model(hp):
       model = tf.keras.Sequential()
       model.add(tf.keras.layers.InputLayer(input_shape=(2048, 3)))
       
       # Layer configuration
       for i in range(hp.Int('num_layers', 1, 5)):
           model.add(tf.keras.layers.Dense(
               hp.Int(f'layer_{i}_units', min_value=64, max_value=1024, step=64),
               activation='relu'
           ))
       
       model.add(tf.keras.layers.Dense(3, activation='sigmoid'))
       
       # Optimizer configuration
       model.compile(
           optimizer=tf.keras.optimizers.Adam(
               learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')
           ),
           loss='mean_squared_error'
       )
       return model
   ```

### Evaluation and Visualization

Generate performance plots and visualizations:

```bash
python ut_run_render.py ev_experiment.yml
python ut_tensorboard_plots.py ev_experiment.yml
```

## Model Architecture

### Network Overview
- Analysis-synthesis architecture with scale hyperprior
- Incorporates GDN/CENIC-GDN activation functions
- Novel 1+2D spatially separable convolutional blocks
- Progressive channel expansion with dimension reduction

### Key Components
- **Analysis Network**: Processes input point clouds through multiple analysis blocks
- **Synthesis Network**: Reconstructs point clouds from compressed representations
- **Hyperprior**: Learns and encodes additional parameters for entropy modeling
- **Custom Activation**: Uses CENIC-GDN for improved efficiency

### Spatially Separable Design
The architecture employs 1+2D convolutions instead of full 3D convolutions, providing:
- More parameter efficiency for same input/output channels
- Reduced operation count
- Better filter utilization
- Encoded knowledge of point cloud surface properties

## Implementation Details

### Point Cloud Metrics

The `pc_metric.py` module provides efficient implementations of standard point cloud metrics:

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

Performance optimizations:
- KD-tree acceleration for nearest neighbor searches
- Parallel computation using Numba
- Efficient memory management for large point clouds

### Data Processing Pipeline

1. **Mesh to Point Cloud Conversion** (`ds_mesh_to_pc.py`):
   ```bash
   python ds_mesh_to_pc.py input.off output.ply \
       --num_points 2048 \
       --compute_normals
   ```
   Features:
   - Surface-aware point sampling
   - Normal computation
   - PLY format output
   
2. **Octree Partitioning** (`ds_pc_octree_blocks.py`):
   ```bash
   python ds_pc_octree_blocks.py input.ply blocks/ \
       --block_size 1.0 \
       --min_points 100
   ```
   Features:
   - Adaptive block sizing
   - Minimum point threshold
   - Parallel block processing

### Model Architecture

The model uses custom transforms defined in `model_transforms.py`:

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

### Parallel Processing

The `parallel_process.py` module provides robust parallel execution:

```python
from parallel_process import parallel_process

results = parallel_process(
    process_function,
    parameter_list,
    num_parallel=4,
    max_retries=3,
    timeout=300
)
```

Features:
- Automatic retry mechanism
- Timeout handling
- Result ordering preservation
- Resource management

2. **Hyperparameter Tuning**:
   ```bash
   python tr_train.py input_dir output_dir --tune --num_epochs 10
   ```
   Tunes key parameters:
   - Number of layers (1-5)
   - Units per layer (64-1024)
   - Learning rate (1e-5 to 1e-3)

3. **Model Training**:
   ```bash
   python tr_train.py input_dir output_dir --batch_size 32 --num_epochs 10
   ```
   Parameters:
   - Optimization: Adam optimizer (β1=0.9, β2=0.999)
   - Learning rates:
     - Reconstruction: 1×10⁻⁴
     - Entropy bottleneck: 1×10⁻³
   - Loss function: Focal loss (α=0.75, γ=2)

4. **Model Optimization**:
   The `ModelOptimizer` class provides:
   - Gradient-based optimization
   - Batch processing
   - Progress tracking
   - Model evaluation

### Running Experiments

```bash
# Train all models
python tr_train.py ev_experiment.yml

# Run experiments with parallel processing
python ev_run_experiment.py ev_experiment.yml --num_parallel 8

# Compare results
python ev_run_compare.py ev_experiment.yml
```

For visualization and analysis:
```bash
# Generate rendered visualizations
python ut_run_render.py ev_experiment.yml

# Create training plots
python ut_tensorboard_plots.py ev_experiment.yml
```

## Evaluation Metrics

We evaluate point cloud compression quality using several geometry preservation and distortion metrics:

### Distance-based Metrics

- **D1 Metric**: Mean closest point distance from predicted points to ground truth
- **D2 Metric**: Mean closest point distance from ground truth to predicted points
- **Chamfer Distance**: Symmetric metric combining D1 and D2 distances

### Normal-based Metrics

- **N1**: Point-to-normal distance between predicted points and ground truth normals
- **N2**: Point-to-normal distance between ground truth points and predicted normals
- **Normal Chamfer**: Combined sum of N1 and N2 distances

### Usage Example

```python
from pc_metric import calculate_metrics

predicted_points = ...  # Your predicted point cloud
ground_truth_points = ... # Your ground truth point cloud

metrics = calculate_metrics(predicted_points, ground_truth_points)
print("D1:", metrics['d1'])
print("D2:", metrics['d2'])
```

## Project Structure

### Source Code (`/src`)

- **Core Processing**
  - `compress_octree.py`: Point cloud octree compression
  - `decompress_octree.py`: Point cloud decompression
  - `ds_mesh_to_pc.py`: Mesh to point cloud conversion
  - `ds_pc_octree_blocks.py`: Octree block partitioning
  - `ds_select_largest.py`: Large point cloud selection

- **Evaluation & Comparison**
  - `ev_compare.py`: Results comparison
  - `ev_experiment.yml`: Experiment configuration
  - `ev_run_experiment.py`: Experiment execution
  - `ev_run_render.py`: Point cloud rendering
  - `experiment.py`: Core experiment utilities

- **Model Components**
  - `model_opt.py`: Model optimization
  - `model_transforms.py`: Model transformations
  - `patch_gaussian_conditional.py`: Gaussian conditional debugging
  - `pc_metric.py`: Point cloud metrics

- **Support Utilities**
  - `colorbar.py`: Visualization colorbars
  - `map_color.py`: Color mapping
  - `mp_report.py`, `mp_run.py`: MPEG-related utilities
  - `octree_coding.py`: Octree encoding
  - `parallel_process.py`: Parallel processing
  - `tr_train.py`: Model training

### Tests (`/test`)

- **Data Processing Tests**
  - `test_ds_mesh_to_pc.py`
  - `test_ds_pc_octree_blocks.py`
  - `test_ds_select_largest.py`

- **Evaluation Tests**
  - `test_ev_compare.py`
  - `test_ev_run_experiment.py`
  - `test_ev_run_render.py`
  - `test_experiment.py`

- **Model Tests**
  - `test_model_opt.py`
  - `test_model_transforms.py`
  - `test_patch_gaussian_conditional.py`
  - `test_pc_metric.py`
  - `test_tr_train.py`

- **Utility Tests**
  - `test_colorbar.py`
  - `test_compress_octree.py`
  - `test_map_color.py`
  - `test_mp_report.py`
  - `test_mp_run.py`
  - `test_octree_coding.py`
  - `test_parallel_process.py`