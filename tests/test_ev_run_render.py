import sys
import os
import tempfile
import pytest
import yaml
import tensorflow as tf

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ev_run_render import load_experiment_config, compute_camera_params, render_point_cloud, save_rendered_image


def create_dummy_experiment_file():
    """Create a dummy YAML experiment file for testing."""
    temp_dataset_dir = tempfile.mkdtemp()
    temp_experiment_dir = tempfile.mkdtemp()

    experiments = {
        'MPEG_DATASET_DIR': temp_dataset_dir,
        'EXPERIMENT_DIR': temp_experiment_dir,
        'model_configs': [{'id': 'model_1', 'config': 'config_1', 'lambdas': [0.01, 0.1]}],
        'vis_comps': [],
        'data': [{'pc_name': 'test_pc', 'input_pc': 'test_pc.ply'}]
    }
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
    with open(temp_file.name, 'w') as f:
        yaml.dump(experiments, f)

    return temp_file.name, temp_dataset_dir, temp_experiment_dir


def test_load_experiment_config():
    """Test loading the YAML experiment configuration."""
    experiment_file, temp_dataset_dir, temp_experiment_dir = create_dummy_experiment_file()
    try:
        config = load_experiment_config(experiment_file)
        assert 'MPEG_DATASET_DIR' in config
        assert 'EXPERIMENT_DIR' in config
        assert 'data' in config
    finally:
        os.remove(experiment_file)
        os.rmdir(temp_dataset_dir)
        os.rmdir(temp_experiment_dir)


def test_compute_camera_params():
    """Test computation of camera parameters."""
    point_cloud = tf.random.uniform((1024, 3), dtype=tf.float32)
    camera_params = compute_camera_params(point_cloud)
    assert "center" in camera_params, "Camera parameters missing center."


def test_render_point_cloud():
    """Test rendering point cloud."""
    point_cloud = tf.random.uniform((1024, 3), dtype=tf.float32)
    camera_params = {"center": [0.5, 0.5, 0.5]}
    image = render_point_cloud(point_cloud, camera_params)
    assert image.shape == (256, 256), "Rendered image has incorrect dimensions."


def test_save_rendered_image():
    """Test saving rendered image."""
    with tempfile.TemporaryDirectory() as temp_dir:
        image = tf.random.uniform((256, 256), dtype=tf.float32).numpy()
        bbox = (0, 0, 256, 256)
        save_path = os.path.join(temp_dir, "test_image.png")
        save_rendered_image(image, bbox, save_path)

        assert os.path.exists(save_path), "Image file not saved."
        assert os.path.exists(save_path + ".bbox.json"), "Bounding box file not saved."
