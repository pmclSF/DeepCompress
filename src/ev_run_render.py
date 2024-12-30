import argparse
import json
import logging
import os
import yaml
import tensorflow as tf
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_experiment_config(experiment_path):
    """Load experiment configuration from YAML file."""
    with open(experiment_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    required_keys = ['MPEG_DATASET_DIR', 'EXPERIMENT_DIR', 'model_configs', 'vis_comps']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required keys in experiment configuration: {missing_keys}")
    return config


def compute_camera_params(point_cloud):
    """Compute dummy camera parameters for rendering."""
    center = tf.reduce_mean(point_cloud, axis=0)
    camera_params = {"center": center.numpy().tolist()}
    return camera_params


def render_point_cloud(point_cloud, camera_params):
    """Render point cloud as an image using TensorFlow."""
    # Dummy rendering: Create a tensor representing a grayscale image
    height, width = 256, 256
    rendered_image = tf.random.uniform((height, width), dtype=tf.float32)
    return rendered_image.numpy()


def save_rendered_image(image_array, bbox, save_path):
    """Save rendered image and bounding box."""
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img = img.crop(bbox)
    img.save(save_path)
    with open(save_path + ".bbox.json", 'w') as f:
        json.dump({"bbox": bbox}, f)


def main(experiment_path):
    """Main function for rendering experiments."""
    # Load experiment configuration
    config = load_experiment_config(experiment_path)
    dataset_dir = config['MPEG_DATASET_DIR']
    experiment_dir = config['EXPERIMENT_DIR']

    # Iterate through data entries
    for data_entry in config['data']:
        pc_name = data_entry['pc_name']
        input_pc_path = os.path.join(dataset_dir, data_entry['input_pc'])
        output_dir = os.path.join(experiment_dir, pc_name)
        os.makedirs(output_dir, exist_ok=True)

        # Dummy point cloud
        point_cloud = tf.random.uniform((1024, 3), dtype=tf.float32)

        # Compute camera parameters
        camera_params = compute_camera_params(point_cloud)

        # Render point cloud
        rendered_image = render_point_cloud(point_cloud, camera_params)

        # Save image and bounding box
        save_path = os.path.join(output_dir, pc_name)
        bbox = (0, 0, 256, 256)  # Dummy bbox
        save_rendered_image(rendered_image, bbox, save_path)

        logger.info(f"Rendered {pc_name} and saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run rendering experiments.")
    parser.add_argument('experiment_path', help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.experiment_path)