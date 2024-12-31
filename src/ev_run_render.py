import argparse
import json
import logging
import os
from typing import Dict, Any, Tuple, Optional
import yaml
import tensorflow as tf
import numpy as np
from PIL import Image
from dataclasses import dataclass

@dataclass
class RenderConfig:
    """Configuration for point cloud rendering."""
    image_width: int = 256
    image_height: int = 256
    point_size: float = 1.0
    focal_length: float = 500.0
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    min_depth: float = 0.1
    max_depth: float = 100.0
    color_map: str = 'viridis'

@dataclass
class CameraParams:
    """Camera parameters for rendering."""
    position: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov: float = 60.0
    aspect: float = 1.0
    near: float = 0.1
    far: float = 100.0

class PointCloudRenderer:
    """Point cloud renderer using TensorFlow."""
    
    def __init__(self, config: RenderConfig):
        self.config = config
        self._setup_colormap()
        
    def _setup_colormap(self):
        """Setup color mapping for point cloud visualization."""
        import matplotlib.pyplot as plt
        self.colormap = plt.get_cmap(self.config.color_map)
    
    def _compute_projection_matrix(self, camera: CameraParams) -> tf.Tensor:
        """Compute perspective projection matrix."""
        f = 1.0 / tf.tan(tf.constant(camera.fov * 0.5 * np.pi / 180.0, dtype=tf.float32))
        aspect = tf.constant(camera.aspect, dtype=tf.float32)
        near = tf.constant(camera.near, dtype=tf.float32)
        far = tf.constant(camera.far, dtype=tf.float32)
        
        return tf.cast(tf.stack([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), 2*far*near/(near-far)],
            [0, 0, -1, 0]
        ]), tf.float32)
        
    def _compute_view_matrix(self, camera: CameraParams) -> tf.Tensor:
        """Compute view matrix from camera parameters."""
        position = tf.cast(camera.position, tf.float32)
        target = tf.cast(camera.target, tf.float32)
        up = tf.cast(camera.up, tf.float32)

        # Expand dimensions to handle vector operations
        position = tf.expand_dims(position, 0)  # Shape: [1, 3]
        target = tf.expand_dims(target, 0)      # Shape: [1, 3]
        up = tf.expand_dims(up, 0)              # Shape: [1, 3]

        # Compute camera axes
        z_axis = position - target
        z_axis = tf.nn.l2_normalize(z_axis, axis=-1)
        
        x_axis = tf.linalg.cross(up, z_axis)
        x_axis = tf.nn.l2_normalize(x_axis, axis=-1)
        
        y_axis = tf.linalg.cross(z_axis, x_axis)

        # Remove extra dimensions
        x_axis = tf.squeeze(x_axis)  # Shape: [3]
        y_axis = tf.squeeze(y_axis)  # Shape: [3]
        z_axis = tf.squeeze(z_axis)  # Shape: [3]
        position = tf.squeeze(position)  # Shape: [3]

        # Compute translation components
        translation = tf.stack([
            -tf.reduce_sum(x_axis * position),
            -tf.reduce_sum(y_axis * position),
            -tf.reduce_sum(z_axis * position)
        ])
        
        # Build view matrix
        view_matrix = tf.stack([
            tf.concat([x_axis, [translation[0]]], axis=0),
            tf.concat([y_axis, [translation[1]]], axis=0),
            tf.concat([z_axis, [translation[2]]], axis=0),
            tf.constant([0.0, 0.0, 0.0, 1.0], dtype=tf.float32)
        ])
        
        return view_matrix
    
    def _project_points(self, points: tf.Tensor, camera: CameraParams) -> tf.Tensor:
        """Project 3D points to 2D image coordinates."""
        # Ensure points are float32
        points = tf.cast(points, tf.float32)
        
        # Homogeneous coordinates
        points_h = tf.concat([points, tf.ones([tf.shape(points)[0], 1], dtype=tf.float32)], axis=1)
        
        # View and projection transformations
        view_matrix = self._compute_view_matrix(camera)
        proj_matrix = self._compute_projection_matrix(camera)
        
        # Transform points
        view_points = tf.matmul(points_h, view_matrix, transpose_b=True)
        proj_points = tf.matmul(view_points, proj_matrix, transpose_b=True)
        
        # Perspective division
        proj_points = proj_points[..., :2] / (proj_points[..., 3:4] + 1e-10)
        
        # Scale to image coordinates
        image_points = (proj_points + 1.0) * 0.5
        image_points = tf.stack([
            image_points[:, 0] * tf.cast(self.config.image_width, tf.float32),
            (1.0 - image_points[:, 1]) * tf.cast(self.config.image_height, tf.float32)
        ], axis=1)
        
        return image_points
    
    def render(self, 
              points: tf.Tensor, 
              colors: Optional[tf.Tensor] = None, 
              camera: Optional[CameraParams] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Render point cloud to image."""
        # Convert points to float32
        points = tf.cast(points, tf.float32)
        
        if camera is None:
            # Compute default camera parameters
            center = tf.reduce_mean(points, axis=0)
            max_dist = tf.reduce_max(tf.norm(points - center, axis=1))
            camera = CameraParams(
                position=center.numpy() + np.array([0., 0., max_dist.numpy() * 2]),
                target=center.numpy(),
                up=np.array([0., 1., 0.], dtype=np.float32),
                fov=45.0,
                aspect=self.config.image_width / self.config.image_height
            )
            
        # Project points to 2D
        image_points = self._project_points(points, camera)
        
        # Initialize image
        image = tf.zeros((self.config.image_height, self.config.image_width, 3), dtype=tf.float32)
        
        # Get point colors
        if colors is None:
            # Color by depth
            depths = tf.norm(points - tf.cast(camera.position, tf.float32), axis=1)
            norm_depths = (depths - tf.reduce_min(depths)) / (tf.reduce_max(depths) - tf.reduce_min(depths) + 1e-10)
            colors = tf.convert_to_tensor([self.colormap(d) for d in norm_depths.numpy()])[:, :3]
        
        # Convert colors to float32
        colors = tf.cast(colors, tf.float32)
        
        # Render points
        valid_mask = tf.logical_and(
            tf.logical_and(
                image_points[:, 0] >= 0,
                image_points[:, 0] < self.config.image_width
            ),
            tf.logical_and(
                image_points[:, 1] >= 0,
                image_points[:, 1] < self.config.image_height
            )
        )
        
        valid_points = tf.boolean_mask(image_points, valid_mask)
        valid_colors = tf.boolean_mask(colors, valid_mask)
        
        # Simple point splatting
        indices = tf.cast(tf.round(valid_points), tf.int32)
        updates = valid_colors
        
        image = tf.tensor_scatter_nd_update(
            image,
            indices,
            updates
        )
        
        render_info = {
            'camera': {
                'position': camera.position.tolist(),
                'target': camera.target.tolist(),
                'up': camera.up.tolist(),
                'fov': float(camera.fov),
                'aspect': float(camera.aspect),
                'near': float(camera.near),
                'far': float(camera.far)
            },
            'render_config': {
                'image_width': self.config.image_width,
                'image_height': self.config.image_height,
                'point_size': self.config.point_size,
                'color_map': self.config.color_map
            }
        }
        
        return image.numpy(), render_info

def load_experiment_config(experiment_path: str) -> Dict[str, Any]:
    """Load and validate experiment configuration."""
    with open(experiment_path, 'r') as f:
        config = yaml.safe_load(f)
        
    required_keys = [
        'MPEG_DATASET_DIR', 
        'EXPERIMENT_DIR', 
        'model_configs', 
        'vis_comps'
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required keys in configuration: {missing_keys}")
        
    return config

def save_rendered_image(
    image_array: np.ndarray,
    render_info: Dict[str, Any],
    save_path: str,
    bbox: Optional[Tuple[int, int, int, int]] = None
):
    """Save rendered image with metadata."""
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    
    if bbox is not None:
        img = img.crop(bbox)
        render_info['bbox'] = bbox
    
    img.save(save_path)
    
    with open(save_path + ".meta.json", 'w') as f:
        json.dump(render_info, f, indent=2)

def main(experiment_path: str):
    """Main rendering pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    config = load_experiment_config(experiment_path)
    renderer = PointCloudRenderer(RenderConfig(**config.get('render_config', {})))
    
    for data_entry in config['data']:
        pc_name = data_entry['pc_name']
        output_dir = os.path.join(config['EXPERIMENT_DIR'], pc_name)
        os.makedirs(output_dir, exist_ok=True)
        
        point_cloud = tf.random.uniform((1024, 3), dtype=tf.float32)
        
        camera = None
        if 'camera_params' in data_entry:
            camera = CameraParams(**data_entry['camera_params'])
        
        image, render_info = renderer.render(point_cloud, camera=camera)
        bbox = data_entry.get('bbox', None)
        
        save_path = os.path.join(output_dir, f"{pc_name}.png")
        save_rendered_image(image, render_info, save_path, bbox)
        
        logger.info(f"Rendered {pc_name} and saved at {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run point cloud rendering experiments.")
    parser.add_argument('experiment_path', help="Path to experiment configuration YAML.")
    args = parser.parse_args()
    main(args.experiment_path)