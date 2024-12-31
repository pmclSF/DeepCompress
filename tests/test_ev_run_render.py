import sys
import os
import tempfile
import pytest
import yaml
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ev_run_render import (
    load_experiment_config,
    RenderConfig,
    CameraParams,
    PointCloudRenderer,
    save_rendered_image
)

def create_test_config():
    """Create test experiment configuration."""
    temp_dataset_dir = tempfile.mkdtemp()
    temp_experiment_dir = tempfile.mkdtemp()

    config = {
        'MPEG_DATASET_DIR': temp_dataset_dir,
        'EXPERIMENT_DIR': temp_experiment_dir,
        'model_configs': [{'id': 'model_1', 'config': 'config_1'}],
        'vis_comps': [],
        'render_config': {
            'image_width': 128,
            'image_height': 128,
            'point_size': 2.0,
            'color_map': 'plasma'
        },
        'data': [{
            'pc_name': 'test_pc',
            'input_pc': 'test_pc.ply',
            'camera_params': {
                'position': [0, 0, 5],
                'target': [0, 0, 0],
                'up': [0, 1, 0],
                'fov': 45.0
            },
            'bbox': [10, 10, 118, 118]
        }]
    }

    return config, temp_dataset_dir, temp_experiment_dir

class TestPointCloudRendering:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        self.config, self.dataset_dir, self.experiment_dir = create_test_config()
        self.render_config = RenderConfig(**self.config['render_config'])
        self.renderer = PointCloudRenderer(self.render_config)
        
        # Create test point cloud
        self.points = tf.random.uniform((100, 3), -1, 1, dtype=tf.float32)
        self.colors = tf.random.uniform((100, 3), 0, 1, dtype=tf.float32)
        
        yield
        
        # Cleanup
        os.rmdir(self.dataset_dir)
        os.rmdir(self.experiment_dir)
        
    def test_render_config(self):
        """Test render configuration."""
        assert self.render_config.image_width == 128
        assert self.render_config.image_height == 128
        assert self.render_config.point_size == 2.0
        assert self.render_config.color_map == 'plasma'
        
    def test_camera_params(self):
        """Test camera parameter handling."""
        cam_config = self.config['data'][0]['camera_params']
        camera = CameraParams(
            position=np.array(cam_config['position']),
            target=np.array(cam_config['target']),
            up=np.array(cam_config['up']),
            fov=cam_config['fov']
        )
        
        assert np.allclose(camera.position, [0, 0, 5])
        assert np.allclose(camera.target, [0, 0, 0])
        assert np.allclose(camera.up, [0, 1, 0])
        assert camera.fov == 45.0
        
    def test_point_cloud_rendering(self):
        """Test basic point cloud rendering."""
        camera = CameraParams(
            position=np.array([0, 0, 5]),
            target=np.array([0, 0, 0]),
            up=np.array([0, 1, 0]),
            fov=45.0
        )
        
        image, render_info = self.renderer.render(
            self.points,
            colors=self.colors,
            camera=camera
        )
        
        assert image.shape == (128, 128, 3)
        assert np.all(image >= 0) and np.all(image <= 1)
        assert 'camera' in render_info
        assert 'render_config' in render_info
        
    def test_save_rendered_image(self):
        """Test saving rendered image with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Render image
            camera = CameraParams(
                position=np.array([0, 0, 5]),
                target=np.array([0, 0, 0]),
                up=np.array([0, 1, 0]),
                fov=45.0
            )
            image, render_info = self.renderer.render(self.points, self.colors, camera)
            
            # Save image
            save_path = os.path.join(temp_dir, "test_render.png")
            bbox = (10, 10, 118, 118)
            save_rendered_image(image, render_info, save_path, bbox)
            
            # Check files exist
            assert os.path.exists(save_path)
            assert os.path.exists(save_path + ".meta.json")
            
            # Check image size
            img = Image.open(save_path)
            assert img.size == (108, 108)  # Size after cropping
            
            # Check metadata
            with open(save_path + ".meta.json", 'r') as f:
                meta = json.load(f)
                assert 'camera' in meta
                assert 'render_config' in meta
                assert 'bbox' in meta
                
    def test_load_experiment_config(self):
        """Test experiment configuration loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.config, f)
            config_path = f.name
            
        try:
            loaded_config = load_experiment_config(config_path)
            assert loaded_config['MPEG_DATASET_DIR'] == self.config['MPEG_DATASET_DIR']
            assert loaded_config['EXPERIMENT_DIR'] == self.config['EXPERIMENT_DIR']
            assert loaded_config['render_config'] == self.config['render_config']
        finally:
            os.unlink(config_path)
            
    def test_missing_config_keys(self):
        """Test handling of missing configuration keys."""
        invalid_config = {
            'MPEG_DATASET_DIR': self.dataset_dir
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name
            
        try:
            with pytest.raises(ValueError):
                load_experiment_config(config_path)
        finally:
            os.unlink(config_path)

if __name__ == '__main__':
    pytest.main([__file__])