import sys
import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from map_color import load_point_cloud, map_colors, save_point_cloud

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def create_dummy_point_cloud(file_path, num_points=100):
    """Create a dummy point cloud CSV file."""
    data = {
        'x': np.random.rand(num_points),
        'y': np.random.rand(num_points),
        'z': np.random.rand(num_points),
        'red': np.random.randint(0, 256, num_points),
        'green': np.random.randint(0, 256, num_points),
        'blue': np.random.randint(0, 256, num_points)
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return df

def test_load_point_cloud():
    """Test loading a point cloud from a CSV file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        create_dummy_point_cloud(temp_file.name)
        points = load_point_cloud(temp_file.name)

    assert not points.empty, "Loaded point cloud should not be empty."
    assert set(['x', 'y', 'z', 'red', 'green', 'blue']).issubset(points.columns), "Missing required columns in point cloud."


def test_map_colors():
    """Test mapping colors from one point cloud to another."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as ori_file, \
         tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as target_file:

        ori_points = create_dummy_point_cloud(ori_file.name)
        target_points = create_dummy_point_cloud(target_file.name, num_points=50)[['x', 'y', 'z']]

        mapped_points = map_colors(ori_points, target_points)

    assert not mapped_points.empty, "Mapped point cloud should not be empty."
    assert set(['x', 'y', 'z', 'red', 'green', 'blue']).issubset(mapped_points.columns), "Missing required columns in mapped point cloud."


def test_save_point_cloud():
    """Test saving a point cloud to a CSV file."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        points = create_dummy_point_cloud(temp_file.name)
        output_path = temp_file.name + '_output.csv'

        save_point_cloud(points, output_path)

        assert os.path.exists(output_path), "Output file was not created."

if __name__ == '__main__':
    pytest.main([__file__])
