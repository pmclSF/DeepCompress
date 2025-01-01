import unittest
import os
import numpy as np

# Local
from map_color import (
    load_point_cloud, 
    load_colors, 
    transfer_colors, 
    save_colored_point_cloud
)

class TestMapColor(unittest.TestCase):
    """Test suite for point cloud color mapping operations."""

    def setUp(self):
        """Set up temporary test data."""
        self.source_ply = "source.ply"
        self.target_ply = "target.ply"
        self.output_ply = "output.ply"

        self.source_points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        self.source_colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255]
        ])
        self.target_points = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0]
        ])

        # Create source PLY file
        with open(self.source_ply, "w") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {self.source_points.shape[0]}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
            file.write("end_header\n")
            for point, color in zip(self.source_points, self.source_colors):
                file.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")

        # Create target PLY file
        with open(self.target_ply, "w") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {self.target_points.shape[0]}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("end_header\n")
            for point in self.target_points:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")

    def tearDown(self):
        """Clean up test files."""
        for file_path in [self.source_ply, self.target_ply, self.output_ply]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_load_point_cloud(self):
        """Test loading point cloud from PLY file."""
        points = load_point_cloud(self.source_ply)
        np.testing.assert_array_equal(points, self.source_points, "Source point cloud loading failed.")

    def test_load_colors(self):
        """Test loading colors from PLY file."""
        colors = load_colors(self.source_ply)
        np.testing.assert_array_equal(colors * 255, self.source_colors, "Source colors loading failed.")

    def test_transfer_colors(self):
        """Test transferring colors from source to target point cloud."""
        target_colors = transfer_colors(self.source_points, self.source_colors, self.target_points)
        self.assertEqual(target_colors.shape, self.target_points.shape, "Transferred colors shape mismatch.")

    def test_save_colored_point_cloud(self):
        """Test saving a colored point cloud to PLY."""
        colors = np.array([
            [100, 150, 200],
            [50, 75, 125]
        ])
        save_colored_point_cloud(self.output_ply, self.target_points, colors / 255.0)
        self.assertTrue(os.path.exists(self.output_ply), "Colored PLY file not created.")

    def test_end_to_end(self):
        """Test end-to-end color transfer and saving."""
        source_points = load_point_cloud(self.source_ply)
        source_colors = load_colors(self.source_ply)
        target_points = load_point_cloud(self.target_ply)

        target_colors = transfer_colors(source_points, source_colors, target_points)
        save_colored_point_cloud(self.output_ply, target_points, target_colors)
        self.assertTrue(os.path.exists(self.output_ply), "Output PLY file not created.")

if __name__ == "__main__":
    unittest.main()
