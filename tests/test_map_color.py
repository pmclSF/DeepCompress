import unittest
import numpy as np
import os
from map_color import load_point_cloud, load_colors, transfer_colors, save_colored_point_cloud

class TestMapColor(unittest.TestCase):
    """Test suite for point cloud color mapping operations."""

    def setUp(self):
        self.test_off_file = "test.off"
        self.test_ply_file = "test.ply"
        self.output_ply = "output.ply"
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float32)

        self.colors = np.array([
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [100, 150, 200],
            [50, 75, 125],
            [200, 100, 50],
            [125, 75, 50],
            [0, 0, 0]
        ])

        with open(self.test_off_file, "w") as file:
            file.write("OFF\n")
            file.write(f"{len(self.vertices)} 0 0\n")
            for vertex in self.vertices:
                file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

    def tearDown(self):
        for file_path in [self.test_off_file, self.test_ply_file, self.output_ply]:
            if os.path.exists(file_path):
                os.remove(file_path)

    def test_load_point_cloud(self):
        points = load_point_cloud(self.test_off_file)
        np.testing.assert_array_equal(points, self.vertices)

    def test_load_colors(self):
        with open(self.test_ply_file, "w") as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {len(self.vertices)}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
            file.write("end_header\n")
            for v, c in zip(self.vertices, self.colors):
                file.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")

        colors = load_colors(self.test_ply_file)
        np.testing.assert_array_almost_equal(colors * 255, self.colors)

    def test_transfer_colors(self):
        target_points = self.vertices[:3]
        transferred_colors = transfer_colors(self.vertices, self.colors / 255.0, target_points)
        self.assertEqual(transferred_colors.shape, target_points.shape)

    def test_save_colored_point_cloud(self):
        save_colored_point_cloud(self.output_ply, self.vertices[:3], self.colors[:3] / 255.0)
        self.assertTrue(os.path.exists(self.output_ply))

        with open(self.output_ply, 'r') as f:
            content = f.read()
            self.assertIn("property uchar red", content)
            self.assertIn("property uchar green", content)
            self.assertIn("property uchar blue", content)

if __name__ == "__main__":
    unittest.main()