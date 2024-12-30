import unittest
import numpy as np
import os
from ds_mesh_to_pc import read_off, sample_points_from_mesh, save_ply, convert_mesh_to_point_cloud

class TestDsMeshToPc(unittest.TestCase):

    def setUp(self):
        """Set up temporary test data."""
        self.test_off_file = "test.off"
        self.test_ply_file = "test.ply"
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ])

        # Write a minimal OFF file
        with open(self.test_off_file, "w") as file:
            file.write("OFF\n")
            file.write(f"{len(self.vertices)} 0 0\n")
            for vertex in self.vertices:
                file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_off_file):
            os.remove(self.test_off_file)
        if os.path.exists(self.test_ply_file):
            os.remove(self.test_ply_file)

    def test_read_off(self):
        """Test reading an OFF file."""
        vertices = read_off(self.test_off_file)
        np.testing.assert_array_equal(vertices, self.vertices, "OFF file reading failed.")

    def test_sample_points_from_mesh(self):
        """Test sampling points from mesh vertices."""
        sampled_points = sample_points_from_mesh(self.vertices, num_points=3)
        self.assertEqual(sampled_points.shape, (3, 3))
        for point in sampled_points:
            self.assertTrue(np.any(np.all(self.vertices == point, axis=1)), "Sampled point not in original vertices.")

    def test_save_ply(self):
        """Test saving a point cloud to a PLY file."""
        save_ply(self.test_ply_file, self.vertices)
        self.assertTrue(os.path.exists(self.test_ply_file), "PLY file not created.")

        # Verify file content
        with open(self.test_ply_file, "r") as file:
            lines = file.readlines()
        self.assertEqual(lines[0].strip(), "ply")
        self.assertEqual(lines[2].strip(), f"element vertex {len(self.vertices)}")

    def test_convert_mesh_to_point_cloud(self):
        """Test converting an OFF mesh to a PLY point cloud."""
        convert_mesh_to_point_cloud(self.test_off_file, self.test_ply_file, num_points=3)
        self.assertTrue(os.path.exists(self.test_ply_file), "PLY file not created during conversion.")

if __name__ == "__main__":
    unittest.main()
