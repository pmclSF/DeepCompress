import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ds_mesh_to_pc import (
    MeshData,
    convert_mesh_to_point_cloud,
    partition_point_cloud,
    read_off,
    sample_points_from_mesh,
    save_ply,
)


class TestDsMeshToPc(unittest.TestCase):
    def setUp(self):
        self.test_off_file = "test.off"
        self.test_ply_file = "test.ply"
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],  # Added points closer together
            [0.2, 0.2, 0.2],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [1.0, 1.0, 1.0]   # One distant point
        ], dtype=np.float32)

        self.normals = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        # Normalize the normals
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True)

        with open(self.test_off_file, "w") as file:
            file.write("OFF\n")
            file.write(f"{len(self.vertices)} 0 0\n")
            for vertex in self.vertices:
                file.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

    def tearDown(self):
        for file_path in [self.test_off_file, self.test_ply_file]:
            if os.path.exists(file_path):
                os.remove(file_path)
        base_path = os.path.splitext(self.test_ply_file)[0]
        for file in os.listdir('.'):
            if file.startswith(f"{base_path}_block_") and file.endswith('.ply'):
                os.remove(file)

    def test_read_off(self):
        mesh_data = read_off(self.test_off_file)
        self.assertIsInstance(mesh_data, MeshData)
        np.testing.assert_array_equal(mesh_data.vertices, self.vertices)

    def test_sample_points_from_mesh(self):
        mesh_data = MeshData(vertices=self.vertices, faces=None, vertex_normals=self.normals)
        points, normals = sample_points_from_mesh(mesh_data, num_points=3, compute_normals=True)

        self.assertEqual(points.shape, (3, 3))
        self.assertEqual(normals.shape, (3, 3))

        for point in points:
            self.assertTrue(
                np.any(np.all(np.abs(self.vertices - point) < 1e-5, axis=1)),
                "Sampled point not in original vertices."
            )

    def test_save_ply(self):
        save_ply(self.test_ply_file, self.vertices, normals=self.normals)
        self.assertTrue(os.path.exists(self.test_ply_file))

        with open(self.test_ply_file, "r") as file:
            lines = file.readlines()
            self.assertEqual(lines[0].strip(), "ply")
            self.assertEqual(lines[2].strip(), f"element vertex {len(self.vertices)}")
            self.assertIn("property float nx", "".join(lines))

    def test_partition_point_cloud(self):
        blocks = partition_point_cloud(
            self.vertices,
            self.normals,
            block_size=0.3,
            min_points=2
        )

        self.assertGreater(len(blocks), 0)
        for block in blocks:
            self.assertIn('points', block)
            self.assertIn('normals', block)
            self.assertEqual(block['points'].shape[1], 3)
            self.assertEqual(block['normals'].shape[1], 3)
            self.assertGreaterEqual(len(block['points']), 2)

    def test_end_to_end(self):
        convert_mesh_to_point_cloud(
            self.test_off_file,
            self.test_ply_file,
            num_points=3,
            compute_normals=True,
            partition_blocks=False,
        )
        self.assertTrue(os.path.exists(self.test_ply_file))

if __name__ == "__main__":
    unittest.main()
