import unittest
import numpy as np
import os
from ds_mesh_to_pc import (
    read_off,
    sample_points_from_mesh,
    save_ply,
    convert_mesh_to_point_cloud,
    MeshData,
    partition_point_cloud
)

class TestDsMeshToPc(unittest.TestCase):
    def setUp(self):
        """Set up temporary test data."""
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

        # Create test normals
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

        # Write a minimal OFF file with vertices
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
        # Clean up any block files that might have been created
        base_path = os.path.splitext(self.test_ply_file)[0]
        for file in os.listdir('.'):
            if file.startswith(f"{base_path}_block_") and file.endswith('.ply'):
                os.remove(file)

    def test_read_off(self):
        """Test reading an OFF file."""
        mesh_data = read_off(self.test_off_file)
        self.assertIsInstance(mesh_data, MeshData)
        np.testing.assert_array_equal(mesh_data.vertices, self.vertices)

    def test_sample_points_from_mesh(self):
        """Test sampling points from mesh vertices."""
        mesh_data = MeshData(vertices=self.vertices, faces=None, vertex_normals=self.normals)
        points, normals = sample_points_from_mesh(mesh_data, num_points=3, compute_normals=True)
        
        self.assertEqual(points.shape, (3, 3))
        self.assertEqual(normals.shape, (3, 3))
        
        # Check sampled points exist in original vertices
        for point in points:
            self.assertTrue(
                np.any(np.all(np.abs(self.vertices - point) < 1e-5, axis=1)),
                "Sampled point not in original vertices."
            )

    def test_save_ply(self):
        """Test saving a point cloud to a PLY file with normals."""
        save_ply(self.test_ply_file, self.vertices, normals=self.normals)
        self.assertTrue(os.path.exists(self.test_ply_file))

        # Verify file content
        with open(self.test_ply_file, "r") as file:
            lines = file.readlines()
            self.assertEqual(lines[0].strip(), "ply")
            self.assertEqual(lines[2].strip(), f"element vertex {len(self.vertices)}")
            self.assertIn("property float nx", "".join(lines))

    def test_convert_mesh_to_point_cloud(self):
        """Test converting an OFF mesh to a PLY point cloud."""
        convert_mesh_to_point_cloud(
            self.test_off_file,
            self.test_ply_file,
            num_points=3,
            compute_normals=True
        )
        self.assertTrue(os.path.exists(self.test_ply_file))

    def test_partition_point_cloud(self):
        """Test partitioning point cloud into blocks."""
        blocks = partition_point_cloud(
            self.vertices,
            self.normals,
            block_size=0.3,  # Adjusted block size to match test data
            min_points=2     # Reduced minimum points per block
        )
        
        self.assertGreater(len(blocks), 0, "No blocks were created")
        for block in blocks:
            self.assertIn('points', block)
            self.assertIn('normals', block)
            self.assertEqual(block['points'].shape[1], 3)
            self.assertEqual(block['normals'].shape[1], 3)
            self.assertGreaterEqual(len(block['points']), 2)  # Check minimum points constraint

    def test_convert_mesh_to_point_cloud_with_blocks(self):
        """Test converting mesh to point cloud with block partitioning."""
        convert_mesh_to_point_cloud(
            self.test_off_file,
            self.test_ply_file,
            num_points=10,
            compute_normals=True,
            partition_blocks=True,
            block_size=0.3,  # Match test_partition_point_cloud parameters
            min_points_per_block=2
        )
        
        # Check that at least one block file was created
        base_path = os.path.splitext(self.test_ply_file)[0]
        block_files = [f for f in os.listdir('.') 
                      if f.startswith(f"{base_path}_block_") and f.endswith('.ply')]
        self.assertGreater(len(block_files), 0)

if __name__ == "__main__":
    unittest.main()