"""
Tests for data pipeline: OFF/PLY file I/O, mesh-to-point-cloud sampling,
and point cloud partitioning.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ds_mesh_to_pc import (
    MeshData,
    compute_face_normals,
    partition_point_cloud,
    read_off,
    sample_points_from_mesh,
    save_ply,
)


class TestReadOFF(tf.test.TestCase):
    """Tests for OFF file reading."""

    @pytest.fixture(autouse=True)
    def inject_tmp_path(self, tmp_path):
        self.tmp_path = tmp_path

    def test_valid_off_file(self):
        """Should parse a valid OFF file correctly."""
        off_path = self.tmp_path / "test.off"
        off_path.write_text(
            "OFF\n"
            "4 2 0\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "0.0 1.0 0.0\n"
            "0.0 0.0 1.0\n"
            "3 0 1 2\n"
            "3 0 1 3\n"
        )

        mesh = read_off(str(off_path))

        self.assertEqual(mesh.vertices.shape, (4, 3))
        self.assertEqual(mesh.faces.shape, (2, 3))

    def test_vertices_only_off(self):
        """Should handle OFF files with vertices but no faces."""
        off_path = self.tmp_path / "verts.off"
        off_path.write_text(
            "OFF\n"
            "3 0 0\n"
            "1.0 2.0 3.0\n"
            "4.0 5.0 6.0\n"
            "7.0 8.0 9.0\n"
        )

        mesh = read_off(str(off_path))

        self.assertIsNotNone(mesh)
        self.assertEqual(mesh.vertices.shape, (3, 3))
        self.assertIsNone(mesh.faces)

    def test_invalid_header_returns_none(self):
        """Non-OFF header should return None."""
        off_path = self.tmp_path / "bad.off"
        off_path.write_text("NOT_OFF\n1 0 0\n0.0 0.0 0.0\n")

        mesh = read_off(str(off_path))
        self.assertIsNone(mesh)

    def test_nonexistent_file_returns_none(self):
        """Missing file should return None (not raise)."""
        mesh = read_off(str(self.tmp_path / "nonexistent.off"))
        self.assertIsNone(mesh)

    def test_ngon_triangulation(self):
        """N-gons (quads etc.) should be triangulated via fan method."""
        off_path = self.tmp_path / "quad.off"
        off_path.write_text(
            "OFF\n"
            "4 1 0\n"
            "0.0 0.0 0.0\n"
            "1.0 0.0 0.0\n"
            "1.0 1.0 0.0\n"
            "0.0 1.0 0.0\n"
            "4 0 1 2 3\n"  # Quad face
        )

        mesh = read_off(str(off_path))

        # Quad should become 2 triangles
        self.assertIsNotNone(mesh.faces)
        self.assertEqual(mesh.faces.shape[0], 2)
        self.assertEqual(mesh.faces.shape[1], 3)


class TestComputeFaceNormals(tf.test.TestCase):
    """Tests for face normal computation."""

    def test_unit_triangle_normal(self):
        """Right triangle in XY plane should have Z-aligned normal."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)

        normals = compute_face_normals(vertices, faces)

        self.assertEqual(normals.shape, (1, 3))
        # Normal should be [0, 0, 1] (Z direction)
        np.testing.assert_allclose(np.abs(normals[0]), [0, 0, 1], atol=1e-5)

    def test_normals_are_unit_length(self):
        """Face normals should be unit length."""
        np.random.seed(42)
        vertices = np.random.randn(10, 3).astype(np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int32)

        normals = compute_face_normals(vertices, faces)
        lengths = np.linalg.norm(normals, axis=1)

        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


class TestSamplePointsFromMesh(tf.test.TestCase):
    """Tests for mesh-to-point-cloud sampling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        self.mesh = MeshData(
            vertices=self.vertices,
            faces=self.faces,
            face_normals=compute_face_normals(self.vertices, self.faces)
        )

    def test_correct_num_points(self):
        """Should return exactly num_points points."""
        points, normals = sample_points_from_mesh(self.mesh, num_points=100)
        self.assertEqual(points.shape[0], 100)
        self.assertEqual(points.shape[1], 3)

    def test_points_within_mesh_bounds(self):
        """Sampled points should be within vertex bounding box."""
        points, _ = sample_points_from_mesh(self.mesh, num_points=1000)

        min_bound = self.vertices.min(axis=0)
        max_bound = self.vertices.max(axis=0)

        for dim in range(3):
            self.assertAllGreaterEqual(points[:, dim], min_bound[dim] - 1e-5)
            self.assertAllLessEqual(points[:, dim], max_bound[dim] + 1e-5)

    def test_normals_unit_length(self):
        """Returned normals should be unit length."""
        points, normals = sample_points_from_mesh(
            self.mesh, num_points=100, compute_normals=True
        )

        self.assertIsNotNone(normals)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)

    def test_no_normals_when_disabled(self):
        """Should return None normals when compute_normals=False."""
        mesh_no_normals = MeshData(
            vertices=self.vertices,
            faces=self.faces
        )
        points, normals = sample_points_from_mesh(
            mesh_no_normals, num_points=100, compute_normals=False
        )
        self.assertIsNone(normals)

    def test_vertex_sampling_when_no_faces(self):
        """Without faces, should sample directly from vertices."""
        mesh_no_faces = MeshData(vertices=self.vertices)
        points, _ = sample_points_from_mesh(
            mesh_no_faces, num_points=10, compute_normals=False
        )
        self.assertEqual(points.shape, (10, 3))

    def test_points_dtype_float32(self):
        """Sampled points should be float32."""
        points, _ = sample_points_from_mesh(self.mesh, num_points=50)
        self.assertEqual(points.dtype, np.float32)


class TestPartitionPointCloud(tf.test.TestCase):
    """Tests for point cloud spatial partitioning."""

    def test_single_block(self):
        """Points within one block_size should create one block."""
        points = np.random.uniform(0, 0.5, (200, 3)).astype(np.float32)

        blocks = partition_point_cloud(points, block_size=1.0, min_points=10)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['points'].shape[1], 3)

    def test_multiple_blocks(self):
        """Spread-out points should create multiple blocks."""
        # Create two clusters far apart
        cluster1 = np.random.uniform(0, 0.5, (100, 3)).astype(np.float32)
        cluster2 = np.random.uniform(5, 5.5, (100, 3)).astype(np.float32)
        points = np.vstack([cluster1, cluster2])

        blocks = partition_point_cloud(points, block_size=1.0, min_points=10)

        self.assertGreaterEqual(len(blocks), 2)

    def test_min_points_filter(self):
        """Blocks with fewer than min_points should be excluded."""
        # Create a big cluster and a tiny cluster
        big_cluster = np.random.uniform(0, 0.5, (200, 3)).astype(np.float32)
        tiny_cluster = np.random.uniform(10, 10.1, (5, 3)).astype(np.float32)
        points = np.vstack([big_cluster, tiny_cluster])

        blocks = partition_point_cloud(points, block_size=1.0, min_points=10)

        # Tiny cluster should be filtered out
        total_points = sum(len(b['points']) for b in blocks)
        self.assertEqual(total_points, 200)

    def test_normals_partitioned(self):
        """Normals should be partitioned along with points."""
        np.random.seed(42)
        points = np.random.uniform(0, 0.5, (200, 3)).astype(np.float32)
        normals = np.random.randn(200, 3).astype(np.float32)

        blocks = partition_point_cloud(
            points, normals=normals, block_size=1.0, min_points=10
        )

        for block in blocks:
            self.assertIn('normals', block)
            self.assertEqual(block['normals'].shape[0], block['points'].shape[0])
            self.assertEqual(block['normals'].shape[1], 3)

    def test_all_points_accounted_for(self):
        """All points in valid blocks should appear exactly once."""
        np.random.seed(42)
        points = np.random.uniform(0, 3, (500, 3)).astype(np.float32)

        blocks = partition_point_cloud(points, block_size=1.0, min_points=1)

        total = sum(len(b['points']) for b in blocks)
        self.assertEqual(total, 500)


class TestSavePly(tf.test.TestCase):
    """Tests for PLY file writing."""

    @pytest.fixture(autouse=True)
    def inject_tmp_path(self, tmp_path):
        self.tmp_path = tmp_path

    def test_write_points_only(self):
        """Should write valid PLY with points only."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        ply_path = str(self.tmp_path / "test.ply")

        save_ply(ply_path, points)

        content = Path(ply_path).read_text()
        self.assertIn("ply", content)
        self.assertIn("element vertex 2", content)
        self.assertNotIn("property float nx", content)

    def test_write_points_with_normals(self):
        """Should write valid PLY with points and normals."""
        points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        normals = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32)
        ply_path = str(self.tmp_path / "test_normals.ply")

        save_ply(ply_path, points, normals)

        content = Path(ply_path).read_text()
        self.assertIn("property float nx", content)
        self.assertIn("property float ny", content)
        self.assertIn("property float nz", content)

    def test_roundtrip_off_to_ply(self):
        """OFF → sample → PLY should produce valid output."""
        # Write OFF
        off_path = str(self.tmp_path / "mesh.off")
        with open(off_path, 'w') as f:
            f.write("OFF\n4 2 0\n")
            f.write("0 0 0\n1 0 0\n0 1 0\n0 0 1\n")
            f.write("3 0 1 2\n3 0 1 3\n")

        # Read and sample
        mesh = read_off(off_path)
        self.assertIsNotNone(mesh)
        points, normals = sample_points_from_mesh(mesh, num_points=50)

        # Write PLY
        ply_path = str(self.tmp_path / "output.ply")
        save_ply(ply_path, points, normals)

        # Verify file exists and has content
        content = Path(ply_path).read_text()
        lines = content.strip().split('\n')
        # Header + 50 data lines
        header_end = lines.index('end_header')
        data_lines = lines[header_end + 1:]
        self.assertEqual(len(data_lines), 50)


if __name__ == '__main__':
    tf.test.main()
