import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MeshData:
    """Container for mesh data including vertices, faces, and normals."""
    vertices: np.ndarray
    faces: Optional[np.ndarray] = None
    vertex_normals: Optional[np.ndarray] = None
    face_normals: Optional[np.ndarray] = None

def read_off(file_path: str) -> Optional[MeshData]:
    """
    Enhanced OFF file reader that handles vertices and faces.

    Args:
        file_path: Path to the OFF file.

    Returns:
        MeshData object containing mesh information.
    """
    try:
        with open(file_path, 'r') as file:
            # Read header
            header = file.readline().strip()
            if header != "OFF":
                raise ValueError("Not a valid OFF file")

            # Read counts
            n_verts, n_faces, _ = map(int, file.readline().strip().split())

            # Read vertices
            vertices = []
            for _ in range(n_verts):
                vertex = list(map(float, file.readline().strip().split()))
                vertices.append(vertex)
            vertices = np.array(vertices, dtype=np.float32)

            # Read faces if present, triangulating n-gons via fan triangulation
            faces = None
            if n_faces > 0:
                triangles = []
                for _ in range(n_faces):
                    indices = list(map(int, file.readline().strip().split()[1:]))
                    if len(indices) < 3:
                        continue
                    # Fan triangulation: (v0, v1, v2), (v0, v2, v3), ...
                    for i in range(1, len(indices) - 1):
                        triangles.append([indices[0], indices[i], indices[i + 1]])
                if triangles:
                    faces = np.array(triangles, dtype=np.int32)

            # Compute face normals if faces are present
            face_normals = None
            if faces is not None:
                face_normals = compute_face_normals(vertices, faces)

            return MeshData(vertices=vertices, faces=faces, face_normals=face_normals)
    except Exception as e:
        logging.error(f"Error reading OFF file {file_path}: {e}")
        return None

def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute face normals for a mesh."""
    normals = []
    for face in faces:
        v1, v2, v3 = vertices[face]
        normal = np.cross(v2 - v1, v3 - v1)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        normals.append(normal)
    return np.array(normals, dtype=np.float32)

def sample_points_from_mesh(
    mesh_data: MeshData,
    num_points: int = 2048,
    compute_normals: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Sample points uniformly from mesh surface with optional normal computation.

    Args:
        mesh_data: MeshData object containing mesh information.
        num_points: Number of points to sample.
        compute_normals: Whether to compute normals for sampled points.

    Returns:
        Tuple of points array and optionally normals array.
    """
    if mesh_data.faces is not None and len(mesh_data.faces) > 0:
        # Sample from faces using area-weighted barycentric sampling
        v1s = mesh_data.vertices[mesh_data.faces[:, 0]]
        v2s = mesh_data.vertices[mesh_data.faces[:, 1]]
        v3s = mesh_data.vertices[mesh_data.faces[:, 2]]

        areas = np.linalg.norm(np.cross(v2s - v1s, v3s - v1s), axis=1) / 2
        probabilities = areas / areas.sum()

        # Sample faces by area
        indices = np.random.choice(len(areas), num_points, p=probabilities)

        # Generate random barycentric coordinates for uniform sampling
        r1 = np.sqrt(np.random.random(num_points))
        r2 = np.random.random(num_points)
        points = (
            (1 - r1)[:, None] * v1s[indices]
            + (r1 * (1 - r2))[:, None] * v2s[indices]
            + (r1 * r2)[:, None] * v3s[indices]
        )

        # Get corresponding normals if requested
        normals = None
        if compute_normals and mesh_data.face_normals is not None:
            normals = mesh_data.face_normals[indices]
    else:
        # If no faces, sample directly from vertices
        if len(mesh_data.vertices) >= num_points:
            indices = np.random.choice(len(mesh_data.vertices), num_points, replace=False)
        else:
            indices = np.random.choice(len(mesh_data.vertices), num_points, replace=True)
        points = mesh_data.vertices[indices]

        normals = None
        if compute_normals and mesh_data.vertex_normals is not None:
            normals = mesh_data.vertex_normals[indices]

    return points.astype(np.float32), normals

def partition_point_cloud(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    block_size: float = 1.0,
    min_points: int = 100
) -> List[Dict[str, np.ndarray]]:
    """
    Partition point cloud into octree blocks.

    Args:
        points: Points array of shape (N, 3).
        normals: Optional normals array of shape (N, 3).
        block_size: Size of octree blocks.
        min_points: Minimum points per block.

    Returns:
        List of dictionaries containing points and normals for each block.
    """
    # Compute point cloud bounds
    min_bound = np.min(points, axis=0)

    # Initialize blocks
    blocks = []

    # Assign points to grid cells
    grid_indices = np.floor((points - min_bound) / block_size).astype(int)

    # Process each occupied grid cell
    unique_indices = np.unique(grid_indices, axis=0)
    for idx in unique_indices:
        # Find points in this cell
        mask = np.all(grid_indices == idx, axis=1)
        block_points = points[mask]

        # Only keep blocks with enough points
        if len(block_points) >= min_points:
            block_data = {'points': block_points}
            if normals is not None:
                block_data['normals'] = normals[mask]
            blocks.append(block_data)

    return blocks

def save_ply(
    file_path: str,
    points: np.ndarray,
    normals: Optional[np.ndarray] = None
):
    """
    Save points and optional normals to PLY file.

    Args:
        file_path: Output PLY file path.
        points: Points array of shape (N, 3).
        normals: Optional normals array of shape (N, 3).
    """
    try:
        with open(file_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if normals is not None:
                f.write("property float nx\n")
                f.write("property float ny\n")
                f.write("property float nz\n")

            f.write("end_header\n")

            # Write data
            for i in range(len(points)):
                line = f"{points[i,0]} {points[i,1]} {points[i,2]}"
                if normals is not None:
                    line += f" {normals[i,0]} {normals[i,1]} {normals[i,2]}"
                f.write(line + "\n")

    except Exception as e:
        logging.error(f"Error saving PLY file {file_path}: {e}")

def convert_mesh_to_point_cloud(
    input_path: str,
    output_path: str,
    num_points: int = 2048,
    compute_normals: bool = True,
    partition_blocks: bool = False,
    block_size: float = 1.0,
    min_points_per_block: int = 100
):
    """
    Convert mesh to point cloud with optional features.

    Args:
        input_path: Input OFF file path.
        output_path: Output PLY file path.
        num_points: Number of points to sample.
        compute_normals: Whether to compute normals.
        partition_blocks: Whether to partition into octree blocks.
        block_size: Size of octree blocks.
        min_points_per_block: Minimum points per block.
    """
    # Read mesh
    mesh_data = read_off(input_path)
    if mesh_data is None:
        return

    # Sample points
    points, normals = sample_points_from_mesh(
        mesh_data,
        num_points,
        compute_normals
    )

    if partition_blocks:
        # Partition into blocks
        blocks = partition_point_cloud(
            points,
            normals,
            block_size,
            min_points_per_block
        )

        # Save each block
        output_base = Path(output_path).with_suffix('')
        for i, block in enumerate(blocks):
            block_path = f"{output_base}_block_{i}.ply"
            save_ply(
                block_path,
                block['points'],
                block.get('normals')
            )
        logging.info(f"Saved {len(blocks)} blocks")
    else:
        # Save single point cloud
        save_ply(output_path, points, normals)
        logging.info(f"Saved point cloud to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D mesh (OFF) to point cloud (PLY) with enhanced features."
    )
    parser.add_argument("input", type=str, help="Path to input OFF file")
    parser.add_argument("output", type=str, help="Path to output PLY file")
    parser.add_argument("--num_points", type=int, default=2048,
                      help="Number of points to sample (default: 2048)")
    parser.add_argument("--no_normals", action="store_false", dest="compute_normals",
                      help="Skip normal computation")
    parser.add_argument("--partition", action="store_true",
                      help="Partition into octree blocks")
    parser.add_argument("--block_size", type=float, default=1.0,
                      help="Size of octree blocks (default: 1.0)")
    parser.add_argument("--min_points", type=int, default=100,
                      help="Minimum points per block (default: 100)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Convert mesh to point cloud
    convert_mesh_to_point_cloud(
        args.input,
        args.output,
        args.num_points,
        args.compute_normals,
        args.partition,
        args.block_size,
        args.min_points
    )

if __name__ == "__main__":
    main()
