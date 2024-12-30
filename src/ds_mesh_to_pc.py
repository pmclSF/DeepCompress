import numpy as np
import argparse
from pathlib import Path
from typing import Optional


def read_off(file_path: str) -> Optional[np.ndarray]:
    """
    Reads a 3D mesh from an OFF file and returns the vertices.

    Args:
        file_path (str): Path to the OFF file.

    Returns:
        np.ndarray: An array of shape (N, 3) containing the vertices of the mesh.
    """
    try:
        with open(file_path, 'r') as file:
            header = file.readline().strip()
            if header != "OFF":
                raise ValueError("Not a valid OFF file")

            n_verts, n_faces, _ = map(int, file.readline().strip().split())
            vertices = []

            for _ in range(n_verts):
                vertex = list(map(float, file.readline().strip().split()))
                vertices.append(vertex)

            return np.array(vertices, dtype=np.float32)

    except Exception as e:
        print(f"Error reading OFF file {file_path}: {e}")
        return None


def sample_points_from_mesh(vertices: np.ndarray, num_points: int = 2048) -> np.ndarray:
    """
    Uniformly samples points from the surface of the mesh represented by its vertices.

    Args:
        vertices (np.ndarray): Array of shape (N, 3) representing vertices of the mesh.
        num_points (int): Number of points to sample.

    Returns:
        np.ndarray: Sampled points as a numpy array of shape (num_points, 3).
    """
    # Placeholder implementation for uniform sampling; can be replaced with an actual algorithm.
    indices = np.random.choice(vertices.shape[0], size=num_points, replace=False)
    return vertices[indices]


def save_ply(file_path: str, point_cloud: np.ndarray):
    """
    Saves a point cloud to a PLY file.

    Args:
        file_path (str): Path to the output PLY file.
        point_cloud (np.ndarray): An array of shape (N, 3) containing the points.
    """
    try:
        with open(file_path, 'w') as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {point_cloud.shape[0]}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("end_header\n")

            for point in point_cloud:
                file.write(f"{point[0]} {point[1]} {point[2]}\n")

    except Exception as e:
        print(f"Error saving PLY file {file_path}: {e}")


def convert_mesh_to_point_cloud(input_path: str, output_path: str, num_points: int = 2048):
    """
    Converts a 3D mesh in OFF format to a point cloud and saves it in PLY format.

    Args:
        input_path (str): Path to the input OFF file.
        output_path (str): Path to the output PLY file.
        num_points (int): Number of points to sample from the mesh.
    """
    vertices = read_off(input_path)
    if vertices is not None:
        sampled_points = sample_points_from_mesh(vertices, num_points=num_points)
        save_ply(output_path, sampled_points)
        print(f"Conversion successful: {output_path}")
    else:
        print(f"Conversion failed for {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert 3D mesh (OFF) to point cloud (PLY) with sampling.")
    parser.add_argument("input", type=str, help="Path to the input OFF file.")
    parser.add_argument("output", type=str, help="Path to the output PLY file.")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points to sample from the mesh (default: 2048).")

    args = parser.parse_args()
    convert_mesh_to_point_cloud(args.input, args.output, args.num_points)


if __name__ == "__main__":
    main()
