import argparse
from typing import Optional

import numpy as np


def load_point_cloud(file_path: str) -> Optional[np.ndarray]:
    """
    Load a point cloud from a PLY file.

    Args:
        file_path (str): Path to the PLY file.

    Returns:
        np.ndarray: An array of shape (N, 3) containing the points of the cloud.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if "end_header\n" not in lines:
                print(f"Error: no end_header found in {file_path}")
                return None
            start = lines.index("end_header\n") + 1
            points = [list(map(float, line.split()[:3])) for line in lines[start:]]
            return np.array(points, dtype=np.float32)
    except Exception as e:
        print(f"Error loading point cloud from {file_path}: {e}")
        return None

def load_colors(file_path: str) -> Optional[np.ndarray]:
    """
    Load colors from a PLY file.

    Args:
        file_path (str): Path to the PLY file.

    Returns:
        np.ndarray: An array of shape (N, 3) containing RGB colors.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if "end_header\n" not in lines:
                print(f"Error: no end_header found in {file_path}")
                return None
            start = lines.index("end_header\n") + 1
            colors = [list(map(float, line.split()[3:6])) for line in lines[start:] if len(line.split()) > 3]
            return np.array(colors, dtype=np.float32) / 255.0
    except Exception as e:
        print(f"Error loading colors from {file_path}: {e}")
        return None

def transfer_colors(source_points: np.ndarray, source_colors: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Transfer colors from a source point cloud to a target point cloud based on nearest neighbors.

    Args:
        source_points (np.ndarray): Array of shape (N, 3) for source point cloud.
        source_colors (np.ndarray): Array of shape (N, 3) for source colors.
        target_points (np.ndarray): Array of shape (M, 3) for target point cloud.

    Returns:
        np.ndarray: Array of shape (M, 3) containing colors for the target cloud.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(source_points)
    _, indices = tree.query(target_points, k=1)
    return source_colors[indices]

def save_colored_point_cloud(file_path: str, points: np.ndarray, colors: np.ndarray):
    """
    Save a colored point cloud to a PLY file.

    Args:
        file_path (str): Path to the output PLY file.
        points (np.ndarray): Array of shape (N, 3) for point coordinates.
        colors (np.ndarray): Array of shape (N, 3) for RGB colors.
    """
    try:
        with open(file_path, 'w') as file:
            file.write("ply\n")
            file.write("format ascii 1.0\n")
            file.write(f"element vertex {points.shape[0]}\n")
            file.write("property float x\n")
            file.write("property float y\n")
            file.write("property float z\n")
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
            file.write("end_header\n")

            for point, color in zip(points, colors):
                r, g, b = (color * 255).astype(int)
                file.write(f"{point[0]} {point[1]} {point[2]} {r} {g} {b}\n")
    except Exception as e:
        print(f"Error saving colored point cloud to {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transfer colors from one point cloud to another.")
    parser.add_argument("source", type=str, help="Path to the source PLY file with colors.")
    parser.add_argument("target", type=str, help="Path to the target PLY file without colors.")
    parser.add_argument("output", type=str, help="Path to the output PLY file with transferred colors.")

    args = parser.parse_args()

    source_points = load_point_cloud(args.source)
    source_colors = load_colors(args.source)
    target_points = load_point_cloud(args.target)

    if source_points is None or source_colors is None or target_points is None:
        print("Error: Unable to load one or more point clouds.")
        return

    target_colors = transfer_colors(source_points, source_colors, target_points)
    save_colored_point_cloud(args.output, target_points, target_colors)
    print(f"Color transfer complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
