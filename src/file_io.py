import logging
from pathlib import Path
from typing import Optional

import numpy as np


def read_off(file_path: str) -> Optional[np.ndarray]:
    """Read vertex coordinates from an OFF file.

    Args:
        file_path: Path to the OFF file.

    Returns:
        Numpy array of shape (N, 3) with vertex positions, or None on error.
    """
    try:
        with open(file_path, 'r') as f:
            header = f.readline().strip()
            if header != "OFF":
                raise ValueError("Not a valid OFF file")

            n_verts, _, _ = map(int, f.readline().strip().split())

            vertices = []
            for _ in range(n_verts):
                values = f.readline().strip().split()
                vertices.append([float(values[0]), float(values[1]), float(values[2])])

            return np.array(vertices, dtype=np.float32)
    except Exception as e:
        logging.error(f"Error reading OFF file {file_path}: {e}")
        return None


def read_ply(file_path: str) -> Optional[np.ndarray]:
    """Read vertex coordinates from an ASCII PLY file.

    Args:
        file_path: Path to the PLY file.

    Returns:
        Numpy array of shape (N, 3) with vertex positions, or None on error.
    """
    try:
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            if line != "ply":
                raise ValueError("Not a valid PLY file")

            n_verts = 0
            while True:
                line = f.readline().strip()
                if line == "end_header":
                    break
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])

            if n_verts == 0:
                return np.array([], dtype=np.float32).reshape(0, 3)

            vertices = []
            for _ in range(n_verts):
                values = f.readline().strip().split()
                vertices.append([float(values[0]), float(values[1]), float(values[2])])

            return np.array(vertices, dtype=np.float32)
    except Exception as e:
        logging.error(f"Error reading PLY file {file_path}: {e}")
        return None


def read_point_cloud(file_path: str) -> Optional[np.ndarray]:
    """Read a point cloud from a file, dispatching by extension.

    Supports .off and .ply formats.

    Args:
        file_path: Path to the point cloud file.

    Returns:
        Numpy array of shape (N, 3) with vertex positions, or None on error.
    """
    ext = Path(file_path).suffix.lower()
    if ext == '.off':
        return read_off(file_path)
    elif ext == '.ply':
        return read_ply(file_path)
    else:
        logging.error(f"Unsupported file format: {ext}")
        return None
