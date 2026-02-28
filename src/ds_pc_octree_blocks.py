import argparse
from pathlib import Path
from typing import List

import tensorflow as tf

from .file_io import read_point_cloud as _read_point_cloud


class PointCloudProcessor:
    """Point cloud processing with TF 2.x operations."""

    def __init__(self, block_size: float = 1.0, min_points: int = 10):
        self.block_size = block_size
        self.min_points = min_points

    def read_point_cloud(self, file_path: str) -> tf.Tensor:
        """Read point cloud from PLY or OFF file."""
        vertices = _read_point_cloud(file_path)
        if vertices is None:
            raise ValueError(f"Failed to read point cloud: {file_path}")
        return tf.convert_to_tensor(vertices, dtype=tf.float32)

    def partition_point_cloud(self, points: tf.Tensor) -> List[tf.Tensor]:
        """Partition point cloud into blocks using TF operations."""
        # Compute bounds
        min_bound = tf.reduce_min(points, axis=0)

        # Compute grid indices
        grid_indices = tf.cast(
            tf.floor((points - min_bound) / self.block_size),
            tf.int32
        )

        # Create unique block keys
        block_keys = (
            grid_indices[:, 0] * 1000000 +
            grid_indices[:, 1] * 1000 +
            grid_indices[:, 2]
        )

        # Get unique blocks
        unique_keys, inverse_indices = tf.unique(block_keys)

        blocks = []
        for key in unique_keys:
            mask = tf.equal(block_keys, key)
            block_points = tf.boolean_mask(points, mask)

            if tf.shape(block_points)[0] >= self.min_points:
                blocks.append(block_points)

        return blocks

    def save_blocks(self, blocks: List[tf.Tensor], output_dir: str, base_name: str):
        """Save point cloud blocks to PLY files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, block in enumerate(blocks):
            file_path = output_dir / f"{base_name}_block_{i}.ply"
            points = block.numpy() if isinstance(block, tf.Tensor) else block

            with open(file_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("end_header\n")
                for point in points:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Partition point clouds into octree blocks."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input PLY file"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save output blocks"
    )
    parser.add_argument(
        "--block_size",
        type=float,
        default=1.0,
        help="Size of each block"
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=10,
        help="Minimum points per block"
    )

    args = parser.parse_args()

    processor = PointCloudProcessor(
        block_size=args.block_size,
        min_points=args.min_points
    )

    # Process point cloud
    points = processor.read_point_cloud(args.input)
    blocks = processor.partition_point_cloud(points)

    # Save blocks
    base_name = Path(args.input).stem
    processor.save_blocks(blocks, args.output_dir, base_name)

    print(f"Saved {len(blocks)} blocks to {args.output_dir}")

if __name__ == "__main__":
    main()
