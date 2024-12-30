import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict, Any

class OctreeCoder:
    def __init__(self, resolution: int = 64):
        """
        Initialize the OctreeCoder with a specific resolution.

        Args:
            resolution (int): The resolution of the voxel grid.
        """
        self.resolution = resolution

    def encode(self, point_cloud: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Encode a point cloud into an octree representation.

        Args:
            point_cloud (tf.Tensor): Input point cloud of shape (N, 3).

        Returns:
            Tuple[tf.Tensor, Dict[str, Any]]: Binary voxel grid and metadata.
        """
        grid = tf.zeros((self.resolution, self.resolution, self.resolution), dtype=tf.bool)
        min_bounds = tf.reduce_min(point_cloud, axis=0)
        max_bounds = tf.reduce_max(point_cloud, axis=0)
        scale = max_bounds - min_bounds

        scaled_points = (point_cloud - min_bounds) / scale * (self.resolution - 1)
        indices = tf.cast(tf.clip_by_value(scaled_points, 0, self.resolution - 1), tf.int32)

        # Create update values for scatter_nd
        updates = tf.ones(tf.shape(indices)[0], dtype=tf.bool)
        grid = tf.tensor_scatter_nd_update(grid, indices, updates)

        metadata = {
            "min_bounds": min_bounds.numpy(),
            "max_bounds": max_bounds.numpy(),
            "scale": scale.numpy()
        }

        return grid, metadata

    def decode(self, grid: tf.Tensor, metadata: Dict[str, Any]) -> tf.Tensor:
        """
        Decode an octree representation back into a point cloud.

        Args:
            grid (tf.Tensor): Binary voxel grid.
            metadata (Dict[str, Any]): Metadata containing min bounds, max bounds, and scale.

        Returns:
            tf.Tensor: Decoded point cloud.
        """
        indices = tf.cast(tf.where(grid), tf.float32)
        scale = tf.constant(metadata["scale"], dtype=tf.float32)
        min_bounds = tf.constant(metadata["min_bounds"], dtype=tf.float32)

        points = tf.cast(indices, tf.float32) / tf.cast(self.resolution - 1, tf.float32)
        points = points * scale + min_bounds
        return points

    def partition_octree(
        self,
        point_cloud: tf.Tensor,
        bbox: Tuple[float, float, float, float, float, float],
        level: int
    ) -> List[Tuple[tf.Tensor, Tuple[float, float, float, float, float, float]]]:
        """
        Partition a point cloud into octree blocks recursively.

        Args:
            point_cloud (tf.Tensor): Input point cloud of shape (N, 3).
            bbox (Tuple[float, float, float, float, float, float]):
                Bounding box as (xmin, xmax, ymin, ymax, zmin, zmax).
            level (int): Level of recursion for octree partitioning.

        Returns:
            List[Tuple[tf.Tensor, Tuple[float, float, float, float, float, float]]]:
                A list of tuples, each containing a point cloud block and its bounding box.
        """
        if level == 0 or tf.shape(point_cloud)[0] == 0:
            return [(point_cloud, bbox)]

        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        blocks = []
        for x_range, y_range, z_range in [
            ((xmin, xmid), (ymin, ymid), (zmin, zmid)),
            ((xmin, xmid), (ymin, ymid), (zmid, zmax)),
            ((xmin, xmid), (ymid, ymax), (zmin, zmid)),
            ((xmin, xmid), (ymid, ymax), (zmid, zmax)),
            ((xmid, xmax), (ymin, ymid), (zmin, zmid)),
            ((xmid, xmax), (ymin, ymid), (zmid, zmax)),
            ((xmid, xmax), (ymid, ymax), (zmin, zmid)),
            ((xmid, xmax), (ymid, ymax), (zmid, zmax)),
        ]:
            epsilon = 1e-10

            # Compute conditions for x, y, and z separately
            x_cond = tf.logical_and(
                point_cloud[:, 0] >= x_range[0] - epsilon,
                point_cloud[:, 0] <= x_range[1] + epsilon
            )
            y_cond = tf.logical_and(
                point_cloud[:, 1] >= y_range[0] - epsilon,
                point_cloud[:, 1] <= y_range[1] + epsilon
            )
            z_cond = tf.logical_and(
                point_cloud[:, 2] >= z_range[0] - epsilon,
                point_cloud[:, 2] <= z_range[1] + epsilon
            )

            # Combine all conditions
            mask = tf.logical_and(x_cond, tf.logical_and(y_cond, z_cond))

            in_block = tf.boolean_mask(point_cloud, mask)

            if tf.shape(in_block)[0] > 0:
                child_bbox = (
                    x_range[0], x_range[1],
                    y_range[0], y_range[1],
                    z_range[0], z_range[1]
                )
                blocks.extend(self.partition_octree(in_block, child_bbox, level - 1))

        return blocks

if __name__ == "__main__":
    # Example usage
    point_cloud = tf.random.uniform((1000, 3), 0, 10)  # Random point cloud
    coder = OctreeCoder(resolution=64)

    # Encoding and decoding
    grid, metadata = coder.encode(point_cloud)
    decoded_points = coder.decode(grid, metadata)

    print("Original Points (Sample):", point_cloud[:5].numpy())
    print("Decoded Points (Sample):", decoded_points[:5].numpy())

    # Partitioning into octree blocks
    initial_bbox = (0, 10, 0, 10, 0, 10)  # Bounding box
    level = 3
    blocks = coder.partition_octree(point_cloud, initial_bbox, level)
    print(f"Number of blocks: {len(blocks)}")
    for i, (block, bbox) in enumerate(blocks[:2]):
        print(f"\nBlock {i}:")
        print(f"Bounding box: {bbox}")
        print(f"Number of points: {tf.shape(block)[0].numpy()}")
        if tf.shape(block)[0] > 0:
            print(f"Point range: [{tf.reduce_min(block, axis=0).numpy()}, {tf.reduce_max(block, axis=0).numpy()}]")