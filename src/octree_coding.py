from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import tensorflow as tf


@dataclass
class OctreeConfig:
    """Configuration for octree encoding."""
    resolution: int = 64
    min_points: int = 100
    epsilon: float = 1e-10

class OctreeCoder(tf.keras.layers.Layer):
    """TensorFlow 2.x implementation of octree coding."""

    def __init__(self, config: OctreeConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

    @tf.function
    def encode(self, point_cloud: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """Encode point cloud into octree representation."""
        # Create empty grid
        grid = tf.zeros(
            (self.config.resolution,) * 3,
            dtype=tf.bool
        )

        # Calculate bounds
        min_bounds = tf.reduce_min(point_cloud, axis=0)
        max_bounds = tf.reduce_max(point_cloud, axis=0)
        scale = max_bounds - min_bounds

        # Handle zero scales
        scale = tf.where(
            tf.equal(scale, 0),
            tf.ones_like(scale) * self.config.epsilon,
            scale
        )

        # Scale points to grid resolution
        scaled_points = (point_cloud - min_bounds) / scale * tf.cast(
            self.config.resolution - 1,
            tf.float32
        )
        indices = tf.cast(
            tf.clip_by_value(
                scaled_points,
                0,
                self.config.resolution - 1
            ),
            tf.int32
        )

        # Create update values
        updates = tf.ones(tf.shape(indices)[0], dtype=tf.bool)

        # Update grid
        grid = tf.tensor_scatter_nd_update(grid, indices, updates)

        metadata = {
            'min_bounds': min_bounds.numpy(),
            'max_bounds': max_bounds.numpy(),
            'scale': scale.numpy()
        }

        return grid, metadata

    @tf.function
    def decode(self,
               grid: tf.Tensor,
               metadata: Dict[str, Any]) -> tf.Tensor:
        """Decode octree representation to point cloud."""
        # Get occupied positions
        indices = tf.cast(
            tf.where(grid),
            tf.float32
        )

        # Scale back to original space
        scale = tf.constant(metadata['scale'], dtype=tf.float32)
        min_bounds = tf.constant(metadata['min_bounds'], dtype=tf.float32)

        points = (
            indices / tf.cast(self.config.resolution - 1, tf.float32)
        ) * scale + min_bounds

        return points

    @tf.function
    def partition_octree(
        self,
        point_cloud: tf.Tensor,
        bbox: Tuple[float, float, float, float, float, float],
        level: tf.Tensor
    ) -> List[Tuple[tf.Tensor, Tuple[float, float, float, float, float, float]]]:
        """Partition point cloud into octree blocks."""
        if tf.equal(level, 0) or tf.equal(tf.shape(point_cloud)[0], 0):
            return [(point_cloud, bbox)]

        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        blocks = []
        ranges = [
            ((xmin, xmid), (ymin, ymid), (zmin, zmid)),
            ((xmin, xmid), (ymin, ymid), (zmid, zmax)),
            ((xmin, xmid), (ymid, ymax), (zmin, zmid)),
            ((xmin, xmid), (ymid, ymax), (zmid, zmax)),
            ((xmid, xmax), (ymin, ymid), (zmin, zmid)),
            ((xmid, xmax), (ymin, ymid), (zmid, zmax)),
            ((xmid, xmax), (ymid, ymax), (zmin, zmid)),
            ((xmid, xmax), (ymid, ymax), (zmid, zmax))
        ]

        for x_range, y_range, z_range in ranges:
            # Compute conditions
            x_cond = tf.logical_and(
                point_cloud[:, 0] >= x_range[0] - self.config.epsilon,
                point_cloud[:, 0] <= x_range[1] + self.config.epsilon
            )
            y_cond = tf.logical_and(
                point_cloud[:, 1] >= y_range[0] - self.config.epsilon,
                point_cloud[:, 1] <= y_range[1] + self.config.epsilon
            )
            z_cond = tf.logical_and(
                point_cloud[:, 2] >= z_range[0] - self.config.epsilon,
                point_cloud[:, 2] <= z_range[1] + self.config.epsilon
            )

            # Combine conditions
            mask = tf.logical_and(x_cond, tf.logical_and(y_cond, z_cond))

            # Get points in block
            in_block = tf.boolean_mask(point_cloud, mask)

            if tf.shape(in_block)[0] > 0:
                child_bbox = (
                    x_range[0], x_range[1],
                    y_range[0], y_range[1],
                    z_range[0], z_range[1]
                )
                blocks.extend(
                    self.partition_octree(
                        in_block,
                        child_bbox,
                        level - 1
                    )
                )

        return blocks
