import tensorflow as tf
import argparse
import os
from typing import List, Tuple
from pathlib import Path

class PointCloudProcessor:
    """Point cloud processing with TF 2.x operations."""
    
    def __init__(self, block_size: float = 1.0, min_points: int = 10):
        self.block_size = block_size
        self.min_points = min_points

    @tf.function
    def read_point_cloud(self, file_path: str) -> tf.Tensor:
        """Read point cloud using TF file operations."""
        raw_data = tf.io.read_file(file_path)
        lines = tf.strings.split(raw_data, '\n')[1:]  # Skip header
        
        def parse_line(line):
            values = tf.strings.split(line)
            return tf.strings.to_number(values[:3], out_type=tf.float32)
            
        points = tf.map_fn(
            parse_line,
            lines,
            fn_output_signature=tf.float32
        )
        return points

    def partition_point_cloud(self, points: tf.Tensor) -> List[tf.Tensor]:
        """Partition point cloud into blocks using TF operations."""
        # Compute bounds
        min_bound = tf.reduce_min(points, axis=0)
        max_bound = tf.reduce_max(points, axis=0)
        
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
            
            header = [
                "ply",
                "format ascii 1.0",
                f"element vertex {block.shape[0]}",
                "property float x",
                "property float y",
                "property float z",
                "end_header"
            ]
            
            with open(file_path, 'w') as f:
                f.write('\n'.join(header) + '\n')
                
                # Convert points to strings and write
                points_str = tf.strings.reduce_join(
                    tf.strings.as_string(block),
                    axis=1,
                    separator=' '
                )
                points_str = tf.strings.join([points_str, tf.constant('\n')], '')
                tf.io.write_file(
                    str(file_path),
                    tf.strings.join([tf.strings.join(header, '\n'), points_str])
                )

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