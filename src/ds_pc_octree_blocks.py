import tensorflow as tf
import argparse
import os

def read_point_cloud(file_path):
    """
    Reads a point cloud from a .ply file into a TensorFlow tensor.
    Args:
        file_path (str): Path to the .ply file.

    Returns:
        tf.Tensor: A tensor of shape (N, 3) containing point coordinates.
    """
    point_cloud = []
    with open(file_path, "r") as f:
        header = True
        for line in f:
            if header:
                if line.startswith("end_header"):
                    header = False
                continue
            point_cloud.append([float(v) for v in line.strip().split()[:3]])
    return tf.convert_to_tensor(point_cloud, dtype=tf.float32)

def partition_point_cloud(points, block_size, min_points):
    """
    Partitions a point cloud into octree blocks.

    Args:
        points (tf.Tensor): Input point cloud of shape (N, 3).
        block_size (float): Size of each block.
        min_points (int): Minimum number of points per block.

    Returns:
        list: A list of blocks, where each block is a Tensor of points.
    """
    # Compute grid bounds
    min_bound = tf.reduce_min(points, axis=0)
    grid_indices = tf.cast(tf.math.floor((points - min_bound) / block_size), tf.int32)

    # Combine indices into unique keys
    unique_indices, idx = tf.unique(
        grid_indices[:, 0] * 1000000 +
        grid_indices[:, 1] * 1000 +
        grid_indices[:, 2]
    )

    # Group points into blocks
    blocks = []
    for unique_idx in unique_indices:
        mask = idx == unique_idx
        block = tf.boolean_mask(points, mask)
        if tf.shape(block)[0] >= min_points:
            blocks.append(block)

    return blocks

def save_blocks(blocks, output_dir, base_name):
    """
    Saves each block as a .ply file.

    Args:
        blocks (list): List of blocks to save.
        output_dir (str): Directory to save the blocks.
        base_name (str): Base name for block files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, block in enumerate(blocks):
        file_path = os.path.join(output_dir, f"{base_name}_block_{i}.ply")
        with open(file_path, "w") as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {block.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            # Write points
            for point in block:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

def main():
    parser = argparse.ArgumentParser(description="Partition a point cloud into octree blocks.")
    parser.add_argument("input", type=str, help="Path to the input .ply file.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output blocks.")
    parser.add_argument("--block_size", type=float, default=1.0, help="Size of each block (default: 1.0).")
    parser.add_argument("--min_points", type=int, default=10, help="Minimum points per block (default: 10).")
    args = parser.parse_args()

    # Read the point cloud
    points = read_point_cloud(args.input)

    # Partition the point cloud
    blocks = partition_point_cloud(points, args.block_size, args.min_points)

    # Save the blocks
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    save_blocks(blocks, args.output_dir, base_name)

    print(f"Saved {len(blocks)} blocks to {args.output_dir}")

if __name__ == "__main__":
    main()
