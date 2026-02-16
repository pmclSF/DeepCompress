import os
import argparse
import tensorflow as tf

def count_points_in_block(file_path):
    """
    Counts the number of points in a block from a `.ply` file.

    Args:
        file_path (str): Path to the `.ply` file.

    Returns:
        int: Number of points in the block.
    """
    with open(file_path, "r") as f:
        header = True
        count = 0
        for line in f:
            if header:
                if line.startswith("end_header"):
                    header = False
                continue
            count += 1
        return count

def prioritize_blocks(input_dir, output_dir, num_blocks, criteria="points"):
    """
    Select and prioritize blocks based on the specified criteria.

    Args:
        input_dir (str): Directory containing block `.ply` files.
        output_dir (str): Directory to save prioritized blocks.
        num_blocks (int): Number of blocks to prioritize.
        criteria (str): Criteria for ranking blocks (default: "points").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Gather all block files and their point counts
    blocks = [
        (block, count_points_in_block(os.path.join(input_dir, block)))
        for block in os.listdir(input_dir) if block.endswith(".ply")
    ]

    # Sort blocks based on the specified criteria (default is point count)
    blocks = sorted(blocks, key=lambda x: x[1], reverse=True)

    # Select the top N blocks
    selected_blocks = blocks[:num_blocks]

    # Copy prioritized blocks to the output directory, maintaining filenames
    for block, count in selected_blocks:
        input_path = os.path.join(input_dir, block)
        output_path = os.path.join(output_dir, block)
        tf.io.gfile.copy(input_path, output_path, overwrite=True)
        print(f"Selected block {block} with {count} points.")

def main():
    parser = argparse.ArgumentParser(description="Prioritize and select the most relevant blocks for training.")
    parser.add_argument("input_dir", type=str, help="Directory containing block `.ply` files.")
    parser.add_argument("output_dir", type=str, help="Directory to save prioritized blocks.")
    parser.add_argument("num_blocks", type=int, help="Number of blocks to prioritize.")
    parser.add_argument("--criteria", type=str, default="points", choices=["points"],
                        help="Criteria for prioritization (default: 'points').")
    args = parser.parse_args()

    prioritize_blocks(args.input_dir, args.output_dir, args.num_blocks, args.criteria)

if __name__ == "__main__":
    main()
