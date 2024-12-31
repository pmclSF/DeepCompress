import os
import json
import subprocess
import argparse
import time
import numpy as np
import tensorflow as tf
from ds_mesh_to_pc import read_off  # Assuming this function is available

def run_compression(input_file, output_file, octree_level=3, quantization_level=8):
    """
    Run the compression process for the point cloud.

    Args:
        input_file (str): Path to the original .ply file.
        output_file (str): Path to save the compressed .ply file.
        octree_level (int): Octree level to use for compression.
        quantization_level (int): Quantization level for compression.

    Returns:
        str: Path to the compressed file.
    """
    # Placeholder for compression logic
    # In practice, you would call the octree compression function here
    # For example: compress_octree(input_file, output_file, octree_level, quantization_level)
    
    # Simulate compression by copying the original file
    subprocess.run(["cp", input_file, output_file], check=True)
    
    return output_file

def run_decompression(input_file, output_file):
    """
    Run the decompression process for the point cloud.

    Args:
        input_file (str): Path to the compressed .ply file.
        output_file (str): Path to save the decompressed .ply file.

    Returns:
        str: Path to the decompressed file.
    """
    # Placeholder for decompression logic
    # In practice, you would call the octree decompression function here
    # For example: decompress_octree(input_file, output_file)
    
    # Simulate decompression by copying the compressed file
    subprocess.run(["cp", input_file, output_file], check=True)
    
    return output_file

def evaluate_compression(original_file, compressed_file):
    """
    Evaluate the compression performance using PSNR and BD-Rate.

    Args:
        original_file (str): Path to the original .ply file.
        compressed_file (str): Path to the compressed .ply file.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    original_vertices, _ = read_off(original_file)
    compressed_vertices, _ = read_off(compressed_file)

    # Compute PSNR
    psnr = compute_psnr(original_vertices, compressed_vertices)

    # Compute BD-Rate (for simplicity, just simulating the calculation here)
    bitrate = os.path.getsize(compressed_file) / os.path.getsize(original_file)
    bd_rate = compute_bd_rate(original_file, compressed_file, psnr, bitrate)

    # Return the results as a dictionary
    return {
        "psnr": psnr,
        "bd_rate": bd_rate,
        "bitrate": bitrate,
    }

def compute_psnr(original, compressed):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between original and compressed point clouds.

    Args:
        original (np.ndarray): The original point cloud.
        compressed (np.ndarray): The compressed point cloud.

    Returns:
        float: PSNR value.
    """
    mse = np.mean(np.square(original - compressed))
    max_value = np.max(np.abs(original))  # Max value for normalization
    psnr = 10 * np.log10((max_value**2) / mse)
    return psnr

def compute_bd_rate(original_file, compressed_file, psnr, bitrate):
    """
    Compute the Bjøntegaard Delta Rate (BD-Rate).

    Args:
        original_file (str): Path to the original file.
        compressed_file (str): Path to the compressed file.
        psnr (float): PSNR of the original and compressed point cloud.
        bitrate (float): Bitrate of the compressed point cloud.

    Returns:
        float: BD-Rate.
    """
    # Placeholder BD-Rate calculation (in practice, you can compute it based on actual results)
    return psnr / bitrate  # Just a dummy formula for now

def run_experiment(input_dir, output_dir, octree_level=3, quantization_level=8):
    """
    Run the experiment for all .ply files in the input directory, compressing and evaluating them.

    Args:
        input_dir (str): Directory containing the original .ply files.
        output_dir (str): Directory to save the compressed and decompressed files.
        octree_level (int): Octree level for compression.
        quantization_level (int): Quantization level for compression.

    Returns:
        dict: Summary of the experiment results.
    """
    results = {}

    # List all .ply files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.ply')]

    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_file)

        # Compression step
        compressed_file_path = os.path.join(output_dir, f"compressed_{input_file}")
        run_compression(input_file_path, compressed_file_path, octree_level, quantization_level)

        # Decompression step
        decompressed_file_path = os.path.join(output_dir, f"decompressed_{input_file}")
        run_decompression(compressed_file_path, decompressed_file_path)

        # Evaluation
        eval_results = evaluate_compression(input_file_path, decompressed_file_path)
        results[input_file] = eval_results

    # Save the results to a JSON file
    report_file = os.path.join(output_dir, "experiment_report.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def main():
    parser = argparse.ArgumentParser(description="Run point cloud compression experiments.")
    parser.add_argument("input_dir", type=str, help="Directory containing original .ply files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the results.")
    parser.add_argument("--octree_level", type=int, default=3, help="Octree depth level (default: 3).")
    parser.add_argument("--quantization_level", type=int, default=8, help="Quantization level (default: 8).")
    args = parser.parse_args()

    # Run the experiment
    start_time = time.time()
    run_experiment(args.input_dir, args.output_dir, args.octree_level, args.quantization_level)
    end_time = time.time()

    print(f"Experiment completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
