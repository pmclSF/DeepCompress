import os
import argparse
import numpy as np
import tensorflow as tf
import time
import psutil
from ds_mesh_to_pc import read_off  # Assuming this function is available

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
    max_value = np.max(np.abs(original))  # Max value for normalization (can be adjusted)
    psnr = 10 * np.log10((max_value**2) / mse)
    return psnr

def compute_bd_rate(original_psnr, compressed_psnr, original_bit_rate, compressed_bit_rate):
    """
    Compute the Bjøntegaard Delta Rate (BD-Rate) between original and compressed point clouds.

    Args:
        original_psnr (float): PSNR of the original point cloud.
        compressed_psnr (float): PSNR of the compressed point cloud.
        original_bit_rate (float): Bitrate of the original point cloud.
        compressed_bit_rate (float): Bitrate of the compressed point cloud.

    Returns:
        float: BD-Rate value.
    """
    return (compressed_bit_rate - original_bit_rate) / original_bit_rate

def compute_compression_ratio(original_size, compressed_size):
    """
    Compute the Compression Ratio between the original and compressed point clouds.

    Args:
        original_size (int): The size of the original file.
        compressed_size (int): The size of the compressed file.

    Returns:
        float: Compression ratio.
    """
    return original_size / compressed_size

def measure_time(func):
    """Measure the time taken for a function to execute."""
    start_time = time.time()
    func()
    end_time = time.time()
    return end_time - start_time

def measure_memory(func):
    """Measure the memory usage during the execution of a function."""
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss  # in bytes
    func()
    end_memory = process.memory_info().rss  # in bytes
    return (end_memory - start_memory) / (1024 * 1024)  # in MB

def evaluate_compression(original_file, compressed_file):
    """
    Evaluate the compression performance of point clouds by comparing original and compressed files.

    Args:
        original_file (str): Path to the original .ply file.
        compressed_file (str): Path to the compressed .ply file.
    """
    # Read original and compressed point clouds
    original_vertices, _ = read_off(original_file)
    compressed_vertices, _ = read_off(compressed_file)

    # Compute PSNR
    psnr = compute_psnr(original_vertices, compressed_vertices)
    print(f"PSNR: {psnr} dB")

    # Compute Bitrate (file size based)
    original_size = os.path.getsize(original_file) / 1024  # in KB
    compressed_size = os.path.getsize(compressed_file) / 1024  # in KB
    bitrate = compressed_size / original_size
    print(f"Bitrate: {bitrate:.2f}")

    # Compute Compression Ratio
    compression_ratio = compute_compression_ratio(original_size, compressed_size)
    print(f"Compression Ratio: {compression_ratio:.2f}")

    # Measure time efficiency
    compression_time = measure_time(lambda: read_off(compressed_file))
    decompression_time = measure_time(lambda: read_off(original_file))
    print(f"Compression Time: {compression_time:.2f} seconds")
    print(f"Decompression Time: {decompression_time:.2f} seconds")

    # Measure memory efficiency
    compression_memory = measure_memory(lambda: read_off(compressed_file))
    decompression_memory = measure_memory(lambda: read_off(original_file))
    print(f"Compression Memory: {compression_memory:.2f} MB")
    print(f"Decompression Memory: {decompression_memory:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Evaluate compression of point cloud models.")
    parser.add_argument("original_file", type=str, help="Path to the original .ply file.")
    parser.add_argument("compressed_file", type=str, help="Path to the compressed .ply file.")
    args = parser.parse_args()

    # Run evaluation
    evaluate_compression(args.original_file, args.compressed_file)

if __name__ == "__main__":
    main()
