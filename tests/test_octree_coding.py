import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pytest
from octree_coding import partition_octree, departition_octree

def test_partition_and_departition():
    """Test basic partitioning and reconstruction of points."""
    bbox_min = [0, 0, 0]
    bbox_max = [16, 16, 16]
    level = 4

    # Test with points that were previously failing
    points = np.array([
        [14, 4, 15], [6, 15, 10], [14, 13, 2], [0, 6, 11],
        [12, 15, 8], [2, 15, 8], [11, 1, 13], [11, 3, 10]
    ], dtype=np.int64)

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    # Sort points for comparison
    original_sorted = np.array(sorted(map(tuple, points)))
    reconstructed_sorted = np.array(sorted(map(tuple, reconstructed_points)))

    np.testing.assert_array_equal(
        original_sorted, 
        reconstructed_sorted,
        err_msg=f"Missing points: {set(map(tuple, points)) - set(map(tuple, reconstructed_points))}, "
               f"Extra points: {set(map(tuple, reconstructed_points)) - set(map(tuple, points))}"
    )

def test_large_partition():
    """Test with a large number of random points."""
    bbox_min = [0, 0, 0]
    bbox_max = [256, 256, 256]
    level = 8

    # Use fixed seed for reproducibility
    np.random.seed(42)
    points = np.random.randint(0, 256, (1000, 3), dtype=np.int64)

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    # Sort points for comparison
    original_sorted = np.array(sorted(map(tuple, points)))
    reconstructed_sorted = np.array(sorted(map(tuple, reconstructed_points)))

    np.testing.assert_array_equal(
        original_sorted,
        reconstructed_sorted,
        err_msg="Large partition failed to reconstruct points exactly"
    )

def test_empty_partition():
    """Test handling of empty point sets."""
    bbox_min = [0, 0, 0]
    bbox_max = [16, 16, 16]
    level = 4

    points = np.array([], dtype=np.int64).reshape(0, 3)

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    assert len(reconstructed_points) == 0, "Empty input should produce empty output"
    assert reconstructed_points.shape == (0,) or reconstructed_points.shape == (0, 3), \
        "Empty output should have correct shape"

def test_single_point():
    """Test with a single point."""
    bbox_min = [0, 0, 0]
    bbox_max = [16, 16, 16]
    level = 4

    points = np.array([[8, 8, 8]], dtype=np.int64)

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    np.testing.assert_array_equal(
        points,
        reconstructed_points,
        err_msg=f"Single point {points} not reconstructed correctly as {reconstructed_points}"
    )

def test_high_dimensional_points():
    """Test with points having additional dimensions beyond xyz."""
    bbox_min = [0, 0, 0]
    bbox_max = [16, 16, 16]
    level = 4

    points = np.array([
        [1, 2, 3, 0.5],
        [4, 5, 6, 0.6],
        [7, 8, 9, 0.7]
    ])

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    np.testing.assert_array_almost_equal(
        points,
        reconstructed_points,
        decimal=6,
        err_msg="High dimensional points not reconstructed correctly"
    )

def test_boundary_points():
    """Test with points at the boundaries of the bounding box."""
    bbox_min = [0, 0, 0]
    bbox_max = [16, 16, 16]
    level = 4

    points = np.array([
        [0, 0, 0],    # Min corner
        [15, 15, 15], # Max corner
        [0, 15, 0],   # Edge points
        [15, 0, 0],
        [0, 0, 15],
        [8, 8, 8],    # Center point
    ], dtype=np.int64)

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    # Sort points for comparison
    original_sorted = np.array(sorted(map(tuple, points)))
    reconstructed_sorted = np.array(sorted(map(tuple, reconstructed_points)))

    np.testing.assert_array_equal(
        original_sorted,
        reconstructed_sorted,
        err_msg="Boundary points not reconstructed correctly"
    )

def test_odd_sized_bbox():
    """Test with a bounding box of odd dimensions."""
    bbox_min = [0, 0, 0]
    bbox_max = [17, 15, 13]  # Odd dimensions
    level = 4

    points = np.array([
        [16, 14, 12],  # Near max corner
        [1, 1, 1],     # Near min corner
        [8, 7, 6],     # Middle points
        [9, 8, 7],
    ], dtype=np.int64)

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    # Sort points for comparison
    original_sorted = np.array(sorted(map(tuple, points)))
    reconstructed_sorted = np.array(sorted(map(tuple, reconstructed_points)))

    np.testing.assert_array_equal(
        original_sorted,
        reconstructed_sorted,
        err_msg="Points in odd-sized bbox not reconstructed correctly"
    )

def test_invalid_inputs():
    """Test handling of invalid inputs."""
    bbox_min = [0, 0, 0]
    bbox_max = [16, 16, 16]
    level = 4

    # Test with points outside bbox
    points = np.array([[20, 20, 20]], dtype=np.int64)
    with pytest.raises(AssertionError):
        blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)

    # Test with invalid level
    points = np.array([[8, 8, 8]], dtype=np.int64)
    with pytest.raises(AssertionError):
        blocks, binstr = partition_octree(points, bbox_min, bbox_max, 5)  # level > geometric level

def test_dense_points():
    """Test with densely packed points."""
    bbox_min = [0, 0, 0]
    bbox_max = [4, 4, 4]
    level = 2

    # Create a dense 4x4x4 grid of points
    x, y, z = np.meshgrid(np.arange(4), np.arange(4), np.arange(4))
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1).astype(np.int64)

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    # Sort points for comparison
    original_sorted = np.array(sorted(map(tuple, points)))
    reconstructed_sorted = np.array(sorted(map(tuple, reconstructed_points)))

    np.testing.assert_array_equal(
        original_sorted,
        reconstructed_sorted,
        err_msg="Dense point grid not reconstructed correctly"
    )

if __name__ == '__main__':
    pytest.main([__file__])