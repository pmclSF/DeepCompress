import numpy as np
import pytest
from ds_mesh_to_pc import partition_octree, departition_octree, split_octree

def test_split_octree():
    bbox_min = [0, 0, 0]
    bbox_max = [8, 8, 8]
    points = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [6, 6, 6],
        [7, 7, 7],
        [4, 4, 4]
    ])

    ret_points, binstr, local_bboxes = split_octree(points, bbox_min, bbox_max)

    assert len(ret_points) == 8, "Should split into 8 octants"
    assert len(ret_points[0]) == 3, "Incorrect point count in octant 0"
    assert len(ret_points[7]) == 2, "Incorrect point count in octant 7"

def test_partition_octree():
    bbox_min = [0, 0, 0]
    bbox_max = [8, 8, 8]
    points = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [6, 6, 6], 
        [7, 7, 7],
        [4, 4, 4]
    ])
    level = 2

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    
    assert len(blocks) == 4, "Should partition into 4 blocks"

def test_departition_octree():
    bbox_min = [0, 0, 0]
    bbox_max = [8, 8, 8]
    points = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [6, 6, 6],
        [7, 7, 7],
        [4, 4, 4]
    ])
    level = 2

    blocks, binstr = partition_octree(points, bbox_min, bbox_max, level)
    reconstructed_points = departition_octree(blocks, binstr, bbox_min, bbox_max, level)

    # Sort both arrays
    points_sorted = points[np.lexsort((points[:,2], points[:,1], points[:,0]))]
    reconstructed_sorted = reconstructed_points[np.lexsort((reconstructed_points[:,2], reconstructed_points[:,1], reconstructed_points[:,0]))]

    assert reconstructed_points.shape == points.shape, "Reconstructed shape mismatch"
    assert np.allclose(reconstructed_sorted, points_sorted), "Reconstructed points mismatch"