from typing import Tuple
import numpy as np

def compute_new_bbox(idx: int, bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    midpoint = (bbox_max + bbox_min) // 2

    cur_bbox_min = bbox_min.copy()
    cur_bbox_max = midpoint.copy()

    if idx & 1:
        cur_bbox_min[0] = midpoint[0]
        cur_bbox_max[0] = bbox_max[0]
    if (idx >> 1) & 1:
        cur_bbox_min[1] = midpoint[1]
        cur_bbox_max[1] = bbox_max[1]
    if (idx >> 2) & 1:
        cur_bbox_min[2] = midpoint[2]
        cur_bbox_max[2] = bbox_max[2]

    return cur_bbox_min, cur_bbox_max

def split_octree(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray):
    if len(points) == 0:
        return [[] for _ in range(8)], 0, []

    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    midpoint = (bbox_max + bbox_min) // 2

    ret_points = [[] for _ in range(8)]
    binstr = 0
    global_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]

    for point in points:
        loc = 0
        if point[0] > midpoint[0]: loc |= 1
        if point[1] > midpoint[1]: loc |= 2
        if point[2] > midpoint[2]: loc |= 4
        ret_points[loc].append(point)
        binstr |= (1 << loc)

    ret_arrays = [np.array(pts) if pts else np.array([], dtype=np.int64).reshape(0, points.shape[1]) 
                 for pts in ret_points]
    return ret_arrays, binstr, global_bboxes

def partition_octree(points: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, level: int):
    if len(points) == 0 or level == 0:
        return [points], []

    ret_points, binstr, local_bboxes = split_octree(points, bbox_min, bbox_max)
    
    blocks = []
    new_binstr = [binstr]

    for i, rp in enumerate(ret_points):
        if len(rp) > 0:
            child_min, child_max = local_bboxes[i]
            sub_blocks, sub_binstr = partition_octree(rp, child_min, child_max, level - 1)
            blocks.extend(sub_blocks)
            new_binstr.extend(sub_binstr)

    return blocks, new_binstr

def departition_octree(blocks: list, binstr_list: list, bbox_min: np.ndarray, bbox_max: np.ndarray, level: int):
    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)

    if not binstr_list or len(blocks) == 0:
        return blocks[0] if blocks else np.array([])

    points = []
    binstr_idx = 0
    block_idx = 0

    def process_node(cur_min, cur_max, cur_level):
        nonlocal binstr_idx, block_idx

        if cur_level == level:
            if block_idx >= len(blocks):
                return
            points.extend(blocks[block_idx])
            block_idx += 1
        else:
            if binstr_idx >= len(binstr_list):
                return

            binstr = binstr_list[binstr_idx]
            binstr_idx += 1

            for oct_idx in range(8):
                if binstr & (1 << oct_idx):
                    child_min, child_max = compute_new_bbox(oct_idx, cur_min, cur_max)
                    process_node(child_min, child_max, cur_level + 1)

    process_node(bbox_min, bbox_max, 0)
    return np.array(points)

    def process_node(cur_min, cur_max, cur_level):
        nonlocal binstr_idx, block_idx

        if cur_level == level:
            if block_idx >= len(blocks):
                return
            global_points = blocks[block_idx].copy()
            points_list.append(global_points)
            block_idx += 1
        else:
            if binstr_idx >= len(binstr_list):
                return

            binstr = binstr_list[binstr_idx]
            binstr_idx += 1

            for oct_idx in range(8):
                if binstr & (1 << oct_idx):
                    child_min, child_max = compute_new_bbox(oct_idx, cur_min, cur_max)
                    process_node(child_min, child_max, cur_level + 1)

    process_node(bbox_min, bbox_max, 0)
    return np.concatenate(points_list, axis=0) if points_list else np.array([])