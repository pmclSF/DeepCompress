import numpy as np

def compute_new_bbox(idx, bbox_min, bbox_max):
    """
    Compute the bounding box for a given octant index.
    """
    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    # Use ceiling division for midpoint to handle odd dimensions
    midpoint = np.ceil((bbox_max - bbox_min) / 2).astype(np.int64) + bbox_min
    
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

def split_octree(points, bbox_min, bbox_max):
    """
    Split points into octants.
    """
    ret_points = [[] for _ in range(8)]
    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    # Validate points are within bbox
    if len(points) > 0:
        if not np.all((points[:, :3] >= bbox_min) & (points[:, :3] < bbox_max)):
            raise ValueError("Points must be within the bounding box")
    
    midpoint = np.ceil((bbox_max - bbox_min) / 2).astype(np.int64) + bbox_min
    
    # Precompute bounding boxes for all octants
    global_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]
    local_bboxes = [(np.zeros(3, dtype=np.int64), x[1] - x[0]) for x in global_bboxes]

    for point in points:
        location = 0
        if point[0] >= midpoint[0]:
            location |= 0b001
        if point[1] >= midpoint[1]:
            location |= 0b010
        if point[2] >= midpoint[2]:
            location |= 0b100
        
        # Convert point to local coordinates
        local_point = point.copy()
        local_point[:3] = point[:3] - global_bboxes[location][0]
        ret_points[location].append(local_point)

    # Create binary string representation
    binstr = 0
    for i, rp in enumerate(ret_points):
        if len(rp) > 0:
            binstr |= (1 << i)

    # Convert lists to numpy arrays
    ret_points = [np.vstack(rp) if len(rp) > 0 else np.zeros((0, points.shape[1]), dtype=points.dtype) 
                 for rp in ret_points]

    return [rp for rp in ret_points if len(rp) > 0], binstr, local_bboxes

def partition_octree(points, bbox_min, bbox_max, level):
    """
    Partition points into an octree structure.
    """
    points = np.asarray(points)
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None

    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    # Calculate geometric level based on actual box dimensions
    geo_level = int(np.ceil(np.log2(np.max(bbox_max - bbox_min))))
    if geo_level < level:
        raise ValueError("Geometric level must be >= partition level")
    
    block_size = 2 ** (geo_level - level)

    # Convert points to block coordinates
    block_ids = points[:, :3] // block_size
    block_ids = block_ids.astype(np.uint32)
    
    # Find unique blocks and their points
    block_ids_unique, block_idx, block_len = np.unique(
        block_ids, return_inverse=True, return_counts=True, axis=0)

    # Sort blocks by Morton code
    sort_key = []
    for x, y, z in block_ids_unique:
        zip_params = [f'{v:0{geo_level - level}b}' for v in [z, y, x]]
        sort_key.append(''.join(i + j + k for i, j, k in zip(*zip_params)))
    
    sort_idx = np.argsort(sort_key)
    block_ids_unique = block_ids_unique[sort_idx]
    block_len = block_len[sort_idx]

    # Update block indices after sorting
    inv_sort_idx = np.zeros_like(sort_idx)
    inv_sort_idx[sort_idx] = np.arange(sort_idx.size)
    block_idx = inv_sort_idx[block_idx]

    # Convert points to local coordinates within their blocks
    local_refs = np.pad(block_ids_unique[block_idx] * block_size, 
                       [[0, 0], [0, points.shape[1] - 3]])
    points_local = points - local_refs

    # Distribute points to their blocks
    blocks = [np.zeros((l, points.shape[1]), dtype=points.dtype) for l in block_len]
    blocks_last_idx = np.zeros(len(block_len), dtype=np.uint32)
    
    for i, b_idx in enumerate(block_idx):
        blocks[b_idx][blocks_last_idx[b_idx]] = points_local[i]
        blocks_last_idx[b_idx] += 1

    # Generate binary string representation
    _, binstr = partition_octree_rec(
        block_ids_unique, [0, 0, 0], 
        (2 ** level) * np.ones(3, dtype=np.int64), level)

    return blocks, binstr

def partition_octree_rec(points, bbox_min, bbox_max, level):
    """
    Recursive helper function for partition_octree.
    """
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None

    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    ret_points, binstr, bboxes = split_octree(points, bbox_min, bbox_max)
    
    result = []
    for rp, bbox in zip(ret_points, bboxes):
        if len(rp) > 0:
            result.append(partition_octree(rp, bbox[0], bbox[1], level - 1))

    blocks = []
    new_binstr = [binstr]
    
    for block_res in result:
        blocks.extend([b for b in block_res[0] if len(b) > 0])
        if block_res[1] is not None:
            new_binstr.extend(block_res[1])

    return blocks, new_binstr

def departition_octree(blocks, binstr_list, bbox_min, bbox_max, level):
    """
    Reconstruct points from octree partition.
    """
    if binstr_list is None:
        return blocks[0] if blocks else np.array([], dtype=np.int64)

    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)

    blocks = [b.copy() for b in blocks]
    binstr_list = binstr_list.copy()
    binstr_idxs = np.zeros(len(binstr_list), dtype=np.uint8)
    children_counts = np.zeros(len(binstr_list), dtype=np.uint32)

    binstr_list_idx = 0
    block_idx = 0
    cur_level = 1

    bbox_stack = [(bbox_min, bbox_max)]
    parents_stack = []

    global_points = []

    while block_idx < len(blocks):
        child_found = False
        while binstr_list[binstr_list_idx] != 0 and not child_found:
            if (binstr_list[binstr_list_idx] & 1) == 1:
                v = binstr_idxs[binstr_list_idx]
                cur_bbox = compute_new_bbox(v, *bbox_stack[-1])

                if cur_level == level:
                    # Transform points back to global coordinates
                    transformed_points = blocks[block_idx].copy()
                    padding = np.pad(cur_bbox[0], [0, blocks[block_idx].shape[1] - 3])
                    transformed_points[:, :3] += padding[:3]
                    global_points.append(transformed_points)
                    block_idx += 1
                else:
                    child_found = True

            binstr_list[binstr_list_idx] >>= 1
            binstr_idxs[binstr_list_idx] += 1

        if child_found:
            bbox_stack.append(cur_bbox)
            parents_stack.append(binstr_list_idx)
            for i in range(len(parents_stack)):
                children_counts[parents_stack[i]] += 1
            cur_level += 1
            binstr_list_idx += children_counts[parents_stack[-1]]
        else:
            if parents_stack:  # Only pop if stack is not empty
                binstr_list_idx = parents_stack.pop()
                cur_level -= 1
                bbox_stack.pop()
            else:
                break  # Exit if we've processed all points

    if not global_points:  # Handle empty input case
        return np.array([], dtype=np.int64)
    
    return np.vstack(global_points)