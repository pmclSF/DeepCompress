import numpy as np

def compute_new_bbox(idx, bbox_min, bbox_max):
    """
    Compute the bounding box for a given octant index.
    
    Args:
        idx (int): Octant index (0-7)
        bbox_min (array-like): Minimum corner of parent bounding box
        bbox_max (array-like): Maximum corner of parent bounding box
        
    Returns:
        tuple: (min_corner, max_corner) of new bounding box
    """
    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    # Use ceiling division for midpoint to handle odd dimensions
    midpoint = np.ceil((bbox_max - bbox_min) / 2).astype(np.int64) + bbox_min
    
    cur_bbox_min = bbox_min.copy()
    cur_bbox_max = midpoint.copy()

    if idx & 1:  # x-dimension
        cur_bbox_min[0] = midpoint[0]
        cur_bbox_max[0] = bbox_max[0]
    if (idx >> 1) & 1:  # y-dimension
        cur_bbox_min[1] = midpoint[1]
        cur_bbox_max[1] = bbox_max[1]
    if (idx >> 2) & 1:  # z-dimension
        cur_bbox_min[2] = midpoint[2]
        cur_bbox_max[2] = bbox_max[2]

    return cur_bbox_min, cur_bbox_max

def split_octree(points, bbox_min, bbox_max):
    """
    Split points into octants.
    
    Args:
        points (np.ndarray): Points to split
        bbox_min (array-like): Minimum corner of bounding box
        bbox_max (array-like): Maximum corner of bounding box
        
    Returns:
        tuple: (list of point arrays, binary string, list of local bounding boxes)
    """
    if len(points) == 0:
        return [], 0, []

    ret_points = [[] for _ in range(8)]
    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    # Validate points are within bbox
    if len(points) > 0:
        within_bounds = np.all((points[:, :3] >= bbox_min) & (points[:, :3] < bbox_max))
        assert within_bounds, "Points must be within the bounding box"
    
    midpoint = bbox_min + np.ceil((bbox_max - bbox_min) / 2).astype(np.int64)
    
    # Calculate octant for each point
    x_mask = points[:, 0] >= midpoint[0]
    y_mask = points[:, 1] >= midpoint[1]
    z_mask = points[:, 2] >= midpoint[2]
    
    locations = x_mask.astype(int) | \
               (y_mask.astype(int) << 1) | \
               (z_mask.astype(int) << 2)
    
    # Precompute bounding boxes
    global_bboxes = [compute_new_bbox(i, bbox_min, bbox_max) for i in range(8)]
    
    # Distribute points to octants
    for i, point in enumerate(points):
        loc = locations[i]
        local_point = point.copy()
        local_point[:3] = point[:3] - global_bboxes[loc][0]
        ret_points[loc].append(local_point)
    
    # Convert to numpy arrays and create binary string
    ret_arrays = []
    local_bboxes = []
    binstr = 0
    
    for i in range(8):
        if len(ret_points[i]) > 0:
            ret_arrays.append(np.stack(ret_points[i]))
            local_bboxes.append((np.zeros(3, dtype=np.int64), 
                               global_bboxes[i][1] - global_bboxes[i][0]))
            binstr |= (1 << i)
    
    return ret_arrays, binstr, local_bboxes

def partition_octree(points, bbox_min, bbox_max, level):
    """
    Partition points into an octree structure.
    
    Args:
        points (np.ndarray): Points to partition
        bbox_min (array-like): Minimum corner of bounding box
        bbox_max (array-like): Maximum corner of bounding box
        level (int): Maximum octree depth
        
    Returns:
        tuple: (list of point blocks, list of binary strings)
    """
    points = np.asarray(points)
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None

    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    # Validate points against bounding box
    assert np.all((points[:, :3] >= bbox_min) & (points[:, :3] < bbox_max)), \
           "Points must be within the bounding box"
    
    # Calculate geometric level and validate requested level
    max_dim = np.max(bbox_max - bbox_min)
    geo_level = int(np.ceil(np.log2(max_dim)))
    assert level <= geo_level, f"Requested level {level} exceeds maximum geometric level {geo_level}"
    
    # Get result from partition_octree_rec
    blocks, binstr = partition_octree_rec(points, bbox_min, bbox_max, level)
    
    # Ensure blocks is always a list
    if not isinstance(blocks, list):
        blocks = [blocks]
    
    return blocks, binstr

def partition_octree_rec(points, bbox_min, bbox_max, level):
    """
    Recursive helper function for partition_octree.
    
    Args:
        points (np.ndarray): Points to partition
        bbox_min (array-like): Minimum corner of bounding box
        bbox_max (array-like): Maximum corner of bounding box
        level (int): Current recursion level
        
    Returns:
        tuple: (list of point blocks, list of binary strings)
    """
    if len(points) == 0:
        return [points], None
    if level == 0:
        return [points], None

    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    ret_points, binstr, bboxes = split_octree(points, bbox_min, bbox_max)
    
    # Recursively partition non-empty octants
    result = []
    for rp, bbox in zip(ret_points, bboxes):
        if len(rp) > 0:
            result.append(partition_octree(rp, bbox[0], bbox[1], level - 1))
    
    # Collect blocks and binary strings
    blocks = []
    new_binstr = [binstr]
    
    for block_res in result:
        blocks.extend(block_res[0])
        if block_res[1] is not None:
            new_binstr.extend(block_res[1])
    
    return blocks, new_binstr

def departition_octree(blocks, binstr_list, bbox_min, bbox_max, level):
    """
    Reconstruct points from octree partition.
    
    Args:
        blocks (list): List of point blocks
        binstr_list (list): List of binary strings describing tree structure
        bbox_min (array-like): Minimum corner of bounding box
        bbox_max (array-like): Maximum corner of bounding box
        level (int): Maximum octree depth
        
    Returns:
        np.ndarray: Reconstructed points
    """
    if binstr_list is None:
        return blocks[0] if blocks else np.array([], dtype=np.int64)
    
    bbox_min = np.asarray(bbox_min, dtype=np.int64)
    bbox_max = np.asarray(bbox_max, dtype=np.int64)
    
    if len(blocks) == 0:
        return np.array([], dtype=np.int64)
    
    points_list = []
    
    # Track current position in the octree
    binstr_idx = 0
    block_idx = 0
    
    def process_node(cur_min, cur_max, cur_level):
        nonlocal binstr_idx, block_idx
        
        if cur_level == level:
            # At leaf level, transform points to global coordinates
            if block_idx >= len(blocks):
                return
            global_points = blocks[block_idx].copy()
            global_points[:, :3] += cur_min
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
    
    # Start recursion from root
    process_node(bbox_min, bbox_max, 0)
    
    if not points_list:
        return np.array([], dtype=blocks[0].dtype)
    
    return np.concatenate(points_list, axis=0)