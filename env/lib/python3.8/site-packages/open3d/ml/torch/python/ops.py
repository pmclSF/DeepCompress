# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2023 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# This file is machine generated. Do not modify.
import torch as _torch
from . import return_types


def voxel_pooling(positions,
                  features,
                  voxel_size,
                  position_fn="average",
                  feature_fn="average",
                  debug=False):
    """Spatial pooling for point clouds by combining points that fall into the same voxel bin.

    The voxel grid used for pooling is always aligned to the origin (0,0,0) to
    simplify building voxel grid hierarchies. The order of the returned voxels is
    not defined as can be seen in the following example::

      import open3d.ml.tf as ml3d

      positions = [
          [0.1,0.1,0.1],
          [0.5,0.5,0.5],
          [1.7,1.7,1.7],
          [1.8,1.8,1.8],
          [0.3,2.4,1.4]]

      features = [[1.0,2.0],
                  [1.1,2.3],
                  [4.2,0.1],
                  [1.3,3.4],
                  [2.3,1.9]]

      ml3d.ops.voxel_pooling(positions, features, 1.0,
                             position_fn='center', feature_fn='max')

      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      positions = torch.Tensor([
          [0.1,0.1,0.1],
          [0.5,0.5,0.5],
          [1.7,1.7,1.7],
          [1.8,1.8,1.8],
          [0.3,2.4,1.4]])

      features = torch.Tensor([
                  [1.0,2.0],
                  [1.1,2.3],
                  [4.2,0.1],
                  [1.3,3.4],
                  [2.3,1.9]])

      ml3d.ops.voxel_pooling(positions, features, 1.0,
                             position_fn='center', feature_fn='max')

      # returns the voxel centers  [[0.5, 2.5, 1.5],
      #                             [1.5, 1.5, 1.5],
      #                             [0.5, 0.5, 0.5]]
      # and the max pooled features for each voxel [[2.3, 1.9],
      #                                             [4.2, 3.4],
      #                                             [1.1, 2.3]]

    position_fn: Defines how the new point positions will be computed.
      The options are
        * "average" computes the center of gravity for the points within one voxel.
        * "nearest_neighbor" selects the point closest to the voxel center.
        * "center" uses the voxel center for the position of the generated point.

    feature_fn: Defines how the pooled features will be computed.
      The options are
        * "average" computes the average feature vector.
        * "nearest_neighbor" selects the feature vector of the point closest to the voxel center.
        * "max" uses the maximum feature among all points within the voxel.

    debug: If true additional checks for debugging will be enabled.

    positions: The point positions with shape [N,3] with N as the number of points.

    features: The feature vector with shape [N,channels].

    voxel_size: The voxel size.

    pooled_positions: The output point positions with shape [M,3] and M <= N.

    pooled_features: The output point features with shape [M,channels] and M <= N.
    """
    return return_types.voxel_pooling(
        *_torch.ops.open3d.voxel_pooling(positions=positions,
                                         features=features,
                                         voxel_size=voxel_size,
                                         position_fn=position_fn,
                                         feature_fn=feature_fn,
                                         debug=debug))


def voxelize(points,
             row_splits,
             voxel_size,
             points_range_min,
             points_range_max,
             max_points_per_voxel=9223372036854775807,
             max_voxels=9223372036854775807):
    """Voxelization for point clouds.

    The function returns the integer coordinates of the voxels that contain
    points and a compact list of the indices that associate the voxels to the
    points. Also supports variable length batching.

    Minimal example::

      import open3d.ml.tf as ml3d

      points = [
          [0.1,0.1,0.1],
          [0.5,0.5,0.5],
          [1.7,1.7,1.7],
          [1.8,1.8,1.8],
          [9.3,9.4,9.4]]

      row_splits = [0, 2, 5]

      ml3d.ops.voxelize(points,
                        row_splits,
                        voxel_size=[1.0,1.0,1.0],
                        points_range_min=[0,0,0],
                        points_range_max=[2,2,2])

      # returns the voxel coordinates  [[0, 0, 0],
      #                                 [1, 1, 1]]
      #
      #         the point indices      [0, 1, 2, 3]
      #
      #         the point row splits   [0, 2, 4]
      #
      #         and the batch splits   [0, 1, 2]
      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      points = torch.Tensor([
          [0.1,0.1,0.1],
          [0.5,0.5,0.5],
          [1.7,1.7,1.7],
          [1.8,1.8,1.8],
          [9.3,9.4,9.4]])

      row_splits = torch.Tensor([0, 2, 5]).to(torch.int64)

      ml3d.ops.voxelize(points,
                        row_splits,
                        voxel_size=torch.Tensor([1.0,1.0,1.0]),
                        points_range_min=torch.Tensor([0,0,0]),
                        points_range_max=torch.Tensor([2,2,2]))

      # returns the voxel coordinates  [[0, 0, 0],
      #                                 [1, 1, 1]]
      #
      #         the point indices      [0, 1, 2, 3]
      #
      #         the point row splits   [0, 2, 4]
      #
      #         and the batch splits   [0, 1, 2]

    points: The point positions with shape [N,D] with N as the number of points and
      D as the number of dimensions, which must be 0 < D < 9.

    row_splits: 1D vector with row splits information if points is batched. This
      vector is [0, num_points] if there is only 1 batch item.

    voxel_size: The voxel size with shape [D].

    points_range_min: The minimum range for valid points to be voxelized. This
      vector has shape [D] and is used as the origin for computing the voxel_indices.

    points_range_min: The maximum range for valid points to be voxelized. This
      vector has shape [D].

    max_points_per_voxel: The maximum number of points to consider for a voxel.

    max_voxels: The maximum number of voxels to generate per batch.

    voxel_coords: The integer voxel coordinates.The shape of this tensor is [M, D]
      with M as the number of voxels and D as the number of dimensions.

    voxel_point_indices: A flat list of all the points that have been voxelized.
      The start and end of each voxel is defined in voxel_point_row_splits.

    voxel_point_row_splits: This is an exclusive prefix sum that includes the total
      number of points in the last element. This can be used to find the start and
      end of the point indices for each voxel. The shape of this tensor is [M+1].

    voxel_batch_splits: This is a prefix sum of number of voxels per batch. This can
      be used to find voxel_coords and row_splits corresponding to any particular
      batch.
    """
    return return_types.voxelize(
        *_torch.ops.open3d.voxelize(points=points,
                                    row_splits=row_splits,
                                    voxel_size=voxel_size,
                                    points_range_min=points_range_min,
                                    points_range_max=points_range_max,
                                    max_points_per_voxel=max_points_per_voxel,
                                    max_voxels=max_voxels))


def reduce_subarrays_sum(values, row_splits):
    """Computes the sum for each subarray in a flat vector of arrays.

    The start and end of the subarrays are defined by an exclusive prefix sum.
    Zero length subarrays are allowed as shown in the following example::

      import open3d.ml.tf as ml3d

      ml3d.ops.reduce_subarrays_sum(
          values = [1,2,3,4],
          row_splits=[0,2,2,4] # defines 3 subarrays with starts and ends 0-2,2-2,2-4
          )
      # returns [3,0,7]


      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      ml3d.ops.reduce_subarrays_sum(
        values = torch.Tensor([1,2,3,4]),
        row_splits=torch.LongTensor([0,2,2,4]) # defines 3 subarrays with starts and ends 0-2,2-2,2-4
        )
      # returns [3,0,7]


    values: Linear memory which stores the values for all arrays.

    row_splits: Defines the start and end of each subarray. This is an exclusive
      prefix sum with 0 as the first element and the length of values as
      additional last element. If there are N subarrays the length of this vector
      is N+1.

    sums: The sum of each subarray. The sum of an empty subarray is 0.
      sums is a zero length vector if values is a zero length vector.
    """
    return _torch.ops.open3d.reduce_subarrays_sum(values=values,
                                                  row_splits=row_splits)


def ragged_to_dense(values, row_splits, out_col_size, default_value):

    return _torch.ops.open3d.ragged_to_dense(values=values,
                                             row_splits=row_splits,
                                             out_col_size=out_col_size,
                                             default_value=default_value)


def radius_search(points,
                  queries,
                  radii,
                  points_row_splits,
                  queries_row_splits,
                  index_dtype=3,
                  metric="L2",
                  ignore_query_point=False,
                  return_distances=False,
                  normalize_distances=False):
    """Computes the indices and distances of all neighbours within a radius.

    This op computes the neighborhood for each query point and returns the indices
    of the neighbors and optionally also the distances. Each query point has an
    individual search radius. Points and queries can be batched with each batch
    item having an individual number of points and queries. The following example
    shows a simple search with just a single batch item::

      import open3d.ml.tf as ml3d

      points = [
          [0.1,0.1,0.1],
          [0.5,0.5,0.5],
          [1.7,1.7,1.7],
          [1.8,1.8,1.8],
          [0.3,2.4,1.4]]

      queries = [
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.2],
      ]

      radii = [1.0,1.0,1.0]

      ml3d.ops.radius_search(points, queries, radii,
                             points_row_splits=[0,5],
                             queries_row_splits=[0,3])
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []


      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      points = torch.Tensor([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]])

      queries = torch.Tensor([
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.1],
      ])

      radii = torch.Tensor([1.0,1.0,1.0])

      ml3d.ops.radius_search(points, queries, radii,
                             points_row_splits=torch.LongTensor([0,5]),
                             queries_row_splits=torch.LongTensor([0,3]))
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []


    metric: Either L1 or L2. Default is L2

    ignore_query_point: If true the points that coincide with the center of the
      search window will be ignored. This excludes the query point if **queries** and
      **points** are the same point cloud.

    return_distances: If True the distances for each neighbor will be returned in
      the output tensor **neighbors_distance**.  If False a zero length Tensor will
      be returned for **neighbors_distances**.

    normalize_distances: If True the returned distances will be normalized with the
      radii.

    points: The 3D positions of the input points.

    queries: The 3D positions of the query points.

    radii: A vector with the individual radii for each query point.

    points_row_splits: 1D vector with the row splits information if points is
      batched. This vector is [0, num_points] if there is only 1 batch item.

    queries_row_splits: 1D vector with the row splits information if queries is
      batched. This vector is [0, num_queries] if there is only 1 batch item.

    neighbors_index: The compact list of indices of the neighbors. The
      corresponding query point can be inferred from the
      **neighbor_count_row_splits** vector.

    neighbors_row_splits: The exclusive prefix sum of the neighbor count for the
      query points including the total neighbor count as the last element. The
      size of this array is the number of queries + 1.

    neighbors_distance: Stores the distance to each neighbor if **return_distances**
      is True. The distances are squared only if metric is L2.
      This is a zero length Tensor if **return_distances** is False.
    """
    return return_types.radius_search(*_torch.ops.open3d.radius_search(
        points=points,
        queries=queries,
        radii=radii,
        points_row_splits=points_row_splits,
        queries_row_splits=queries_row_splits,
        index_dtype=index_dtype,
        metric=metric,
        ignore_query_point=ignore_query_point,
        return_distances=return_distances,
        normalize_distances=normalize_distances))


def nms(boxes, scores, nms_overlap_thresh):
    """Performs non-maximum suppression of bounding boxes.

    This function performs non-maximum suppression for the input bounding boxes
    considering the the per-box score and overlaps. It returns the indices of the
    selected boxes.

    Minimal example::

      import open3d.ml.tf as ml3d
      import numpy as np

      boxes = np.array([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                        [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                        [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                        [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                        [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                        [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]],
                       dtype=np.float32)
      scores = np.array([3, 1.1, 5, 2, 1, 0], dtype=np.float32)
      nms_overlap_thresh = 0.7
      keep_indices = ml3d.ops.nms(boxes, scores, nms_overlap_thresh)
      print(keep_indices)

      # PyTorch example.
      import torch
      import open3d.ml.torch as ml3d

      boxes = torch.Tensor([[15.0811, -7.9803, 15.6721, -6.8714, 0.5152],
                            [15.1166, -7.9261, 15.7060, -6.8137, 0.6501],
                            [15.1304, -7.8129, 15.7069, -6.8903, 0.7296],
                            [15.2050, -7.8447, 15.8311, -6.7437, 1.0506],
                            [15.1343, -7.8136, 15.7121, -6.8479, 1.0352],
                            [15.0931, -7.9552, 15.6675, -7.0056, 0.5979]])
      scores = torch.Tensor([3, 1.1, 5, 2, 1, 0])
      nms_overlap_thresh = 0.7
      keep_indices = ml3d.ops.nms(boxes, scores, nms_overlap_thresh)
      print(keep_indices)

    boxes: (N, 5) float32 tensor. Bounding boxes are represented as (x0, y0, x1, y1, rotate).

    scores: (N,) float32 tensor. A higher score means a more confident bounding box.

    nms_overlap_thresh: float value between 0 and 1. When a high-score box is
      selected, other remaining boxes with IoU > nms_overlap_thresh will be discarded.
      A higher nms_overlap_thresh means more boxes will be kept.

    keep_indices: (M,) int64 tensor. The selected box indices.
    """
    return _torch.ops.open3d.nms(boxes=boxes,
                                 scores=scores,
                                 nms_overlap_thresh=nms_overlap_thresh)


def knn_search(points,
               queries,
               k,
               points_row_splits,
               queries_row_splits,
               index_dtype=3,
               metric="L2",
               ignore_query_point=False,
               return_distances=False):
    """Computes the indices of k nearest neighbors.

    This op computes the neighborhood for each query point and returns the indices
    of the neighbors. The output format is compatible with the radius_search and
    fixed_radius_search ops and supports returning less than k neighbors if there
    are less than k points or ignore_query_point is enabled and the **queries** and
    **points** arrays are the same point cloud. The following example shows the usual
    case where the outputs can be reshaped to a [num_queries, k] tensor::

      import tensorflow as tf
      import open3d.ml.tf as ml3d

      points = [
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]]

      queries = [
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.2],
      ]

      ans = ml3d.ops.knn_search(points, queries, k=2,
                          points_row_splits=[0,5],
                          queries_row_splits=[0,3],
                          return_distances=True)
      # returns ans.neighbors_index      = [1, 2, 4, 2, 4, 2]
      #         ans.neighbors_row_splits = [0, 2, 4, 6]
      #         ans.neighbors_distance   = [0.75 , 1.47, 0.56, 1.62, 0.77, 1.85]
      # Since there are more than k points and we do not ignore any points we can
      # reshape the output to [num_queries, k] with
      neighbors_index = tf.reshape(ans.neighbors_index, [3,2])
      neighbors_distance = tf.reshape(ans.neighbors_distance, [3,2])


      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      points = torch.Tensor([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]])

      queries = torch.Tensor([
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.2],
      ])

      radii = torch.Tensor([1.0,1.0,1.0])

      ans = ml3d.ops.knn_search(points, queries, k=2,
                                points_row_splits=torch.LongTensor([0,5]),
                                queries_row_splits=torch.LongTensor([0,3]),
                                return_distances=True)
      # returns ans.neighbors_index      = [1, 2, 4, 2, 4, 2]
      #         ans.neighbors_row_splits = [0, 2, 4, 6]
      #         ans.neighbors_distance   = [0.75 , 1.47, 0.56, 1.62, 0.77, 1.85]
      # Since there are more than k points and we do not ignore any points we can
      # reshape the output to [num_queries, k] with
      neighbors_index = ans.neighbors_index.reshape(3,2)
      neighbors_distance = ans.neighbors_distance.reshape(3,2)

    metric: Either L1 or L2. Default is L2

    ignore_query_point: If true the points that coincide with the center of the
      search window will be ignored. This excludes the query point if **queries** and
     **points** are the same point cloud.

    return_distances: If True the distances for each neighbor will be returned in
      the output tensor **neighbors_distances**. If False a zero length Tensor will
      be returned for **neighbors_distances**.

    points: The 3D positions of the input points.

    queries: The 3D positions of the query points.

    k: The number of nearest neighbors to search.

    points_row_splits: 1D vector with the row splits information if points is
      batched. This vector is [0, num_points] if there is only 1 batch item.

    queries_row_splits: 1D vector with the row splits information if queries is
      batched. This vector is [0, num_queries] if there is only 1 batch item.

    neighbors_index: The compact list of indices of the neighbors. The
      corresponding query point can be inferred from the
      **neighbor_count_prefix_sum** vector. Neighbors for the same point are sorted
      with respect to the distance.

      Note that there is no guarantee that there will be exactly k neighbors in some cases.
      These cases are:
        * There are less than k points.
        * **ignore_query_point** is True and there are multiple points with the same position.

    neighbors_row_splits: The exclusive prefix sum of the neighbor count for the
      query points including the total neighbor count as the last element. The
      size of this array is the number of queries + 1.

    neighbors_distance: Stores the distance to each neighbor if **return_distances**
      is True. The distances are squared only if metric is L2. This is a zero length
      Tensor if **return_distances** is False.
    """
    return return_types.knn_search(
        *_torch.ops.open3d.knn_search(points=points,
                                      queries=queries,
                                      k=k,
                                      points_row_splits=points_row_splits,
                                      queries_row_splits=queries_row_splits,
                                      index_dtype=index_dtype,
                                      metric=metric,
                                      ignore_query_point=ignore_query_point,
                                      return_distances=return_distances))


def invert_neighbors_list(num_points, inp_neighbors_index,
                          inp_neighbors_row_splits, inp_neighbors_attributes):
    """Inverts a neighbors list made of neighbors_index and neighbors_row_splits.

    This op inverts the neighbors list as returned from the neighbor search ops.
    The role of query points and input points is reversed in the returned list.
    The following example illustrates this::

      import open3d.ml.tf as ml3d

      # in this example we have 4 points and 3 query points with 3, 1, and 2 neighbors
      # the mapping is 0->(0,1,2), 1->(2), 2->(1,3)
      neighbors_index = [0, 1, 2, 2, 1, 3]
      neighbors_row_splits = [0, 3, 4, 6]
      # optional attributes for each pair
      neighbors_attributes = [10, 20, 30, 40, 50, 60]

      ans = ml3d.ops.invert_neighbors_list(4,
                                           neighbors_index,
                                           neighbors_row_splits,
                                           neighbors_attributes)
      # returns ans.neighbors_index      = [0, 0, 2, 0, 1, 2]
      #         ans.neighbors_row_splits = [0, 1, 3, 5, 6]
      #         ans.neighbors_attributes = [10, 20, 50, 30, 40, 60]
      # which is the mapping 0->(0), 1->(0,2), 2->(0,1), 3->(2)
      # note that the order of the neighbors can be permuted

      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      # in this example we have 4 points and 3 query points with 3, 1, and 2 neighbors
      # the mapping is 0->(0,1,2), 1->(2), 2->(1,3)
      neighbors_index = torch.IntTensor([0, 1, 2, 2, 1, 3])
      neighbors_row_splits = torch.LongTensor([0, 3, 4, 6])
      # optional attributes for each pair
      neighbors_attributes = torch.Tensor([10, 20, 30, 40, 50, 60])

      ans = ml3d.ops.invert_neighbors_list(4,
                                           neighbors_index,
                                           neighbors_row_splits,
                                           neighbors_attributes)
      # returns ans.neighbors_index      = [0, 0, 2, 0, 1, 2]
      #         ans.neighbors_row_splits = [0, 1, 3, 5, 6]
      #         ans.neighbors_attributes = [10, 20, 50, 30, 40, 60]
      # which is the mapping 0->(0), 1->(0,2), 2->(0,1), 3->(2)
      # note that the order of the neighbors can be permuted

    num_points: Scalar integer with the number of points that have been tested in a neighbor
      search. This is the number of the points in the second point cloud (not the
      query point cloud) in a neighbor search.
      The size of the output **neighbors_row_splits** will be **num_points** +1.

    inp_neighbors_index: The input neighbor indices stored linearly.

    inp_neighbors_row_splits: The number of neighbors for the input queries as
      exclusive prefix sum. The prefix sum includes the total number as last
      element.

    inp_neighbors_attributes: Array that stores an attribute for each neighbor.
      The size of the first dim must match the first dim of inp_neighbors_index.
      To ignore attributes pass a 1D Tensor with zero size.

    neighbors_index: The output neighbor indices stored
      linearly.

    neighbors_row_splits: Stores the number of neighbors for the new queries,
      previously the input points, as exclusive prefix sum including the total
      number in the last element.

    neighbors_attributes: Array that stores an attribute for each neighbor.
      If the inp_neighbors_attributes Tensor is a zero length vector then the output
      will be a zero length vector as well.
    """
    return return_types.invert_neighbors_list(
        *_torch.ops.open3d.invert_neighbors_list(
            num_points=num_points,
            inp_neighbors_index=inp_neighbors_index,
            inp_neighbors_row_splits=inp_neighbors_row_splits,
            inp_neighbors_attributes=inp_neighbors_attributes))


def fixed_radius_search(points,
                        queries,
                        radius,
                        points_row_splits,
                        queries_row_splits,
                        hash_table_splits,
                        hash_table_index,
                        hash_table_cell_splits,
                        index_dtype=3,
                        metric="L2",
                        ignore_query_point=False,
                        return_distances=False):
    """Computes the indices of all neighbors within a radius.

    This op computes the neighborhood for each query point and returns the indices
    of the neighbors and optionally also the distances. The same fixed radius is
    used for each query point. Points and queries can be batched with each batch
    item having an individual number of points and queries. The following example
    shows a simple search with just a single batch item::


      import open3d.ml.tf as ml3d

      points = [
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]]

      queries = [
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.1],
      ]

      radius = 1.0

      # build the spatial hash table for fixex_radius_search
      table = ml3d.ops.build_spatial_hash_table(points,
                                                radius,
                                                points_row_splits=torch.LongTensor([0,5]),
                                                hash_table_size_factor=1/32)

      # now run the fixed radius search
      ml3d.ops.fixed_radius_search(points,
                                   queries,
                                   radius,
                                   points_row_splits=torch.LongTensor([0,5]),
                                   queries_row_splits=torch.LongTensor([0,3]),
                                   **table._asdict())
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []

      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      points = torch.Tensor([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]])

      queries = torch.Tensor([
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.1],
      ])

      radius = 1.0

      # build the spatial hash table for fixex_radius_search
      table = ml3d.ops.build_spatial_hash_table(points,
                                                radius,
                                                points_row_splits=torch.LongTensor([0,5]),
                                                hash_table_size_factor=1/32)

      # now run the fixed radius search
      ml3d.ops.fixed_radius_search(points,
                                   queries,
                                   radius,
                                   points_row_splits=torch.LongTensor([0,5]),
                                   queries_row_splits=torch.LongTensor([0,3]),
                                   **table._asdict())
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []


    index_dtype:
      The data type for the returned neighbor_index Tensor. Either int32 or int64.
      Default is int32.

    metric:
      Either L1, L2 or Linf. Default is L2

    ignore_query_point:
      If true the points that coincide with the center of the search window will be
      ignored. This excludes the query point if 'queries' and 'points' are the same
      point cloud.

    return_distances:
      If True the distances for each neighbor will be returned in the tensor
      'neighbors_distance'.
      If False a zero length Tensor will be returned for 'neighbors_distance'.

    points:
      The 3D positions of the input points.

    queries:
      The 3D positions of the query points.

    radius:
      A scalar with the neighborhood radius

    points_row_splits:
      1D vector with the row splits information if points is batched.
      This vector is [0, num_points] if there is only 1 batch item.

    queries_row_splits:
      1D vector with the row splits information if queries is batched.
      This vector is [0, num_queries] if there is only 1 batch item.

    hash_table_splits: Array defining the start and end the hash table
      for each batch item. This is [0, number of cells] if there is only
      1 batch item or [0, hash_table_cell_splits_size-1] which is the same.

    hash_table_index: Stores the values of the hash table, which are the indices of
      the points. The start and end of each cell is defined by hash_table_cell_splits.

    hash_table_cell_splits: Defines the start and end of each hash table cell.

    neighbors_index:
      The compact list of indices of the neighbors. The corresponding query point
      can be inferred from the 'neighbor_count_row_splits' vector.

    neighbors_row_splits:
      The exclusive prefix sum of the neighbor count for the query points including
      the total neighbor count as the last element. The size of this array is the
      number of queries + 1.

    neighbors_distance:
      Stores the distance to each neighbor if 'return_distances' is True.
      Note that the distances are squared if metric is L2.
      This is a zero length Tensor if 'return_distances' is False.
    """
    return return_types.fixed_radius_search(
        *_torch.ops.open3d.fixed_radius_search(
            points=points,
            queries=queries,
            radius=radius,
            points_row_splits=points_row_splits,
            queries_row_splits=queries_row_splits,
            hash_table_splits=hash_table_splits,
            hash_table_index=hash_table_index,
            hash_table_cell_splits=hash_table_cell_splits,
            index_dtype=index_dtype,
            metric=metric,
            ignore_query_point=ignore_query_point,
            return_distances=return_distances))


def build_spatial_hash_table(points,
                             radius,
                             points_row_splits,
                             hash_table_size_factor,
                             max_hash_table_size=33554432):
    """Creates a spatial hash table meant as input for fixed_radius_search


    The following example shows how **build_spatial_hash_table** and
    **fixed_radius_search** are used together::

      import open3d.ml.tf as ml3d

      points = [
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]]

      queries = [
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.1],
      ]

      radius = 1.0

      # build the spatial hash table for fixed_radius_search
      table = ml3d.ops.build_spatial_hash_table(points,
                                                radius,
                                                points_row_splits=torch.LongTensor([0,5]),
                                                hash_table_size_factor=1/32)

      # now run the fixed radius search
      ml3d.ops.fixed_radius_search(points,
                                   queries,
                                   radius,
                                   points_row_splits=torch.LongTensor([0,5]),
                                   queries_row_splits=torch.LongTensor([0,3]),
                                   **table._asdict())
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []

      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      points = torch.Tensor([
        [0.1,0.1,0.1],
        [0.5,0.5,0.5],
        [1.7,1.7,1.7],
        [1.8,1.8,1.8],
        [0.3,2.4,1.4]])

      queries = torch.Tensor([
          [1.0,1.0,1.0],
          [0.5,2.0,2.0],
          [0.5,2.1,2.1],
      ])

      radius = 1.0

      # build the spatial hash table for fixex_radius_search
      table = ml3d.ops.build_spatial_hash_table(points,
                                                radius,
                                                points_row_splits=torch.LongTensor([0,5]),
                                                hash_table_size_factor=1/32)

      # now run the fixed radius search
      ml3d.ops.fixed_radius_search(points,
                                   queries,
                                   radius,
                                   points_row_splits=torch.LongTensor([0,5]),
                                   queries_row_splits=torch.LongTensor([0,3]),
                                   **table._asdict())
      # returns neighbors_index      = [1, 4, 4]
      #         neighbors_row_splits = [0, 1, 2, 3]
      #         neighbors_distance   = []



    max_hash_table_size: The maximum hash table size.

    points: The 3D positions of the input points.

    radius: A scalar which defines the spatial cell size of the hash table.

    points_row_splits: 1D vector with the row splits information if points is
      batched. This vector is [0, num_points] if there is only 1 batch item.

    hash_table_size_factor:
      The size of the hash table as a factor of the number of input points.

    hash_table_index: Stores the values of the hash table, which are the indices of
      the points. The start and end of each cell is defined by
      **hash_table_cell_splits**.

    hash_table_cell_splits: Defines the start and end of each hash table cell within
      a hash table.

    hash_table_splits: Defines the start and end of each hash table in the
      hash_table_cell_splits array. If the batch size is 1 then there is only one
      hash table and this vector is [0, number of cells].
    """
    return return_types.build_spatial_hash_table(
        *_torch.ops.open3d.build_spatial_hash_table(
            points=points,
            radius=radius,
            points_row_splits=points_row_splits,
            hash_table_size_factor=hash_table_size_factor,
            max_hash_table_size=max_hash_table_size))


def sparse_conv_transpose(filters,
                          out_importance,
                          inp_features,
                          inp_neighbors_index,
                          inp_neighbors_importance_sum,
                          inp_neighbors_row_splits,
                          neighbors_index,
                          neighbors_kernel_index,
                          neighbors_importance,
                          neighbors_row_splits,
                          normalize=False,
                          max_temp_mem_MB=64):
    """Sparse tranpose convolution of two pointclouds.

    normalize:
      If True the input feature values will be normalized using
      'inp_neighbors_importance_sum'.


    max_temp_mem_MB:
      Defines the maximum temporary memory in megabytes to be used for the GPU
      implementation. More memory means fewer kernel invocations. Note that the
      a minimum amount of temp memory will always be allocated even if this
      variable is set to 0.


    filters:
      The filter parameters.
      The shape of the filter is [depth, height, width, in_ch, out_ch].
      The dimensions 'depth', 'height', 'width' define the spatial resolution of
      the filter. The spatial size of the filter is defined by the parameter
      'extents'.


    out_importance:
      An optional scalar importance for each output point. The output features of
      each point will be multiplied with the corresponding value.
      The shape is [num input points]. Use a zero length Tensor to disable.


    inp_features:
      A 2D tensor which stores a feature vector for each input point.


    inp_neighbors_index:
      The inp_neighbors_index stores a list of indices of neighbors for each input point as nested lists.
      The start and end of each list can be computed using 'inp_neighbors_row_splits'.


    inp_neighbors_importance_sum:
      1D tensor of the same length as 'inp_features' or zero length if
      neighbors_importance is empty. This is the sum of the values in
      'neighbors_importance' for each input point.


    inp_neighbors_row_splits:
      The exclusive prefix sum of the neighbor count for the input points including
      the total neighbor count as the last element. The size of this array is the
      number of input points + 1.


    neighbors_index:
      The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
      The start and end of each list can be computed using 'neighbors_row_splits'.


    neighbors_kernel_index:
      Defines which kernel element to use for each neighbor. This array has the same length as neighbors_index.


    neighbors_importance:
      Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
      the features of each neighbor.


    neighbors_row_splits:
      The exclusive prefix sum of the neighbor count for the output points including
      the total neighbor count as the last element. The size of this array is the
      number of output points + 1.

    output_type: The type for the output.

    out_features:
      A Tensor with the output feature vectors for each output point.
    """
    return _torch.ops.open3d.sparse_conv_transpose(
        filters=filters,
        out_importance=out_importance,
        inp_features=inp_features,
        inp_neighbors_index=inp_neighbors_index,
        inp_neighbors_importance_sum=inp_neighbors_importance_sum,
        inp_neighbors_row_splits=inp_neighbors_row_splits,
        neighbors_index=neighbors_index,
        neighbors_kernel_index=neighbors_kernel_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        normalize=normalize,
        max_temp_mem_MB=max_temp_mem_MB)


def sparse_conv(filters,
                inp_features,
                inp_importance,
                neighbors_index,
                neighbors_kernel_index,
                neighbors_importance,
                neighbors_row_splits,
                normalize=False,
                max_temp_mem_MB=64):
    """General sparse convolution.

    This op computes the features for the forward pass.
    This example shows how to use this op::

      import tensorflow as tf
      import open3d.ml.tf as ml3d

      # This filter has 3 "spatial" elements with 8 input and 16 output channels
      filters = tf.random.normal([3,8,16])

      inp_features = tf.random.normal([5,8])


      out_features = ml3d.ops.sparse_conv(
          filters,
          inp_features=inp_features,
          inp_importance=[],
          neighbors_index=[0,1,2, 1,2,3, 2,3,4],
          # neighbors_kernel_index defines which of the "spatial"
          # elements of the filter to use
          neighbors_kernel_index=tf.convert_to_tensor([0,1,2, 0,1,2, 0,1,2], dtype=tf.uint8),
          neighbors_importance=[],
          neighbors_row_splits=[0,3,6,9]
      )

      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      # This filter has 3 "spatial" elements with 8 input and 16 output channels
      filters = torch.randn([3,8,16])

      inp_features = torch.randn([5,8])


      out_features = ml3d.ops.sparse_conv(
          filters,
          inp_features=inp_features,
          inp_importance=torch.FloatTensor([]),
          neighbors_index=torch.IntTensor([0,1,2, 1,2,3, 2,3,4]),
          # neighbors_kernel_index defines which of the "spatial"
          # elements of the filter to use
          neighbors_kernel_index=torch.ByteTensor([0,1,2, 0,1,2, 0,1,2]),
          neighbors_importance=torch.FloatTensor([]),
          neighbors_row_splits=torch.LongTensor([0,3,6,9])
      )

    normalize:
      If True the output feature values will be normalized using the sum for
      'neighbors_importance' for each output point.


    max_temp_mem_MB:
      Defines the maximum temporary memory in megabytes to be used for the GPU
      implementation. More memory means fewer kernel invocations. Note that the
      a minimum amount of temp memory will always be allocated even if this
      variable is set to 0.


    filters:
      The filter parameters.
      The shape of the filter is [depth, height, width, in_ch, out_ch].
      The dimensions 'depth', 'height', 'width' define the spatial resolution of
      the filter. The spatial size of the filter is defined by the parameter
      'extents'.


    inp_features:
      A 2D tensor which stores a feature vector for each input point.


    inp_importance:
      An optional scalar importance for each input point. The features of each point
      will be multiplied with the corresponding value. The shape is [num input points].
      Use a zero length Tensor to disable.


    neighbors_index:
      The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
      The start and end of each list can be computed using 'neighbors_row_splits'.


    neighbors_kernel_index:
      Defines which kernel element to use for each neighbor. This array has the same length as neighbors_index.


    neighbors_importance:
      Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
      the features of each neighbor. Use a zero length Tensor to weigh each neighbor
      with 1.


    neighbors_row_splits:
      The exclusive prefix sum of the neighbor count for the output points including
      the total neighbor count as the last element. The size of this array is the
      number of output points + 1.

    output_type: The type for the output.

    out_features:
      A Tensor with the output feature vectors for each output point.
    """
    return _torch.ops.open3d.sparse_conv(
        filters=filters,
        inp_features=inp_features,
        inp_importance=inp_importance,
        neighbors_index=neighbors_index,
        neighbors_kernel_index=neighbors_kernel_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        normalize=normalize,
        max_temp_mem_MB=max_temp_mem_MB)


def continuous_conv_transpose(filters,
                              out_positions,
                              out_importance,
                              extents,
                              offset,
                              inp_positions,
                              inp_features,
                              inp_neighbors_index,
                              inp_neighbors_importance_sum,
                              inp_neighbors_row_splits,
                              neighbors_index,
                              neighbors_importance,
                              neighbors_row_splits,
                              align_corners=False,
                              coordinate_mapping="ball_to_cube_radial",
                              normalize=False,
                              interpolation="linear",
                              max_temp_mem_MB=64):
    """Continuous tranpose convolution of two pointclouds.

    align_corners:
      If True the outer voxel centers of the filter grid are aligned with the boundary of the spatial shape.


    coordinate_mapping:
      Defines how the relative positions of the neighbors are mapped before computing
      filter indices.
      For all mappings relative coordinates will be scaled with the inverse extent,
      i.e. the extent becomes a unit cube.
      After that one of the following mappings will be applied:
        'ball_to_cube_radial': maps a unit ball to a unit cube by radial stretching.
        'ball_to_cube_volume_preserving': maps a unit ball to a unit cube preserving the volume.
        'identity': the identity mapping.
      Use 'ball_to_cube_radial' for a spherical or ellipsoidal filter window
      and 'identity' for a rectangular filter window.


    normalize:
      If True the input feature values will be normalized using
      'inp_neighbors_importance_sum'.


    interpolation:
      If interpolation is 'linear' then each filter value lookup is a trilinear interpolation.
      If interpolation is 'nearest_neighbor' only the spatially closest value is considered.
      This makes the filter and therefore the convolution discontinuous.


    max_temp_mem_MB:
      Defines the maximum temporary memory in megabytes to be used for the GPU
      implementation. More memory means fewer kernel invocations. Note that the
      a minimum amount of temp memory will always be allocated even if this
      variable is set to 0.


    filters:
      The filter parameters.
      The shape of the filter is [depth, height, width, in_ch, out_ch].
      The dimensions 'depth', 'height', 'width' define the spatial resolution of
      the filter. The spatial size of the filter is defined by the parameter
      'extents'.


    out_positions:
      A 2D tensor with the 3D point positions of each output point.
      The coordinates for each point is a vector with format [x,y,z].


    out_positions:
      A 1D tensor with the 3D point positions of each output point.
      The coordinates for each point is a vector with format [x,y,z].

    out_importance:
      An optional scalar importance for each output point. The output features of
      each point will be multiplied with the corresponding value.
      The shape is [num input points]. Use a zero length Tensor to disable.

    extents:
      The extent defines the spatial size of the filter for each input point.
      It is a 2D vector of the form [[x_size, y_size, z_size], ..].
      For 'ball to cube' coordinate mappings the extent defines the bounding box
      of the ball.
      Broadcasting is supported for all axes. E.g. providing only the extent for a
      single point as well as only providing 'x_size' is valid.


    offset:
      A 1D tensor which defines the offset in voxel units to shift the input points.
      Offsets will be ignored if align_corners is True.


    inp_positions:
      A 2D tensor with the 3D point positions of each input point.
      The coordinates for each point is a vector with format [x,y,z].


    inp_features:
      A 2D tensor which stores a feature vector for each input point.


    inp_neighbors_index:
      The inp_neighbors_index stores a list of indices of neighbors for each input point as nested lists.
      The start and end of each list can be computed using 'inp_neighbors_row_splits'.


    inp_neighbors_importance_sum:
      1D tensor of the same length as 'inp_positions' or zero length if
      neighbors_importance is empty. This is the sum of the values in
      'neighbors_importance' for each input point.


    inp_neighbors_row_splits:
      The exclusive prefix sum of the neighbor count for the input points including
      the total neighbor count as the last element. The size of this array is the
      number of input points + 1.


    neighbors_index:
      The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
      The start and end of each list can be computed using 'neighbors_row_splits'.


    neighbors_importance:
      Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
      the features of each neighbor.


    neighbors_row_splits:
      The exclusive prefix sum of the neighbor count for the output points including
      the total neighbor count as the last element. The size of this array is the
      number of output points + 1.

    output_type: The type for the output.

    out_features:
      A Tensor with the output feature vectors for each output point.
    """
    return _torch.ops.open3d.continuous_conv_transpose(
        filters=filters,
        out_positions=out_positions,
        out_importance=out_importance,
        extents=extents,
        offset=offset,
        inp_positions=inp_positions,
        inp_features=inp_features,
        inp_neighbors_index=inp_neighbors_index,
        inp_neighbors_importance_sum=inp_neighbors_importance_sum,
        inp_neighbors_row_splits=inp_neighbors_row_splits,
        neighbors_index=neighbors_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        align_corners=align_corners,
        coordinate_mapping=coordinate_mapping,
        normalize=normalize,
        interpolation=interpolation,
        max_temp_mem_MB=max_temp_mem_MB)


def continuous_conv(filters,
                    out_positions,
                    extents,
                    offset,
                    inp_positions,
                    inp_features,
                    inp_importance,
                    neighbors_index,
                    neighbors_importance,
                    neighbors_row_splits,
                    align_corners=False,
                    coordinate_mapping="ball_to_cube_radial",
                    normalize=False,
                    interpolation="linear",
                    max_temp_mem_MB=64):
    """Continuous convolution of two pointclouds.

    This op computes the features for the forward pass.
    This example shows how to use this op::

      import tensorflow as tf
      import open3d.ml.tf as ml3d

      filters = tf.random.normal([3,3,3,8,16])

      inp_positions = tf.random.normal([20,3])
      inp_features = tf.random.normal([20,8])
      out_positions = tf.random.normal([10,3])

      nsearch = ml3d.layers.FixedRadiusSearch()
      radius = 1.2
      neighbors = nsearch(inp_positions, out_positions, radius)

      ml3d.ops.continuous_conv(filters,
                               out_positions,
                               extents=[[2*radius]],
                               offset=[0,0,0],
                               inp_positions=inp_positions,
                               inp_features=inp_features,
                               inp_importance=[],
                               neighbors_index=neighbors.neighbors_index,
                               neighbors_row_splits=neighbors.neighbors_row_splits,
                               neighbors_importance=[]
                              )

      # or with pytorch
      import torch
      import open3d.ml.torch as ml3d

      filters = torch.randn([3,3,3,8,16])

      inp_positions = torch.randn([20,3])
      inp_features = torch.randn([20,8])
      out_positions = torch.randn([10,3])

      nsearch = ml3d.nn.FixedRadiusSearch()
      radius = 1.2
      neighbors = nsearch(inp_positions, out_positions, radius)

      ml3d.ops.continuous_conv(filters,
                               out_positions,
                               extents=torch.FloatTensor([[2*radius]]),
                               offset=torch.FloatTensor([0,0,0]),
                               inp_positions=inp_positions,
                               inp_features=inp_features,
                               inp_importance=torch.FloatTensor([]),
                               neighbors_index=neighbors.neighbors_index,
                               neighbors_row_splits=neighbors.neighbors_row_splits,
                               neighbors_importance=torch.FloatTensor([]),
                              )

    align_corners: If True the outer voxel centers of the filter grid are aligned
      with the boundary of the spatial shape.


    coordinate_mapping: Defines how the relative positions of the neighbors are
      mapped before computing filter indices.
      For all mappings relative coordinates will be scaled with the inverse extent,
      i.e. the extent becomes a unit cube.
      After that one of the following mappings will be applied:
        "ball_to_cube_radial": maps a unit ball to a unit cube by radial stretching.
        "ball_to_cube_volume_preserving": maps a unit ball to a unit cube preserving the volume.
        "identity": the identity mapping.
      Use "ball_to_cube_radial" for a spherical or ellipsoidal filter window
      and "identity" for a rectangular filter window.


    normalize: If True the output feature values will be normalized using the sum
      for **neighbors_importance** for each output point.


    interpolation: If interpolation is "linear" then each filter value lookup is a
      trilinear interpolation. If interpolation is "nearest_neighbor" only the
      spatially closest value is considered. This makes the filter and therefore
      the convolution discontinuous.


    max_temp_mem_MB: Defines the maximum temporary memory in megabytes to be used
      for the GPU implementation. More memory means fewer kernel invocations. Note
      that the a minimum amount of temp memory will always be allocated even if
      this variable is set to 0.


    filters: The filter parameters. The shape of the filter is
      [depth, height, width, in_ch, out_ch]. The dimensions 'depth', 'height',
      'width' define the spatial resolution of the filter. The spatial size of the
      filter is defined by the parameter 'extents'.


    out_positions: A 2D tensor with the 3D point positions of each output point.
      The coordinates for each point is a vector with format [x,y,z].


    extents: The extent defines the spatial size of the filter for each output
      point.  It is a 2D vector of the form [[x_size, y_size, z_size], ..].
      For 'ball to cube' coordinate mappings the extent defines the bounding box
      of the ball.
      Broadcasting is supported for all axes. E.g. providing only the extent for a
      single point as well as only providing 'x_size' is valid.


    offset: A 1D tensor which defines the offset in voxel units to shift the input
      points. Offsets will be ignored if align_corners is True.


    inp_positions: A 2D tensor with the 3D point positions of each input point.
      The coordinates for each point is a vector with format [x,y,z].


    inp_features: A 2D tensor which stores a feature vector for each input point.


    inp_importance: An optional scalar importance for each input point. The
      features of each point will be multiplied with the corresponding value. The
      shape is [num input points]. Use a zero length Tensor to disable.


    neighbors_index: The neighbors_index stores a list of indices of neighbors for
      each output point as nested lists. The start and end of each list can be
      computed using 'neighbors_row_splits'.


    neighbors_importance: Tensor of the same shape as 'neighbors_index' with a
      scalar value that is used to scale the features of each neighbor. Use a
      zero length Tensor to weigh each neighbor with 1.


    neighbors_row_splits: The exclusive prefix sum of the neighbor count for the
      output points including the total neighbor count as the last element. The
      size of this array is the number of output points + 1.

    output_type: The type for the output.

    out_features: A Tensor with the output feature vectors for each output point.
    """
    return _torch.ops.open3d.continuous_conv(
        filters=filters,
        out_positions=out_positions,
        extents=extents,
        offset=offset,
        inp_positions=inp_positions,
        inp_features=inp_features,
        inp_importance=inp_importance,
        neighbors_index=neighbors_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        align_corners=align_corners,
        coordinate_mapping=coordinate_mapping,
        normalize=normalize,
        interpolation=interpolation,
        max_temp_mem_MB=max_temp_mem_MB)
