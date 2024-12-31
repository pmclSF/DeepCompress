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
import tensorflow as _tf
from .lib import _lib


def ball_query(xyz, center, radius, nsample, name=None):
    """ TODO

      Args:
        xyz: A `Tensor` of type `float32`.
        center: A `Tensor` of type `float32`.
        radius: A `float`.
        nsample: An `int`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `int32`.
  
    """
    return _lib.open3d_ball_query(xyz=xyz,
                                  center=center,
                                  radius=radius,
                                  nsample=nsample,
                                  name=name)


def batch_grid_subsampling(points, batches, dl, name=None):
    """TODO: add doc.

      Args:
        points: A `Tensor` of type `float32`.
        batches: A `Tensor` of type `int32`.
        dl: A `Tensor` of type `float32`.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (sub_points, sub_batches).

        sub_points: A `Tensor` of type `float32`.
        sub_batches: A `Tensor` of type `int32`.
  
    """
    return _lib.open3d_batch_grid_subsampling(points=points,
                                              batches=batches,
                                              dl=dl,
                                              name=name)


def build_spatial_hash_table(points,
                             radius,
                             points_row_splits,
                             hash_table_size_factor,
                             max_hash_table_size=33554432,
                             name=None):
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

      Args:
        points: A `Tensor`. Must be one of the following types: `float32`, `float64`.
          The 3D positions of the input points.
        radius: A `Tensor`. Must have the same type as `points`.
          A scalar which defines the spatial cell size of the hash table.
        points_row_splits: A `Tensor` of type `int64`.
          1D vector with the row splits information if points is
          batched. This vector is [0, num_points] if there is only 1 batch item.
        hash_table_size_factor: A `Tensor` of type `float64`. 
          The size of the hash table as a factor of the number of input points.
        max_hash_table_size: An optional `int`. Defaults to `33554432`.
          The maximum hash table size.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (hash_table_index, hash_table_cell_splits, hash_table_splits).

        hash_table_index: A `Tensor` of type `uint32`. Stores the values of the hash table, which are the indices of
          the points. The start and end of each cell is defined by
          **hash_table_cell_splits**.
        hash_table_cell_splits: A `Tensor` of type `uint32`. Defines the start and end of each hash table cell within
          a hash table.
        hash_table_splits: A `Tensor` of type `uint32`. Defines the start and end of each hash table in the
          hash_table_cell_splits array. If the batch size is 1 then there is only one
          hash table and this vector is [0, number of cells].
  
    """
    return _lib.open3d_build_spatial_hash_table(
        points=points,
        radius=radius,
        points_row_splits=points_row_splits,
        hash_table_size_factor=hash_table_size_factor,
        max_hash_table_size=max_hash_table_size,
        name=name)


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
                    output_type=_tf.float32,
                    align_corners=True,
                    coordinate_mapping="ball_to_cube_radial",
                    normalize=False,
                    interpolation="linear",
                    max_temp_mem_MB=64,
                    name=None):
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

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.
          The filter parameters. The shape of the filter is
          [depth, height, width, in_ch, out_ch]. The dimensions 'depth', 'height',
          'width' define the spatial resolution of the filter. The spatial size of the
          filter is defined by the parameter 'extents'.
        out_positions: A `Tensor`. Must be one of the following types: `float32`, `float64`.
          A 2D tensor with the 3D point positions of each output point.
          The coordinates for each point is a vector with format [x,y,z].
        extents: A `Tensor`. Must have the same type as `out_positions`.
          The extent defines the spatial size of the filter for each output
          point.  It is a 2D vector of the form [[x_size, y_size, z_size], ..].
          For 'ball to cube' coordinate mappings the extent defines the bounding box
          of the ball.
          Broadcasting is supported for all axes. E.g. providing only the extent for a
          single point as well as only providing 'x_size' is valid.
        offset: A `Tensor`. Must have the same type as `out_positions`.
          A 1D tensor which defines the offset in voxel units to shift the input
          points. Offsets will be ignored if align_corners is True.
        inp_positions: A `Tensor`. Must have the same type as `out_positions`.
          A 2D tensor with the 3D point positions of each input point.
          The coordinates for each point is a vector with format [x,y,z].
        inp_features: A `Tensor`. Must have the same type as `filters`.
          A 2D tensor which stores a feature vector for each input point.
        inp_importance: A `Tensor`. Must have the same type as `filters`.
          An optional scalar importance for each input point. The
          features of each point will be multiplied with the corresponding value. The
          shape is [num input points]. Use a zero length Tensor to disable.
        neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.
          The neighbors_index stores a list of indices of neighbors for
          each output point as nested lists. The start and end of each list can be
          computed using 'neighbors_row_splits'.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`.
          Tensor of the same shape as 'neighbors_index' with a
          scalar value that is used to scale the features of each neighbor. Use a
          zero length Tensor to weigh each neighbor with 1.
        neighbors_row_splits: A `Tensor` of type `int64`.
          The exclusive prefix sum of the neighbor count for the
          output points including the total neighbor count as the last element. The
          size of this array is the number of output points + 1.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.bfloat16`. Defaults to `tf.float32`.
          The type for the output.
        align_corners: An optional `bool`. Defaults to `True`.
          If True the outer voxel centers of the filter grid are aligned
          with the boundary of the spatial shape.
        coordinate_mapping: An optional `string` from: `"ball_to_cube_radial", "ball_to_cube_volume_preserving", "identity"`. Defaults to `"ball_to_cube_radial"`.
          Defines how the relative positions of the neighbors are
          mapped before computing filter indices.
          For all mappings relative coordinates will be scaled with the inverse extent,
          i.e. the extent becomes a unit cube.
          After that one of the following mappings will be applied:
            "ball_to_cube_radial": maps a unit ball to a unit cube by radial stretching.
            "ball_to_cube_volume_preserving": maps a unit ball to a unit cube preserving the volume.
            "identity": the identity mapping.
          Use "ball_to_cube_radial" for a spherical or ellipsoidal filter window
          and "identity" for a rectangular filter window.
        normalize: An optional `bool`. Defaults to `False`.
          If True the output feature values will be normalized using the sum
          for **neighbors_importance** for each output point.
        interpolation: An optional `string` from: `"linear", "linear_border", "nearest_neighbor"`. Defaults to `"linear"`.
          If interpolation is "linear" then each filter value lookup is a
          trilinear interpolation. If interpolation is "nearest_neighbor" only the
          spatially closest value is considered. This makes the filter and therefore
          the convolution discontinuous.
        max_temp_mem_MB: An optional `int`. Defaults to `64`.
          Defines the maximum temporary memory in megabytes to be used
          for the GPU implementation. More memory means fewer kernel invocations. Note
          that the a minimum amount of temp memory will always be allocated even if
          this variable is set to 0.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`.
        A Tensor with the output feature vectors for each output point.
  
    """
    return _lib.open3d_continuous_conv(
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
        output_type=output_type,
        align_corners=align_corners,
        coordinate_mapping=coordinate_mapping,
        normalize=normalize,
        interpolation=interpolation,
        max_temp_mem_MB=max_temp_mem_MB,
        name=name)


def continuous_conv_backprop_filter(filters,
                                    out_positions,
                                    extents,
                                    offset,
                                    inp_positions,
                                    inp_features,
                                    inp_importance,
                                    neighbors_index,
                                    neighbors_importance,
                                    neighbors_row_splits,
                                    out_features_gradient,
                                    output_type=_tf.float32,
                                    align_corners=True,
                                    coordinate_mapping="ball_to_cube_radial",
                                    normalize=False,
                                    interpolation="linear",
                                    max_temp_mem_MB=64,
                                    debug=False,
                                    name=None):
    """Computes the backprop for the filter of the ContinuousConv

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.

          The filter parameters.
          The shape of the filter is [depth, height, width, in_ch, out_ch].
          The dimensions 'depth', 'height', 'width' define the spatial resolution of
          the filter. The spatial size of the filter is defined by the parameter
          'extents'.
        out_positions: A `Tensor`. Must be one of the following types: `float32`, `float64`.

          A 2D tensor with the 3D point positions of each output point.
          The coordinates for each point is a vector with format [x,y,z].
        extents: A `Tensor`. Must have the same type as `out_positions`. 
          The extent defines the spatial size of the filter for each output point.
          It is a 2D vector of the form [[x_size, y_size, z_size], ..].
          For 'ball to cube' coordinate mappings the extent defines the bounding box
          of the ball.
          Broadcasting is supported for all axes. E.g. providing only the extent for a
          single point as well as only providing 'x_size' is valid.
        offset: A `Tensor`. Must have the same type as `out_positions`. 
          A 1D tensor which defines the offset in voxel units to shift the input points.
          Offsets will be ignored if align_corners is True.
        inp_positions: A `Tensor`. Must have the same type as `out_positions`. 
          A 2D tensor with the 3D point positions of each input point.
          The coordinates for each point is a vector with format [x,y,z].
        inp_features: A `Tensor`. Must have the same type as `filters`. 
          A 2D tensor which stores a feature vector for each input point.
        inp_importance: A `Tensor`. Must have the same type as `filters`.
        neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.

          The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
          The start and end of each list can be computed using 'neighbors_row_splits'.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`. 
          Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
          the features of each neighbor.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the output points including
          the total neighbor count as the last element. The size of this array is the
          number of output points + 1.
        out_features_gradient: A `Tensor`. Must have the same type as `filters`. 
          A Tensor with the gradient for the outputs of the DCConv in the forward pass.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
          The type for the output.
        align_corners: An optional `bool`. Defaults to `True`. 
          If True the outer voxel centers of the filter grid are aligned with the boundady of the spatial shape.
        coordinate_mapping: An optional `string` from: `"ball_to_cube_radial", "ball_to_cube_volume_preserving", "identity"`. Defaults to `"ball_to_cube_radial"`.

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
        normalize: An optional `bool`. Defaults to `False`. 
          If True the output feature values will be normalized by the number of neighbors.
        interpolation: An optional `string` from: `"linear", "linear_border", "nearest_neighbor"`. Defaults to `"linear"`.

          If interpolation is 'linear' then each filter value lookup is a trilinear interpolation.
          If interpolation is 'nearest_neighbor' only the spatially closest value is considered.
          This makes the filter and therefore the convolution discontinuous.
        max_temp_mem_MB: An optional `int`. Defaults to `64`. 
          Defines the maximum temporary memory in megabytes to be used for the GPU
          implementation. More memory means fewer kernel invocations. Note that the
          a minimum amount of temp memory will always be allocated even if this
          variable is set to 0.
        debug: An optional `bool`. Defaults to `False`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`. 
        The gradients for the filter
  
    """
    return _lib.open3d_continuous_conv_backprop_filter(
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
        out_features_gradient=out_features_gradient,
        output_type=output_type,
        align_corners=align_corners,
        coordinate_mapping=coordinate_mapping,
        normalize=normalize,
        interpolation=interpolation,
        max_temp_mem_MB=max_temp_mem_MB,
        debug=debug,
        name=name)


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
                              output_type=_tf.float32,
                              align_corners=True,
                              coordinate_mapping="ball_to_cube_radial",
                              normalize=False,
                              interpolation="linear",
                              max_temp_mem_MB=64,
                              debug=False,
                              name=None):
    """Continuous tranpose convolution of two pointclouds.

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.

          The filter parameters.
          The shape of the filter is [depth, height, width, in_ch, out_ch].
          The dimensions 'depth', 'height', 'width' define the spatial resolution of
          the filter. The spatial size of the filter is defined by the parameter
          'extents'.
        out_positions: A `Tensor`. Must be one of the following types: `float32`, `float64`.

          A 1D tensor with the 3D point positions of each output point.
          The coordinates for each point is a vector with format [x,y,z].
        out_importance: A `Tensor`. Must have the same type as `filters`. 
          An optional scalar importance for each output point. The output features of
          each point will be multiplied with the corresponding value.
          The shape is [num input points]. Use a zero length Tensor to disable.
        extents: A `Tensor`. Must have the same type as `out_positions`. 
          The extent defines the spatial size of the filter for each input point.
          It is a 2D vector of the form [[x_size, y_size, z_size], ..].
          For 'ball to cube' coordinate mappings the extent defines the bounding box
          of the ball.
          Broadcasting is supported for all axes. E.g. providing only the extent for a
          single point as well as only providing 'x_size' is valid.
        offset: A `Tensor`. Must have the same type as `out_positions`. 
          A 1D tensor which defines the offset in voxel units to shift the input points.
          Offsets will be ignored if align_corners is True.
        inp_positions: A `Tensor`. Must have the same type as `out_positions`. 
          A 2D tensor with the 3D point positions of each input point.
          The coordinates for each point is a vector with format [x,y,z].
        inp_features: A `Tensor`. Must have the same type as `filters`. 
          A 2D tensor which stores a feature vector for each input point.
        inp_neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.

          The inp_neighbors_index stores a list of indices of neighbors for each input point as nested lists.
          The start and end of each list can be computed using 'inp_neighbors_row_splits'.
        inp_neighbors_importance_sum: A `Tensor`. Must have the same type as `filters`.

          1D tensor of the same length as 'inp_positions' or zero length if
          neighbors_importance is empty. This is the sum of the values in
          'neighbors_importance' for each input point.
        inp_neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the input points including
          the total neighbor count as the last element. The size of this array is the
          number of input points + 1.
        neighbors_index: A `Tensor`. Must have the same type as `inp_neighbors_index`.

          The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
          The start and end of each list can be computed using 'neighbors_row_splits'.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`. 
          Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
          the features of each neighbor.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the output points including
          the total neighbor count as the last element. The size of this array is the
          number of output points + 1.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
          The type for the output.
        align_corners: An optional `bool`. Defaults to `True`. 
          If True the outer voxel centers of the filter grid are aligned with the boundary of the spatial shape.
        coordinate_mapping: An optional `string` from: `"ball_to_cube_radial", "ball_to_cube_volume_preserving", "identity"`. Defaults to `"ball_to_cube_radial"`.

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
        normalize: An optional `bool`. Defaults to `False`. 
          If True the input feature values will be normalized using
          'inp_neighbors_importance_sum'.
        interpolation: An optional `string` from: `"linear", "linear_border", "nearest_neighbor"`. Defaults to `"linear"`.

          If interpolation is 'linear' then each filter value lookup is a trilinear interpolation.
          If interpolation is 'nearest_neighbor' only the spatially closest value is considered.
          This makes the filter and therefore the convolution discontinuous.
        max_temp_mem_MB: An optional `int`. Defaults to `64`. 
          Defines the maximum temporary memory in megabytes to be used for the GPU
          implementation. More memory means fewer kernel invocations. Note that the
          a minimum amount of temp memory will always be allocated even if this
          variable is set to 0.
        debug: An optional `bool`. Defaults to `False`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`. 
        A Tensor with the output feature vectors for each output point.
  
    """
    return _lib.open3d_continuous_conv_transpose(
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
        output_type=output_type,
        align_corners=align_corners,
        coordinate_mapping=coordinate_mapping,
        normalize=normalize,
        interpolation=interpolation,
        max_temp_mem_MB=max_temp_mem_MB,
        debug=debug,
        name=name)


def continuous_conv_transpose_backprop_filter(
        filters,
        out_positions,
        out_importance,
        extents,
        offset,
        inp_positions,
        inp_features,
        inp_neighbors_importance_sum,
        inp_neighbors_row_splits,
        neighbors_index,
        neighbors_importance,
        neighbors_row_splits,
        out_features_gradient,
        output_type=_tf.float32,
        align_corners=True,
        coordinate_mapping="ball_to_cube_radial",
        normalize=False,
        interpolation="linear",
        max_temp_mem_MB=64,
        debug=False,
        name=None):
    """Computes the backrop for the filter of the ContinuousConvTranspose

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.

          The filter parameters.
          The shape of the filter is [depth, height, width, in_ch, out_ch].
          The dimensions 'depth', 'height', 'width' define the spatial resolution of
          the filter. The spatial size of the filter is defined by the parameter
          'extents'.
        out_positions: A `Tensor`. Must be one of the following types: `float32`, `float64`.

          A 2D tensor with the 3D point positions of each output point.
          The coordinates for each point is a vector with format [x,y,z].
        out_importance: A `Tensor`. Must have the same type as `filters`.
        extents: A `Tensor`. Must have the same type as `out_positions`. 
          The extent defines the spatial size of the filter for each input point.
          It is a 2D vector of the form [[x_size, y_size, z_size], ..].
          For 'ball to cube' coordinate mappings the extent defines the bounding box
          of the ball.
          Broadcasting is supported for all axes. E.g. providing only the extent for a
          single point as well as only providing 'x_size' is valid.
        offset: A `Tensor`. Must have the same type as `out_positions`. 
          A 1D tensor which defines the offset in voxel units to shift the output points.
          Offsets will be ignored if align_corners is True.
        inp_positions: A `Tensor`. Must have the same type as `out_positions`. 
          A 2D tensor with the 3D point positions of each input point.
          The coordinates for each point is a vector with format [x,y,z].
        inp_features: A `Tensor`. Must have the same type as `filters`. 
          A 2D tensor which stores a feature vector for each input point.
        inp_neighbors_importance_sum: A `Tensor`. Must have the same type as `filters`.
        inp_neighbors_row_splits: A `Tensor` of type `int64`. 
          The number of neighbors for each input point as exclusive prefix sum.
        neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.

          The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
          The start and end of each list can be computed using 'neighbors_row_splits'.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The number of neighbors for each output point as exclusive prefix sum.
        out_features_gradient: A `Tensor`. Must have the same type as `filters`. 
          A Tensor with the gradient for the outputs of the DCConv in the forward pass.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64`. Defaults to `tf.float32`.
          The type for the output.
        align_corners: An optional `bool`. Defaults to `True`. 
          If True the outer voxel centers of the filter grid are aligned with the boundady of the spatial shape.
        coordinate_mapping: An optional `string` from: `"ball_to_cube_radial", "ball_to_cube_volume_preserving", "identity"`. Defaults to `"ball_to_cube_radial"`.

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
        normalize: An optional `bool`. Defaults to `False`. 
          If True the input feature values will be normalized by the number of neighbors.
        interpolation: An optional `string` from: `"linear", "linear_border", "nearest_neighbor"`. Defaults to `"linear"`.

          If interpolation is 'linear' then each filter value lookup is a trilinear interpolation.
          If interpolation is 'nearest_neighbor' only the spatially closest value is considered.
          This makes the filter and therefore the convolution discontinuous.
        max_temp_mem_MB: An optional `int`. Defaults to `64`. 
          Defines the maximum temporary memory in megabytes to be used for the GPU
          implementation. More memory means fewer kernel invocations. Note that the
          a minimum amount of temp memory will always be allocated even if this
          variable is set to 0.
        debug: An optional `bool`. Defaults to `False`. 
          If True then some additional checks will be enabled.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`. 
        The gradients for the filter
  
    """
    return _lib.open3d_continuous_conv_transpose_backprop_filter(
        filters=filters,
        out_positions=out_positions,
        out_importance=out_importance,
        extents=extents,
        offset=offset,
        inp_positions=inp_positions,
        inp_features=inp_features,
        inp_neighbors_importance_sum=inp_neighbors_importance_sum,
        inp_neighbors_row_splits=inp_neighbors_row_splits,
        neighbors_index=neighbors_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        out_features_gradient=out_features_gradient,
        output_type=output_type,
        align_corners=align_corners,
        coordinate_mapping=coordinate_mapping,
        normalize=normalize,
        interpolation=interpolation,
        max_temp_mem_MB=max_temp_mem_MB,
        debug=debug,
        name=name)


def fixed_radius_search(points,
                        queries,
                        radius,
                        points_row_splits,
                        queries_row_splits,
                        hash_table_splits,
                        hash_table_index,
                        hash_table_cell_splits,
                        index_dtype=_tf.int32,
                        metric="L2",
                        ignore_query_point=False,
                        return_distances=False,
                        name=None):
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

      Args:
        points: A `Tensor`. Must be one of the following types: `float32`, `float64`.

          The 3D positions of the input points.
        queries: A `Tensor`. Must have the same type as `points`. 
          The 3D positions of the query points.
        radius: A `Tensor`. Must have the same type as `points`. 
          A scalar with the neighborhood radius
        points_row_splits: A `Tensor` of type `int64`. 
          1D vector with the row splits information if points is batched.
          This vector is [0, num_points] if there is only 1 batch item.
        queries_row_splits: A `Tensor` of type `int64`. 
          1D vector with the row splits information if queries is batched.
          This vector is [0, num_queries] if there is only 1 batch item.
        hash_table_splits: A `Tensor` of type `uint32`.
          Array defining the start and end the hash table
          for each batch item. This is [0, number of cells] if there is only
          1 batch item or [0, hash_table_cell_splits_size-1] which is the same.
        hash_table_index: A `Tensor` of type `uint32`.
          Stores the values of the hash table, which are the indices of
          the points. The start and end of each cell is defined by hash_table_cell_splits.
        hash_table_cell_splits: A `Tensor` of type `uint32`.
          Defines the start and end of each hash table cell.
        index_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.

          The data type for the returned neighbor_index Tensor. Either int32 or int64.
          Default is int32.
        metric: An optional `string` from: `"L1", "L2", "Linf"`. Defaults to `"L2"`.

          Either L1, L2 or Linf. Default is L2
        ignore_query_point: An optional `bool`. Defaults to `False`. 
          If true the points that coincide with the center of the search window will be
          ignored. This excludes the query point if 'queries' and 'points' are the same
          point cloud.
        return_distances: An optional `bool`. Defaults to `False`. 
          If True the distances for each neighbor will be returned in the tensor
          'neighbors_distance'.
          If False a zero length Tensor will be returned for 'neighbors_distance'.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (neighbors_index, neighbors_row_splits, neighbors_distance).

        neighbors_index: A `Tensor` of type `index_dtype`. 
          The compact list of indices of the neighbors. The corresponding query point
          can be inferred from the 'neighbor_count_row_splits' vector.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the query points including
          the total neighbor count as the last element. The size of this array is the
          number of queries + 1.
        neighbors_distance: A `Tensor`. Has the same type as `points`. 
          Stores the distance to each neighbor if 'return_distances' is True.
          Note that the distances are squared if metric is L2.
          This is a zero length Tensor if 'return_distances' is False.
  
    """
    return _lib.open3d_fixed_radius_search(
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
        return_distances=return_distances,
        name=name)


def furthest_point_sampling(points, sample_size, name=None):
    """ TODO

      Args:
        points: A `Tensor` of type `float32`.
        sample_size: An `int`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `int32`.
  
    """
    return _lib.open3d_furthest_point_sampling(points=points,
                                               sample_size=sample_size,
                                               name=name)


def grid_subsampling(points, dl, name=None):
    """TODO: add doc.

      Args:
        points: A `Tensor` of type `float32`.
        dl: A `Tensor` of type `float32`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `float32`.
  
    """
    return _lib.open3d_grid_subsampling(points=points, dl=dl, name=name)


def invert_neighbors_list(num_points,
                          inp_neighbors_index,
                          inp_neighbors_row_splits,
                          inp_neighbors_attributes,
                          name=None):
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

      Args:
        num_points: A `Tensor` of type `int64`.
          Scalar integer with the number of points that have been tested in a neighbor
          search. This is the number of the points in the second point cloud (not the
          query point cloud) in a neighbor search.
          The size of the output **neighbors_row_splits** will be **num_points** +1.
        inp_neighbors_index: A `Tensor`. Must be one of the following types: `int32`.
          The input neighbor indices stored linearly.
        inp_neighbors_row_splits: A `Tensor` of type `int64`.
          The number of neighbors for the input queries as
          exclusive prefix sum. The prefix sum includes the total number as last
          element.
        inp_neighbors_attributes: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
          Array that stores an attribute for each neighbor.
          The size of the first dim must match the first dim of inp_neighbors_index.
          To ignore attributes pass a 1D Tensor with zero size.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (neighbors_index, neighbors_row_splits, neighbors_attributes).

        neighbors_index: A `Tensor`. Has the same type as `inp_neighbors_index`. The output neighbor indices stored
          linearly.
        neighbors_row_splits: A `Tensor` of type `int64`. Stores the number of neighbors for the new queries,
          previously the input points, as exclusive prefix sum including the total
          number in the last element.
        neighbors_attributes: A `Tensor`. Has the same type as `inp_neighbors_attributes`. Array that stores an attribute for each neighbor.
          If the inp_neighbors_attributes Tensor is a zero length vector then the output
          will be a zero length vector as well.
  
    """
    return _lib.open3d_invert_neighbors_list(
        num_points=num_points,
        inp_neighbors_index=inp_neighbors_index,
        inp_neighbors_row_splits=inp_neighbors_row_splits,
        inp_neighbors_attributes=inp_neighbors_attributes,
        name=name)


def knn_search(points,
               queries,
               k,
               points_row_splits,
               queries_row_splits,
               index_dtype=_tf.int32,
               metric="L2",
               ignore_query_point=False,
               return_distances=False,
               name=None):
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

      Args:
        points: A `Tensor`. Must be one of the following types: `float32`, `float64`.
          The 3D positions of the input points.
        queries: A `Tensor`. Must have the same type as `points`.
          The 3D positions of the query points.
        k: A `Tensor` of type `int32`. The number of nearest neighbors to search.
        points_row_splits: A `Tensor` of type `int64`.
          1D vector with the row splits information if points is
          batched. This vector is [0, num_points] if there is only 1 batch item.
        queries_row_splits: A `Tensor` of type `int64`.
          1D vector with the row splits information if queries is
          batched. This vector is [0, num_queries] if there is only 1 batch item.
        index_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
        metric: An optional `string` from: `"L1", "L2"`. Defaults to `"L2"`.
          Either L1 or L2. Default is L2
        ignore_query_point: An optional `bool`. Defaults to `False`.
          If true the points that coincide with the center of the
           search window will be ignored. This excludes the query point if **queries** and
          **points** are the same point cloud.
        return_distances: An optional `bool`. Defaults to `False`.
          If True the distances for each neighbor will be returned in
          the output tensor **neighbors_distances**. If False a zero length Tensor will
          be returned for **neighbors_distances**.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (neighbors_index, neighbors_row_splits, neighbors_distance).

        neighbors_index: A `Tensor` of type `index_dtype`. The compact list of indices of the neighbors. The
          corresponding query point can be inferred from the
          **neighbor_count_prefix_sum** vector. Neighbors for the same point are sorted
          with respect to the distance.

          Note that there is no guarantee that there will be exactly k neighbors in some cases.
          These cases are:
            * There are less than k points.
            * **ignore_query_point** is True and there are multiple points with the same position.
        neighbors_row_splits: A `Tensor` of type `int64`. The exclusive prefix sum of the neighbor count for the
          query points including the total neighbor count as the last element. The
          size of this array is the number of queries + 1.
        neighbors_distance: A `Tensor`. Has the same type as `points`. Stores the distance to each neighbor if **return_distances**
          is True. The distances are squared only if metric is L2. This is a zero length
          Tensor if **return_distances** is False.
  
    """
    return _lib.open3d_knn_search(points=points,
                                  queries=queries,
                                  k=k,
                                  points_row_splits=points_row_splits,
                                  queries_row_splits=queries_row_splits,
                                  index_dtype=index_dtype,
                                  metric=metric,
                                  ignore_query_point=ignore_query_point,
                                  return_distances=return_distances,
                                  name=name)


def nms(boxes, scores, nms_overlap_thresh, name=None):
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

      Args:
        boxes: A `Tensor`. Must be one of the following types: `float32`.
          (N, 5) float32 tensor. Bounding boxes are represented as (x0, y0, x1, y1, rotate).
        scores: A `Tensor`. Must have the same type as `boxes`.
          (N,) float32 tensor. A higher score means a more confident bounding box.
        nms_overlap_thresh: A `float`.
          float value between 0 and 1. When a high-score box is
          selected, other remaining boxes with IoU > nms_overlap_thresh will be discarded.
          A higher nms_overlap_thresh means more boxes will be kept.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `int64`. (M,) int64 tensor. The selected box indices.
  
    """
    return _lib.open3d_nms(boxes=boxes,
                           scores=scores,
                           nms_overlap_thresh=nms_overlap_thresh,
                           name=name)


def radius_search(points,
                  queries,
                  radii,
                  points_row_splits,
                  queries_row_splits,
                  index_dtype=_tf.int32,
                  metric="L2",
                  ignore_query_point=False,
                  return_distances=False,
                  normalize_distances=False,
                  name=None):
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

      Args:
        points: A `Tensor`. Must be one of the following types: `float32`, `float64`.
          The 3D positions of the input points.
        queries: A `Tensor`. Must have the same type as `points`.
          The 3D positions of the query points.
        radii: A `Tensor`. Must have the same type as `points`.
          A vector with the individual radii for each query point.
        points_row_splits: A `Tensor` of type `int64`.
          1D vector with the row splits information if points is
          batched. This vector is [0, num_points] if there is only 1 batch item.
        queries_row_splits: A `Tensor` of type `int64`.
          1D vector with the row splits information if queries is
          batched. This vector is [0, num_queries] if there is only 1 batch item.
        index_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
        metric: An optional `string` from: `"L1", "L2"`. Defaults to `"L2"`.
          Either L1 or L2. Default is L2
        ignore_query_point: An optional `bool`. Defaults to `False`.
          If true the points that coincide with the center of the
          search window will be ignored. This excludes the query point if **queries** and
          **points** are the same point cloud.
        return_distances: An optional `bool`. Defaults to `False`.
          If True the distances for each neighbor will be returned in
          the output tensor **neighbors_distance**.  If False a zero length Tensor will
          be returned for **neighbors_distances**.
        normalize_distances: An optional `bool`. Defaults to `False`.
          If True the returned distances will be normalized with the
          radii.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (neighbors_index, neighbors_row_splits, neighbors_distance).

        neighbors_index: A `Tensor` of type `index_dtype`. The compact list of indices of the neighbors. The
          corresponding query point can be inferred from the
          **neighbor_count_row_splits** vector.
        neighbors_row_splits: A `Tensor` of type `int64`. The exclusive prefix sum of the neighbor count for the
          query points including the total neighbor count as the last element. The
          size of this array is the number of queries + 1.
        neighbors_distance: A `Tensor`. Has the same type as `points`. Stores the distance to each neighbor if **return_distances**
          is True. The distances are squared only if metric is L2.
          This is a zero length Tensor if **return_distances** is False.
  
    """
    return _lib.open3d_radius_search(points=points,
                                     queries=queries,
                                     radii=radii,
                                     points_row_splits=points_row_splits,
                                     queries_row_splits=queries_row_splits,
                                     index_dtype=index_dtype,
                                     metric=metric,
                                     ignore_query_point=ignore_query_point,
                                     return_distances=return_distances,
                                     normalize_distances=normalize_distances,
                                     name=name)


def reduce_subarrays_sum(values, row_splits, name=None):
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

      Args:
        values: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
          Linear memory which stores the values for all arrays.
        row_splits: A `Tensor` of type `int64`.
          Defines the start and end of each subarray. This is an exclusive
          prefix sum with 0 as the first element and the length of values as
          additional last element. If there are N subarrays the length of this vector
          is N+1.
        name: A name for the operation (optional).

      Returns:
        A `Tensor`. Has the same type as `values`.
        The sum of each subarray. The sum of an empty subarray is 0.
        sums is a zero length vector if values is a zero length vector.
  
    """
    return _lib.open3d_reduce_subarrays_sum(values=values,
                                            row_splits=row_splits,
                                            name=name)


def roi_pool(xyz, boxes3d, pts_feature, sampled_pts_num, name=None):
    """ TODO

      Args:
        xyz: A `Tensor` of type `float32`.
        boxes3d: A `Tensor` of type `float32`.
        pts_feature: A `Tensor` of type `float32`.
        sampled_pts_num: An `int`.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (feats, flags).

        feats: A `Tensor` of type `float32`.
        flags: A `Tensor` of type `int32`.
  
    """
    return _lib.open3d_roi_pool(xyz=xyz,
                                boxes3d=boxes3d,
                                pts_feature=pts_feature,
                                sampled_pts_num=sampled_pts_num,
                                name=name)


def sparse_conv(filters,
                inp_features,
                inp_importance,
                neighbors_index,
                neighbors_kernel_index,
                neighbors_importance,
                neighbors_row_splits,
                output_type=_tf.float32,
                normalize=False,
                max_temp_mem_MB=64,
                name=None):
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

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.

          The filter parameters.
          The shape of the filter is [depth, height, width, in_ch, out_ch].
          The dimensions 'depth', 'height', 'width' define the spatial resolution of
          the filter. The spatial size of the filter is defined by the parameter
          'extents'.
        inp_features: A `Tensor`. Must have the same type as `filters`. 
          A 2D tensor which stores a feature vector for each input point.
        inp_importance: A `Tensor`. Must have the same type as `filters`. 
          An optional scalar importance for each input point. The features of each point
          will be multiplied with the corresponding value. The shape is [num input points].
          Use a zero length Tensor to disable.
        neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.

          The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
          The start and end of each list can be computed using 'neighbors_row_splits'.
        neighbors_kernel_index: A `Tensor`. Must be one of the following types: `uint8`, `int16`.

          Defines which kernel element to use for each neighbor. This array has the same length as neighbors_index.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`. 
          Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
          the features of each neighbor. Use a zero length Tensor to weigh each neighbor
          with 1.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the output points including
          the total neighbor count as the last element. The size of this array is the
          number of output points + 1.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.bfloat16`. Defaults to `tf.float32`.
          The type for the output.
        normalize: An optional `bool`. Defaults to `False`. 
          If True the output feature values will be normalized using the sum for
          'neighbors_importance' for each output point.
        max_temp_mem_MB: An optional `int`. Defaults to `64`. 
          Defines the maximum temporary memory in megabytes to be used for the GPU
          implementation. More memory means fewer kernel invocations. Note that the
          a minimum amount of temp memory will always be allocated even if this
          variable is set to 0.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`. 
        A Tensor with the output feature vectors for each output point.
  
    """
    return _lib.open3d_sparse_conv(
        filters=filters,
        inp_features=inp_features,
        inp_importance=inp_importance,
        neighbors_index=neighbors_index,
        neighbors_kernel_index=neighbors_kernel_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        output_type=output_type,
        normalize=normalize,
        max_temp_mem_MB=max_temp_mem_MB,
        name=name)


def sparse_conv_backprop_filter(filters,
                                inp_features,
                                inp_importance,
                                neighbors_index,
                                neighbors_kernel_index,
                                neighbors_importance,
                                neighbors_row_splits,
                                out_features_gradient,
                                output_type=_tf.float32,
                                normalize=False,
                                max_temp_mem_MB=64,
                                name=None):
    """Computes the backprop for the filter of the SparseConv

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.

          The filter parameters.
          The shape of the filter is [depth, height, width, in_ch, out_ch].
          The dimensions 'depth', 'height', 'width' define the spatial resolution of
          the filter. The spatial size of the filter is defined by the parameter
          'extents'.
        inp_features: A `Tensor`. Must have the same type as `filters`. 
          A 2D tensor which stores a feature vector for each input point.
        inp_importance: A `Tensor`. Must have the same type as `filters`.
        neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.

          The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
          The start and end of each list can be computed using 'neighbors_row_splits'.
        neighbors_kernel_index: A `Tensor`. Must be one of the following types: `uint8`, `int16`.

          Defines which kernel element to use for each neighbor. This array has the same length as neighbors_index.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`. 
          Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
          the features of each neighbor.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the output points including
          the total neighbor count as the last element. The size of this array is the
          number of output points + 1.
        out_features_gradient: A `Tensor`. Must have the same type as `filters`. 
          A Tensor with the gradient for the outputs of the SparseConv in the forward pass.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.bfloat16`. Defaults to `tf.float32`.
          The type for the output.
        normalize: An optional `bool`. Defaults to `False`. 
          If True the output feature values will be normalized by the number of neighbors.
        max_temp_mem_MB: An optional `int`. Defaults to `64`. 
          Defines the maximum temporary memory in megabytes to be used for the GPU
          implementation. More memory means fewer kernel invocations. Note that the
          a minimum amount of temp memory will always be allocated even if this
          variable is set to 0.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`. 
        The gradients for the filter
  
    """
    return _lib.open3d_sparse_conv_backprop_filter(
        filters=filters,
        inp_features=inp_features,
        inp_importance=inp_importance,
        neighbors_index=neighbors_index,
        neighbors_kernel_index=neighbors_kernel_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        out_features_gradient=out_features_gradient,
        output_type=output_type,
        normalize=normalize,
        max_temp_mem_MB=max_temp_mem_MB,
        name=name)


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
                          output_type=_tf.float32,
                          normalize=False,
                          max_temp_mem_MB=64,
                          name=None):
    """Sparse tranpose convolution of two pointclouds.

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.

          The filter parameters.
          The shape of the filter is [depth, height, width, in_ch, out_ch].
          The dimensions 'depth', 'height', 'width' define the spatial resolution of
          the filter. The spatial size of the filter is defined by the parameter
          'extents'.
        out_importance: A `Tensor`. Must have the same type as `filters`. 
          An optional scalar importance for each output point. The output features of
          each point will be multiplied with the corresponding value.
          The shape is [num input points]. Use a zero length Tensor to disable.
        inp_features: A `Tensor`. Must have the same type as `filters`. 
          A 2D tensor which stores a feature vector for each input point.
        inp_neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.

          The inp_neighbors_index stores a list of indices of neighbors for each input point as nested lists.
          The start and end of each list can be computed using 'inp_neighbors_row_splits'.
        inp_neighbors_importance_sum: A `Tensor`. Must have the same type as `filters`.

          1D tensor of the same length as 'inp_features' or zero length if
          neighbors_importance is empty. This is the sum of the values in
          'neighbors_importance' for each input point.
        inp_neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the input points including
          the total neighbor count as the last element. The size of this array is the
          number of input points + 1.
        neighbors_index: A `Tensor`. Must have the same type as `inp_neighbors_index`.

          The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
          The start and end of each list can be computed using 'neighbors_row_splits'.
        neighbors_kernel_index: A `Tensor`. Must be one of the following types: `uint8`, `int16`.

          Defines which kernel element to use for each neighbor. This array has the same length as neighbors_index.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`. 
          Tensor of the same shape as 'neighbors_index' with a scalar value that is used to scale
          the features of each neighbor.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The exclusive prefix sum of the neighbor count for the output points including
          the total neighbor count as the last element. The size of this array is the
          number of output points + 1.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.bfloat16`. Defaults to `tf.float32`.
          The type for the output.
        normalize: An optional `bool`. Defaults to `False`. 
          If True the input feature values will be normalized using
          'inp_neighbors_importance_sum'.
        max_temp_mem_MB: An optional `int`. Defaults to `64`. 
          Defines the maximum temporary memory in megabytes to be used for the GPU
          implementation. More memory means fewer kernel invocations. Note that the
          a minimum amount of temp memory will always be allocated even if this
          variable is set to 0.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`. 
        A Tensor with the output feature vectors for each output point.
  
    """
    return _lib.open3d_sparse_conv_transpose(
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
        output_type=output_type,
        normalize=normalize,
        max_temp_mem_MB=max_temp_mem_MB,
        name=name)


def sparse_conv_transpose_backprop_filter(filters,
                                          out_importance,
                                          inp_features,
                                          inp_neighbors_importance_sum,
                                          inp_neighbors_row_splits,
                                          neighbors_index,
                                          neighbors_kernel_index,
                                          neighbors_importance,
                                          neighbors_row_splits,
                                          out_features_gradient,
                                          output_type=_tf.float32,
                                          normalize=False,
                                          max_temp_mem_MB=64,
                                          name=None):
    """Computes the backrop for the filter of the SparseConvTranspose

      Args:
        filters: A `Tensor`. Must be one of the following types: `float32`, `float64`, `bfloat16`.

          The filter parameters.
          The shape of the filter is [depth, height, width, in_ch, out_ch].
          The dimensions 'depth', 'height', 'width' define the spatial resolution of
          the filter. The spatial size of the filter is defined by the parameter
          'extents'.
        out_importance: A `Tensor`. Must have the same type as `filters`.
        inp_features: A `Tensor`. Must have the same type as `filters`. 
          A 2D tensor which stores a feature vector for each input point.
        inp_neighbors_importance_sum: A `Tensor`. Must have the same type as `filters`.
        inp_neighbors_row_splits: A `Tensor` of type `int64`. 
          The number of neighbors for each input point as exclusive prefix sum.
        neighbors_index: A `Tensor`. Must be one of the following types: `int32`, `int64`.

          The neighbors_index stores a list of indices of neighbors for each output point as nested lists.
          The start and end of each list can be computed using 'neighbors_row_splits'.
        neighbors_kernel_index: A `Tensor`. Must be one of the following types: `uint8`, `int16`.

          Defines which kernel element to use for each neighbor. This array has the same length as neighbors_index.
        neighbors_importance: A `Tensor`. Must have the same type as `filters`.
        neighbors_row_splits: A `Tensor` of type `int64`. 
          The number of neighbors for each output point as exclusive prefix sum.
        out_features_gradient: A `Tensor`. Must have the same type as `filters`. 
          A Tensor with the gradient for the outputs of the SparseConvTranspose in the forward pass.
        output_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.bfloat16`. Defaults to `tf.float32`.
          The type for the output.
        normalize: An optional `bool`. Defaults to `False`. 
          If True the input feature values will be normalized by the number of neighbors.
        max_temp_mem_MB: An optional `int`. Defaults to `64`. 
          Defines the maximum temporary memory in megabytes to be used for the GPU
          implementation. More memory means fewer kernel invocations. Note that the
          a minimum amount of temp memory will always be allocated even if this
          variable is set to 0.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `output_type`. 
        The gradients for the filter
  
    """
    return _lib.open3d_sparse_conv_transpose_backprop_filter(
        filters=filters,
        out_importance=out_importance,
        inp_features=inp_features,
        inp_neighbors_importance_sum=inp_neighbors_importance_sum,
        inp_neighbors_row_splits=inp_neighbors_row_splits,
        neighbors_index=neighbors_index,
        neighbors_kernel_index=neighbors_kernel_index,
        neighbors_importance=neighbors_importance,
        neighbors_row_splits=neighbors_row_splits,
        out_features_gradient=out_features_gradient,
        output_type=output_type,
        normalize=normalize,
        max_temp_mem_MB=max_temp_mem_MB,
        name=name)


def three_interpolate(points, idx, weights, name=None):
    """ TODO

      Args:
        points: A `Tensor` of type `float32`.
        idx: A `Tensor` of type `int32`.
        weights: A `Tensor` of type `float32`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `float32`.
  
    """
    return _lib.open3d_three_interpolate(points=points,
                                         idx=idx,
                                         weights=weights,
                                         name=name)


def three_interpolate_grad(grad_out, idx, weights, M, name=None):
    """ TODO

      Args:
        grad_out: A `Tensor` of type `float32`.
        idx: A `Tensor` of type `int32`.
        weights: A `Tensor` of type `float32`.
        M: An `int`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `float32`.
  
    """
    return _lib.open3d_three_interpolate_grad(grad_out=grad_out,
                                              idx=idx,
                                              weights=weights,
                                              M=M,
                                              name=name)


def three_nn(query_pts, data_pts, name=None):
    """ TODO

      Args:
        query_pts: A `Tensor` of type `float32`.
        data_pts: A `Tensor` of type `float32`.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (out_dist2, out_idx).

        out_dist2: A `Tensor` of type `float32`.
        out_idx: A `Tensor` of type `int32`.
  
    """
    return _lib.open3d_three_nn(query_pts=query_pts,
                                data_pts=data_pts,
                                name=name)


def trilinear_devoxelize(coords, features, resolution, is_training, name=None):
    """Trilinear Devoxelize.

      This function takes a 3D voxel grid and a list of coordinates and
      computes interpolated features corresponding to each point.

      Minimal example::
        import open3d.ml.tf as ml3d

        coords = tf.Tensor(
              [[[0.2 0.0 0.0 1.0 1.5]
                [1.0 1.2 0.9 0.0 0.7]
                [0.2 0.6 0.8 0.7 1.1]]], shape=(1, 3, 5), dtype=float32)

        features = tf.Tensor(
        [[[[[0.  0.5]
            [0.6 0.4]]

           [[0.5 0.7]
            [0.6 0.5]]]

          [[[0.4 0.8]
            [0.6 0.3]]

           [[0.4 0.2]
            [0.8 0.6]]]

          [[[0.1 0.2]
            [0.  0.6]]

           [[0.9 0. ]
            [0.2 0.3]]]]], shape=(1, 3, 2, 2, 2), dtype=float32)

        ml3d.ops.trilinear_devoxelize(coords,
                                      features,
                                      resolution,
                                      is_training)

        # returns output tf.Tensor(
        #                array([[[0.564     , 0.508     , 0.436     , 0.64      , 0.5005    ],
        #                        [0.58400005, 0.39200002, 0.396     , 0.26000002, 0.47900003],
        #                        [0.14      , 0.36      , 0.45000002, 0.27      , 0.0975    ]]],
        #                shape=(1, 3, 5), dtype=float32)
        #
        #         indices tf.Tensor([[[ 2,  2,  0,  4,  5],
        #                             [ 3,  3,  1,  5,  6],
        #                             [ 2,  4,  2,  4,  7],
        #                             [ 3,  5,  3,  5,  8],
        #                             [ 6,  2,  0,  4,  9],
        #                             [ 7,  3,  1,  5, 10],
        #                             [ 6,  4,  2,  4, 11],
        #                             [ 7,  5,  3,  5, 12]]],
        #                 shape=(1, 8, 5), dtype=float32)
        #
        #         weights tf.Tensor([[[0.64000005, 0.31999996, 0.02      , 0.3       , 0.135     ],
        #                             [0.16000001, 0.48      , 0.08000002, 0.7       , 0.015     ],
        #                             [0.        , 0.08000001, 0.17999998, 0.        , 0.315     ],
        #                             [0.        , 0.12000003, 0.71999997, 0.        , 0.03500001],
        #                             [0.16000001, 0.        , 0.        , 0.        , 0.135     ],
        #                             [0.04      , 0.        , 0.        , 0.        , 0.015     ],
        #                             [0.        , 0.        , 0.        , 0.        , 0.315     ],
        #                             [0.        , 0.        , 0.        , 0.        , 0.03500001]]],
        #                 shape=(1, 8, 5), dtype=float32)

      Args:
        coords: A `Tensor` of type `float32`.
          List of 3D coordinates for which features to be interpolated.
          The shape of this tensor is [B, 3, N]. The range of coordinates is
          [0, resolution-1]. If all of the adjacent position of any coordinate are out
          of range, then the interpolated features will be 0. Voxel centers are at integral
          values of voxel grid.
        features: A `Tensor` of type `float32`.
          A voxel grid of shape [B, C, R, R, R]. Here R is resolution.
        resolution: An `int`.
          Integer attribute defining resolution of the voxel grid.
        is_training: A `bool`. Boolean variable for training phase.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (outputs, indices, weights).

        outputs: A `Tensor` of type `float32`. Features for each point. The shape of this tensor is [B, C, N].
        indices: A `Tensor` of type `int32`. Indices which are used to interpolate features. Shape is [B, 8, N].
        weights: A `Tensor` of type `float32`. Weights for each index used to interpolate features. Shape is [B, 8, N].
  
    """
    return _lib.open3d_trilinear_devoxelize(coords=coords,
                                            features=features,
                                            resolution=resolution,
                                            is_training=is_training,
                                            name=name)


def trilinear_devoxelize_grad(grad_y, indices, weights, resolution, name=None):
    """Gradient function for Trilinear Devoxelize op.

      This function takes feature gradients and indices, weights returned from
      the op and computes gradient for voxelgrid.

      Args:
        grad_y: A `Tensor` of type `float32`.
          Gradients for the interpolated features. Shape is [B, C, N].
        indices: A `Tensor` of type `int32`.
          Indices which are used to interpolate features. Shape is [B, 8, N].
        weights: A `Tensor` of type `float32`.
          Weights for each index used to interpolate features. Shape is [B, 8, N].
        resolution: An `int`. Integer attribute defining resolution of the grid.
        name: A name for the operation (optional).

      Returns:
        A `Tensor` of type `float32`.
        Output gradients for the voxel grid. Shape is [B, C, R, R, R]
  
    """
    return _lib.open3d_trilinear_devoxelize_grad(grad_y=grad_y,
                                                 indices=indices,
                                                 weights=weights,
                                                 resolution=resolution,
                                                 name=name)


def voxel_pooling(positions,
                  features,
                  voxel_size,
                  position_fn="average",
                  feature_fn="average",
                  debug=False,
                  name=None):
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

      Args:
        positions: A `Tensor`. Must be one of the following types: `float32`, `float64`.
          The point positions with shape [N,3] with N as the number of points.
        features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
          The feature vector with shape [N,channels].
        voxel_size: A `Tensor`. Must have the same type as `positions`.
          The voxel size.
        position_fn: An optional `string` from: `"average", "nearest_neighbor", "center"`. Defaults to `"average"`.
          Defines how the new point positions will be computed.
          The options are
            * "average" computes the center of gravity for the points within one voxel.
            * "nearest_neighbor" selects the point closest to the voxel center.
            * "center" uses the voxel center for the position of the generated point.
        feature_fn: An optional `string` from: `"average", "nearest_neighbor", "max"`. Defaults to `"average"`.
          Defines how the pooled features will be computed.
          The options are
            * "average" computes the average feature vector.
            * "nearest_neighbor" selects the feature vector of the point closest to the voxel center.
            * "max" uses the maximum feature among all points within the voxel.
        debug: An optional `bool`. Defaults to `False`.
          If true additional checks for debugging will be enabled.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (pooled_positions, pooled_features).

        pooled_positions: A `Tensor`. Has the same type as `positions`. The output point positions with shape [M,3] and M <= N.
        pooled_features: A `Tensor`. Has the same type as `features`. The output point features with shape [M,channels] and M <= N.
  
    """
    return _lib.open3d_voxel_pooling(positions=positions,
                                     features=features,
                                     voxel_size=voxel_size,
                                     position_fn=position_fn,
                                     feature_fn=feature_fn,
                                     debug=debug,
                                     name=name)


def voxel_pooling_grad(positions,
                       features,
                       voxel_size,
                       pooled_positions,
                       pooled_features_gradient,
                       position_fn="average",
                       feature_fn="average",
                       name=None):
    """Gradient for features in VoxelPooling. For internal use only.

      Args:
        positions: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`.
        voxel_size: A `Tensor`. Must have the same type as `positions`.
        pooled_positions: A `Tensor`. Must have the same type as `positions`.
        pooled_features_gradient: A `Tensor`. Must have the same type as `features`.
        position_fn: An optional `string` from: `"average", "nearest_neighbor", "center"`. Defaults to `"average"`.
        feature_fn: An optional `string` from: `"average", "nearest_neighbor", "max"`. Defaults to `"average"`.
        name: A name for the operation (optional).

      Returns:
        A `Tensor`. Has the same type as `features`.
  
    """
    return _lib.open3d_voxel_pooling_grad(
        positions=positions,
        features=features,
        voxel_size=voxel_size,
        pooled_positions=pooled_positions,
        pooled_features_gradient=pooled_features_gradient,
        position_fn=position_fn,
        feature_fn=feature_fn,
        name=name)


def voxelize(points,
             row_splits,
             voxel_size,
             points_range_min,
             points_range_max,
             max_points_per_voxel=9223372036854775807,
             max_voxels=9223372036854775807,
             name=None):
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

      Args:
        points: A `Tensor`. Must be one of the following types: `float32`, `float64`.
          The point positions with shape [N,D] with N as the number of points and
          D as the number of dimensions, which must be 0 < D < 9.
        row_splits: A `Tensor` of type `int64`.
          1D vector with row splits information if points is batched. This
          vector is [0, num_points] if there is only 1 batch item.
        voxel_size: A `Tensor`. Must have the same type as `points`.
          The voxel size with shape [D].
        points_range_min: A `Tensor`. Must have the same type as `points`.
          The maximum range for valid points to be voxelized. This
          vector has shape [D].
        points_range_max: A `Tensor`. Must have the same type as `points`.
        max_points_per_voxel: An optional `int`. Defaults to `9223372036854775807`.
          The maximum number of points to consider for a voxel.
        max_voxels: An optional `int`. Defaults to `9223372036854775807`.
          The maximum number of voxels to generate per batch.
        name: A name for the operation (optional).

      Returns:
        A tuple of `Tensor` objects (voxel_coords, voxel_point_indices, voxel_point_row_splits, voxel_batch_splits).

        voxel_coords: A `Tensor` of type `int32`. The integer voxel coordinates.The shape of this tensor is [M, D]
          with M as the number of voxels and D as the number of dimensions.
        voxel_point_indices: A `Tensor` of type `int64`. A flat list of all the points that have been voxelized.
          The start and end of each voxel is defined in voxel_point_row_splits.
        voxel_point_row_splits: A `Tensor` of type `int64`. This is an exclusive prefix sum that includes the total
          number of points in the last element. This can be used to find the start and
          end of the point indices for each voxel. The shape of this tensor is [M+1].
        voxel_batch_splits: A `Tensor` of type `int64`. This is a prefix sum of number of voxels per batch. This can
          be used to find voxel_coords and row_splits corresponding to any particular
          batch.
  
    """
    return _lib.open3d_voxelize(points=points,
                                row_splits=row_splits,
                                voxel_size=voxel_size,
                                points_range_min=points_range_min,
                                points_range_max=points_range_max,
                                max_points_per_voxel=max_points_per_voxel,
                                max_voxels=max_voxels,
                                name=name)
