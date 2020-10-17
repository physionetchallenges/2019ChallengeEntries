"""Implements various metric learning losses."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.summary import summary
try:
  # pylint: disable=g-import-not-at-top
  from sklearn import metrics
  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False


def pairwise_distance(feature, squared=False):
  """Computes the pairwise distance matrix with numerical stability.
  output[i, j] = || feature[i, :] - feature[j, :] ||_2
  Args:
    feature: 2-D Tensor of size [number of data, feature dimension].
    squared: Boolean, whether or not to square the pairwise distances.
  Returns:
    pairwise_distances: 2-D Tensor of size [number of data, number of data].
  """
  pairwise_distances_squared = math_ops.add(
      math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
      math_ops.reduce_sum(
          math_ops.square(array_ops.transpose(feature)),
          axis=[0],
          keepdims=True)) - 2.0 * math_ops.matmul(feature,
                                                  array_ops.transpose(feature))

  # Deal with numerical inaccuracies. Set small negatives to zero.
  pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
  # Get the mask where the zero distances are at.
  error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

  # Optionally take the sqrt.
  if squared:
    pairwise_distances = pairwise_distances_squared
  else:
    pairwise_distances = math_ops.sqrt(
        pairwise_distances_squared +
        math_ops.cast(error_mask, dtypes.float32) * 1e-16)    # type conversion

  # Undo conditionally adding 1e-16.
  pairwise_distances = math_ops.multiply(
      pairwise_distances,
      math_ops.cast(math_ops.logical_not(error_mask), dtypes.float32))

  num_data = array_ops.shape(feature)[0]
  # Explicitly set diagonals to zero.
  mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
      array_ops.ones([num_data]))
  pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
  return pairwise_distances


def contrastive_loss(labels, embeddings_anchor, embeddings_positive,
                     margin=1.0):
  """Computes the contrastive loss.
  This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
  See: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      binary labels indicating positive vs negative pair.
    embeddings_anchor: 2-D float `Tensor` of embedding vectors for the anchor
      images. Embeddings should be l2 normalized.
    embeddings_positive: 2-D float `Tensor` of embedding vectors for the
      positive images. Embeddings should be l2 normalized.
    margin: margin term in the loss definition.
  Returns:
    contrastive_loss: tf.float32 scalar.
  """
  # Get per pair distances
  distances = math_ops.sqrt(
      math_ops.reduce_sum(
          math_ops.squared_difference(embeddings_anchor, embeddings_positive),
          1))

  # Add contrastive loss for the siamese network.
  #   label here is {0,1} for neg, pos.
  return math_ops.reduce_mean(
      math_ops.cast(labels, dtypes.float32) * math_ops.square(distances) +
      (1. - math_ops.cast(labels, dtypes.float32)) *
      math_ops.square(math_ops.maximum(margin - distances, 0.)),
      name='contrastive_loss')


def masked_maximum(data, mask, dim=1):
  """Computes the axis wise maximum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the maximum.
  Returns:
    masked_maximums: N-D `Tensor`.
      The maximized dimension is of size 1 after the operation.
  """
  axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
  masked_maximums = math_ops.reduce_max(
      math_ops.multiply(data - axis_minimums, mask), dim,
      keepdims=True) + axis_minimums
  return masked_maximums


def masked_minimum(data, mask, dim=1):
  """Computes the axis wise minimum over chosen elements.
  Args:
    data: 2-D float `Tensor` of size [n, m].
    mask: 2-D Boolean `Tensor` of size [n, m].
    dim: The dimension over which to compute the minimum.
  Returns:
    masked_minimums: N-D `Tensor`.
      The minimized dimension is of size 1 after the operation.
  """
  axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
  masked_minimums = math_ops.reduce_min(
      math_ops.multiply(data - axis_maximums, mask), dim,
      keepdims=True) + axis_maximums
  return masked_minimums


def triplet_semihard_loss(labels, embeddings, margin=1.0):
  """Computes the triplet loss with semi-hard negative mining.
  The loss encourages the positive distances (between a pair of embeddings with
  the same labels) to be smaller than the minimum negative distance among
  which are at least greater than the positive distance plus the margin constant
  (called semi-hard negative) in the mini-batch. If no such negative exists,
  uses the largest negative distance instead.
  See: https://arxiv.org/abs/1503.03832.
  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
      be l2 normalized.
    margin: Float, margin term in the loss definition.
  Returns:
    triplet_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pdist_matrix = pairwise_distance(embeddings, squared=True)
  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  # Compute the mask.
  pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
  mask = math_ops.logical_and(
      array_ops.tile(adjacency_not, [batch_size, 1]),
      math_ops.greater(
          pdist_matrix_tile, array_ops.reshape(
              array_ops.transpose(pdist_matrix), [-1, 1])))
  mask_final = array_ops.reshape(
      math_ops.greater(
          math_ops.reduce_sum(
              math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
          0.0), [batch_size, batch_size])
  mask_final = array_ops.transpose(mask_final)

  adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  mask = math_ops.cast(mask, dtype=dtypes.float32)

  # negatives_outside: smallest D_an where D_an > D_ap.
  negatives_outside = array_ops.reshape(
      masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
  negatives_outside = array_ops.transpose(negatives_outside)

  # negatives_inside: largest D_an.
  negatives_inside = array_ops.tile(
      masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
  semi_hard_negatives = array_ops.where(
      mask_final, negatives_outside, negatives_inside)

  loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

  mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

  # In lifted-struct, the authors multiply 0.5 for upper triangular
  #   in semihard, they take all positive pairs except the diagonal.
  num_positives = math_ops.reduce_sum(mask_positives)

  triplet_loss = math_ops.truediv(
      math_ops.reduce_sum(
          math_ops.maximum(
              math_ops.multiply(loss_mat, mask_positives), 0.0)),
      num_positives,
      name='triplet_semihard_loss')

  return triplet_loss


def lifted_struct_loss(labels, embeddings, margin=1.0):
  """Computes the lifted structured loss.
  The loss encourages the positive distances (between a pair of embeddings
  with the same labels) to be smaller than any negative distances (between a
  pair of embeddings with different labels) in the mini-batch in a way
  that is differentiable with respect to the embedding vectors.
  See: https://arxiv.org/abs/1511.06452.
  Args:
    labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
      multiclass integer labels.
    embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should not
      be l2 normalized.
    margin: Float, margin term in the loss definition.
  Returns:
    lifted_loss: tf.float32 scalar.
  """
  # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
  lshape = array_ops.shape(labels)
  assert lshape.shape == 1
  labels = array_ops.reshape(labels, [lshape[0], 1])

  # Build pairwise squared distance matrix.
  pairwise_distances = pairwise_distance(embeddings)

  # Build pairwise binary adjacency matrix.
  adjacency = math_ops.equal(labels, array_ops.transpose(labels))
  # Invert so we can select negatives only.
  adjacency_not = math_ops.logical_not(adjacency)

  batch_size = array_ops.size(labels)

  diff = margin - pairwise_distances
  mask = math_ops.cast(adjacency_not, dtype=dtypes.float32)
  # Safe maximum: Temporarily shift negative distances
  #   above zero before taking max.
  #     this is to take the max only among negatives.
  row_minimums = math_ops.reduce_min(diff, 1, keepdims=True)
  row_negative_maximums = math_ops.reduce_max(
      math_ops.multiply(diff - row_minimums, mask), 1,
      keepdims=True) + row_minimums

  # Compute the loss.
  # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
  #   where m_i is the max of alpha - negative D_i's.
  # This matches the Caffe loss layer implementation at:
  #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp  # pylint: disable=line-too-long

  max_elements = math_ops.maximum(
      row_negative_maximums, array_ops.transpose(row_negative_maximums))
  diff_tiled = array_ops.tile(diff, [batch_size, 1])
  mask_tiled = array_ops.tile(mask, [batch_size, 1])
  max_elements_vect = array_ops.reshape(
      array_ops.transpose(max_elements), [-1, 1])

  loss_exp_left = array_ops.reshape(
      math_ops.reduce_sum(
          math_ops.multiply(
              math_ops.exp(diff_tiled - max_elements_vect), mask_tiled),
          1,
          keepdims=True), [batch_size, batch_size])

  loss_mat = max_elements + math_ops.log(
      loss_exp_left + array_ops.transpose(loss_exp_left))
  # Add the positive distance.
  loss_mat += pairwise_distances

  mask_positives = math_ops.cast(
      adjacency, dtype=dtypes.float32) - array_ops.diag(
          array_ops.ones([batch_size]))

  # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
  num_positives = math_ops.reduce_sum(mask_positives) / 2.0

  lifted_loss = math_ops.truediv(
      0.25 * math_ops.reduce_sum(
          math_ops.square(
              math_ops.maximum(
                  math_ops.multiply(loss_mat, mask_positives), 0.0))),
      num_positives,
      name='liftedstruct_loss')
  return lifted_loss


