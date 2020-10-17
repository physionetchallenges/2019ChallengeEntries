# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for triplet_semihard_loss."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
# import session as sess
from tensorflow.contrib.losses.python import metric_learning as metric_loss_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
try:
  # pylint: disable=g-import-not-at-top
  from sklearn import datasets
  from sklearn import metrics
  HAS_SKLEARN = True
except ImportError:
  HAS_SKLEARN = False


def pairwise_distance_np(feature, squared=False):
  """Computes the pairwise distance matrix in numpy.

  Args:
    feature: 2-D numpy array of size [number of data, feature dimension]
    squared: Boolean. If true, output is the pairwise squared euclidean
      distance matrix; else, output is the pairwise euclidean distance matrix.

  Returns:
    pairwise_distances: 2-D numpy array of size
      [number of data, number of data].
  """
  triu = np.triu_indices(feature.shape[0], 1)
  upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
  if squared:
    upper_tri_pdists **= 2.
  num_data = feature.shape[0]
  pairwise_distances = np.zeros((num_data, num_data))
  pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
  # Make symmetrical.
  pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
      pairwise_distances.diagonal())
  return pairwise_distances


class TripletSemiHardLossTest():
    def testTripletSemiHard(self):
      with sess.as_default():
          num_data = 10
          feat_dim = 6
          margin = 1.0
          num_classes = 4

          embedding = np.random.rand(num_data, feat_dim).astype(np.float32)  # uniform distribution over [0, 1]
          labels = np.random.randint(
              0, num_classes, size=(num_data)).astype(np.float32)

          # Reshape labels to compute adjacency matrix.
          labels_reshaped = np.reshape(labels, (labels.shape[0], 1))
          # Compute the loss in NP.
          adjacency = np.equal(labels_reshaped, labels_reshaped.T)

          pdist_matrix = pairwise_distance_np(embedding, squared=True)
          loss_np = 0.0
          num_positives = 0.0
          for i in range(num_data):
            for j in range(num_data):
              if adjacency[i][j] > 0.0 and i != j:
                num_positives += 1.0

                pos_distance = pdist_matrix[i][j]
                neg_distances = []

                for k in range(num_data):
                  if adjacency[i][k] == 0:
                    neg_distances.append(pdist_matrix[i][k])

                # Sort by distance.
                neg_distances.sort()
                chosen_neg_distance = neg_distances[0]

                for l in range(len(neg_distances)):
                  chosen_neg_distance = neg_distances[l]
                  if chosen_neg_distance > pos_distance:
                    break

                loss_np += np.maximum(
                    0.0, margin - chosen_neg_distance + pos_distance)

          loss_np /= num_positives

          # Compute the loss in TF.
          loss_tf = metric_loss_ops.triplet_semihard_loss(
              labels=ops.convert_to_tensor(labels),
              embeddings=ops.convert_to_tensor(embedding),
              margin=margin)
          loss_tf = loss_tf.eval()


def convert_to_list_of_sparse_tensor(np_matrix):
  list_of_sparse_tensors = []
  nrows, ncols = np_matrix.shape
  for i in range(nrows):
    sp_indices = []
    for j in range(ncols):
      if np_matrix[i][j] == 1:
        sp_indices.append([j])

    num_non_zeros = len(sp_indices)
    list_of_sparse_tensors.append(sparse_tensor.SparseTensor(
        indices=np.array(sp_indices),
        values=np.ones((num_non_zeros,)),
        dense_shape=np.array([ncols,])))

  return list_of_sparse_tensors


def compute_ground_truth_cluster_score(feat, y):
  y_unique = np.unique(y)
  score_gt_np = 0.0
  for c in y_unique:
    feat_subset = feat[y == c, :]
    pdist_subset = pairwise_distance_np(feat_subset)
    score_gt_np += -1.0 * np.min(np.sum(pdist_subset, axis=0))
  score_gt_np = score_gt_np.astype(np.float32)
  return score_gt_np


if __name__ == '__main__':
    a=TripletSemiHardLossTest()
    a.testTripletSemiHard()