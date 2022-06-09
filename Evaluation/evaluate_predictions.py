import argparse
import glob
import os
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KDTree
from Evaluation.icp import icp


# Original code found on
# https://github.com/olalium/face-reconstruction/blob/master/Evaluation/evaluate_predicitons.py
# Slightly adapted to current work

def apply_homogenous_tform(tform, vertices):
    n, m = vertices.shape
    vertices_affine = np.ones((n, m + 1))
    vertices_affine[:, :3] = vertices.copy()
    vertices = np.dot(tform, vertices_affine.T).T
    return vertices[:, :3]


class prediction_evaluater:

    def __call__(self, predicted_vertices, ground_truth_vertices, alignment_data=None, save_vertices=False,
                 save_output='aligned_vertices.obj'):
        if alignment_data is not None:
            init_pose = alignment_data[:4]
            scale = alignment_data[4][0]
        else:
            init_pose = None
            scale = 1.0
        original_predicted_vertices = predicted_vertices.copy() * scale
        original_f_vertices = ground_truth_vertices.copy()
        if (predicted_vertices.shape[0] > ground_truth_vertices.shape[0]):
            diff = predicted_vertices.shape[0] - ground_truth_vertices.shape[0]
            predicted_vertices = predicted_vertices[diff:, :] * scale
        else:
            diff = ground_truth_vertices.shape[0] - predicted_vertices.shape[0]
            ground_truth_vertices = ground_truth_vertices[diff:, :] * scale

        tform, distances, i = icp(predicted_vertices, ground_truth_vertices,
                                  max_iterations=100, tolerance=0.0001, init_pose=init_pose)

        aligned_predicted_vertices = apply_homogenous_tform(tform, predicted_vertices)
        aligned_original_vertices = apply_homogenous_tform(tform, original_predicted_vertices)

        error = self.nmse(aligned_original_vertices, original_f_vertices)
        return error

    def nmse(self, predicted_vertices, ground_truth_vertices, normalization_factor=None):
        # calculate the normalized mean squared error between a predicted and ground truth mesh
        if not normalization_factor:
            mins = np.amin(ground_truth_vertices, axis=0)
            maxes = np.amax(ground_truth_vertices, axis=0)
            bbox = np.sqrt((maxes[0] - mins[0]) ** 2 + (maxes[1] - mins[1]) ** 2 + (maxes[2] - mins[2]) ** 2)
            normalization_factor = bbox

        v_tree = KDTree(ground_truth_vertices)
        error_array = np.zeros(predicted_vertices.shape[0])
        for i, v in enumerate(predicted_vertices):
            dst, ind = v_tree.query([v], k=1)
            gt_v = ground_truth_vertices[ind[0][0]]
            error_array[i] = distance.euclidean(v, gt_v)

        nmse = np.mean(error_array) / normalization_factor
        print(nmse)
        return nmse
