from .. import metric

from scipy.spatial import KDTree
import numpy
from numpy import linalg
from ..util import vectorized_dot_product


class EberlyBaseComputations:
    def __init__(self, image_function):
        self.image_function = image_function

    def compute_hessian_eigendecomp(self, points_moving, hessians=None):
        if hessians is None:
            hessians = self.image_function.hessian(points_moving)
        evals = numpy.empty((len(points_moving), 3))
        evecs = numpy.empty((len(points_moving), 3, 3))

        for i in xrange(len(points_moving)):
            try:
                evals_, evecs_ = linalg.eig(hessians[i])
            except numpy.linalg.LinAlgError:
                print hessians[i]
                evals[i] = 0
                evecs[i] = numpy.eye(3)
            order = evals_.argsort()[::-1]
            evals[i] = evals_[order]
            evecs[i] = evecs_[:, order]

        return hessians, evals, evecs


class SurfaceValley(metric.Metric, EberlyBaseComputations):
    def __init__(self, image_function, transform=None, points_moving=None):
        self.image_function = image_function
        self.transform = transform
        self.points_moving = points_moving
        self.image_function.transform = transform
        self.image_function.points_moving = points_moving
        self.one_dimensional_search = False

    def tangent_plane_normal(self, points_moving, use_transform=False):
        '''
        Return the tangent plane and the normal direction
        '''
        f, gradient = self.image_function.metric_gradient(points_moving, use_transform=use_transform)

        hessians, evals, evecs = self.compute_hessian_eigendecomp(points_moving)

        v11 = vectorized_dot_product(evecs[:, :, 0][:, :, None], evecs[:, :, 0][:, None, :])
        v22 = vectorized_dot_product(evecs[:, :, 1][:, :, None], evecs[:, :, 1][:, None, :])
        v33 = vectorized_dot_product(evecs[:, :, 2][:, :, None], evecs[:, :, 2][:, None, :])

        tangent_planes = (v22 + v33) / 2.
        return tangent_planes, evecs[:, :, 0]

    def metric_gradient(self, points_moving, use_transform=False):
        f, gradient = self.image_function.metric_gradient(points_moving, use_transform=use_transform)

        if self.one_dimensional_search:
            normal_planes = self.normal_planes
        else:
            hessians, evals, evecs = self.compute_hessian_eigendecomp(points_moving)

            v11 = vectorized_dot_product(evecs[:, :, 0][:, :, None], evecs[:, :, 0][:, None, :])
            #v22 = vectorized_dot_product(evecs[:, 1, :][:, :, None], evecs[:, 1, :][:, None, :])
            #v33 = vectorized_dot_product(evecs[:, 2, :][:, :, None], evecs[:, 2, :][:, None, :])

            normal_planes = v11
        ridge_gradients = vectorized_dot_product(normal_planes, gradient[:, :, None])[:, :, 0]
        return f, ridge_gradients

    def start_one_dimensional_search(self, points_moving):
        hessians, evals, evecs = self.compute_hessian_eigendecomp(points_moving)

        v11 = vectorized_dot_product(evecs[:, 0, :][:, :, None], evecs[:, 0, :][:, None, :])
        self.normal_planes = v11
        self.one_dimensional_search = True

    def stop_one_dimensional_search(self):
        self.one_dimensional_search = False


class TensorFieldCrease(metric.Metric):
    def __init__(self, crease_function, sigma):
        self.crease_function = crease_function
        self.sigma = sigma
        self.sigma2 = self.sigma ** 2
        self.normalizing_constant = self.sigma * numpy.sqrt(2 * numpy.pi)

    def tensor_field_value_gradient(self, points, points_to_sample):
        tensors = numpy.zeros((len(points_to_sample), 3, 3))
        tensor_gradient = numpy.zeros((len(points_to_sample), 3, 3))

        tangent_planes, _ = self.crease_function.tangent_plane_normal(points, use_transform=False)
        tree = KDTree(points)
        tree_to_sample = KDTree(points_to_sample)
        closest_points = tree_to_sample.query_ball_tree(tree, self.sigma * 4)

        for i, p in enumerate(points_to_sample):
            if len(closest_points[i]) == 0:
                continue
            dists2 = ((points[closest_points[i]] - p) ** 2).sum(-1)
            weights = numpy.exp(- .5 * dists2 / self.sigma2)
            #weights /= len(dists2)
            #weights /= self.normalizing_constant
            #weights /= weights.sum()
            tensors[i] = (
                tangent_planes[closest_points[i]] *
                weights[:, None, None]
            ).sum(0)

        return tensors

    def tensor_field_value_normal(self, points, points_to_sample, sigma_tangent=0, invert=False):
        tensors = numpy.zeros((len(points_to_sample), 3, 3))

        tangent_planes, normals = self.crease_function.tangent_plane_normal(points, use_transform=False)

        normal_tensors = vectorized_dot_product(normals[:, :, None], normals[:, None, :])

        if sigma_tangent > 0:
            normal_tensors += sigma_tangent * tangent_planes

        tree = KDTree(points)
        tree_to_sample = KDTree(points_to_sample)
        closest_points = tree_to_sample.query_ball_tree(tree, self.sigma * 4)

        for i, p in enumerate(points_to_sample):
            if len(closest_points[i]) == 0:
                continue
            dists2 = ((points[closest_points[i]] - p) ** 2).sum(-1)
            weights = numpy.exp(- .5 * dists2 / self.sigma2)
            #weights /= len(dists2)
            #weights /= self.normalizing_constant
            #weights /= weights.sum()
            tensors[i] = (
                normal_tensors[closest_points[i]] *
                weights[:, None, None]
            ).sum(0)

        return tensors
