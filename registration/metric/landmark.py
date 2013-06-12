from .basis import Metric
import numpy
from numpy import linalg


__all__ = ['ExactLandmarkL2', 'InexactLandmarkL2']


class ExactLandmarkL2(Metric):
    r'''
    Extact Landmark matching metric based in the L2 metric.

    For two pointsets :math:`X,Y \in \Re^{N\times d}` where :math:`N` is the number of points
    and :math:`d` the dimension. Let :math:`X` be the moving point set and :math:`Y` the fixed one

    .. math::
        m(X, Y) = \sum_{i \leq N} \sum_{j \leq d} (X_{i,j} - Y_{i, j})^2

        \nabla_{X_i} m(X, Y) = 2(X_i - Y_i)
    '''
    def __init__(self, points_moving, points_fixed, transform=None):
        super(ExactLandmarkL2, self).__init__(points_moving, transform=transform)
        self.points_fixed = numpy.ascontiguousarray(points_fixed, dtype=numpy.float64)
        self.points_moving = numpy.ascontiguousarray(points_moving, dtype=numpy.float64)
        self.transform = transform
        self.points_moving = points_moving
        self.tensors = None
        self.vectors = None

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        difference = points_moving - self.points_fixed
        cost = (difference ** 2).sum()
        grad = 2 * difference

        return cost, grad


class InexactLandmarkL2(Metric):
    r'''
    Inexact Landmark matching metric based in the L2 metric (Joshi et al. 2000).

    For two pointsets :math:`X,Y \in \Re^{N\times d}` where :math:`N` is the number of points
    and :math:`d` the dimension. Let :math:`X` be the moving point set and :math:`Y` the fixed one
    and let :math:`\{\Sigma\}_{1\ldots N}` be the covariance matrix denoting the precision of the
    location with respect to the fixed landmarks :math:`Y`.

    .. math::
        m(X, Y) = \sum_{i \leq N} (X_i - Y_i)^T \Sigma^{-1}_i (X_i - Y_i)

        \nabla_{X_i} m(X, Y) = 2(X_i - Y_i)^T \Sigma_i^{-1}
    '''
    def __init__(self, points_moving, points_fixed, covariances_fixed, transform=None):
        super(InexactLandmarkL2, self).__init__(points_moving, transform=transform)
        self.points_fixed = numpy.ascontiguousarray(points_fixed, dtype=numpy.float64)
        self.points_moving = numpy.ascontiguousarray(points_moving, dtype=numpy.float64)
        self.covariances_fixed = covariances_fixed
        self.inv_covariances_fixed = numpy.empty_like(self.covariances_fixed)
        for i, c in enumerate(self.covariances_fixed):
            self.inv_covariances_fixed[i] = linalg.inv(c)

        self.transform = transform
        self.points_moving = points_moving
        self.tensors = None
        self.vectors = None

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        cost = 0
        grad = numpy.empty_like(points_moving)
        difference = points_moving - self.points_fixed
        for i, d in enumerate(difference):
            cov = self.inv_covariances_fixed[i]
            dot = numpy.dot(cov, d)
            cost += (dot * d).sum()
            grad[i] = 2 * dot

        return cost, grad
