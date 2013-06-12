from _metrics import gauss_transform
from .basis import Metric
import numpy

from ..decorator import skip_from_test

__all__ = ['SquaredDifference', 'Correlation', 'CorrelationWithTensorFeatures']


class SquaredDifference(Metric):
    def __init__(self, points_moving, points_fixed, sigma, transform=None):
        super(SquaredDifference, self).__init__(points_moving, transform=transform)
        self.points_fixed = numpy.ascontiguousarray(points_fixed, dtype=numpy.float64)
        self.points_moving = numpy.ascontiguousarray(points_moving, dtype=numpy.float64)
        self.sigma = sigma
        self.fixed_norm, self.value_ff, _, _ = gauss_transform(self.points_fixed, self.points_fixed, sigma)
        self.transform = transform
        self.points_moving = points_moving
        self.tensors = None
        self.vectors = None

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        gf, value_gf, _, grad_gf = gauss_transform(
            points_moving, self.points_fixed, self.sigma
        )

        moving_norm, value_gg, _, grad_gg = gauss_transform(
            points_moving, points_moving, self.sigma
        )

        f = self.fixed_norm + moving_norm - 2 * gf
        grad = 2 * grad_gg - 2 * grad_gf

        return f, grad


class Correlation(Metric):
    r"""
    Fixed variance correlation metric:
    Let :math:`Y\in \Re^{N\times 3}` be the fixed set of points and :math:`X\in\Re^{M\times 3}` the
    moving set of points. Then the metric is

    .. math::
        m(X, Y) = -\frac{\langle X, Y \rangle^2} {\|X\|^2\|Y\|^2} \propto -\frac{\langle X, Y \rangle^2} {\|X\|^2}

    where

    .. math::
        \langle X, Y\rangle =  \frac 1 {2\pi\sigma^2}\sum_{ij} \int
        e^{-.5 \left(\frac {X_i-\eta} \sigma\right)^2}  e^{-.5 \left(\frac {Y_j-\eta} \sigma\right)^2} d\eta

        \|X\|^2 = \langle X, X \rangle
    """
    def __init__(self,  points_moving, points_fixed, sigma, transform=None):
        self.points_fixed = numpy.require(points_fixed, requirements='C', dtype=numpy.float64)
        self.sigma = sigma
        self.fixed_norm, self.value_ff, _, _ = gauss_transform(self.points_fixed, self.points_fixed, sigma)
        self.transform = transform

        if points_moving is not None:
            self.points_moving = numpy.require(points_moving, requirements='C', dtype=numpy.float64)
        else:
            self.points_moving = None

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        fg, value_gf, value_fg, grad_gf = gauss_transform(
            points_moving, self.points_fixed, self.sigma
        )

        moving_norm, value_gg, _, grad_gg = gauss_transform(
            points_moving, points_moving, self.sigma
        )

        f = -fg * fg / moving_norm
        grad = 2 * fg / moving_norm * (
            grad_gg * (fg / moving_norm) -
            grad_gf
        )

        return f, grad


@skip_from_test
class CorrelationWithTensorFeatures(Metric):
    def __init__(self,  points_moving, tensors_moving, points_fixed, tensors_fixed, sigma, transform=None):
        self.points_fixed = numpy.require(points_fixed, requirements='C', dtype=numpy.float64)
        self.tensors_fixed = numpy.require(tensors_fixed, requirements='C', dtype=numpy.float64)
        self.sigma = sigma
        self.fixed_norm, self.value_ff, _, _ = gauss_process_w_covar_transform(
            self.points_fixed, self.points_fixed,
            self.tensors_fixed, self.tensors_fixed,
            None, None,
            sigma
        )
        self.transform = transform

        if points_moving is not None:
            self.points_moving = numpy.require(points_moving, requirements='C', dtype=numpy.float64)
            self.tensors_moving = numpy.require(tensors_moving, requirements='C', dtype=numpy.float64)
        else:
            self.points_moving = None

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = numpy.require(
                self.transform.transform_points(points_moving),
                requirements='C'
            )
            tensors = numpy.require(
                self.transform.transform_tensors(points_moving, tensors),
                requirements='C'
            )

        fg, value_gf, _, grad_gf = gauss_process_w_covar_transform(
            points_moving, self.points_fixed,
            tensors, self.tensors_fixed,
            None, None,
            self.sigma
        )

        moving_norm, value_gg, _, _ = gauss_process_w_covar_transform(
            points_moving, points_moving,
            self.tensors_moving, self.tensors_moving,
            None, None,
            self.sigma
        )

        fg_sqr = fg ** 2

        f = -fg_sqr / (self.fixed_norm * moving_norm)

        grad = (2 * fg / (self.fixed_norm * moving_norm) * (value_gg - fg / moving_norm * value_gf) * grad_gf)

        return f, grad, None
