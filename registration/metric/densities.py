from _metrics import gauss_transform
from .basis import Metric
import numpy

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

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

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
    def __init__(self,  points_moving, points_fixed, sigma, transform=None):
        self.points_fixed = numpy.require(points_fixed, requirements='C', dtype=numpy.float64)
        self.sigma = sigma
        self.fixed_norm, self.value_ff, _, _ = gauss_transform(self.points_fixed, self.points_fixed, sigma)
        self.transform = transform

        if points_moving is not None:
            self.points_moving = numpy.require(points_moving, requirements='C', dtype=numpy.float64)
        else:
            self.points_moving = None

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None or self.points_moving is None:
            return ValueError('transform and points_moving must be set for this')

        self.transform.parameter = parameter
        points_moving = numpy.ascontiguousarray(self.transform.transform_points(self.points_moving))

        fg, value_gf, _, grad_gf = gauss_transform(
            points_moving, self.points_fixed, self.sigma
        )

        moving_norm, value_gg, _, _ = gauss_transform(
            points_moving, points_moving, self.sigma
        )

        fg_sqr = fg ** 2

        f = -fg_sqr / (self.fixed_norm * moving_norm)

        grad = -(2 * fg / (self.fixed_norm * moving_norm) * (value_gg - fg / moving_norm * value_gf) * grad_gf)

        jacobian = self.transform.jacobian(points_moving)
        grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

        return f, grad

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        fg, value_gf, value_fg, grad_gf = gauss_transform(
            points_moving, self.points_fixed, self.sigma
        )

        moving_norm, value_gg, _, _ = gauss_transform(
            points_moving, points_moving, self.sigma
        )

        fg_sqr = fg ** 2

        f = -fg_sqr / (self.fixed_norm * moving_norm)

        grad = (2 * fg / (self.fixed_norm * moving_norm) * (value_gg - fg / moving_norm * value_gf) * grad_gf)

        return f, grad


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
