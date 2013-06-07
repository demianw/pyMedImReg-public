from .basis import Metric
import numpy


__all__ = ['ExactLandmarkL2']


class ExactLandmarkL2(Metric):
    r'''
    Extact Landmark matching metric based in the L2 metric.

    For two pointsets :math:`X,Y \in \Re^{N\times d}` where :math:`N` is the number of points
    and :math:`d` the dimension

    .. math::
        S(X, Y) = \sum_{i \leq N} \sum_{j \leq d} (X_{i,j} - Y_{i, j})^2
        \nabla_{X_i} S(X, Y) = 2(X_i - Y_i)
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
