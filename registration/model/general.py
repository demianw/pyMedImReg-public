r'''
General Unclassified Transforms
'''
import numpy
from .basis import Model
from ..decorator import doc_inherit, skip_from_test

__all__ = ['DenseTransform', 'DenseTransformWithHessian']


@skip_from_test
class DenseTransform(Model):
    r"""
        Simple transform for a fixed set of :math:`K` :math:`N`-dimensional points :math:`X \in \Re^{K\times N}`
        with parameter :math:`\theta = \operatorname{vec}(D), D\in \Re^{K\times N}`. The transformation is
        defined as

        .. math::
            Y_{i\cdot} = \phi(X_{i\cdot}; \theta) = X + D


        Attributes
        ----------
        `number_of_points`: the number of points :math:`K`

    """
    def __init__(self, number_of_points):
        self.number_of_points = number_of_points
        self.parameter = self.identity

    @property
    def identity(self):
        return numpy.zeros(self.number_of_points * 3, dtype=float)

    @doc_inherit
    def transform_points(self, points):
        displacements = self.parameter.reshape(self.number_of_points, 3)
        return points + displacements

    @doc_inherit
    def jacobian(self, points):
        jacobian = numpy.zeros((len(points), 3 * self.number_of_points, 3))
        for i in xrange(self.number_of_points):
            j = i * 3
            jacobian[i, (j, j + 1, j + 2), (0, 1, 2)] = 1

        return jacobian

    @doc_inherit
    def jacobian_position(self, points):
        raise NotImplementedError()

    @property
    def bounds(self):
        return numpy.c_[
            numpy.repeat(numpy.inf, len(self.number_of_points * 9)),
            numpy.repeat(numpy.inf, len(self.number_of_points * 9))
        ]


@skip_from_test
class DenseTransformWithHessian(Model):
    def __init__(self, number_of_points):
        self.number_of_points = number_of_points
        self.parameter = self.identity

    @property
    def identity(self):
        identity = numpy.zeros(self.number_of_points * 12, dtype=float)
        identity[3::9] = 1
        identity[7::9] = 1
        identity[11::9] = 1

    def transform_points(self, points):
        displacements = self.parameter.reshape(self.number_of_points, 9)[:, :3]
        return points + displacements

    def jacobian(self, points):
        jacobian = numpy.zeros((len(points), 3 * self.number_of_points, 3))
        for i in xrange(self.number_of_points):
            j = i * 3
            jacobian[i, (j, j + 1, j + 2), (0, 1, 2)] = 1

        return jacobian

    def jacobian_position(self, points):
        return self.parameter.reshape(self.number_of_points, 9)[:, 3:].reshape(self.number_of_points, 3, 3)

    @property
    def bounds(self):
        return numpy.c_[
            numpy.repeat(numpy.inf, len(self.number_of_points * 3)),
            numpy.repeat(numpy.inf, len(self.number_of_points * 3))
        ]
