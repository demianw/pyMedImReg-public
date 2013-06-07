import numpy
from .basis import Model
from numpy import cos, sin

from ..decorator import skip_from_test


__all__ = ['LinearTransform', 'Rigid', 'Translation', 'Affine', 'Similarity', 'Rotation', 'AnisotropicScale', 'IsotropicScale']


def affine_transform_points(affine_parameter, points):
    scales = affine_parameter[:3]
    cos_phi, cos_theta, cos_psi = cos(affine_parameter[3:-3])
    sin_phi, sin_theta, sin_psi = sin(affine_parameter[3:-3])

    displacement = affine_parameter[-3:][None, :]

    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    z = points[:, 2][:, None]
    return scales * numpy.c_[
        cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
        cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
        -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
    ] + displacement


def affine_jacobian_parameter_jacobian_position(parameter, center, points):
    scales = parameter[0: 3]
    cos_phi, cos_theta, cos_psi = cos(parameter[3: 6])
    sin_phi, sin_theta, sin_psi = sin(parameter[3: 6])

    jacobian = numpy.zeros((len(points), len(parameter), 3, 3))

    jacobian[:, 0, 0, 0] = cos_theta * cos_psi
    jacobian[:, 0, 0, 1] = cos_theta * sin_psi
    jacobian[:, 0, 0, 2] = -sin_theta

    jacobian[:, 1, 1, 0] = (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi)
    jacobian[:, 1, 1, 1] = (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi)
    jacobian[:, 1, 1, 2] = cos_theta * sin_phi

    jacobian[:, 2, 2, 0] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)
    jacobian[:, 2, 2, 1] = (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)
    jacobian[:, 2, 2, 2] = cos_phi * cos_theta

    jacobian[:, 3, 1, 0] = scales[1] * (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)
    jacobian[:, 3, 2, 0] = scales[2] * (cos_phi * sin_psi + -sin_phi * sin_theta * cos_psi)

    jacobian[:, 3, 1, 1] = scales[1] * (-sin_phi * cos_psi + sin_psi * sin_theta * cos_phi)
    jacobian[:, 3, 2, 1] = scales[2] * (-cos_phi * cos_psi + -sin_phi * sin_theta * sin_psi)

    jacobian[:, 3, 1, 2] = scales[1] * cos_theta * cos_phi
    jacobian[:, 3, 2, 2] = scales[2] * -sin_phi * cos_theta

    jacobian[:, 4, 0, 0] = scales[0] * -sin_theta * cos_psi
    jacobian[:, 4, 1, 0] = scales[1] * (sin_phi * cos_theta * cos_psi)
    jacobian[:, 4, 2, 0] = scales[2] * (cos_phi * cos_theta * cos_psi)

    jacobian[:, 4, 0, 1] = scales[0] * -sin_theta * sin_psi
    jacobian[:, 4, 1, 1] = scales[1] * (sin_psi * cos_theta * sin_phi)
    jacobian[:, 4, 2, 1] = scales[2] * (cos_phi * cos_theta * sin_psi)

    jacobian[:, 4, 0, 2] = scales[0] * -cos_theta
    jacobian[:, 4, 1, 2] = scales[1] * -sin_theta * sin_phi
    jacobian[:, 4, 2, 2] = scales[2] * cos_phi * -sin_theta

    jacobian[:, 5, 0, 0] = scales[0] * cos_theta * -sin_psi
    jacobian[:, 5, 1, 0] = scales[1] * (-cos_phi * cos_psi + sin_phi * sin_theta * -sin_psi)
    jacobian[:, 5, 2, 0] = scales[2] * (sin_phi * cos_psi + cos_phi * sin_theta * -sin_psi)

    jacobian[:, 5, 0, 1] = scales[0] * cos_theta * cos_psi
    jacobian[:, 5, 1, 1] = scales[1] * (cos_phi * -sin_psi + cos_psi * sin_theta * sin_phi)
    jacobian[:, 5, 2, 1] = scales[2] * (-sin_phi * -sin_psi + cos_phi * sin_theta * cos_psi)

    return jacobian


def affine_jacobian_position(parameter, center, points):
    scales = parameter[0: 3]
    cos_phi, cos_theta, cos_psi = cos(parameter[3: 6])
    sin_phi, sin_theta, sin_psi = sin(parameter[3: 6])

    jacobian = numpy.empty((len(points), 3, 3))

    jacobian[:, 0, 0] = scales[0] * cos_theta * cos_psi
    jacobian[:, 1, 0] = scales[1] * (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi)
    jacobian[:, 2, 0] = scales[2] * (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

    jacobian[:, 0, 1] = scales[0] * cos_theta * sin_psi
    jacobian[:, 1, 1] = scales[1] * (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi)
    jacobian[:, 2, 1] = scales[2] * (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)

    jacobian[:, 0, 2] = scales[0] * -sin_theta
    jacobian[:, 1, 2] = scales[1] * cos_theta * sin_phi
    jacobian[:, 2, 2] = scales[2] * cos_phi * cos_theta

    return jacobian


def affine_jacobian(parameter, center, points):
    centered_points = numpy.atleast_2d(points) - center

    x = centered_points[:, 0][:, None].flatten()
    y = centered_points[:, 1][:, None].flatten()
    z = centered_points[:, 2][:, None].flatten()

    scales = parameter[0: 3]
    cos_phi, cos_theta, cos_psi = cos(parameter[3: 6])
    sin_phi, sin_theta, sin_psi = sin(parameter[3: 6])

    jacobian = numpy.zeros((len(points), 9, 3))

    jacobian[:, 0, 0] = cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z
    jacobian[:, 1, 1] = cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z
    jacobian[:, 2, 2] = -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z

    jacobian_rotation(parameter[3: 6], points, jacobian[:, 3: 6, :])
    jacobian[:, 3: 6, :] *= scales
    jacobian_translation(parameter[6: 9], points, jacobian[:, 6: 9, :])

    return jacobian


def jacobian_rotation(angles, points, jacobian_out):
    cos_phi, cos_theta, cos_psi = cos(angles)
    sin_phi, sin_theta, sin_psi = sin(angles)

    x = points[:, 0].flatten()
    y = points[:, 1].flatten()
    z = points[:, 2].flatten()

    jacobian_out[:, 0, 0] = (cos_psi * sin_theta * cos_phi + sin_psi * sin_phi) * y + (-cos_psi * sin_theta * sin_phi + sin_psi * cos_phi) * z
    jacobian_out[:, 0, 1] = (sin_psi * sin_theta * cos_phi - cos_psi * sin_phi) * y + (-sin_psi * sin_theta * sin_phi - cos_psi * cos_phi) * z
    jacobian_out[:, 0, 2] = (cos_theta * cos_phi) * y + (-cos_theta * sin_phi) * z
    jacobian_out[:, 1, 0] = (-cos_psi * sin_theta) * x + (cos_psi * cos_theta * sin_phi) * y + (cos_psi * cos_theta * cos_phi) * z
    jacobian_out[:, 1, 1] = (-sin_psi * sin_theta) * x + (sin_psi * cos_theta * sin_phi) * y + (sin_psi * cos_theta * cos_phi) * z
    jacobian_out[:, 1, 2] = (-cos_theta) * x + (-sin_theta * sin_phi) * y + (-sin_theta * cos_phi) * z
    jacobian_out[:, 2, 0] = (-sin_psi * cos_theta) * x + (-sin_psi * sin_theta * sin_phi - cos_psi * cos_phi) * y + (-sin_psi * sin_theta * cos_phi + cos_psi * sin_phi) * z
    jacobian_out[:, 2, 1] = (cos_psi * cos_theta) * x + (cos_psi * sin_theta * sin_phi - sin_psi * cos_phi) * y + (cos_psi * sin_theta * cos_phi + sin_psi * sin_phi) * z
    jacobian_out[:, 2, 2] = 0


def jacobian_translation(translation, points, jacobian_out):
    jacobian_out[...] = 0
    jacobian_out[:, (0, 1, 2), (0, 1, 2)] = 1


class LinearTransform(Model):
    def norm(self, points):
        return numpy.norm(self.parameter)


class Affine(LinearTransform):
    def __init__(self, center=None):
        self.parameter = self.identity
        if center is None:
            self.center = numpy.zeros(3)
        else:
            self.center = center

    @property
    def identity(self):
        return numpy.array((1., 1., 1., 0., 0., 0., 0., 0., 0.))

    def transform_points(self, points):
        centered_points = numpy.atleast_2d(points) - self.center
        return affine_transform_points(self.parameter, centered_points) + self.center

#    def transform_vectors(self, points, vectors):
#        return affine_transform_vectors(self.parameter, vectors)

    def jacobian(self, points):
        return affine_jacobian(self.parameter, self.center, points)

    def jacobian_position(self, points):
        return affine_jacobian_position(self.parameter, self.center, points)

    def jacobian_parameter_jacobian_position(self, points):
        return affine_jacobian_parameter_jacobian_position(self.parameter, self.center, points)

    def homogeneous_matrix(self):
        scales = self.parameter[:3]
        cos_phi, cos_theta, cos_psi = cos(self.parameter[3:-3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[3:-3])

        displacement = self.parameter[-3:][None, :]

        affine_matrix = numpy.zeros((4, 4))
        affine_matrix[0, 0] = scales[0] * (cos_theta * cos_psi)
        affine_matrix[0, 1] = scales[0] * (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi)
        affine_matrix[0, 2] = scales[0] * (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

        affine_matrix[1, 0] = scales[1] * (cos_theta * sin_psi)
        affine_matrix[1, 1] = scales[1] * (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi)
        affine_matrix[1, 2] = scales[1] * (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)

        affine_matrix[2, 0] = scales[2] * (-sin_theta)
        affine_matrix[2, 1] = scales[2] * (sin_phi * cos_theta)
        affine_matrix[2, 2] = scales[2] * (cos_phi * cos_theta)

        affine_matrix[:-1, -1] = displacement
        affine_matrix[-1, -1] = 1

        return affine_matrix

    @property
    def bounds(self):
        return numpy.array([
            [1e-10, numpy.inf],
            [1e-10, numpy.inf],
            [1e-10, numpy.inf],
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf]
        ])


@skip_from_test
class Similarity(LinearTransform):
    def __init__(self):
        self.parameter = self.identity

    @property
    def identity(self):
        return numpy.array((1., 0., 0., 0., 0., 0., 0.))

    def transform_points(self, points):
        scale = self.parameter[0]
        cos_phi, cos_theta, cos_psi = cos(self.parameter[1:-3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[1:-3])

        displacement = self.parameter[-3:][None, :]

        x = points[:, 0][:, None]
        y = points[:, 1][:, None]
        z = points[:, 2][:, None]

        return scale * numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + cos_theta * sin_phi * y + cos_phi * cos_theta * z
        ] + displacement

    def transform_vectors(self, points, vectors):
        scale = self.parameter[0]
        cos_phi, cos_theta, cos_psi = cos(self.parameter[1:-3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[1:-3])

        x = vectors[:, 0][:, None]
        y = vectors[:, 1][:, None]
        z = vectors[:, 2][:, None]

        return scale * numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + cos_theta * sin_phi * y + cos_phi * cos_theta * z
        ]

    def jacobian(self, points):
        centered_points = numpy.atleast_2d(points)

        x = centered_points[:, 0][:, None].flatten()
        y = centered_points[:, 1][:, None].flatten()
        z = centered_points[:, 2][:, None].flatten()

        s = self.parameter[0]
        cos_phi, cos_theta, cos_psi = cos(self.parameter[1: 4])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[1: 4])

        jacobian = numpy.zeros((len(points), 7, 3))

        jacobian[:, 0, 0] = (
            cos_theta * cos_psi * x +
            (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y +
            (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z
        )
        jacobian[:, 0, 1] = (
            cos_theta * sin_psi * x +
            (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi) * y +
            (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z
        )
        jacobian[:, 0, 2] = (
            -sin_theta * x + cos_theta * sin_psi * y + cos_phi * cos_theta * z
        )

        jacobian_rotation(self.parameter[1: 4], points, jacobian[:, 1: 4, :])
        jacobian[:, 1: 4, :] *= s
        jacobian_translation(self.parameter[4: 7], points, jacobian[:, 4: 7, :])

        return jacobian

    def jacobian_position(self, points):
        scale = self.parameter[0]
        cos_phi, cos_theta, cos_psi = cos(self.parameter[1: -3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[1: -3])

        jacobian = numpy.empty((len(points), 3, 3))

        jacobian[:, 0, 0] = cos_theta * cos_psi
        jacobian[:, 1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        jacobian[:, 2, 0] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

        jacobian[:, 0, 1] = cos_theta * sin_psi
        jacobian[:, 1, 1] = cos_phi * cos_psi + sin_psi * sin_theta * sin_phi
        jacobian[:, 2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi

        jacobian[:, 0, 2] = -sin_theta
        jacobian[:, 1, 2] = cos_theta * sin_phi
        jacobian[:, 2, 2] = cos_phi * cos_theta

        return scale * jacobian

    def jacobian_parameter_jacobian_position(self, points):
        scale = self.parameter[0]
        cos_phi, cos_theta, cos_psi = cos(self.parameter[1: -3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[1: -3])

        jacobian = numpy.empty((len(points), 7, 3, 3))

        jacobian[:, 0, 0, 0] = cos_theta * cos_psi
        jacobian[:, 0, 1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        jacobian[:, 0, 2, 0] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

        jacobian[:, 0, 0, 1] = cos_theta * sin_psi
        jacobian[:, 0, 1, 1] = cos_phi * cos_psi + sin_psi * sin_theta * sin_phi
        jacobian[:, 0, 2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi

        jacobian[:, 0, 0, 2] = -sin_theta
        jacobian[:, 0, 1, 2] = cos_theta * sin_phi
        jacobian[:, 0, 2, 2] = cos_phi * cos_theta

        jacobian[:, 1, 0, 0] = 0
        jacobian[:, 1, 1, 0] = sin_phi * sin_psi - cos_phi * sin_theta * cos_psi
        jacobian[:, 1, 2, 0] = (-cos_phi * sin_psi - sin_phi * sin_theta * cos_psi)

        jacobian[:, 1, 0, 1] = 0
        jacobian[:, 1, 1, 1] = -sin_phi * cos_psi + sin_psi * sin_theta * -cos_phi
        jacobian[:, 1, 2, 1] = cos_phi * cos_psi - sin_phi * sin_theta * sin_psi

        jacobian[:, 1, 0, 2] = 0
        jacobian[:, 1, 1, 2] = cos_theta * -cos_phi
        jacobian[:, 1, 2, 2] = -sin_phi * cos_theta

        jacobian[:, 2, 0, 0] = sin_theta * cos_psi
        jacobian[:, 2, 1, 0] = sin_phi * -cos_theta * cos_psi
        jacobian[:, 2, 2, 0] = cos_phi * -cos_theta * cos_psi

        jacobian[:, 2, 0, 1] = sin_theta * sin_psi
        jacobian[:, 2, 1, 1] = sin_psi * -cos_theta * sin_phi
        jacobian[:, 2, 2, 1] = cos_phi * -cos_theta * sin_psi

        jacobian[:, 2, 0, 2] = cos_theta
        jacobian[:, 2, 1, 2] = sin_theta * sin_phi
        jacobian[:, 2, 2, 2] = cos_phi * sin_theta

        jacobian[:, 3, 0, 0] = cos_theta * sin_psi
        jacobian[:, 3, 1, 0] = -cos_phi * -cos_psi + sin_phi * sin_theta * sin_psi
        jacobian[:, 3, 2, 0] = (sin_phi * -cos_psi + cos_phi * sin_theta * sin_psi)

        jacobian[:, 3, 0, 1] = cos_theta * -cos_psi
        jacobian[:, 3, 1, 1] = cos_phi * sin_psi - cos_psi * sin_theta * sin_phi
        jacobian[:, 3, 2, 1] = -sin_phi * sin_psi + cos_phi * sin_theta * -cos_psi

        jacobian[:, 1:] *= scale

        return jacobian

    @property
    def bounds(self):
        return numpy.array([
            [1e-10, numpy.inf],
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf]
        ])


class Rigid(LinearTransform):
    def __init__(self, center=[0, 0, 0]):
        self.parameter = self.identity
        self.center = numpy.array(center, dtype=float)

    @property
    def identity(self):
        return numpy.array((0., 0., 0., 0., 0., 0.))

    def transform_points(self, points):
        centered_points = numpy.atleast_2d(points) - self.center
        cos_phi, cos_theta, cos_psi = cos(self.parameter[0:-3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[0:-3])

        displacement = self.parameter[-3:][None, :]

        x = centered_points[:, 0][:, None]
        y = centered_points[:, 1][:, None]
        z = centered_points[:, 2][:, None]

        return numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
        ] + displacement + self.center

    def transform_vectors(self, points, vectors):
        cos_phi, cos_theta, cos_psi = cos(self.parameter[0:-3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[0:-3])

        x = vectors[:, 0][:, None]
        y = vectors[:, 1][:, None]
        z = vectors[:, 2][:, None]

        return - numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
        ]

    def jacobian(self, points):
        centered_points = numpy.atleast_2d(points) - self.center
        jacobian = numpy.empty((len(points), 6, 3))
        jacobian_rotation(self.parameter[0: 3], centered_points, jacobian[:, 0: 3, :])
        jacobian_translation(self.parameter[-3:], centered_points, jacobian[:, 3: 6, :])
        return jacobian

    def jacobian_position(self, points):
        cos_phi, cos_theta, cos_psi = cos(self.parameter[0:-3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[0:-3])

        jacobian = numpy.empty((len(points), 3, 3))

        jacobian[:, 0, 0] = cos_theta * cos_psi
        jacobian[:, 1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        jacobian[:, 2, 0] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

        jacobian[:, 0, 1] = cos_theta * sin_psi
        jacobian[:, 1, 1] = cos_phi * cos_psi + sin_psi * sin_theta * sin_phi
        jacobian[:, 2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi

        jacobian[:, 0, 2] = -sin_theta
        jacobian[:, 1, 2] = cos_theta * sin_phi
        jacobian[:, 2, 2] = cos_phi * cos_theta

        return jacobian

    def jacobian_parameter_jacobian_position(self, points):
        cos_phi, cos_theta, cos_psi = cos(self.parameter[0:-3])
        sin_phi, sin_theta, sin_psi = sin(self.parameter[0:-3])

        #displacement = self.parameter[-3:][None, :]

        jacobian = numpy.zeros((len(points), len(self.parameter), 3, 3))

        jacobian[:, 0, 0, 0] = 0
        jacobian[:, 0, 1, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
        jacobian[:, 0, 2, 0] = (cos_phi * sin_psi - sin_phi * sin_theta * cos_psi)

        jacobian[:, 0, 0, 1] = 0
        jacobian[:, 0, 1, 1] = -sin_phi * cos_psi + sin_psi * sin_theta * -cos_phi
        jacobian[:, 0, 2, 1] = -(cos_phi * cos_psi - sin_phi * sin_theta * sin_psi)

        jacobian[:, 0, 0, 2] = 0
        jacobian[:, 0, 1, 2] = cos_theta * cos_phi
        jacobian[:, 0, 2, 2] = -sin_phi * cos_theta

        jacobian[:, 1, 0, 0] = -sin_theta * cos_psi
        jacobian[:, 1, 1, 0] = sin_phi * cos_theta * cos_psi
        jacobian[:, 1, 2, 0] = cos_phi * cos_theta * cos_psi

        jacobian[:, 1, 0, 1] = -sin_theta * sin_psi
        jacobian[:, 1, 1, 1] = sin_psi * cos_theta * sin_phi
        jacobian[:, 1, 2, 1] = cos_phi * cos_theta * sin_psi

        jacobian[:, 1, 0, 2] = -cos_theta
        jacobian[:, 1, 1, 2] = -sin_theta * sin_phi
        jacobian[:, 1, 2, 2] = -cos_phi * sin_theta

        jacobian[:, 2, 0, 0] = -cos_theta * sin_psi
        jacobian[:, 2, 1, 0] = -cos_phi * cos_psi - sin_phi * sin_theta * sin_psi
        jacobian[:, 2, 2, 0] = (sin_phi * cos_psi - cos_phi * sin_theta * sin_psi)

        jacobian[:, 2, 0, 1] = cos_theta * cos_psi
        jacobian[:, 2, 1, 1] = cos_phi * -sin_psi + cos_psi * sin_theta * sin_phi
        jacobian[:, 2, 2, 1] = -sin_phi * -sin_psi + cos_phi * sin_theta * cos_psi

        #jacobian[:, 3, :, :] = 0
        #jacobian[:, 4, :, :] = 0
        #jacobian[:, 5, :, :] = 0

        return jacobian

    @property
    def bounds(self):
        return numpy.array([
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf]
        ])


@skip_from_test
class Rotation(LinearTransform):
    def __init__(self, center=numpy.array((0., 0., 0.))):
        self.parameter = self.identity
        self.center = numpy.asanyarray(center)

    @property
    def identity(self):
        return numpy.array((0., 0., 0.))

    def transform_points(self, points):
        cos_phi, cos_theta, cos_psi = cos(self.parameter)
        sin_phi, sin_theta, sin_psi = sin(self.parameter)

        x = points[:, 0][:, None] - self.center[0]
        y = points[:, 1][:, None] - self.center[1]
        z = points[:, 2][:, None] - self.center[2]

        return numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + cos_theta * sin_phi * y + cos_phi * cos_theta * z
        ] + self.center

    def transform_vectors(self, points, vectors):
        cos_phi, cos_theta, cos_psi = cos(self.parameter)
        sin_phi, sin_theta, sin_psi = sin(self.parameter)

        x = vectors[:, 0][:, None]
        y = vectors[:, 1][:, None]
        z = vectors[:, 2][:, None]

        return numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + cos_theta * sin_phi * y + cos_phi * cos_theta * z
        ]

    def jacobian(self, points):
        points = numpy.atleast_2d(points) - self.center
        jacobian = numpy.empty((len(points), 3, 3))
        jacobian_rotation(self.parameter, points, jacobian)

        return jacobian

    def jacobian_position(self, points):
        cos_phi, cos_theta, cos_psi = cos(self.parameter)
        sin_phi, sin_theta, sin_psi = sin(self.parameter)

        jacobian = numpy.empty((len(points), 3, 3))

        jacobian[:, 0, 0] = cos_theta * cos_psi
        jacobian[:, 1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
        jacobian[:, 2, 0] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

        jacobian[:, 0, 1] = cos_theta * sin_psi
        jacobian[:, 1, 1] = cos_phi * cos_psi + sin_psi * sin_theta * sin_phi
        jacobian[:, 2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi

        jacobian[:, 0, 2] = -sin_theta
        jacobian[:, 1, 2] = cos_theta * sin_phi
        jacobian[:, 2, 2] = cos_phi * cos_theta

        return jacobian

    @property
    def bounds(self):
        return numpy.array([
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
            [-numpy.pi, numpy.pi],
        ])


@skip_from_test
class AnisotropicScale(LinearTransform):
    def __init__(self, center=numpy.array((0., 0., 0.))):
        self.parameter = self.identity
        self.center = numpy.asanyarray(center)

    @property
    def identity(self):
        return numpy.array((1., 1., 1.))

    def transform_points(self, points):
        return (points - self.center) * self.parameter + self.center

    def jacobian(self, points):
        centered_points = numpy.atleast_2d(points) - self.center
        jacobian = numpy.zeros((len(points), 3, 3))
        jacobian[:, 0, 0] = centered_points[:, 0]
        jacobian[:, 1, 1] = centered_points[:, 1]
        jacobian[:, 2, 2] = centered_points[:, 2]
        return jacobian

    def jacobian_position(self, points):
        jacobian = numpy.zeros((len(points), 3, 3))
        jacobian[:, 0, 0] = self.parameter[0]
        jacobian[:, 1, 1] = self.parameter[1]
        jacobian[:, 2, 2] = self.parameter[2]
        return jacobian

    @property
    def bounds(self):
        return numpy.array([
            [1e-10, numpy.inf],
            [1e-10, numpy.inf],
            [1e-10, numpy.inf]
        ])


@skip_from_test
class IsotropicScale(LinearTransform):
    def __init__(self, center=numpy.array((0., 0., 0.))):
        self.parameter = self.identity
        self.center = numpy.asanyarray(center)

    @property
    def identity(self):
        return numpy.array([1.])

    def transform_points(self, points):
        return points * self.parameter

    def jacobian(self, points):
        centered_points = numpy.atleast_2d(points) - self.center
        return centered_points[:, None, :]

    def jacobian_position(self, points):
        jacobian = numpy.zeros((len(points), 3, 3))
        jacobian[:, (0, 1, 2), (0, 1, 2)] = self.parameter[0]
        return jacobian

    @property
    def bounds(self):
        return numpy.array([[1e-10, numpy.inf]])


class Translation(LinearTransform):
    def __init__(self):
        self.parameter = self.identity

    @property
    def identity(self):
        return numpy.array((0., 0., 0.))

    def transform_points(self, points):
        return points + self.parameter

    def jacobian(self, points):
        jacobian = numpy.zeros((len(points), 3, 3))
        jacobian[:, (0, 1, 2), (0, 1, 2)] = 1

        return jacobian

    def jacobian_position(self, points):
        jacobian = numpy.zeros((len(points), 3, 3))
        jacobian[:, (0, 1, 2), (0, 1, 2)] = 1

        return jacobian

    def jacobian_parameter_jacobian_position(self, points):
        jacobian = numpy.zeros((len(points), 3, 3, 3))

        return jacobian

    def norm(self, points):
        return numpy.norm(self.parameter)

    @property
    def bounds(self):
        return numpy.array([
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf],
            [-numpy.inf, numpy.inf]
        ])
