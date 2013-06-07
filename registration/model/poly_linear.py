import numpy
from numpy import cos, sin
from scipy.linalg import logm, expm
from scipy.spatial import KDTree
from . import linear, basis

from ..decorator import skip_from_test


__all__ = ['PolyAffine', 'PolyRigidDiffeomorphism', 'PolyRigid']


def affine_transform_vectors(affine_parameter, vectors):
    scales = affine_parameter[:3]
    cos_phi, cos_theta, cos_psi = cos(affine_parameter[3:-3])
    sin_phi, sin_theta, sin_psi = sin(affine_parameter[3:-3])

    x = vectors[:, 0][:, None]
    y = vectors[:, 1][:, None]
    z = vectors[:, 2][:, None]
    return scales * numpy.c_[
        cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
        cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
        -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
    ]


def affine_from_rotations(rotations):
    cos_phi, cos_theta, cos_psi = cos(rotations)
    sin_phi, sin_theta, sin_psi = sin(rotations)

    affine_matrix = numpy.zeros((4, 4))
    affine_matrix[0, 0] = (cos_theta * cos_psi)
    affine_matrix[0, 1] = (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi)
    affine_matrix[0, 2] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

    affine_matrix[1, 0] = (cos_theta * sin_psi)
    affine_matrix[1, 1] = (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi)
    affine_matrix[1, 2] = (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)

    affine_matrix[2, 0] = (-sin_theta)
    affine_matrix[2, 1] = (sin_phi * cos_theta)
    affine_matrix[2, 2] = (cos_phi * cos_theta)

    affine_matrix[-1, -1] = 1

    return affine_matrix


def scales_from_matrix(matrix):
    scale_x = numpy.linalg.norm(matrix[0, 0: 3])
    scale_y = numpy.linalg.norm(matrix[1, 0: 3])
    scale_z = numpy.linalg.norm(matrix[2, 0: 3])

    return scale_x, scale_y, scale_z


def parameters_from_affine(affine):
    parameters = numpy.empty(9)
    parameters[:3] = scales_from_matrix(affine)
    parameters[3: 6] = rotations_from_matrix(affine)
    parameters[6: 9] = affine[:3, -1]

    return parameters


def affine_jacobian_from_parameters(affine_parameter):
    scales = affine_parameter[:3]
    cos_phi, cos_theta, cos_psi = cos(affine_parameter[3:-3])
    sin_phi, sin_theta, sin_psi = sin(affine_parameter[3:-3])

    jacobian = numpy.zeros((len(affine_parameter), 3, 3))
    jacobian[0, 0, 0] = (cos_theta * cos_psi)
    jacobian[0, 0, 1] = (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi)
    jacobian[0, 0, 2] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

    jacobian[1, 1, 0] = (cos_theta * sin_psi)
    jacobian[1, 1, 1] = (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi)
    jacobian[1, 1, 2] = (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)

    jacobian[2, 2, 0] = (-sin_theta)
    jacobian[2, 2, 1] = (sin_phi * cos_theta)
    jacobian[2, 2, 2] = (cos_phi * cos_theta)

    jacobian[3, 0, 1] = scales[0] * (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)
    jacobian[3, 0, 2] = scales[0] * (cos_phi * sin_psi + -sin_phi * sin_theta * cos_psi)

    jacobian[3, 1, 1] = scales[1] * (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)
    jacobian[3, 1, 2] = scales[1] * (-cos_phi * cos_psi + -sin_phi * sin_theta * sin_psi)

    jacobian[3, 2, 1] = scales[2] * (cos_phi * cos_theta)
    jacobian[3, 2, 2] = scales[2] * (-sin_phi * cos_theta)

    jacobian[4, 0, 0] = scales[0] * (-sin_theta * cos_psi)
    jacobian[4, 0, 1] = scales[0] * (sin_phi * cos_theta * cos_psi)
    jacobian[4, 0, 2] = scales[0] * (cos_phi * cos_theta * cos_psi)

    jacobian[4, 1, 0] = scales[1] * (-sin_theta * sin_psi)
    jacobian[4, 1, 1] = scales[1] * (sin_phi * cos_theta * sin_psi)
    jacobian[4, 1, 2] = scales[1] * (cos_phi * cos_theta * sin_psi)

    jacobian[4, 2, 0] = scales[2] * (-cos_theta)
    jacobian[4, 2, 1] = scales[2] * (sin_phi * -sin_theta)
    jacobian[4, 2, 2] = scales[2] * (cos_phi * -sin_theta)

    jacobian[5, 0, 0] = scales[0] * (cos_theta * -sin_psi)
    jacobian[5, 0, 1] = scales[0] * (-cos_phi * -cos_psi + sin_phi * sin_theta * -sin_psi)
    jacobian[5, 0, 2] = scales[0] * (sin_phi * -cos_psi + cos_phi * sin_theta * -sin_psi)

    jacobian[5, 1, 0] = scales[1] * (cos_theta * -cos_psi)
    jacobian[5, 1, 1] = scales[1] * (cos_phi * -sin_psi + sin_phi * sin_theta * -cos_psi)
    jacobian[5, 1, 2] = scales[1] * (-sin_phi * -sin_psi + cos_phi * sin_theta * -cos_psi)

    return jacobian.swapaxes(-1, -2)


def affine_from_parameters(affine_parameter):
    scales = affine_parameter[:3]
    cos_phi, cos_theta, cos_psi = cos(affine_parameter[3:-3])
    sin_phi, sin_theta, sin_psi = sin(affine_parameter[3:-3])

    displacement = affine_parameter[-3:][None, :]

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


def parameters_from_rigid(affine):
    parameters = numpy.empty(6)
    parameters[0: 3] = rotations_from_matrix(affine)
    parameters[3: 6] = affine[:3, -1]

    return parameters


def rigid_from_parameters(affine_parameter):
    cos_phi, cos_theta, cos_psi = cos(affine_parameter[0: 3])
    sin_phi, sin_theta, sin_psi = sin(affine_parameter[0: 3])

    displacement = affine_parameter[-3:][None, :]

    affine_matrix = numpy.zeros((4, 4))
    affine_matrix[0, 0] = (cos_theta * cos_psi)
    affine_matrix[0, 1] = (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi)
    affine_matrix[0, 2] = (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi)

    affine_matrix[1, 0] = (cos_theta * sin_psi)
    affine_matrix[1, 1] = (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi)
    affine_matrix[1, 2] = (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi)

    affine_matrix[2, 0] = (-sin_theta)
    affine_matrix[2, 1] = (sin_phi * cos_theta)
    affine_matrix[2, 2] = (cos_phi * cos_theta)

    affine_matrix[:-1, -1] = displacement
    affine_matrix[-1, -1] = 1

    return affine_matrix


def rotations_from_matrix(matrix):
    angle_y = -numpy.arcsin(matrix[2, 0])
    C = numpy.cos(angle_y)
    if (numpy.fabs(C) > 0.00005):
        x = matrix[2, 2] / C
        y = matrix[2, 1] / C
        angle_x = numpy.arctan2(y, x)
        x = matrix[0, 0] / C
        y = matrix[1, 0] / C
        angle_z = numpy.arctan2(y, x)
    else:
        angle_x = 0.
        x = matrix[1, 1]
        y = -matrix[0, 1]
        angle_z = numpy.arctan2(y, x)

    return angle_x, angle_y, angle_z


@skip_from_test
class PolyAffine(basis.Model):
    def __init__(self, control_points, smooth, points=None):
        self.update_control_points(control_points)
        self.smooth = smooth
        self.parameter = self.identity
        self.points = points
        if points is not None:
            self.tree = KDTree(points)

    def update_control_points(self, control_points):
        self.control_points = control_points.copy()
        self.control_tree = KDTree(self.control_points)
        self._identity = numpy.tile(
            numpy.array((1., 1., 1., 0., 0., 0., 0., 0., 0.)),
            len(self.control_points)
        )

    def __getstate__(self):
        result = self.__dict__.copy()
        if 'tree' in result:
            del result['tree']
        if 'control_tree' in result:
            del result['control_tree']
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        for name, value in state.iteritems():
            setattr(self, name, value)
        self.control_tree = KDTree(self.control_points)
        if 'points' in state and self.points is not None:
            self.tree = KDTree(self.points)

    @property
    def identity(self):
        return self._identity

    @property
    def bounds(self):
        return numpy.tile(
            numpy.array([
                [1e-10, numpy.inf],
                [1e-10, numpy.inf],
                [1e-10, numpy.inf],
                [-numpy.pi, numpy.pi],
                [-numpy.pi, numpy.pi],
                [-numpy.pi, numpy.pi],
                [-numpy.inf, numpy.inf],
                [-numpy.inf, numpy.inf],
                [-numpy.inf, numpy.inf]
            ]),
            (len(self.control_points), 1)
        )

    def transform_points(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        displacements_uw = numpy.zeros_like(points, dtype=float)
        displacements = numpy.zeros_like(points, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            displacement = points_to_control - linear.affine_transform_points(
                self.parameter[i * 9: (i + 1) * 9],
                points_to_control
            )

            displacements_uw[point_indices_to_control] += (
                displacement
            )

            displacements[point_indices_to_control] += (
                displacement * weights
            )

        total_weights_mask = (total_weights > 0).ravel()
        displacements[total_weights_mask] /= total_weights[total_weights_mask]
        return points - displacements

    def transform_vectors(self, points, vectors):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        displacements_uw = numpy.zeros_like(points, dtype=float)
        displacements = numpy.zeros_like(points, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            vectors_to_control = vectors[point_indices_to_control]

            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            displacement = vectors_to_control - affine_transform_vectors(
                self.parameter[i * 9: (i + 1) * 9],
                vectors_to_control
            )

            displacements_uw[point_indices_to_control] += (
                displacement
            )

            displacements[point_indices_to_control] += (
                displacement * weights
            )

        total_weights_mask = (total_weights > 0).ravel()
        displacements[total_weights_mask] /= total_weights[total_weights_mask]
        return vectors - displacements

    def jacobian(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        jacobian = numpy.zeros((len(points), 9 * len(self.control_points), 3))
        jacobian_uw = numpy.zeros_like(jacobian, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            control_point_jacobian = self.affine_jacobian(
                self.parameter[i * 9: (i + 1) * 9],
                points_to_control
            )

            jacobian_uw[point_indices_to_control, i * 9: (i + 1) * 9, :] += (
                control_point_jacobian
            )

            jacobian[point_indices_to_control, i * 9: (i + 1) * 9, :] += (
                control_point_jacobian * weights[:, None]
            )

        total_weights_mask = (total_weights > 0).ravel()
        jacobian[total_weights_mask] /= 2 * total_weights[:, None][total_weights_mask]
        return jacobian

    def jacobian_position(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        jacobian = numpy.zeros((len(points), 3, 3))
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            control_point_parameter = self.parameter[i * 9: (i + 1) * 9].copy()
            control_point_jacobian = affine_from_parameters(control_point_parameter)[:-1, :-1].T

            jacobian[point_indices_to_control, :, :] += (
                control_point_jacobian * weights[:, None]
            )

        total_weights_mask = (total_weights > 0).ravel()
        jacobian[total_weights_mask] /= total_weights[:, None][total_weights_mask]
        return jacobian

    def jacobian_parameter_jacobian_position(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        n_control_points = len(self.control_points)

        jacobian = numpy.zeros((len(points), n_control_points * 9, 3, 3))
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            control_point_parameter = self.parameter[i * 9: (i + 1) * 9].copy()
            control_point_jacobian = affine_jacobian_from_parameters(control_point_parameter).swapaxes(-1, -2)

            jacobian[point_indices_to_control,  i * 9: (i + 1) * 9, :, :] += (
                control_point_jacobian * weights[:, None, None]
            )

        total_weights_mask = (total_weights > 0).ravel()
        jacobian[total_weights_mask] /= 2 * total_weights[:, None, None][total_weights_mask]
        return jacobian

    def affine_matrices(self):
        affine_matrices = numpy.empty((len(self.control_points), 4, 4))
        for i in xrange(len(self.control_points)):
            affine_matrices[i] = affine_from_parameters(self.parameter[i * 9: (i + 1) * 9])

        return affine_matrices

    def kernel(self, distances2):
        return  numpy.exp(-distances2 / self.smooth ** 2)

    @staticmethod
    def log(parameter):
        n_matrices = len(parameter) / 9
        matrices = numpy.empty((n_matrices, 4, 4))
        for i in xrange(n_matrices):
            p = parameter[i * 9: (i + 1) * 9].copy()
            matrices[i, :, :] = logm(
                affine_from_parameters(p)
            )
        return matrices

    @staticmethod
    def gradient_log(parameter):
        n_matrices = len(parameter) / 9
        matrices = numpy.empty((n_matrices, 4, 4))
        for i in xrange(n_matrices):
            p = parameter[i * 9: (i + 1) * 9].copy()
            #In the case that it is the gradient, we
            #take the parameter as a distance from the
            #identity, so we add the non-scale to it
            p[0: 3] += 1
            matrices[i, :, :] = logm(
                affine_from_parameters(p)
            )
        return matrices

    @staticmethod
    def exp(log_parameter):
        parameter = numpy.empty(9 * len(log_parameter))
        for i in xrange(len(log_parameter)):
            parameter[i * 9: (i + 1) * 9] = parameters_from_affine(
                expm(log_parameter[i])
            )

        return parameter

    @staticmethod
    def affine_jacobian(affine_parameter, points):
        centered_points = numpy.atleast_2d(points)

        x = centered_points[:, 0][:, None].flatten()
        y = centered_points[:, 1][:, None].flatten()
        z = centered_points[:, 2][:, None].flatten()

        scales = affine_parameter[0: 3]
        cos_phi, cos_theta, cos_psi = cos(affine_parameter[3: 6])
        sin_phi, sin_theta, sin_psi = sin(affine_parameter[3: 6])

        jacobian = numpy.zeros((len(points), 9, 3))

        jacobian[:, 0, 0] = cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z
        jacobian[:, 1, 1] = cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z
        jacobian[:, 2, 2] = -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z

        linear.jacobian_rotation(affine_parameter[3: 6], points, jacobian[:, 3: 6, :])
        jacobian[:, 3: 6, :] *= scales
        linear.jacobian_translation(affine_parameter[6: 9], points, jacobian[:, 6: 9, :])

        return jacobian


@skip_from_test
class PolyRigidDiffeomorphism(basis.Model):
    def __init__(self, control_points, smooth, points=None, number_of_steps=1., step=1.):
        self.control_points = control_points.copy()
        self.control_tree = KDTree(self.control_points)
        self.smooth = smooth
        self.parameter = self.identity
        self.points = points
        self.number_of_steps = float(number_of_steps)
        self.step = float(step)
        self.time = self.step / self.number_of_steps
        if points is not None:
            self.tree = KDTree(points)

    def __getstate__(self):
        result = self.__dict__.copy()
        if 'tree' in result:
            del result['tree']
        if 'control_tree' in result:
            del result['control_tree']
        return result

    def __setstate__(self, dict):
        self.__dict__ = dict
        for name, value in dict.iteritems():
            setattr(self, name, value)
        self.control_tree = KDTree(self.control_points)
        if 'points' in dict and self.points is not None:
            self.tree = KDTree(self.points)

    @property
    def identity(self):
        return numpy.tile(
            numpy.array((0., 0., 0., 0., 0., 0.)),
            len(self.control_points)
        )

    @property
    def bounds(self):
        return numpy.tile(
            numpy.array([
                [-numpy.pi, numpy.pi],
                [-numpy.pi, numpy.pi],
                [-numpy.pi, numpy.pi],
                [-numpy.inf, numpy.inf],
                [-numpy.inf, numpy.inf],
                [-numpy.inf, numpy.inf]
            ]),
            (len(self.control_points), 1)
        )

    def transform_points(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        displacements_uw = numpy.zeros_like(points, dtype=float)
        displacements = numpy.zeros_like(points, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights

            matrix = rigid_from_parameters(self.parameter[i * 6: (i + 1) * 6])
            rotations = matrix[:3, :3]
            translation = matrix[:3, -1]
            displacement_matrix = expm(rotations / self.number_of_steps)
            displacement_matrix[(0, 1, 2), (0, 1, 2)] -= 1
            displacement = (
                translation / self.number_of_steps +
                numpy.dot(displacement_matrix, (
                    points_to_control - translation * self.time
                ).T).T
            )

            displacements_uw[point_indices_to_control] += (
                displacement
            )

            displacements[point_indices_to_control] += (
                displacement * weights
            )

        total_weights_mask = (total_weights > 0).ravel()
        displacements[total_weights_mask] /= total_weights[total_weights_mask]
        return displacements + points

    def jacobian(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        jacobian = numpy.zeros((len(points), 6 * len(self.control_points), 3))
        jacobian_uw = numpy.zeros_like(jacobian, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            control_point_jacobian = self.rigid_jacobian(
                self.parameter[i * 6: (i + 1) * 6],
                points_to_control
            )

            jacobian_uw[point_indices_to_control, i * 6: (i + 1) * 6, :] += (
                control_point_jacobian
            )

            jacobian[point_indices_to_control, i * 6: (i + 1) * 6, :] += (
                control_point_jacobian * weights[:, None]
            )

        total_weights_mask = (total_weights > 0).ravel()
        jacobian[total_weights_mask] /= total_weights[:, None][total_weights_mask]
        return jacobian

    def jacobian_position(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        jacobian = numpy.zeros((len(points), 3, 3))
        jacobian_uw = numpy.zeros_like(jacobian, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            control_point_parameter = self.parameter[i * 6: (i + 1) * 6].copy()
            control_point_jacobian = rigid_from_parameters(control_point_parameter)[:-1, :-1].T

            jacobian_uw[point_indices_to_control, :, :] += (
                control_point_jacobian
            )

            jacobian[point_indices_to_control, :, :] += (
                control_point_jacobian * weights[:, None]
            )

        total_weights_mask = (total_weights > 0).ravel()
        jacobian[total_weights_mask] /= total_weights[:, None][total_weights_mask]
        return jacobian

    def affine_matrices(self):
        affine_matrices = numpy.empty((len(self.control_points), 4, 4))
        for i in xrange(len(self.control_points)):
            affine_matrices[i] = rigid_from_parameters(self.parameter[i * 6: (i + 1) * 6])

        return affine_matrices

    def kernel(self, distances2):
        return  numpy.exp(-distances2 / self.smooth ** 2)

    @staticmethod
    def log(parameter):
        n_matrices = len(parameter) / 6
        matrices = numpy.empty((n_matrices, 4, 4))
        for i in xrange(n_matrices):
            p = parameter[i * 6: (i + 1) * 6].copy()
            matrices[i, :, :] = logm(
                rigid_from_parameters(p)
            )
        return matrices

    @staticmethod
    def rigid_transform_points(affine_parameter, points):
        cos_phi, cos_theta, cos_psi = cos(affine_parameter[0: 3])
        sin_phi, sin_theta, sin_psi = sin(affine_parameter[0: 3])

        displacement = affine_parameter[-3:][None, :]

        x = points[:, 0][:, None]
        y = points[:, 1][:, None]
        z = points[:, 2][:, None]
        return numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
        ] + displacement

    @staticmethod
    def rigid_transform_vectors(affine_parameter, vectors):
        cos_phi, cos_theta, cos_psi = cos(affine_parameter[0: 3])
        sin_phi, sin_theta, sin_psi = sin(affine_parameter[0: 3])

        x = vectors[:, 0][:, None]
        y = vectors[:, 1][:, None]
        z = vectors[:, 2][:, None]
        return numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
        ]

    @staticmethod
    def rigid_jacobian(affine_parameter, points):
        centered_points = numpy.atleast_2d(points)

        x = centered_points[:, 0][:, None].flatten()
        y = centered_points[:, 1][:, None].flatten()
        z = centered_points[:, 2][:, None].flatten()

        cos_phi, cos_theta, cos_psi = cos(affine_parameter[0: 3])
        sin_phi, sin_theta, sin_psi = sin(affine_parameter[0: 3])

        jacobian = numpy.zeros((len(points), 6, 3))

        jacobian[:, 0, 0] = cos_theta * sin_psi * x
        jacobian[:, 0, 1] = (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi) * y
        jacobian[:, 0, 2] = (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z
        jacobian[:, 1, 0] = -sin_theta * x
        jacobian[:, 1, 1] = cos_theta * sin_phi * y
        jacobian[:, 1, 2] = cos_phi * cos_theta * z

        linear.jacobian_rotation(affine_parameter[0: 3], points, jacobian[:, 0: 3, :])
        linear.jacobian_translation(affine_parameter[3: 6], points, jacobian[:, 3: 6, :])

        return jacobian

    def compute_rigid_rotation_derivatives(matrix, N):
        rotations = numpy.array([
            [
                [0, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ], [
                [0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0]
            ], [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 0]
            ]
        ])

        scaling_matrices = [numpy.eye(3)]
        for i in xrange(N):
            scaling_matrices.append(numpy.dot(
                scaling_matrices[i],
                matrix
            ))

        derivative_matrices = rotations * 0.
        for n in xrange(1, N):
            for i in xrange(1, n + 1):
                for k in xrange(3):
                    derivative_matrices += numpy.dot(
                        numpy.dot(
                            scaling_matrices[i - 1], rotations[k]
                        ), scaling_matrices[n - i]
                    )

        return derivative_matrices


@skip_from_test
class PolyRigid(basis.Model):
    def __init__(self, control_points, smooth, points=None):
        self.control_points = control_points.copy()
        self.control_tree = KDTree(self.control_points)
        self.smooth = smooth
        self.parameter = self.identity
        self.points = points
        if points is not None:
            self.tree = KDTree(points)

    def __getstate__(self):
        result = self.__dict__.copy()
        if 'tree' in result:
            del result['tree']
        if 'control_tree' in result:
            del result['control_tree']
        return result

    def __setstate__(self, dict):
        self.__dict__ = dict
        for name, value in dict.iteritems():
            setattr(self, name, value)
        self.control_tree = KDTree(self.control_points)
        if 'points' in dict and self.points is not None:
            self.tree = KDTree(self.points)

    @property
    def identity(self):
        return numpy.tile(
            numpy.array((0., 0., 0., 0., 0., 0.)),
            len(self.control_points)
        )

    @property
    def bounds(self):
        return numpy.tile(
            numpy.array([
                [-numpy.pi, numpy.pi],
                [-numpy.pi, numpy.pi],
                [-numpy.pi, numpy.pi],
                [-numpy.inf, numpy.inf],
                [-numpy.inf, numpy.inf],
                [-numpy.inf, numpy.inf]
            ]),
            (len(self.control_points), 1)
        )

    def transform_points(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        displacements = numpy.zeros_like(points, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            displacement = points_to_control - self.rigid_transform_points(
                self.parameter[i * 6: (i + 1) * 6],
                points_to_control
            )

            displacements[point_indices_to_control] += (
                displacement * weights
            )

        total_weights_mask = (total_weights > 0).ravel()
        displacements[total_weights_mask] /= total_weights[total_weights_mask]
        return points - displacements

    def transform_vectors(self, points, vectors):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        displacements_uw = numpy.zeros_like(points, dtype=float)
        displacements = numpy.zeros_like(points, dtype=float)
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            vectors_to_control = vectors[point_indices_to_control]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            displacement = vectors_to_control - self.rigid_transform_vectors(
                self.parameter[i * 6: (i + 1) * 6],
                vectors_to_control
            )

            displacements_uw[point_indices_to_control] += (
                displacement
            )

            displacements[point_indices_to_control] += (
                displacement * weights
            )

        total_weights_mask = (total_weights > 0).ravel()
        displacements[total_weights_mask] /= total_weights[total_weights_mask]
        return vectors - displacements

    def jacobian(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        jacobian = numpy.zeros((len(points), 6 * len(self.control_points), 3))
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            control_point_jacobian = self.rigid_jacobian(
                self.parameter[i * 6: (i + 1) * 6],
                points_to_control
            )

            jacobian[point_indices_to_control, i * 6: (i + 1) * 6, :] += (
                control_point_jacobian * weights[:, None]
            )

        total_weights_mask = (total_weights > 0).ravel()
        jacobian[total_weights_mask] /= total_weights[:, None][total_weights_mask]
        return jacobian

    def jacobian_position(self, points):
        if not points is self.points:
            tree = KDTree(points)
        else:
            tree = self.tree

        jacobian = numpy.zeros((len(points), 3, 3))
        total_weights = numpy.zeros((len(points), 1))

        points_to_each_control = self.control_tree.query_ball_tree(tree, self.smooth * 3)

        for i, point_indices_to_control in enumerate(points_to_each_control):
            if len(point_indices_to_control) == 0:
                continue

            points_to_control = points[point_indices_to_control] - self.control_points[i]
            distances2 = (points_to_control ** 2).sum(1)

            weights = self.kernel(distances2)[:, None]
            total_weights[point_indices_to_control] += weights
            control_point_parameter = self.parameter[i * 6: (i + 1) * 6].copy()
            control_point_jacobian = rigid_from_parameters(control_point_parameter)[:-1, :-1].T

            jacobian[point_indices_to_control, :, :] += (
                control_point_jacobian * weights[:, None]
            )

        total_weights_mask = (total_weights > 0).ravel()
        jacobian[total_weights_mask] /= total_weights[:, None][total_weights_mask]
        return jacobian

    def affine_matrices(self):
        affine_matrices = numpy.empty((len(self.control_points), 4, 4))
        for i in xrange(len(self.control_points)):
            affine_matrices[i] = rigid_from_parameters(self.parameter[i * 6: (i + 1) * 6])

        return affine_matrices

    def kernel(self, distances2):
        return  numpy.exp(-distances2 / self.smooth ** 2)

    @staticmethod
    def log(parameter):
        n_matrices = len(parameter) / 6
        matrices = numpy.empty((n_matrices, 4, 4))
        for i in xrange(n_matrices):
            p = parameter[i * 6: (i + 1) * 6].copy()
            matrices[i, :, :] = logm(
                rigid_from_parameters(p)
            )
        return matrices

    @staticmethod
    def gradient_log(parameter):
        n_matrices = len(parameter) / 6
        matrices = numpy.empty((n_matrices, 4, 4))
        for i in xrange(n_matrices):
            p = parameter[i * 6: (i + 1) * 6].copy()
            #In the case that it is the gradient, we
            #take the parameter as a distance from the
            #identity, so we add the non-scale to it
            p[0: 3] += 1
            matrices[i, :, :] = logm(
                rigid_from_parameters(p)
            )
        return matrices

    @staticmethod
    def exp(log_parameter):
        parameter = numpy.empty(6 * len(log_parameter))
        for i in xrange(len(log_parameter)):
            parameter[i * 6: (i + 1) * 6] = parameters_from_rigid(
                expm(log_parameter[i])
            )

        return parameter

    @staticmethod
    def rigid_transform_points(affine_parameter, points):
        cos_phi, cos_theta, cos_psi = cos(affine_parameter[0: 3])
        sin_phi, sin_theta, sin_psi = sin(affine_parameter[0: 3])

        displacement = affine_parameter[-3:][None, :]

        x = points[:, 0][:, None]
        y = points[:, 1][:, None]
        z = points[:, 2][:, None]
        return numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
        ] + displacement

    @staticmethod
    def rigid_transform_vectors(affine_parameter, vectors):
        cos_phi, cos_theta, cos_psi = cos(affine_parameter[0: 3])
        sin_phi, sin_theta, sin_psi = sin(affine_parameter[0: 3])

        x = vectors[:, 0][:, None]
        y = vectors[:, 1][:, None]
        z = vectors[:, 2][:, None]
        return numpy.c_[
            cos_theta * cos_psi * x + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * y + (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * z,
            cos_theta * sin_psi * x + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * y + (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z,
            -sin_theta * x + sin_phi * cos_theta * y + cos_phi * cos_theta * z
        ]

    @staticmethod
    def rigid_jacobian(affine_parameter, points):
        centered_points = numpy.atleast_2d(points)

        x = centered_points[:, 0][:, None].flatten()
        y = centered_points[:, 1][:, None].flatten()
        z = centered_points[:, 2][:, None].flatten()

        cos_phi, cos_theta, cos_psi = cos(affine_parameter[0: 3])
        sin_phi, sin_theta, sin_psi = sin(affine_parameter[0: 3])

        jacobian = numpy.zeros((len(points), 6, 3))

        jacobian[:, 0, 0] = cos_theta * sin_psi * x
        jacobian[:, 0, 1] = (cos_phi * cos_psi + sin_psi * sin_theta * sin_phi) * y
        jacobian[:, 0, 2] = (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * z
        jacobian[:, 1, 0] = -sin_theta * x
        jacobian[:, 1, 1] = cos_theta * sin_phi * y
        jacobian[:, 1, 2] = cos_phi * cos_theta * z

        linear.jacobian_rotation(affine_parameter[0: 3], points, jacobian[:, 0: 3, :])
        linear.jacobian_translation(affine_parameter[3: 6], points, jacobian[:, 3: 6, :])

        return jacobian
