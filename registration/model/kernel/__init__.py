import numpy
from scipy.spatial import KDTree, distance_matrix
from .. import basis
from .kernels import *
from .kernels import __all__ as kernels_all


__all__ = [
    'KernelBasedTransform', 'KernelBasedTransformNoBasis', 'KernelBasedTransformNoBasisMovingAnchors',
] + kernels_all


class KernelBasedTransformNoBasisMovingAnchors(basis.Model):
    def __init__(self, control_points, kernel_function):
        self.control_points = control_points
        self.control_points_need_update = False
        self.parameter_separation_index = len(control_points) * 3
        self.kernel_function = kernel_function
        self.K = self.kernel_function(distance_matrix(self.control_points, self.control_points))
        self.last_points = None
        self.kernel_deriv_matrix_needs_update = True
        self._identity = numpy.zeros(len(control_points) * 6)
        self._identity[3 * len(control_points):] = self.control_points.ravel()
        self._parameter = self.identity.copy()
        self._parameter.flags.writeable = False

    def _update_point_movement(self, points):
        update_points = points is not self.last_points
        update_control_points = self.control_points_need_update
        if update_points or update_control_points:
            if update_control_points:
                self.control_points[:, 0] = self.parameter[0 + self.parameter_separation_index::3]
                self.control_points[:, 1] = self.parameter[1 + self.parameter_separation_index::3]
                self.control_points[:, 2] = self.parameter[2 + self.parameter_separation_index::3]
                self.K = self.kernel_function(distance_matrix(self.control_points, self.control_points))
                self.control_points_need_update = False

            self.last_points = points
            self.last_distance_matrix = distance_matrix(points, self.control_points)
            self.last_kernel_matrix = self.kernel_function(self.last_distance_matrix ** 2)
            self.kernel_deriv_matrix_needs_update = True

    def _update_kernel_deriv_matrix(self, points):
        self._update_point_movement(points)
        if self.kernel_deriv_matrix_needs_update:
            self.last_kernel_deriv_matrix = self.kernel_function.derivative(
                self.last_distance_matrix ** 2
            )

            if not hasattr(self.kernel_function, 'second_derivative_is_multiple'):
                self.last_kernel_second_deriv_matrix = self.kernel_function.second_derivative(
                    self.last_distance_matrix ** 2
                )
            self.kernel_deriv_matrix_needs_update = False

    @property
    def parameter(self):
        return self._parameter

    @parameter.setter
    def parameter(self, parameter):
        s = self.parameter_separation_index
        if not numpy.allclose(parameter[s:], self.parameter[s:]):
            self.control_points_need_update = True
        self._parameter.flags.writeable = True
        self._parameter[:] = parameter
        self._parameter.flags.writeable = False

    @property
    def identity(self):
        return self._identity

    def transform_points(self, points):
        self._update_point_movement(points)
        param = self._parameter.reshape(2 * len(self.control_points), 3)

        return points + numpy.dot(self.last_kernel_matrix, param[:len(self.control_points)])

    def jacobian(self, points):
        self._update_point_movement(points)
        self._update_kernel_deriv_matrix(points)
        jacobian = numpy.zeros((len(points), len(self.identity), 3))
        s = self.parameter_separation_index

        for i in xrange(3):
            aux = -2 * ((self.control_points[:, i][None, :] - points[:, i][:, None]) * self.last_kernel_deriv_matrix)
            jacobian[:, i:s:3, i] = self.last_kernel_matrix
            jacobian[:, s + i::3, i] = (aux * self.parameter[i:s:3])

        return jacobian

    def jacobian_position(self, points):
        self._update_point_movement(points)
        self._update_kernel_deriv_matrix(points)
        jac_position = numpy.zeros((len(points), 3, 3))
        s = self.parameter_separation_index

        for i in xrange(3):
            aux = -2 * ((self.control_points[:, i][None, :] - points[:, i][:, None]) * self.last_kernel_deriv_matrix)
            jac_position[:, i, 0] = (aux * self.parameter[0:s:3]).sum(1)
            jac_position[:, i, 1] = (aux * self.parameter[1:s:3]).sum(1)
            jac_position[:, i, 2] = (aux * self.parameter[2:s:3]).sum(1)
            jac_position[:, i, i] += 1

        return jac_position

    def jacobian_parameter_jacobian_position(self, points):
        self._update_point_movement(points)
        self._update_kernel_deriv_matrix(points)
        jacobian = numpy.zeros((len(points), len(self.identity), 3, 3))
        s = self.parameter_separation_index
        for i in xrange(3):
            point_difference = (self.control_points[:, i][None, :] - points[:, i][:, None])
            kernel_deriv = -2 * (point_difference * self.last_kernel_deriv_matrix)
            jacobian[:, 0:s:3, i, 0] = kernel_deriv
            jacobian[:, 1:s:3, i, 1] = kernel_deriv
            jacobian[:, 2:s:3, i, 2] = kernel_deriv

            if not hasattr(self.kernel_function, 'second_derivative_is_multiple'):
                raise NotImplementedError()

        kernel_2nd_deriv = self.last_kernel_deriv_matrix * self.kernel_function.second_derivative_is_multiple
        jacobian[:, 0 + s::3, :, :] = -4 * kernel_2nd_deriv[..., None, None]
        jacobian[:, 1 + s::3, :, :] = -4 * kernel_2nd_deriv[..., None, None]
        jacobian[:, 2 + s::3, :, :] = -4 * kernel_2nd_deriv[..., None, None]

        for p in xrange(len(self.control_points)):
            point_diff = (points - self.control_points[p][None, :])
            for i in xrange(3):
                jacobian[:, s + p * 3 + i, :, :] *= point_diff[:, i][..., None, None]

                for j in xrange(3):
                    jacobian[:, s + p * 3 + i, j, :] *= point_diff[:, j][..., None]

                jacobian[:, s + p * 3 + i, i, :] -= 2 * self.last_kernel_deriv_matrix[:, p][:, None]

        for i in xrange(3):
            jacobian[:, (0 + s)::3, :, i] *= self.parameter[i:s:3][None, :, None]
            jacobian[:, (1 + s)::3, :, i] *= self.parameter[i:s:3][None, :, None]
            jacobian[:, (2 + s)::3, :, i] *= self.parameter[i:s:3][None, :, None]

        return jacobian

    def __getstate__(self):
        state = {
            'control_points': self.control_points,
            'kernel_function': self.kernel_function,
            'K': self.K,
            'parameter': self.parameter
        }
        return state

    def __setstate__(self, state):
        self.control_points = state['control_points']
        self.kernel_function = state['kernel_function']
        self._parameter = state['parameter']
        self.K = state['K']

        self._identity = numpy.zeros(len(self.control_points) * 3)
        self.last_points = None
        self.kernel_deriv_matrix_needs_update = True

        self.control_points_need_update = False
        self.parameter_separation_index = len(self.control_points) * 3
        self._identity = numpy.zeros(len(self.control_points) * 6)
        self._identity[3 * len(self.control_points):] = self.control_points.ravel()
        self._parameter.flags.writeable = False


class KernelBasedTransformNoBasis(basis.Model):
    def __init__(self, control_points, kernel_function):
        self.control_points = control_points
        self.kernel_function = kernel_function
        self.K = self.kernel_function(distance_matrix(self.control_points, self.control_points))
        self.last_points = None
        self.kernel_deriv_matrix_needs_update = True
        self._identity = numpy.zeros(len(control_points) * 3)
        self.parameter = self.identity.copy()
        self._bounds = numpy.c_[self.identity, self.identity]
        self._bounds[:, 0] = -self.kernel_function.support
        self._bounds[:, 1] = +self.kernel_function.support

    @property
    def bounds(self):
        return self._bounds

    def _update_point_movement(self, points):
        if points is not self.last_points:
            self.last_points = points
            self.last_distance_matrix = distance_matrix(points, self.control_points)
            self.last_kernel_matrix = self.kernel_function(self.last_distance_matrix ** 2)
            self.kernel_deriv_matrix_needs_update = True

    def _update_kernel_deriv_matrix(self, points):
        self._update_point_movement(points)
        if self.kernel_deriv_matrix_needs_update:
            self.last_kernel_deriv_matrix = self.kernel_function.derivative(
                self.last_distance_matrix ** 2
            )
            self.kernel_deriv_matrix_needs_update = False

    @property
    def identity(self):
        return self._identity

    def transform_points(self, points):
        self._update_point_movement(points)
        param = self.parameter.reshape(len(self.control_points), 3)

        return points + numpy.dot(self.last_kernel_matrix, param)

    def jacobian(self, points):
        self._update_point_movement(points)
        jacobian = numpy.zeros((len(points), len(self.identity), 3))
        jacobian[:, 0::3, 0] = self.last_kernel_matrix
        jacobian[:, 1::3, 1] = self.last_kernel_matrix
        jacobian[:, 2::3, 2] = self.last_kernel_matrix

        return jacobian

    def jacobian_position(self, points):
        self._update_point_movement(points)
        self._update_kernel_deriv_matrix(points)
        jac_position = numpy.zeros((len(points), 3, 3))
        param = self.parameter.reshape(len(self.control_points), 3)

        for i in xrange(3):
            aux = -2 * ((self.control_points[:, i][None, :] - points[:, i][:, None]) * self.last_kernel_deriv_matrix)
            jac_position[:, i, 0] = (aux * param[:, 0]).sum(1)
            jac_position[:, i, 1] = (aux * param[:, 1]).sum(1)
            jac_position[:, i, 2] = (aux * param[:, 2]).sum(1)
            jac_position[:, i, i] += 1

        return jac_position

    def jacobian_parameter_jacobian_position(self, points):
        self._update_point_movement(points)
        self._update_kernel_deriv_matrix(points)
        jacobian = numpy.zeros((len(points), len(self.control_points) * 3, 3, 3))

        for i in xrange(3):
            aux = -2 * ((self.control_points[:, i][None, :] - points[:, i][:, None]) * self.last_kernel_deriv_matrix)
            jacobian[:, 0::3, i, 0] = aux
            jacobian[:, 1::3, i, 1] = aux
            jacobian[:, 2::3, i, 2] = aux

        return jacobian

    def __getstate__(self):
        state = {
            'control_points': self.control_points,
            'kernel_function': self.kernel_function,
            'K': self.K,
            'parameter': self.parameter
        }
        return state

    def __setstate__(self, state):
        self.control_points = state['control_points']
        self.kernel_function = state['kernel_function']
        self.parameter = state['parameter']
        self.K = state['K']
        self._identity = numpy.zeros(len(self.control_points) * 3)
        self.last_points = None
        self.kernel_deriv_matrix_needs_update = True


class KernelBasedTransform(basis.Model):
    def __init__(self, control_points, kernel_function, optimize_linear_part=True, regularized=False):
        self.control_points = control_points
        self.kernel_function = kernel_function
        self.optimize_linear_part = optimize_linear_part

        self.__initialize_variables()
        self.parameter = self.identity.copy()

    def __initialize_variables(self):
        n, d = self.control_points.shape

        self.control_tree = KDTree(self.control_points)

        self.K = self.kernel_function(
            distance_matrix(self.control_points, self.control_points) ** 2
        )
        #self.K = self.kernel_function(
        #    numpy.asarray(self.control_tree.sparse_distance_matrix(
        #        self.control_tree, self.kernel_function.support_radius
        #    ).todense()) ** 2
        #)

        self.Pn = numpy.c_[
            numpy.ones((len(self.control_points), 1)),
            self.control_points
        ]

        u, s, vh = numpy.linalg.svd(self.Pn)
        self.PP = u  # [:, self.control_points.shape[1] + 1:]

        self.kernel = numpy.dot(self.PP.T, numpy.dot(self.K, self.PP))

        if self.optimize_linear_part:
            self._identity = numpy.r_[
                numpy.tile(numpy.r_[numpy.zeros(d, dtype=float), 1.], (1, d))[0],
                numpy.zeros(d * n, dtype=float)

            ]
        else:
            self._identity = numpy.zeros(d * n, dtype=float)

        self._points_cache = None
        self._points_cache_len = None
        self._update_jacobian_bases = True

    def bending_energy_jacobian(self, parameter):
        d = self.control_points.shape[1]

        bending_energy = 0
        bending_energy_jacobian = numpy.zeros(len(parameter))
        start = d * (d + 1)
        for i in xrange(d):
            slice_ = slice(start + i, None, 3)
            ps = parameter[slice_]
            bending_energy_jacobian[slice_] = 2 * numpy.dot(ps.T, self.K)
            bending_energy += numpy.dot(ps.T, numpy.dot(self.K, ps))

        return bending_energy, bending_energy_jacobian

    def update_bases(self, points, point_bases_only=False):
        if (
            not hasattr(self, '_points_cache') or
            self._points_cache is None or
            (
                points is not self._points_cache or
                (points.shape[0] != self._points_cache.shape[0]) or
                numpy.any(points != self._points_cache)
            ) or (
                not point_bases_only
                and self._update_jacobian_bases
            )
        ):
            distances2 = self.__update_bases_points(points)

            if not point_bases_only:
                self._update_jacobian_bases = False
                self.__update_bases_jacobian_parameter(points)
                self.__update_jacobian_pos(distances2, points)
                self.__update_jac_param_jac_pos(points)
            else:
                self._update_jacobian_bases = True

    def __update_bases_points(self, points):
        self._points_cache = points
        self._points_cache_tree = KDTree(self._points_cache)

        #distances2 = numpy.asarray(self._points_cache_tree.sparse_distance_matrix(
        #    self.control_tree, self.kernel_function.support_radius
        #).todense()) ** 2

        distances2 = distance_matrix(self._points_cache, self.control_points) ** 2

        self.U = self.kernel_function(distances2)

        if self.optimize_linear_part:
            self.Pm = numpy.c_[
                numpy.ones((len(points), 1)),
                points
            ]

            self.basis = numpy.asarray(numpy.c_[self.Pm, numpy.dot(self.U, self.PP)])
        else:
            self.basis = numpy.dot(self.U, self.PP)
        return distances2

    def __update_bases_jacobian_parameter(self, points):
        _, d = points.shape

        self._jacobian_parameter = numpy.zeros((len(points), len(self.identity), d))

        for i in xrange(len(self.identity)):
            self._jacobian_parameter[:, i, i % 3] = self.basis[:, i // 3]

    def __update_jacobian_pos(self, distances2, points):
        n, d = points.shape
        m, _ = self.control_points.shape

        if self._points_cache_len is None or self._points_cache_len != n:
            self._points_cache_len = n

            self.basis_jac_pos = numpy.zeros((d, len(points), (d + 1) + m))

            self.jac_pos_Pm = numpy.zeros((d, len(points), d + 1))

            for i in xrange(d):
                self.jac_pos_Pm[i, :, 0] = 0
                self.jac_pos_Pm[i, :, 1: i + 1] = 0
                self.jac_pos_Pm[i, :, i + 1] = 1
                self.jac_pos_Pm[i, :, i + 2:] = 0
            self.basis_jac_pos[:, :, :d + 1] = self.jac_pos_Pm

            self.jac_param_jac_pos = numpy.empty(
                (len(points), len(self.identity), d, d),
            )

        self.U_derivative = self.kernel_function.derivative(distances2)
        differences = self._points_cache[None, :, :] - self.control_points[:, None, :]

        self.U_derivative_differences = (
            self.U_derivative.T[..., None] *
            differences * 2
        ).T

        if self.optimize_linear_part:
            for i in xrange(d):
                self.basis_jac_pos[i, :, d + 1:] = numpy.dot(
                    self.U_derivative_differences[i], self.PP
                )

        else:
            for i in xrange(d):
                self.basis_jac_pos.append(numpy.dot(self.U_derivative_differences[i], self.PP))

    def __update_jac_param_jac_pos(self, points):
        _, d = points.shape
        for i in xrange(len(self.identity)):
            for j in xrange(d):
                self.jac_param_jac_pos[:, i, j, :] = 0
                self.jac_param_jac_pos[:, i, j, i % 3] = self.basis_jac_pos[j, :, i // 3]

    @property
    def identity(self):
        return self._identity

    def transform_points(self, points):
        self.update_bases(points, point_bases_only=True)

        transformed_points = numpy.dot(
            self.basis, self.parameter_to_matrix(self.parameter)
        )

        if not self.optimize_linear_part:
            transformed_points += points

        return transformed_points

    def parameter_to_matrix(self, parameter):
        d = self.control_points.shape[1]
        return parameter.reshape(len(parameter) / d, d)

    def jacobian(self, points):
        self.update_bases(points)
        return self._jacobian_parameter

    def jacobian_position(self, points):
        self.update_bases(points)
        d = points.shape[-1]
        jacobian_position = numpy.empty((len(points), d, d))
        ptm = self.parameter_to_matrix(self.parameter)
        for i in xrange(points.shape[1]):
            jacobian_position[:, i, :] = numpy.dot(
                self.basis_jac_pos[i], ptm
            )

        return jacobian_position

    def jacobian_parameter_jacobian_position(self, points):
        self.update_bases(points)
        return self.jac_param_jac_pos

    def __getstate__(self):
        state = {
            'control_points': self.control_points,
            'kernel_function': self.kernel_function,
            'optimize_linear_part': self.optimize_linear_part,
            'parameter': self.parameter
        }
        return state

    def __setstate__(self, state):
        self.control_points = state['control_points']
        self.kernel_function = state['kernel_function']
        self.optimize_linear_part = state['optimize_linear_part']
        self.parameter = state['parameter']
        self.__initialize_variables()
