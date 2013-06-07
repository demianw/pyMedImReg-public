import numpy
from ..util import vectorized_dot_product
from . import basis


__all__ = ['ComposedTransform', 'CompositiveStepTransform', 'DiffeomorphicTransformScalingSquaring']


class ComposedTransform(object):
    def __init__(self, transforms):
        self.transforms = transforms

    @property
    def identity(self):
        return [t.identity for t in self.transforms]

    @property
    def parameter(self):
        return [t.parameter for t in self.transforms]

    def transform_points(self, points):
        points_buf = points
        for t in self.transforms:
            points_buf = t.transform_points(points_buf)
        return points_buf

    def transform_vectors(self, points, vectors):
        vectors_buf = vectors
        for t in self.transforms:
            vectors_buf = t.transform_vectors(points, vectors_buf)
        return vectors_buf

    def transform_tensors(self, points, vectors):
        vectors_buf = vectors
        for t in self.transforms:
            vectors_buf = t.transform_tensors(points, vectors_buf)
        return vectors_buf

    def jacobian_position(self, points):
        jacobian_position = self.transforms[0].jacobian_position(points)
        points = self.transforms[0].transform_points(points)

        for transform in self.transforms[1:]:
            jacobian_position[:] = (
                vectorized_dot_product(
                    transform.jacobian_position(points),
                    jacobian_position
                )
            )

            points = transform.jacobian_position(points)

        return jacobian_position


class DiffeomorphicTransformScalingSquaring(basis.Model):
    def __init__(self, transform_class, number_of_steps, invert_transform=False, *args, **kwargs):
        self.number_of_steps = int(number_of_steps)
        self.transform = transform_class(*args, **kwargs)
        self._identity = self.transform.identity.copy()
        self.step_length = 1. / number_of_steps
        self._inverse_multiplier = -1 if invert_transform else 1

    def inverse_transform(self):
        from copy import deepcopy
        it = deepcopy(self)
        it.is_inverse_transform = not self.is_inverse_transform
        it.parameter = self.parameter
        return it

    @property
    def identity(self):
        return self.transform.identity

    @property
    def parameter(self):
        return self.transform.parameter

    @parameter.setter
    def parameter(self, value):
        self.transform.parameter = value

    @property
    def is_inverse_transform(self):
        return self._inverse_multiplier == -1

    @is_inverse_transform.setter
    def is_inverse_transform(self, value):
        if value is True:
            self._inverse_multiplier = -1
            self.step_length = -1 * abs(self.step_length)
        elif value is False:
            self._inverse_multiplier = 1
            self.step_length = abs(self.step_length)
        else:
            raise ValueError('Value must be true or false')

    def transform_points(self, points):
        for i in xrange(0, self.number_of_steps):
            points = self.small_transform_points(points)

        return points

    def small_update_points(self, points):
        return self.step_length * (self.transform.transform_points(points) - points)

    def small_transform_points(self, points):
        return points + self.small_update_points(points)

    def jacobian(self, points):
        jacobian = self.small_jacobian(points)

        for i in xrange(1, self.number_of_steps):
            points = self.small_transform_points(points)

            jacobian = self.small_jacobian(points) + vectorized_dot_product(
                jacobian,
                self.small_jacobian_position(points)
            )

        return jacobian

    def small_jacobian(self, points):
        return self.step_length * self.transform.jacobian(points)

    def jacobian_position(self, points):
        jacobian_position = self.small_jacobian_position(points)

        for i in xrange(1, self.number_of_steps):
            points = self.small_transform_points(points)
            jacobian_position[:] = (
                vectorized_dot_product(
                    self.small_jacobian_position(points),
                    jacobian_position
                )

            )

        return jacobian_position

    def small_jacobian_position(self, points):
        sjp = self.step_length * self.transform.jacobian_position(points)
        sjp[:, (0, 1, 2), (0, 1, 2)] += 1. - self.step_length
        return sjp

    def jacobian_parameter_jacobian_position(self, points):
        jacobian = self.small_jacobian_parameter_jacobian_position(points)
        jacobian_position = self.small_jacobian_position(points)

        for i in xrange(1, self.number_of_steps):
            points = self.small_transform_points(points)

            jacobian = (
                vectorized_dot_product(
                    self.small_jacobian_parameter_jacobian_position(points),
                    jacobian_position[:, None, ...]
                ) + vectorized_dot_product(
                    jacobian,
                    self.small_jacobian_position(points)[:, None, ...],
                )
            )

            jacobian_position[:] = (
                vectorized_dot_product(
                    self.small_jacobian_position(points),
                    jacobian_position
                )

            )

        return jacobian

    def small_jacobian_parameter_jacobian_position(self, points):
        sjp = self.step_length * self.transform.jacobian_parameter_jacobian_position(points)
        return sjp

    def __setstate__(self, state):
        self.__dict__ = state
        for name, value in state.iteritems():
            setattr(self, name, value)
        if hasattr(self, 'transform') and hasattr(self, 'parameter'):
            self.transform.parameter = self.parameter.copy()
            self.transform.parameter = self.parameter.copy()


class CompositiveStepTransform(basis.Model):
    def __init__(self, transform_class, step_number, *args, **kwargs):
        self.step_number = step_number
        self.transforms = [
            transform_class(*args, **kwargs)
            for _ in xrange(self.step_number)
        ]

        self._identity = numpy.hstack([
            t.identity
            for t in self.transforms
        ])
        self.parameters_per_transform = len(t.identity)
        self.parameter = self.identity.copy()

    @property
    def identity(self):
        return self._identity

    def transform_points(self, points):
        for i, transform in enumerate(self.transforms):
            beg = i * self.parameters_per_transform
            end = beg + self.parameters_per_transform
            if not hasattr(transform, 'parameter'):
                transform.parameter = self.parameter[beg: end]
            else:
                transform.parameter = self.parameter[beg: end]

            transform.parameter[i * self.parameters_per_transform:]
            points = transform.transform_points(points)
        return points

    def jacobian(self, points):
        self.transforms[0].parameter = self.parameter[
            0: self.parameters_per_transform
        ]

        jacobian = [self.transforms[0].jacobian(points)]
        jacobian_position = self.transforms[0].jacobian_position(points)
        points = self.transforms[0].transform_points(points)

        for i, transform in enumerate(self.transforms[1:]):
            i += 1
            beg = i * self.parameters_per_transform
            end = beg + self.parameters_per_transform
            transform.parameter = self.parameter[beg: end]

            jacobian.append(vectorized_dot_product(
                transform.jacobian(points),
                jacobian_position
            ))

            jacobian_position[:] = vectorized_dot_product(
                transform.jacobian_position(points),
                jacobian_position
            )
            points = transform.transform_points(points)

        return numpy.hstack(jacobian)

    def jacobian_position(self, points):
        self.transforms[0].parameter = self.parameter[
            0: self.parameters_per_transform
        ]

        jacobian_position = self.transforms[0].jacobian_position(points)
        points = self.transforms[0].transform_points(points)

        for i, transform in enumerate(self.transforms[1:]):
            i += 1
            beg = i * self.parameters_per_transform
            end = beg + self.parameters_per_transform
            transform.parameter = self.parameter[beg: end]

            jacobian_position[:] = vectorized_dot_product(
                transform.jacobian_position(points),
                jacobian_position
            )
            points = transform.transform_points(points)

        return jacobian_position

    def jacobian_parameter_jacobian_position(self, points):
        self.transforms[0].parameter = self.parameter[
            0: self.parameters_per_transform
        ]

        jacobian = [self.transforms[0].jacobian_parameter_jacobian_position(points)]
        jacobian_position = self.transforms[0].jacobian_position(points)
        points = self.transforms[0].transform_points(points)

        for i, transform in enumerate(self.transforms[1:]):
            i += 1
            beg = i * self.parameters_per_transform
            end = beg + self.parameters_per_transform
            transform.parameter = self.parameter[beg: end]

            jacobian.append(vectorized_dot_product(
                transform.jacobian_parameter_jacobian_position(points),
                jacobian_position[:, None, :, :]
            ))

            jacobian_position[:] = vectorized_dot_product(
                transform.jacobian_position(points),
                jacobian_position
            )
            points = transform.transform_points(points)

        return numpy.hstack(jacobian)
