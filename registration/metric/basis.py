from itertools import izip

import numpy
from scipy import ndimage
from .. import util


__all__ = [
    'Metric', 'AdditiveMetric',
    'ImageMeanSquares', 'VectorImageMeanSquares', 'RosenGradientProjection'
]


class Metric(object):
    r"""
    Base Class for Metrics. We assume that an object is a tuple of locations, vectors and tensors

    .. math::
        (X, V, T) \in \Re^{K\times N}\times \Re^{K\times N}\times \Re^{K\times N \times N}

    where :math:`T_{i\cdot\cdot} \in Sym^N`. Where :math:`N` is the dimension.
    Then, a metric is an objective function

    .. math::
        m(\phi^* X, \phi^* V, \phi^*T) \in \Re

    where :math:`\phi^*` is the pullback application of the transform :math:`\phi:\Re^N\mapsto \Re^N`
    which has a parameter vector :math:`\theta \in \Re^M`. By default the most general definition of the
    pullback operations is implemented in the transform class:

    .. math::
        \phi^*X_i = \phi(X_i; \theta)\\
        \phi^*V_i = D_x^T\phi(X_i; \theta)\cdot V_i\\
        \phi^*T_i = D_x^T\phi(X_i; \theta)\cdot T_i \cdot D_x\phi(X_i; \theta)

    This class is designed to provide the gradient of the metric with respect to the transform parameters:

    .. math::
        \nabla_\theta m(\phi^* X, \phi^* V, \phi^*T) \in \Re^M

    which is calculated for each element as (abusing notation):

    .. math::
        \nabla_\theta m(\phi^* X_i, \phi^* V_i, \phi^*T_i) =\\
            \nabla_{X_i}m(\phi^* X_i, \phi^* V_i, \phi^*T_i)\cdot D^T_\theta (\phi^* X_i) +\\
            \nabla_{V_i}m(\phi^* X_i, \phi^* V_i, \phi^*T_i)\cdot D^T_\theta (\phi^* V_i)+\\
            \nabla_{T_i}m(\phi^* X_i, \phi^* V_i, \phi^*T_i)\cdot D^T_\theta (\phi^* T_i)
    """
    def __init__(self, points_moving, vectors=None, tensors=None, transform=None):
        self.points_moving = points_moving
        self.vectors = vectors
        self.tensors = tensors
        self.transform = transform

    def metric_gradient(self, points_moving, tensors=None, vectors=None, use_transform=False):
        r"""
            Metric value and gradient of the metric with respect to the parameters:

            Parameters
            ----------
            points_moving : array-like, shape (n_points, n_dimensions)
                Locations :math:`X`

            tensors : array-like, shape (n_points, n_dimensions)
                Tensors :math:`T`

            vectors : array-like, shape (n_points, n_dimensions, n_dimensions)
                Vectors :math:`V`

            use_transform : bool
                Apply the transform to the parameters before computing the
                metric and gradients

            Returns
            ----------
                metric_value, point_gradients, vector_gradients, tensor_gradients

                metric_value : float
                    :math:`m(X, V, T)`.

                point_gradients : array_like (n_points, n_dimensions)
                    each row is :math:`\nabla_{X_i} m(X_i, V_i, T_i)`

                vector_gradients : array_like (n_points, n_dimensions)
                    each row is :math:`\nabla_{V_i} m(X_i, V_i, T_i)`

                tensor_gradients : array_like (n_points, n_dimensions, n_dimensions)
                    each row is :math:`\nabla_{T_i} m(X_i, V_i, T_i)`

        """
        raise NotImplementedError()

    def metric_gradient_transform_parameters(self, parameter):
        r"""
            metric value and gradient of the ensemble with respect to the
            transform parameters :math:`\theta`:

            Parameters
            ----------
            parameter : array-like, shape (n_transform_parameters)
                Transform parameter :math:`\theta`

            Returns
            ----------
                metric_value, parameter_gradient

                metric_value : float
                    :math:`\sum_i m(\phi^* X_i, \phi^* V_i, \phi^* T_i)`.

                parameter_gradient : array_like (n_transform_parameters)
                    :math:`\sum_i \nabla_{\theta} m(\phi^* X_i, \phi^* V_i, \phi^* T_i)`

        """

        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.sum(0))

    def metric_jacobian_transform_parameters(self, parameter):
        r"""
            metric value and jacobian of the ensemble with respect to the
            transform parameters :math:`\theta`:

            Parameters
            ----------
            parameter : array-like, shape (n_transform_parameters)
                Transform parameters :math:`\theta`

            Returns
            ----------
                metric_value, parameter_jacobian

                metric_value : float
                    :math:`\sum_i m(\phi^* X_i, \phi^* V_i, \phi^* T_i)`.

                parameter_jacobian : array_like (n_elements, n_transform_parameters)
                    each row :math:`i` is :math:`\nabla_{\theta} m(\phi^* X_i, \phi^* V_i, \phi^* T_i)`

        """
        if self.transform is None or self.points_moving is None:
            return ValueError('transform and points_moving must be set for this')

        with_tensors = hasattr(self, 'tensors') and self.tensors is not None
        with_vectors = hasattr(self, 'vectors') and self.vectors is not None

        self.transform.parameter = parameter
        points_moving = numpy.ascontiguousarray(self.transform.transform_points(self.points_moving))

        if not (with_tensors or with_vectors):
            f, grad = self.metric_gradient(points_moving, use_transform=False)
            jacobian = self.transform.jacobian(points_moving)
            grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

            return f, grad
        elif with_vectors:
            vectors = self.transform.transform_vectors(self.points_moving, self.vectors)
            f, grad_pos, grad_vectors = self.metric_gradient(points_moving, vectors, use_transform=False)

            jacobian_parameters = self.transform.jacobian(self.points_moving)

            jacobian_param_position = numpy.asfortranarray(
                (jacobian_parameters * grad_pos[:, None, :]).sum(-1)
            )

            jacobian_vectors = self.transform.jacobian_tensor_matrices(
                self.points_moving, self.vectors
            )
            jacobian_param_vectors = util.vectorized_dot_product(
                jacobian_vectors,
                grad_vectors[:, None, :, None]
            ).sum(-1).mean(-1)

            return f, jacobian_param_position + jacobian_param_vectors

        elif with_tensors:
            tensors = self.transform.transform_tensors(self.points_moving, self.tensors)
            f, grad_pos, grad_tensors = self.metric_gradient(points_moving, tensors=tensors, use_transform=False)

            jacobian_parameters = self.transform.jacobian(self.points_moving)

            jacobian_param_position = numpy.asfortranarray(
                (jacobian_parameters * grad_pos[:, None, :]).sum(-1)
            )

            jacobian_param = jacobian_param_position

            if grad_tensors is not None:
                jacobian_tensor = self.transform.jacobian_tensor_matrices(
                    self.points_moving, self.tensors
                )
                jacobian_param_tensors = util.vectorized_dot_product(
                    jacobian_tensor,
                    grad_tensors[:, None, :, :]
                )[:, :, (0, 1, 2), (0, 1, 2)].sum(-1)
                jacobian_param += jacobian_param_tensors

            return f, jacobian_param
        else:
            raise NotImplementedError()


class AdditiveMetric(Metric):
    def __init__(self, metrics, weights, inverse, transform=None):
        if not (
                all(isinstance(metric, Metric) for metric in metrics) and
                all(numpy.isscalar(weight) for weight in weights)
        ):
            raise ValueError("Parameters of the wrong type")

        if len(metrics) != len(weights):
            raise ValueError("The parameter must be a list of pairs (metric, weight")

        self.metrics = metrics
        self.inverse = inverse
        self.transform = transform
        self.weights = numpy.array(weights)
        self.weights /= self.weights.sum()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform
        for i, metric in enumerate(self.metrics):
            if self._transform is not None and self.inverse[i]:
                metric.transform = self._transform.inverse_transform()
            else:
                metric.transform = self._transform

    def metric_jacobian_transform_parameters(self, parameter):
        mjtp = list(self.metrics[0].metric_jacobian_transform_parameters(parameter))
        mjtp[0] *= self.weights[0]
        for g in mjtp[1:]:
            g *= self.weights[0]

        for metric, weight in izip(self.metrics[1:], self.weights[1:]):
            mjtpm = metric.metric_jacobian_transform_parameters(parameter)
            mjtp[0] += mjtpm[0] * weight
            for i, g in enumerate(mjtp[1:]):
                g += mjtpm[i + 1] * weight

        return tuple(mjtp)

    def metric_gradient_transform_parameters(self, parameter):
        mjtp = list(self.metrics[0].metric_gradient_transform_parameters(parameter))
        mjtp[0] *= self.weights[0]
        for g in mjtp[1:]:
            g *= self.weights[0]

        for metric, weight in izip(self.metrics[1:], self.weights[1:]):
            mjtpm = metric.metric_gradient_transform_parameters(parameter)
            mjtp[0] += mjtpm[0] * weight
            for i, g in enumerate(mjtp[1:]):
                g += mjtpm[i + 1] * weight

        return tuple(mjtp)

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            if tensors is not None:
                tensors = self.transform_tensors(points_moving, tensors)
            if vectors is not None:
                vectors = self.transform_vectors(points_moving, vectors)
            points_moving = self.transform.transform_points(points_moving)

        metric_gradients = self.metrics[0].metric_gradient(points_moving, use_transform=False)
        for t_ in metric_gradients:
            t_ *= self.weights[0]

        for metric, weight in izip(self.metrics[1:], self.weights[1:]):
            metric_gradients_new = metric.metric_gradient(points_moving, use_transform=False)
            for i, t_ in enumerate(metric_gradients_new):
                metric_gradients[i] += t_ * weight

        return metric_gradients


class UnifyingAdditiveMetric(Metric):
    def __init__(self, metrics, weights, inverse, transform=None, points_moving=None):
        if not (
                all(isinstance(metric, Metric) for metric in metrics) and
                all(numpy.isscalar(weight) for weight in weights)
        ):
            raise ValueError("Parameters of the wrong type")

        if len(metrics) != len(weights):
            raise ValueError("The parameter must be a list of pairs (metric, weight")

        self.metrics = metrics
        self.inverse = inverse
        self.transform = transform
        self.points_moving = points_moving
        self.weights = numpy.array(weights)
        self.weights /= self.weights.sum()

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform
        for i, metric in enumerate(self.metrics):
            if self._transform is not None and self.inverse[i]:
                metric.transform = self._transform.inverse_transform()
            else:
                metric.transform = self._transform

    @property
    def points_moving(self):
        return self._moving

    @points_moving.setter
    def points_moving(self, points_moving):
        self._moving = points_moving
        for metric in self.metrics:
            metric.points_moving = self._moving

    def metric_jacobian_transform_parameters(self, parameter):
        mjtp = list(self.metrics[0].metric_jacobian_transform_parameters(parameter))
        print mjtp
        mjtp[0] *= self.weights[0]
        for g in mjtp[1:]:
            g *= self.weights[0]

        for metric, weight in izip(self.metrics[1:], self.weights[1:]):
            mjtpm = metric.metric_jacobian_transform_parameters(parameter)
            mjtp[0] += mjtpm[0] * weight
            for i, g in enumerate(mjtp[1:]):
                g += mjtpm[i] * weight

        return tuple(mjtp)

    def metric_gradient_transform_parameters(self, parameter):
        metric_jac = self.metric_jacobian_transform_parameters(parameter)
        res = (metric_jac[0],) + tuple(m.mean(0) for m in metric_jac[1:])
        return res

    def metric_gradient(self, points_moving, vectors=None, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            if tensors is not None:
                tensors = self.transform_tensors(points_moving, tensors)
            if vectors is not None:
                vectors = self.transform_vectors(points_moving, vectors)
            points_moving = self.transform.transform_points(points_moving)

        metric_gradients = self.metrics[0].metric_gradient(points_moving, use_transform=False)
        for t_ in metric_gradients:
            t_ *= self.weights[0]

        for metric, weight in izip(self.metrics[1:], self.weights[1:]):
            metric_gradients_new = metric.metric_gradient(points_moving, use_transform=False)
            for i, t_ in enumerate(metric_gradients_new):
                metric_gradients[i] += t_ * weight

        return metric_gradients

    def metric_gradient_per_point(self, points_moving, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        f, grad = self.metrics[0].metric_gradient_per_point(points_moving, use_transform=False)
        f *= self.weights[0]
        grad *= self.weights[0]

        for metric, weight in izip(self.metrics[1:], self.weights[1:]):
            fm, gradm = metric.metric_gradient_per_point(points_moving, use_transform=False)
            f += fm * weight
            grad += gradm * weight

        return f, grad


class LagrangeConstrained(Metric):
    def __init__(self, metrics, transform=None, points_moving=None):
        """
        Assuming metrics_i, i>0 are the constraints
        """
        if (not (all(isinstance(metric, Metric) for metric in metrics))):
            raise ValueError("The parameter must be a list of metrics")

        self.metrics = metrics
        self.n_lagrange_multipliers = len(metrics) - 1
        self.lagrange_multipliers = numpy.zeros(self.n_lagrange_multipliers, dtype=float)
        self.transform = transform
        self.points_moving = points_moving

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform
        for metric in self.metrics:
            metric.transform = self._transform

    @property
    def points_moving(self):
        return self._moving

    @points_moving.setter
    def points_moving(self, points_moving):
        self._moving = points_moving
        for metric in self.metrics:
            metric.points_moving = self._moving

    @property
    def identity(self):
        return numpy.r_[
            self.transform.identity,
            numpy.zeros(self.n_lagrange_multipliers)
        ]

    @property
    def parameter(self):
        return numpy.r_[
            self.transform.parameter,
            self.lagrange_multipliers
        ]

    def metric_jacobian_transform_parameters(self, parameter_w_lagrange):
        parameter = parameter_w_lagrange[:-self.n_lagrange_multipliers]
        self.lagrange_multipliers = parameter_w_lagrange[-self.n_lagrange_multipliers:]
        f, grad = self.metrics[0].metric_jacobian_transform_parameters(parameter)

        lagrange_multiplier_grad = numpy.empty((grad.shape[0], self.n_lagrange_multipliers), dtype=float)
        for i, metric, lagrange_multiplier in izip(xrange(self.n_lagrange_multipliers), self.metrics[1:], self.lagrange_multipliers):
            fm, gradm = metric.metric_jacobian_transform_parameters(parameter)
            f -= fm * lagrange_multiplier
            grad -= gradm * lagrange_multiplier
            lagrange_multiplier_grad[:, i] = -fm

        return f ** 2, (2 * f) * numpy.c_[grad, lagrange_multiplier_grad / grad.shape[0]]

    def metric_gradient(self, points_moving, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        f, grad = self.metrics[0].metric_gradient(points_moving, use_transform=False)
        f *= self.weights[0]
        grad *= self.weights[0]

        for metric, lagrange_multiplier in izip(self.metrics[1:], self.lagrange_multiplier):
            fm, gradm = metric.metric_gradient(points_moving, use_transform=False)
            f += fm * lagrange_multiplier
            grad += gradm * lagrange_multiplier

        return f, grad


class RosenGradientProjection(Metric):
    def __init__(self, metric_to_project, projection_function, transform=None, points_moving=None):
        self.transform = transform
        self.points_moving = points_moving
        self.tensors = None
        self.vectors = None
        self.metric_to_project = metric_to_project
        self.projection_function = projection_function

    def metric_gradient(self, points_moving, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        f, grad = self.metric_to_project.metric_gradient(points_moving, use_transform=False)
        tangent_planes, normals = self.projection_function.tangent_plane_normal(points_moving)

        grad = util.vectorized_dot_product(tangent_planes, grad[:, :, None])[..., 0]

        return f, grad

    def start_one_dimensional_search(self, points_moving, use_transform=True):
        hessians, evals, evecs = self.compute_hessian_eigendecomp(points_moving)

        v11 = util.vectorized_dot_product(evecs[:, 0, :][:, :, None], evecs[:, 0, :][:, None, :])
        self.normal_planes = v11
        self.start_one_dimensional_search = True

    def one_dimensional_search_off(self):
        self.one_dimensional_search = False


class ImageMeanSquares(Metric):
    def __init__(self, points_fixed, transform=None, points_moving=None, fixed_points=None,
                 gradient_operator=numpy.gradient, interpolator=ndimage.map_coordinates
                 ):
        self.points_moving = numpy.ascontiguousarray(points_moving)
        self.transformed_moving_buffer = numpy.zeros_like(points_moving)
        self.transform = transform
        self.points_fixed = points_fixed
        self.fixed_points = fixed_points
        self.fixed_points_tuple = tuple(fixed_points.T)
        self.fixed_points_values = points_fixed[self.fixed_points_tuple]
        self.gradient_operator = gradient_operator
        self.interpolator = interpolator

        self.fixed_gradient = numpy.empty(self.points_fixed.shape + (3,))
        self.fixed_points_gradients_buffer = numpy.empty(self.fixed_points.shape)

        fixed_gradient = self.gradient_operator(self.points_fixed)
        for i in xrange(3):
            self.fixed_gradient[:, :, :, i] = fixed_gradient[i]
            self.fixed_points_gradients_buffer[:, i] =\
                self.fixed_gradient[:, :, :, i][self.fixed_points_tuple]

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.sum(0))

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None or self.points_fixed is None:
            return ValueError('transform and points_fixed must be set for this')

        self.transform.parameter = parameter
        transformed_points = self.transform.transform_points(self.fixed_points)

        moving_points_values = self.interpolator(
            self.points_moving,
            transformed_points.T,
            order=1
        )

        diff = moving_points_values - self.fixed_points_values
        diff = diff[:, None, ...]
        metric = ((diff) ** 2).sum()

        jacobian = self.transform.jacobian(transformed_points)

        grad = 2. * (diff * self.fixed_points_gradients_buffer)[:, None, :]
        grad = (jacobian * grad).sum(-1)

        return metric, grad


class VectorImageMeanSquares(Metric):
    def __init__(self, points_fixed, transform=None, points_moving=None, fixed_points=None,
                 gradient_operator=numpy.gradient, interpolator=ndimage.map_coordinates,
                 smooth=0
                 ):
        self._moving = numpy.ascontiguousarray(points_moving)
        self.transformed_moving_buffer = numpy.zeros_like(points_moving)
        self.transform = transform
        self.points_fixed = points_fixed
        self.fixed_points = fixed_points
        self.fixed_points_tuple = tuple(fixed_points.T)
        self.fixed_points_values = points_fixed[self.fixed_points_tuple]
        self.gradient_operator = gradient_operator
        self.interpolator = interpolator
        self.smooth = smooth

        if smooth > 0:
            self.points_fixed = points_fixed.copy()
            for i in xrange(points_fixed.shape[-1]):
                self.points_fixed[..., i] = ndimage.gaussian_filter(points_fixed[..., i], smooth)

        self.fixed_gradient = numpy.empty(self.points_fixed.shape + (3,))
        self.fixed_points_gradients_buffer = numpy.empty(self.fixed_points.shape + self.points_fixed.shape[-1:])

        fixed_gradient = self.gradient_operator(self.points_fixed)
        for i in xrange(3):
            self.fixed_gradient[..., i, :] = fixed_gradient[i]
            self.fixed_points_gradients_buffer[:, i] =\
                self.fixed_gradient[..., i][self.fixed_points_tuple]

    @property
    def points_moving(self):
        return self._moving

    @points_moving.setter
    def points_moving(self, points_moving):
        if self.smooth == 0:
            self._moving = numpy.ascontiguousarray(points_moving)
        else:
            self._moving = numpy.ascontiguousarray(points_moving.copy())
            for i in xrange(points_moving.shape[-1]):
                self._moving[..., i] = ndimage.gaussian_filter(points_moving[..., i], self.smooth)

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        grad = grad.sum(0)
        return f, numpy.asfortranarray(grad)

    def metric_transform_parameters(self, parameter):
        if self.transform is None or self.points_fixed is None:
            return ValueError('transform and points_fixed must be set for this')

        self.transform.parameter = parameter
        transformed_points = self.transform.transform_points(self.fixed_points)

        moving_points_values = self.interpolator(
            self.points_moving,
            transformed_points.T,
            order=1
        )

        diff = moving_points_values - self.fixed_points_values
        metric = ((diff) ** 2).sum()

        return metric

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None or self.points_fixed is None:
            return ValueError('transform and points_fixed must be set for this')

        self.transform.parameter = parameter
        transformed_points = self.transform.transform_points(self.fixed_points)

        moving_points_values = self.transform.transform_vectors(
            transformed_points,
            self.interpolator(
                self.points_moving,
                transformed_points.T,
                order=1
            )
        )

        diff = moving_points_values - self.fixed_points_values
        metric = ((diff) ** 2).sum()

        jacobian = self.transform.jacobian(transformed_points)

        grad = 2. * (diff[..., None, :] * self.fixed_points_gradients_buffer).sum(-1)[:, None, :]
        grad = (jacobian * grad).sum(-1)

        return metric, grad


class RegularizerParameterSquare(object):
    def __init__(self, metric, weight):
        self.metric = metric
        self.weight = float(weight)

    @property
    def transform(self):
        return self.metric.transform

    def metric_gradient_transform_parameters(self, parameter):
        m, g = self.metric.metric_gradient_transform_parameters(parameter)
        m += self.weight * (parameter ** 2).sum()
        g += self.weight * 2 * parameter

        return m, g
