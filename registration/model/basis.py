r"""
Module with the basic classes defining transformation models
"""

import numpy
from ..util import vectorized_dot_product

__all__ = ['Model']

class Model(object):
    r"""Base Class for Transformations

    A transformation is defined as a map:

    .. math::
        \phi: \Omega \mapsto \Omega


    where :math:`\Omega \subseteq \Re^N` and the
    transform has a parameter vector :math:`\theta \in \Re^M`
    with :math:`M` the number of parameters

    Notes
    ----------
    We define :math:`\phi(x; \theta) = (\phi_1(x;\theta),\ldots, \phi_N(x;\theta))`, then
    the jacobian of the transform with respect to the parameter :math:`\theta` as

    .. math::
        [D_\theta\phi(x; \theta)]_{ij} = \frac{\partial \phi_i(x; \theta)}{\partial \theta_j},
        i=1\ldots N, j=1\ldots M

    and the jacobian of the transform with respect to the location :math:`x` as

    .. math::
        [D_x\phi(x; \theta)]_{ij} = \frac{\partial \phi_i(x; \theta)}{\partial x_j},
        i, j =1\ldots N


    attributes
    ----------
    `parameter` : array-like, shape (n_parameters)
        Stores the parameter vector :math:`\theta` of the transform.

    `identity` : array-like, shape (n_parameters)
        Stores the parameter value :math:`\theta_0` such that :math:`\phi(x; \theta_0) = x`.

    `bounds` : array-like, shape (n_parameters, 2)
        Stores the upper and lower bounds for each component of the parameter vectors
        :math:`\theta` such that :math:`\text{bounds}_{i0} \leq \theta_i \leq \text{bounds}_{i1}`


    References
    ----------
    """

    def __init__(self):
        self.parameter = self.identity

    @property
    def identity(self):
        r"""
        Stores the parameter value :math:`\theta_0` such that :math:`\phi(x; \theta_0) = x`.
        """
        return None

    def transform_points(self, points):
        r"""Transform a set of points.


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Points to be transformed


        Returns
        -------
        y :  array-like, shape (n_points, n_dimensions)
            :math:`y = \phi(x)`

       """
        raise NotImplementedError()

    def transform_vectors(self, points, vectors):
        r"""Transform a set of vectors located in space.


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Location of the vectors to be transformed

        v :  array-like, shape (n_points, n_dimensions)
            Vectors to be transformed

        Returns
        -------
        w :  array-like, shape (n_points, n_dimensions)
            :math:`w = D_x^T\phi(x) \cdot w`

            where :math:`D_x\phi(x)` is the Jacobian of :math:`\phi(x)`
            with respect to the spatial position :math:`x`
       """

        jacobians = self.jacobian_position(points)
        res = vectorized_dot_product(jacobians, vectors[..., None])[..., 0]
        return numpy.atleast_2d(res)

    def transform_tensors(self, points, tensors):
        r"""Transform a set of tensors located in space.


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Location of the vectors to be transformed

        T :  array-like, shape (n_points, n_dimensions, n_dimensions)
            Tensors to be transformed

        Returns
        -------
        S :  array-like, shape (n_points, n_dimensions)
            :math:`S = D^T_x\phi(x) \cdot T \cdot D_x\phi(x)`

            where :math:`D_x\phi(x)` is the Jacobian of :math:`\phi(x)`
            with respect to the spatial position :math:`x`
       """
        jacobians = self.jacobian_position(points)
        return vectorized_dot_product(
            vectorized_dot_product(jacobians.swapaxes(-1, -2), tensors),
            jacobians
        )

    def jacobian(self, points):
        r"""Transposed Jacobian of the transform with respect to its parameters


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Location of the Jacobian to be calculated

        Returns
        -------
        J :  array-like, shape (n_points, n_parameters, n_dimensions)
            :math:`J = D^T_\theta\phi(x)`
       """

        raise NotImplementedError()

    def jacobian_position(self, points):
        r"""Transposed Jacobian of the transform with respect to its location


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Location of the Jacobian to be calculated

        Returns
        -------
        J :  array-like, shape (n_points, n_dimensions, n_dimensions)
            :math:`J = D^T_x\phi(x)`
       """

        raise NotImplementedError()

    def jacobian_parameter_jacobian_position(self, points):
        r"""Iterated Transposed Jacobian of the transform with respect to
        its parameter and Location


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Location of the Jacobian to be calculated

        Returns
        -------
        J :  array-like, shape (n_points, n_parameters, n_dimensions, n_dimensions)
            :math:`J_{ijk} = \frac{\partial \phi_k(x)}{\partial \theta_i \partial x_j}`
       """
        raise NotImplementedError()

    def jacobian_vector_matrices(self, points, vectors):
        r"""Transposed Jacobian with respect to the transform parameter
        of the expression :math:`D^T_x \phi(x) \cdot v`


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Location of the Jacobian to be calculated

        v :  array-like, shape (n_points, n_dimensions)
            Vectors at each point of x


        Returns
        -------
        J :  array-like, shape (n_points, n_parameters, n_dimensions, n_dimensions)
            :math:`J = D^T_\theta[D^T_x\phi(x) \cdot v]`
       """
        jacobian_parameter_jacobian_position = self.jacobian_parameter_jacobian_position(points)

        DjacT_vector = vectorized_dot_product(
            jacobian_parameter_jacobian_position,  # .swapaxes(-1, -2),
            vectors[:, None, :, None]
        )[:, :, :, 0]

        return DjacT_vector

    def jacobian_tensor_matrices(self, points, tensors):
        r"""Transposed Jacobian with respect to the transform parameter
        of the expression :math:`D_x^T \phi(x) \cdot T \cdot D_x\phi(x)`


        Parameters
        ----------
        x :  array-like, shape (n_points, n_dimensions)
            Location of the Jacobian to be calculated

        T :  array-like, shape (n_points, n_dimensions, n_dimensions)
            Tensors at each point of x


        Returns
        -------
        J :  array-like, shape (n_points, n_parameters, n_dimensions, n_dimensions)
            :math:`J = D^T_\theta[D^T_x\phi(x) \cdot T\cdot D_x\phi(x)]`
       """
        jacobians = self.jacobian_position(points)
        jacobian_parameter_jacobian_position = self.jacobian_parameter_jacobian_position(points)

        tensor_jac = vectorized_dot_product(tensors, jacobians)
        DjacT_tensor_jac = vectorized_dot_product(
            jacobian_parameter_jacobian_position.swapaxes(-1, -2),
            tensor_jac[:, None, :, :]
        )

        return DjacT_tensor_jac + DjacT_tensor_jac.swapaxes(-1, -2)

    def norm(self, points):
        raise NotImplementedError()

    @property
    def bounds(self):
        r"""
        Stores the upper and lower bounds for each component of the parameter vectors
        :math:`\theta` such that :math:`\text{bounds}_{i0} \leq \theta_i \leq \text{bounds}_{i1}`
        """
        return None
