from .. import model

import warnings
import inspect
import unittest

import numpy
from numpy import testing

from nose.tools import with_setup
import registration.model.linear
import registration.model.poly_linear
from registration.util import vectorized_dot_product
import registration.model.kernel.kernels
import registration.model.util

warnings.simplefilter("ignore")


initialization = None
models_to_test = None


def setup_transforms():
    global initialization
    global models_to_test
    print "Setting up"

    excluded_models = ['LinearTransform', 'Model']
    models_to_test = []
    for m in model.__all__:
        class_ = getattr(model, m)

        if (
            inspect.isclass(class_) and
            issubclass(class_, model.Model) and
            m not in excluded_models and
            not hasattr(class_, 'skip_from_tests')
        ):
            models_to_test.append(m)
        elif hasattr(class_, 'skip_from_tests'):
            print "Model %s signaled to be skipped" % m

    grid = registration.model.util.grid_from_bounding_box_and_resolution(
        numpy.c_[[-.5, -.5, -.5], [.5, .5, .5]], 1)
    initialization = {
        'default': (tuple(), dict()),
        'PolyRigid': (
            (
                grid,
                2.
            ),
            {}
        ),
        'PolyAffine': (
            (
                grid,
                1.
            ),
            {}
        ),
        'KernelBasedTransform': (
            (
                grid,
                registration.model.kernel.kernels.ThinPlateSpline3DKernel(1)
            ),
            {'optimize_linear_part': True}
        ),
        'KernelBasedTransformNoBasis': (
            (
                grid,
                registration.model.kernel.kernels.InverseExp3DKernel(1)
            ),
            {}
        ),
        'KernelBasedTransformNoBasisMovingAnchors': (
            (
                grid,
                registration.model.kernel.kernels.InverseExp3DKernel(1)
            ),
            {}
        ),
        'CompositiveStepTransform': (
            (
                # model.Rigid,
                registration.model.linear.Rigid,
                5,
            ),
            {}
        ),
        'DiffeomorphicTransformScalingSquaring': (
            (
                model.Rigid,
                8,
                False,
            ),
            {}
        ),

    }
setup_transforms()


@with_setup(setup=setup_transforms)
def test_model_jacobians_one_parameter():
    for model_name in models_to_test:
        if model_name in initialization:
            model_initialization = initialization[model_name]
        else:
            model_initialization = initialization['default']
        n_params = len(getattr(model, model_name)(
            *(model_initialization[0]),
            **(model_initialization[1])
        ).identity)
        for n_param in xrange(n_params):
            yield model_jacobian, model_name, n_param, model_initialization


def model_jacobian(model_name, n_param, model_initialization, eps_max=11, n_points=200, random=numpy.random.RandomState(0)):
    model_ = getattr(model, model_name)(
        *model_initialization[0], **model_initialization[1])
    points = random.randn(n_points, 3)
    approx = numpy.empty((n_points, 3))
    jac = numpy.empty((n_points, 3))

    eps = 10. ** (-eps_max)
    parameter = model_.identity.copy()
    parameter[n_param] = model_.identity[n_param] - eps
    model_.parameter = parameter
    f_minus = model_.transform_points(points)
    parameter[n_param] = model_.identity[n_param] + eps
    model_.parameter = parameter
    f_plus = model_.transform_points(points)
    parameter[n_param] = model_.identity[n_param]
    model_.parameter = parameter
    approx[:] = (f_plus - f_minus) / (2 * eps)
    jac[:] = model_.jacobian(points)[:, n_param, :]

    testing.assert_array_almost_equal(
        approx, jac, decimal=4,
        err_msg="Transform %s parameter %d did not pass the jacobian test" % (
            model_name, n_param
        ), verbose=True
    )


@with_setup(setup_transforms)
def test_model_jacobian_positions_one_parameter():
    for model_name in models_to_test:
        if model_name in initialization:
            model_initialization = initialization[model_name]
        else:
            model_initialization = initialization['default']
        n_params = len(getattr(model, model_name)(
            *model_initialization[0],
            **model_initialization[1]
        ).identity)
        for n_param in xrange(n_params):
            yield model_jacobian_position, model_name, n_param, model_initialization


def model_jacobian_position(model_name, n_param, model_initialization, eps_max=5, n_points=5, random=numpy.random.RandomState(0)):
    model_ = getattr(model, model_name)(
        *model_initialization[0], **model_initialization[1])
    points = random.randn(n_points, 3)
    jac = numpy.empty((n_points, 3, 3))
    jac_approx = numpy.empty_like(jac)
    eps = 10. ** (-eps_max)
    identity = model_.identity
    identity[n_param] += 1e-2
    model_.parameter = identity

    stencil = numpy.vstack([
        points - (eps, 0, 0),
        points + (eps, 0, 0),
        points - (0, eps, 0),
        points + (0, eps, 0),
        points - (0, 0, eps),
        points + (0, 0, eps),
    ])

    transformed_stencil = model_.transform_points(stencil)

    f_minus_x = transformed_stencil[:len(points)]
    f_plus_x = transformed_stencil[len(points): 2 * len(points)]

    f_minus_y = transformed_stencil[2 * len(points): 3 * len(points)]
    f_plus_y = transformed_stencil[3 * len(points): 4 * len(points)]

    f_minus_z = transformed_stencil[4 * len(points): 5 * len(points)]
    f_plus_z = transformed_stencil[5 * len(points): 6 * len(points)]

    # f_minus_x = model_.transform_points(points - (eps, 0, 0))
    # f_plus_x = model_.transform_points(points + (eps, 0, 0))
    #
    # f_minus_y = model_.transform_points(points - (0, eps, 0))
    # f_plus_y = model_.transform_points(points + (0, eps, 0))
    #
    # f_minus_z = model_.transform_points(points - (0, 0, eps))
    # f_plus_z = model_.transform_points(points + (0, 0, eps))

    jac_approx[:, 0, :] = (f_plus_x - f_minus_x) / (2 * eps)
    jac_approx[:, 1, :] = (f_plus_y - f_minus_y) / (2 * eps)
    jac_approx[:, 2, :] = (f_plus_z - f_minus_z) / (2 * eps)

    jac[:] = model_.jacobian_position(points)

    model_.parameter = model_.identity.copy()

    testing.assert_array_almost_equal(
        jac_approx, jac, decimal=3,
        err_msg="Transform %s parameter %d did not pass the jacobian with respect to position test" % (
            model_name, n_param
        ), verbose=True
    )


@with_setup(setup_transforms)
def test_model_jacobian_positions_jacobian_parameter_one_parameter():
    for model_name in models_to_test:
        if model_name in initialization:
            model_initialization = initialization[model_name]
        else:
            model_initialization = initialization['default']
        n_params = len(getattr(model, model_name)(
            *model_initialization[0],
            **model_initialization[1]
        ).identity)
        for n_param in xrange(n_params):
            yield model_jacobian_position_jacobian_parameter, model_name, n_param, model_initialization


def model_jacobian_position_jacobian_parameter(model_name, n_param, model_initialization, eps_max=5, n_points=5, random=numpy.random.RandomState(0)):
    model_ = getattr(model, model_name)(
        *model_initialization[0], **model_initialization[1])
    points = random.randn(n_points, 3)
    jac_jac = numpy.empty((n_points, len(model_.parameter), 3, 3))
    jac_jac_approx = numpy.empty_like(jac_jac)
    eps = 10. ** (-eps_max)
    parameter = model_.identity.copy()
    parameter[n_param] += 1e-2
    model_.parameter = parameter

    stencil = numpy.vstack([
        points - (eps, 0, 0),
        points + (eps, 0, 0),
        points - (0, eps, 0),
        points + (0, eps, 0),
        points - (0, 0, eps),
        points + (0, 0, eps),
    ])

    transformed_stencil = model_.jacobian(stencil)

    f_minus_x = transformed_stencil[:len(points)]
    f_plus_x = transformed_stencil[len(points): 2 * len(points)]

    f_minus_y = transformed_stencil[2 * len(points): 3 * len(points)]
    f_plus_y = transformed_stencil[3 * len(points): 4 * len(points)]

    f_minus_z = transformed_stencil[4 * len(points): 5 * len(points)]
    f_plus_z = transformed_stencil[5 * len(points): 6 * len(points)]

    jac_jac_approx[:, :, 0, :] = (f_plus_x - f_minus_x) / (2 * eps)
    jac_jac_approx[:, :, 1, :] = (f_plus_y - f_minus_y) / (2 * eps)
    jac_jac_approx[:, :, 2, :] = (f_plus_z - f_minus_z) / (2 * eps)

    jac_jac[:] = model_.jacobian_parameter_jacobian_position(points)

    model_.parameter = model_.identity.copy()

    testing.assert_array_almost_equal(
        jac_jac_approx, jac_jac, decimal=4,
        err_msg="Transform %s parameter %d did not pass the jacobian with respect to position test" % (
            model_name, n_param
        ), verbose=True
    )


@unittest.skip
@with_setup(setup_transforms)
def test_model_jacobians_two_parameters():
    for model_name in models_to_test:
        if model_name in initialization:
            model_initialization = initialization[model_name]
        else:
            model_initialization = initialization['default']
        n_params = len(getattr(model, model_name)(
            *model_initialization[0],
            **model_initialization[1]
        ).identity)

        if n_params < 2:
            continue
        for n_param in xrange(n_params):
            # n_param2 = random.randint(0, len(model_.parameter))
            # while n_param2 == n_param:
            #    n_param2 = random.randint(0, len(model_.parameter))

            for n_param2 in xrange(n_params):
                if n_param2 == n_param:
                    continue
                yield model_jacobian_two_parameters, model_name, n_param, n_param2, model_initialization


def model_jacobian_two_parameters(model_name, n_param, n_param2, model_initialization, eps_max=11, n_points=200, random=numpy.random.RandomState(0)):
    model_ = getattr(model, model_name)(
        *model_initialization[0], **model_initialization[1])
    points = random.randn(n_points, 3)
    approx = numpy.empty((n_points, 3))
    jac = numpy.empty((n_points, 3))

    eps = 10. ** (-eps_max)
    parameter = model_.identity.copy()
    parameter[n_param] = model_.identity[n_param] - eps
    model_.parameter = parameter

    if model_.bounds is not None:
        model_.parameter[n_param2] = (
            model_.identity[n_param2] + random.randn() * .1
        ).clip(model_.bounds[n_param2, 0], model_.bounds[n_param2, 1])
    else:
        model_.parameter[n_param2] = (
            model_.identity[n_param2] + random.randn() * .1
        )

    f_minus = model_.transform_points(points)
    model_.parameter[n_param] = model_.identity[n_param] + eps
    f_plus = model_.transform_points(points)
    model_.parameter[n_param] = model_.identity[n_param]
    approx[:] = (f_plus - f_minus) / (2 * eps)
    jac[:] = model_.jacobian(points)[:, n_param, :]

    testing.assert_array_almost_equal(
        approx, jac, decimal=3,
        err_msg="Transform %s parameters %d, %d did not pass the jacobian test" % (
            model_name, n_param, n_param2
        ), verbose=True
    )


@with_setup(setup_transforms)
def test_model_jacobian_vector_matrices():
    for model_name in models_to_test:
        if model_name in initialization:
            model_initialization = initialization[model_name]
        else:
            model_initialization = initialization['default']
        n_params = len(getattr(model, model_name)(
            *model_initialization[0],
            **model_initialization[1]
        ).identity)
        for n_param in xrange(n_params):
            yield model_jacobian_vector_matrices, model_name, n_param, model_initialization


def model_jacobian_vector_matrices(model_name, n_param, model_initialization, eps_max=11, n_points=200, random=numpy.random.RandomState(0)):
    model_ = getattr(model, model_name)(
        *model_initialization[0], **model_initialization[1])
    points = random.randn(n_points, 3)
    vectors = random.rand(n_points, 3) - .5
    approx = numpy.empty((n_points, 3))
    jac = numpy.empty((n_points, 3))

    eps = 10. ** (-eps_max)
    parameter = model_.identity.copy()
    parameter[n_param] = model_.identity[n_param] - eps
    model_.parameter = parameter.copy()
    f_minus = model_.transform_vectors(points, vectors)
    parameter[n_param] = model_.identity[n_param] + eps
    model_.parameter = parameter.copy()
    f_plus = model_.transform_vectors(points, vectors)
    parameter[n_param] = model_.identity[n_param]
    model_.parameter = parameter.copy()
    approx[:] = (f_plus - f_minus) / (2 * eps)
    jac[:] = model_.jacobian_vector_matrices(points, vectors)[:, n_param, :]

    testing.assert_array_almost_equal(
        approx, jac, decimal=4,
        err_msg="Transform %s parameter %d did not pass the jacobian test" % (
            model_name, n_param
        ), verbose=True
    )


@with_setup(setup_transforms)
def test_model_jacobian_tensor_matrices():
    for model_name in models_to_test:
        if model_name in initialization:
            model_initialization = initialization[model_name]
        else:
            model_initialization = initialization['default']
        n_params = len(getattr(model, model_name)(
            *model_initialization[0],
            **model_initialization[1]
        ).identity)
        for n_param in xrange(n_params):
            yield model_jacobian_tensor_matrices, model_name, n_param, model_initialization


def model_jacobian_tensor_matrices(model_name, n_param, model_initialization, eps_max=11, n_points=200, random=numpy.random.RandomState(0)):
    model_ = getattr(model, model_name)(
        *model_initialization[0], **model_initialization[1])
    points = random.randn(n_points, 3)
    tensors = random.rand(n_points, 3) - .5
    tensors = vectorized_dot_product(tensors[:, :, None], tensors[:, None, :])
    approx = numpy.empty((n_points, 3, 3))
    jac = numpy.empty((n_points, 3, 3))

    eps = 10. ** (-eps_max)
    parameter = model_.identity.copy()
    parameter[n_param] = model_.identity[n_param] - eps
    model_.parameter = parameter
    f_minus = model_.transform_tensors(points, tensors)
    parameter[n_param] = model_.identity[n_param] + eps
    model_.parameter = parameter
    f_plus = model_.transform_tensors(points, tensors)
    parameter[n_param] = model_.identity[n_param]
    model_.parameter = parameter
    approx[:] = (f_plus - f_minus) / (2 * eps)
    jac[:] = model_.jacobian_tensor_matrices(points, tensors)[:, n_param, :]

    testing.assert_array_almost_equal(
        approx, jac, decimal=4,
        err_msg="Transform %s parameter %d did not pass the jacobian test" % (
            model_name, n_param
        ), verbose=True
    )
