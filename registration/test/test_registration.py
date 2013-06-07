from .. import *

import pprint

import numpy
from numpy import testing

from unittest import skip


pp = pprint.PrettyPrinter(indent=4)


def sphere_sample(N, random=numpy.random.RandomState()):
    phi = random.random_sample(N) * 2 * numpy.pi
    costheta = random.random_sample(N) * 2 - 1

    theta = numpy.arccos(costheta)
    r = random.rand(N).clip(1e-10, 1)

    x = numpy.sin(theta) * numpy.cos(phi) * r
    y = numpy.sin(theta) * numpy.sin(phi) * r
    z = numpy.cos(theta) * r

    return numpy.c_[x, y, z]


@skip
def test_translation(N=100, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random)
    registration = Registration(
        model=Translation(),
        metric=Correlation(fixed, sigma)
    )

    translation = random.randn(3) * sigma / 8.
    moving = fixed + translation

    registration.register(moving, return_all=True)
    parameter = registration.model.parameter

    testing.assert_allclose(parameter, registration.model.parameter, rtol=1e-1)


@skip
def test_isotropic_scale(N=100, sigma=2., random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random)

    registration = Registration(
        model=IsotropicScale(),
        metric=Correlation(fixed, sigma)
    )

    scale = numpy.clip(random.rand() * sigma / 2. + 1, 1e-10, numpy.inf)
    moving = fixed * scale

    registration.register(moving, return_all=True, **kwargs)
    parameter = registration.model.parameter
    print scale, parameter
    testing.assert_allclose(scale, parameter, rtol=1e-1)


@skip
def test_anisotropic_scale(N=200, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random) * (1, .1, .5)

    registration = Registration(
        model=AnisotropicScale(),
        metric=Correlation(fixed, sigma)
    )

    scale = numpy.clip(random.rand(3) * sigma + 1, 1e-10, numpy.inf)
    moving = fixed * scale

    kwargs['pgtol'] = 1e-20
    registration.register(moving, return_all=True, **kwargs)
    parameter = registration.model.parameter
    testing.assert_allclose(scale, parameter, rtol=1e-1)


@skip
def test_rotation(N=100, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random)

    registration = Registration(
        model=Rotation(),
        metric=Correlation(fixed, sigma)
    )

    for axis in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        rotation = (random.rand(
            3) * sigma * numpy.pi - numpy.pi / 2 * sigma / 2.) * axis
        registration.model.parameter = rotation
        moving = registration.model.transform_points(fixed)

        kwargs['pgtol'] = 1e-20
        registration.register(moving, return_all=True, **kwargs)
        parameter = registration.model.parameter
        testing.assert_allclose(rotation, parameter, rtol=1e-1, atol=1e-3)


@skip
def test_rigid_translation(N=400, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random)

    registration = Registration(
        model=Rigid(),
        metric=Correlation(fixed, sigma)
    )

    translation = (random.randn(3) * sigma / 5.) * (1, 1, 0)
    initial_parameter = registration.model.identity.copy()
    initial_parameter[-3:] = translation
    registration.model.parameter = initial_parameter
    moving = registration.model.transform_points(fixed)

    registration.register(moving, return_all=True, **kwargs)
    parameter = registration.model.parameter
    initial_parameter[-3:] *= -1
    testing.assert_allclose(initial_parameter, parameter, rtol=1e-1, atol=1e-3)


@skip
def test_rigid_rotation(N=400, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random) * (1, .2, .8)

    registration = Registration(
        model=Rigid(),
        metric=Correlation(fixed, sigma)
    )

    for param_position in (0, 1):
        rotation = numpy.clip(
            random.rand() * sigma, -numpy.pi / 2., numpy.pi / 2.)

        initial_parameter = registration.model.identity.copy()
        initial_parameter[param_position] = rotation

        registration.model.parameter = initial_parameter
        moving = registration.model.transform_points(fixed)

        registration.register(moving, return_all=True, **kwargs)
        parameter = registration.model.parameter

        initial_parameter[param_position] *= -1
        testing.assert_allclose(
            initial_parameter, parameter, rtol=1e-1, atol=1e-3)


@skip
def test_similarity_scale(N=100, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random) * (1, .2, .6)

    registration = Registration(
        model=Similarity(),
        metric=Correlation(fixed, sigma)
    )

    scale = numpy.clip(random.rand() * sigma / 4. + 1, 1e-10, numpy.inf)
    initial_parameter = registration.model.identity.copy()
    initial_parameter = scale
    registration.model.parameter[0] = initial_parameter
    moving = registration.model.transform_points(fixed)

    registration.register(moving, return_all=True, **kwargs)
    parameter = registration.model.parameter
    initial_parameter = 1. / initial_parameter
    testing.assert_allclose(initial_parameter, parameter[
                            0], rtol=1e-1, atol=1e-3)


@skip
def test_similarity_translation(N=100, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random) * (1, .2, .8)

    registration = Registration(
        model=Similarity(),
        metric=Correlation(fixed, sigma)
    )

    translation = (random.randn(3) * sigma / 5.)
    initial_parameter = registration.model.identity.copy()
    initial_parameter[-3:] = translation
    registration.model.parameter = initial_parameter
    moving = registration.model.transform_points(fixed)

    registration.register(moving, return_all=True, **kwargs)
    parameter = registration.model.parameter
    initial_parameter[-3:] *= -1
    testing.assert_allclose(initial_parameter, parameter, rtol=1e-1, atol=1e-3)


@skip
def test_similarity_rotation(N=100, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random) * (1, .2, .8)

    for param_position in (1, 2, 3):
        registration = Registration(
            model=Similarity(),
            metric=Correlation(fixed, sigma),
        )

        rotation = numpy.clip(
            random.rand() * sigma, -numpy.pi / 2., numpy.pi / 2.)
        initial_parameter = registration.model.identity.copy()
        initial_parameter[param_position] = rotation
        yield (
            internal_registration,
            initial_parameter,
            registration,
            fixed
        )


@skip
def test_affine_translation(N=100, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random) * (1, .2, .8)

    registration = Registration(
        model=Affine(),
        metric=Correlation(fixed, sigma)
    )

    translation = (random.randn(3) * sigma / 5.) * (1, 1, 0)
    initial_parameter = registration.model.identity.copy()
    initial_parameter[-3:] = translation
    registration.model.parameter = initial_parameter
    moving = registration.model.transform_points(fixed)

    registration.register(moving, return_all=True, **kwargs)
    parameter = registration.model.parameter
    initial_parameter[-3:] *= -1
    testing.assert_allclose(initial_parameter, parameter, rtol=1e-1, atol=1e-3)


@skip
def test_affine_rotation(N=100, sigma=.05, random=numpy.random.RandomState()):
    fixed = sphere_sample(N, random) * (1, .2, .8)

    for param_position in (3, 4, 5):
        registration = Registration(
            model=Affine(),
            metric=Correlation(fixed, sigma)
        )

        rotation = numpy.clip(
            random.rand() * sigma, -numpy.pi / 2., numpy.pi / 2.)
        initial_parameter = registration.model.identity.copy()
        initial_parameter[param_position] = rotation

        yield (
            internal_registration,
            initial_parameter,
            registration,
            fixed
        )


@skip
def test_affine_scale(N=100, sigma=.1, random=numpy.random.RandomState(), **kwargs):
    fixed = sphere_sample(N, random) * (1, .2, .8)

    registration = Registration(
        model=Affine(),
        metric=Correlation(fixed, sigma)
    )

    scales = numpy.clip(random.rand(3) * sigma + 1, 1e-10, numpy.inf)

    initial_parameter = registration.model.identity.copy()
    initial_parameter[:3] = scales
    registration.model.parameter = initial_parameter
    moving = registration.model.transform_points(fixed)

    registration.register(moving, return_all=True, **kwargs)
    parameter = registration.model.parameter
    initial_parameter[:3] = 1. / initial_parameter[:3]
    testing.assert_allclose(initial_parameter, parameter, rtol=1e-1, atol=1e-3)


@skip
def internal_registration(initial_parameter, registration, fixed):
    registration.model.parameter[:] = initial_parameter
    registration.optimizer.optimizer_args['pgtol'] = 1e-20
    registration.optimizer.optimizer_args['factr'] = 1e-3
    moving = registration.model.transform_points(fixed)
    registration.register(moving, return_all=True)
    testing.assert_almost_equal(
        fixed, registration.model.transform_points(moving), decimal=4)

    # initial_parameter[param_position] *= -1
    # testing.assert_allclose(initial_parameter, parameter, rtol=1e-1,
    # atol=1e-3)
