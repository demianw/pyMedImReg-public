from .. import metric
from .. import model
from .. import image_tools
#from .. import currents

import numpy
from scipy import ndimage, optimize

from numpy import testing
from ..model import linear

from unittest import skip

numpy.set_printoptions(precision=2)
random = numpy.random.RandomState(0)

pgtol = 1e-20
factr = 1


def image(N=80, radius=10, sigma=1):
    s = slice(int(-N / 2), int(N / 2), 1)
    i, j, k = numpy.mgrid[s, s, s]

    image = ndimage.gaussian_filter(random.rand(*i.shape), 1)
    image[(numpy.abs(i) + numpy.abs(j * 1.1) + numpy.abs(k * 0.9)) > radius] = 0

    return image


@skip
def test_transforms():
    fixed = image()
    center = numpy.array(fixed.shape) / 2.

    transforms = [
        (linear.Translation, (4, 0, 0), None),
        (linear.Translation, (1, 2, 0), None),
        (linear.Rotation, (numpy.pi / 20., 0, 0), (center,)),
        (linear.Rotation, (0, numpy.pi / 20., 0), (center,)),
        (linear.Rotation, (0, numpy.pi / 20., numpy.pi / 20.), (center,)),
        (linear.Rotation, (0, 0, numpy.pi / 20.), (center,)),
        (linear.Rigid, (0, 0, 0, 0, 0, 2), (center,)),
        (linear.Rigid, (0, 0, numpy.pi / 20, 0, 0, 0), (center,)),
    ]

    for transform_class, parameter, transform_parameters in transforms:
        yield according_to_transforms, metric.ImageMeanSquares, transform_class, parameter, fixed, transform_parameters


def according_to_transforms(metric, transform_class, parameter, moving, transform_parameters):
    if transform_parameters is None:
        transform = transform_class()
    else:
        transform = transform_class(*transform_parameters)

    transform.parameter[:] = parameter

    fixed = image_tools.transform_image(transform, moving)

    m = metric(
        fixed, moving=moving,
        fixed_points=numpy.transpose((fixed).nonzero()),
        transform=transform
    )

    def f(x):
        return m.metric_gradient_transform_parameters(x)[0]

    r = optimize.fmin_l_bfgs_b(m.metric_gradient_transform_parameters, m.transform.identity, disp=1, pgtol=pgtol, factr=factr)

    testing.assert_almost_equal(r[0], parameter, decimal=4)


def curve_current_image(N=80, sigma=1, curve_points=1000):

    curve = numpy.c_[
        numpy.linspace(int(N * 1. / 4.), int(N * 3. / 4.), curve_points),
        numpy.zeros(curve_points) + N / 2,
        numpy.zeros(curve_points) + N / 2
    ]

    image = numpy.zeros((N, N, N, 3))

    currents.curve_current(curve, image, numpy.eye(3), sigma)
    return image, curve


@skip
def test_curve_current_transforms():
    current_image_size = 20
    fixed, fixed_curve = curve_current_image(N=current_image_size)
    center = numpy.array(fixed.shape[:-1]) / 2.

    transforms = [
        (linear.Translation, (4, 0, 0), None),
        (linear.Translation, (1, 2, 0), None),
        (linear.Rotation, (numpy.pi / 20., 0, 0), (center,)),
        (linear.Rotation, (0, numpy.pi / 20., 0), (center,)),
        (linear.Rotation, (0, numpy.pi / 20., numpy.pi / 20.), (center,)),
        (linear.Rotation, (0, 0, numpy.pi / 20.), (center,)),
        (linear.AnisotropicScale, (1.2, 1., 1.), (center,)),
        (linear.AnisotropicScale, (1., 1.2, 1.), (center,)),
        (linear.AnisotropicScale, (1., 1., 1.2), (center,)),
        (linear.Rigid, (0, 0, 0, 0, 0, 2), (center,)),
        (linear.Rigid, (0, 0, numpy.pi / 20, 0, 0, 0), (center,)),
        (model.Affine, (1., 1., 1.01, 0, 0, 0, 0, 0, 0), (center,)),
    ]

    for transform_class, parameter, transform_parameters in transforms:
        yield curve_current_according_to_transforms, metric.VectorImageMeanSquares, transform_class, parameter, transform_parameters, fixed, fixed_curve


def curve_current_according_to_transforms(metric_class, transform_class, parameter, transform_parameters, moving, moving_curve):
    if transform_parameters is None:
        transform = transform_class()
    else:
        transform = transform_class(*transform_parameters)

    transform.parameter[:] = parameter

    fixed = currents.transform_image(transform, moving)
    fixed_curve = transform.transform_points(moving_curve)
    fixed_points = numpy.transpose((fixed).sum(-1).nonzero())

    m = metric_class(
        fixed, moving=moving,
        fixed_points=fixed_points,
        transform=transform,
        gradient_operator=currents.gradient,
        interpolator=currents.map_coordinates
    )

    def f(x):
        return m.metric_transform_parameters(x)

    #x0 = m.transform.identity
    #apg = optimize.approx_fprime(x0, f, 1e-10)
    #testing.assert_almost_equal(apg, m.metric_gradient_transform_parameters(x0)[1], decimal=4)

    r = optimize.fmin_l_bfgs_b(
        m.metric_gradient_transform_parameters,
        m.transform.identity, disp=1, pgtol=pgtol, factr=factr
    )[0]
    #r = optimize.fmin_powell(lambda x: m.metric_gradient_transform_parameters(x)[0], m.transform.identity, disp=1)

    transformed_moving_curve = transform.transform_points(moving_curve)
    sse = ((fixed_curve - transformed_moving_curve) ** 2).sum(1)
    print "Curve SSE: %04f +- %04f" % (sse.mean(), sse.std())
    testing.assert_almost_equal(r, parameter, decimal=4)
