from .. import metric
from ..metric import _metrics

import inspect

import numpy
from numpy import testing

from scipy.optimize import approx_fprime

from nose.tools import with_setup

# import warnings
# warnings.simplefilter("error")

initialization = None
metrics_to_test = None


def eye_tensors(n):
    t = numpy.zeros((n, 3, 3))
    t[:, (0, 1, 2), (0, 1, 2)] = 1
    return t


def random_tensors(n, random=numpy.random.RandomState(0)):
    r1 = random.randn(n, 3, 3)
    for r in r1:
        r[:] = numpy.dot(r, r.T)
    return r1


def setup_metrics(N=10, noise=1e-4, random=numpy.random.RandomState(0)):
    global initialization
    global metrics_to_test
    print "Setting up"

    points = numpy.random.rand(N, 3)
    vectors = numpy.ones_like(points)
    points_fixed = numpy.random.rand(N, 3)
    vectors_fixed = (
        numpy.ones_like(points_fixed) +
        numpy.random.randn(*points_fixed.shape) * noise
    )
    excluded_metrics = ['Metric', 'AdditiveMetric']
    metrics_to_test = []
    for m in metric.__all__:
        class_ = getattr(metric, m)

        if (
            inspect.isclass(class_) and
            issubclass(class_, metric.Metric) and
            m not in excluded_metrics and
            not hasattr(class_, 'skip_from_tests')
        ):
            metrics_to_test.append(m)
        elif hasattr(class_, 'skip_from_tests'):
            print "Metric %s signaled to be skipped" % m

    initialization = {
        'default': (tuple(), dict()),
        'SquaredDifference': (
            (lambda x: (x, x, 1))(random.randn(N, 3)),
            {}
        ),
        'Correlation': (
            (
                points,
                points_fixed,
                1.
            ),
            {}
        ),
        'ExactLandmarkL2': (
            (
                points,
                points_fixed,
            ),
            {}
        ),
        'VectorPatchParticlesL2': (
            (
                points,
                vectors,
                1.,
                points_fixed,
                vectors_fixed,
                1.
            ),
            {}
        ),

    }


setup_metrics()


def get_metric_class(metric_name):
    if hasattr(metric, metric_name):
        metric_ = getattr(metric, metric_name)
    else:
        raise ImportError("Metric %s not found" % metric_name)

    return metric_


@with_setup(setup=setup_metrics)
def test_metric_pos_gradients_one_parameter(sigma_noise=1):
    for metric_name in metrics_to_test:
        if metric_name in initialization:
            metric_initialization = initialization[metric_name]
        else:
            metric_initialization = initialization['default']
        # metric_ = get_metric_class(metric_name)(*metric_initialization[0], **metric_initialization[1])
        # points_moving = metric_.points_moving
        yield metric_gradient_pos, metric_name, metric_initialization
        yield metric_gradient_pos, metric_name, metric_initialization, sigma_noise
#        for index in numpy.ndindex(points_moving.shape):
#            yield metric_gradient_pos, metric_name, index, metric_initialization
#        for index in numpy.ndindex(points_moving.shape):
# yield metric_gradient_pos, metric_name, index, metric_initialization,
# sigma_noise


@with_setup(setup=setup_metrics)
def test_metric_tensor_gradients_one_parameter(sigma_noise=1):
    for metric_name in metrics_to_test:
        if metric_name in initialization:
            metric_initialization = initialization[metric_name]
        else:
            metric_initialization = initialization['default']
        metric_ = get_metric_class(metric_name)(
            *metric_initialization[0], **metric_initialization[1])

        if hasattr(metric_, 'tensors') and metric_.tensors is not None:
            yield metric_gradient_tensors, metric_name, metric_initialization
            yield metric_gradient_tensors, metric_name, metric_initialization, sigma_noise


def metric_gradient_pos(metric_name, metric_initialization, sigma_noise=0, eps=1e-8):
    metric_ = get_metric_class(metric_name)(
        *metric_initialization[0], **metric_initialization[1])
    points_moving = metric_.points_fixed.copy() + sigma_noise * numpy.random.randn(
        *metric_.points_fixed.shape)
    if hasattr(metric_, 'vectors'):
        moving_vectors = metric_.vectors
    else:
        moving_vectors = None
    if hasattr(metric_, 'tensors'):
        moving_tensors = metric_.tensors
    else:
        moving_tensors = None

    f_pos = lambda x: metric_.metric_gradient(x.reshape(len(
        x) / 3, 3), vectors=moving_vectors, tensors=moving_tensors)[0]
    g_pos = lambda x: metric_.metric_gradient(x.reshape(len(
        x) / 3, 3), vectors=moving_vectors, tensors=moving_tensors)[1]

    g_approx = approx_fprime(
        points_moving.ravel(), f_pos, eps).reshape(*points_moving.shape)

    testing.assert_array_almost_equal(
        g_pos(points_moving.ravel()), g_approx, decimal=4,
        err_msg="Metric %s did not pass the gradient test with displacement %g" % (
            metric_name, sigma_noise
        ), verbose=True
    )


def metric_gradient_tensors(metric_name, metric_initialization, sigma=0, eps=1e-5):
    metric_ = get_metric_class(metric_name)(
        *metric_initialization[0], **metric_initialization[1])

    points_moving = metric_.points_fixed.copy()
    moving_vectors = metric_.vectors
    moving_tensors = metric_.tensors.copy()
    if sigma > 0:
        moving_tensors += sigma * random_tensors(len(moving_tensors))

    f_pos = lambda x: metric_.metric_gradient(
        points_moving, vectors=moving_vectors, tensors=x.reshape(len(x) / 9, 3, 3))[0]
    g_pos = lambda x: metric_.metric_gradient(
        points_moving, vectors=moving_vectors, tensors=x.reshape(len(x) / 9, 3, 3))[2]

    g_approx = approx_fprime(
        moving_tensors.ravel(), f_pos, eps).reshape(*moving_tensors.shape)

    testing.assert_array_almost_equal(
        g_pos(moving_tensors.ravel()), g_approx, decimal=5,
        err_msg="Metric %s did not pass the tensor gradient test with displacement %g" % (
            metric_name, sigma
        ), verbose=True
    )


def _metric_gradient_tensors_centered(metric_name, index, metric_initialization, sigma=0, eps=1e-5):
    metric_ = get_metric_class(metric_name)(
        *metric_initialization[0], **metric_initialization[1])
    # m2 = lambda m: (m[3], m[4], m[5], m[0], m[1], m[2])
    # metric2_ =
    # get_metric_class(metric_name)(*(m2(metric_initialization[0])),
    # **metric_initialization[1])

    points_moving = metric_.points_fixed.copy()
    moving_vectors = metric_.vectors
    moving_tensors = metric_.tensors.copy()
    if sigma > 0:
        moving_tensors += sigma * random_tensors(len(moving_tensors))

    grad = metric_.metric_gradient(
        points_moving, vectors=moving_vectors, tensors=moving_tensors)[2]

    moving_tensors[index] = metric_.tensors[index] + eps
    moving_tensors[index[0], index[2], index[1]] = moving_tensors[index]

    m_plus = metric_.metric_gradient(
        points_moving, vectors=moving_vectors, tensors=moving_tensors)[0]

    moving_tensors[index] = metric_.tensors_fixed[index] - eps
    moving_tensors[index[0], index[2], index[1]] = moving_tensors[index]
    m_minus = metric_.metric_gradient(
        points_moving, vectors=moving_vectors, tensors=moving_tensors)[0]

    approx_gradient = (m_plus - m_minus) / (2 * eps)
    if index[1] != index[2]:
        approx_gradient /= 2

    testing.assert_array_almost_equal(
        grad[index], approx_gradient, decimal=5,
        err_msg="Metric %s parameter %s did not pass the centered gradient test" % (
            metric_name, str(index)
        ), verbose=True
    )


def test_gauss_transform(N=10, M=10, sigma=1, eps=1e-10, random=numpy.random.RandomState(0)):
    points_fixed = random.randn(N, 3)
    points_moving = random.randn(M, 3)
    f = lambda x: _metrics.gauss_transform(x.reshape(
        len(x) / 3, 3), points_fixed, sigma)[0]
    g = lambda x: _metrics.gauss_transform(x.reshape(
        len(x) / 3, 3), points_fixed, sigma)[-1]

    approx_g = approx_fprime(points_fixed.ravel(), f, eps).reshape(N, 3)
    grad = g(points_fixed.ravel())

    testing.assert_array_almost_equal(
        grad, approx_g, decimal=4,
        err_msg="Gauss transform gradient failed",
        verbose=True
    )

    approx_g = approx_fprime(points_moving.ravel(), f, eps).reshape(M, 3)
    grad = g(points_moving.ravel())

    testing.assert_array_almost_equal(
        grad, approx_g, decimal=4,
        err_msg="Gauss transform gradient non-centered failed",
        verbose=True
    )


