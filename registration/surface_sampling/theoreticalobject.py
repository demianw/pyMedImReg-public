from .. import metric
import numpy


class SphereRadius(metric.Metric):
    def __init__(self, radius, transform=None, points_moving=None):
        self.radius = radius
        self.radius2 = radius ** 2
        self.transform = transform
        self.points_moving = points_moving

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None or self.points_moving is None:
            return ValueError('transform and points_moving must be set for this')

        self.transform.parameter[:] = parameter
        points_moving = numpy.ascontiguousarray(self.transform.transform_points(self.points_moving))

        f, grad = self.metric_gradient(points_moving, use_transform=False)

        jacobian = self.transform.jacobian(points_moving)
        grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

        return f, grad

    def metric_gradient(self, points_moving, use_transform=True):
        if use_transform and (self.transform is not None):
            points_moving = self.transform.transform_points(points_moving)

        f = (((points_moving ** 2).sum(1) - self.radius2) ** 2).mean()
        grad = 2 * (((points_moving ** 2).sum(1) - self.radius2))[:, None] * 2 * points_moving / len(points_moving)

        return f, grad

    def hessian(self, points_moving):
        hessian = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T

        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2

        hessian[:, 0, 0] = 4 * (3 * x2 + y2 + z2 - 1)
        hessian[:, 0, 1] = 8 * x * y
        hessian[:, 0, 2] = 8 * x * z

        hessian[:, 1, 0] = hessian[:, 0, 1]
        hessian[:, 1, 1] = 4 * (x2 + 3 * y2 + z2 - 1)
        hessian[:, 1, 2] = 8 * y * z

        hessian[:, 2, 0] = hessian[:, 0, 2]
        hessian[:, 2, 1] = hessian[:, 1, 2]
        hessian[:, 2, 2] = 4 * (x2 + y2 + 3 * z2 - 1)

        return hessian

    def hessian_eigendecomp(self, points_moving, sort=True):
        evals = numpy.empty((len(points_moving), 1, 3), dtype=float)
        evecs = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T
        x = numpy.maximum(x, 1e-100)
        y = numpy.maximum(y, 1e-100)
        z = numpy.maximum(z, 1e-100)

        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2

        sx = numpy.sign(x)
        sz = numpy.sign(z)

        ax = numpy.abs(x)
        az = numpy.abs(z)

        dxyz = numpy.sqrt(x2 + y2 + z2)
        dxy = numpy.sqrt(x2 + y2)

        evecs[:, 0, 0] = sz * x / dxyz
        evecs[:, 1, 0] = sz * y / dxyz
        evecs[:, 2, 0] = az / dxyz

        evecs[:, 0, 1] = -sx * y / dxy
        evecs[:, 1, 1] = ax / dxy
        evecs[:, 2, 1] = 0

        evecs[:, 0, 2] = -z * x / (dxyz * dxy)
        evecs[:, 1, 2] = -y * z / (dxyz * dxy)
        evecs[:, 2, 2] = dxy / dxyz

        evals[:, 0, 0] = 12 * (x2 + y2 + z2) - 4.
        evals[:, 0, 1] = 4. * (x2 + y2 + z2 - 1.)
        evals[:, 0, 2] = evals[:, 0, 1]

        if sort:
            ix = numpy.arange(len(evals))
            evals_order = numpy.argsort(evals, 2).squeeze()[:, ::-1]

            evals_sorted = numpy.c_[
                evals[ix, 0, evals_order[:, 0]],
                evals[ix, 0, evals_order[:, 1]],
                evals[ix, 0, evals_order[:, 2]],
            ]

            evecs_sorted = numpy.empty_like(evecs)
            evecs_sorted[ix, :, 0] = evecs[ix, :, evals_order[:, 0]]
            evecs_sorted[ix, :, 1] = evecs[ix, :, evals_order[:, 1]]
            evecs_sorted[ix, :, 2] = evecs[ix, :, evals_order[:, 2]]
            evals = evals_sorted
            evecs = evecs_sorted

        return evals, evecs

    def valley_surface(self, points_moving):

        evals, evecs = self.hessian_eigendecomp(self, points_moving)

        ridge_surface = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T

        x2 = numpy.maximum(x ** 2, 1e-10)
        y2 = numpy.maximum(y ** 2, 1e-10)
        z2 = numpy.maximum(z ** 2, 1e-10)

        ridge_surface[:, 0, 0] = 1 / (x2 / z2 + 1) ** 2 + 1 / (y2 / x2 + z2 / x2 + 1) ** 2
        ridge_surface[:, 0, 1] = y / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)
        ridge_surface[:, 0, 2] = -x / ((x2 / z2 + 1) ** 2 * z) + z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)

        ridge_surface[:, 1, 0] = y / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)
        ridge_surface[:, 1, 1] = y2 / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)
        ridge_surface[:, 1, 2] = y * z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)

        ridge_surface[:, 2, 0] = -x / ((x2 / z2 + 1) ** 2 * z) + z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)
        ridge_surface[:, 2, 1] = y * z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)
        ridge_surface[:, 2, 2] = x2 / ((x2 / z2 + 1) ** 2 * z2) + z2 / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)

        return ridge_surface


class BumpedCube(metric.Metric):
    def __init__(self, radius, transform=None, points_moving=None):
        self.radius = radius
        self.radius2 = radius ** 2
        self.transform = transform
        self.points_moving = points_moving

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None or self.points_moving is None:
            return ValueError('transform and points_moving must be set for this')

        self.transform.parameter[:] = parameter
        points_moving = numpy.ascontiguousarray(self.transform.transform_points(self.points_moving))

        f, grad = self.metric_gradient(points_moving, use_transform=False)

        jacobian = self.transform.jacobian(points_moving)
        grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

        return f, grad

    def metric_gradient(self, points_moving, use_transform=True):
        if use_transform and (self.transform is not None):
            points_moving = self.transform.transform_points(points_moving)

        f = (((points_moving ** 2).sum(1) - self.radius2) ** 2).mean()
        grad = 2 * (((points_moving ** 2).sum(1) - self.radius2))[:, None] * 2 * points_moving / len(points_moving)

        return f, grad

    def hessian(self, points_moving):
        hessian = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T

        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2

        hessian[:, 0, 0] = 4 * (3 * x2 + y2 + z2 - 1)
        hessian[:, 0, 1] = 8 * x * y
        hessian[:, 0, 2] = 8 * x * z

        hessian[:, 1, 0] = hessian[:, 0, 1]
        hessian[:, 1, 1] = 4 * (x2 + 3 * y2 + z2 - 1)
        hessian[:, 1, 2] = 8 * y * z

        hessian[:, 2, 0] = hessian[:, 0, 2]
        hessian[:, 2, 1] = hessian[:, 2, 1]
        hessian[:, 2, 2] = 4 * (x2 + y2 + 3 * z2 - 1)

        return hessian


class Segments(metric.Metric):
    def __init__(self, starts_ends, transform=None, points_moving=None):
        starts, ends = zip(*starts_ends)
        self.starts = numpy.atleast_2d(starts)
        self.ends = numpy.atleast_2d(ends)
        self.line_vectors = self.ends - self.starts
        self.line_norms2 = (self.line_vectors ** 2).sum(1)
        self.line_norms = numpy.sqrt(self.line_norms2)
        self.transform = transform
        self.points_moving = points_moving

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None or self.points_moving is None:
            return ValueError('transform and points_moving must be set for this')

        self.transform.parameter[:] = parameter
        points_moving = numpy.ascontiguousarray(self.transform.transform_points(self.points_moving))

        f, grad = self.metric_gradient(points_moving, use_transform=False)

        jacobian = self.transform.jacobian(points_moving)
        grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

        return f.mean(), grad

    def metric_gradient(self, points_moving, use_transform=True):
        points_moving = numpy.atleast_2d(points_moving)
        if use_transform and (self.transform is not None):
            points_moving = self.transform.transform_points(points_moving)

        grad = numpy.empty((len(points_moving), 3), dtype=float)
        f = numpy.empty(len(points_moving), dtype=float)
        f[:] = numpy.inf
        grad_segment = numpy.empty((len(points_moving), 3), dtype=float)

        for i in xrange(len(self.starts)):
            pos = (i,)
            f_segment, grad_segment = self.segment_metric_grad(points_moving, pos, grad_segment)
            distance_to_keep = f_segment < f
            f[distance_to_keep] = f_segment[distance_to_keep]
            grad[distance_to_keep, :] = grad_segment[distance_to_keep, :]

        return f.mean(), grad / len(grad)

    def metric_gradient_per_point(self, points_moving, use_transform=True):
        points_moving = numpy.atleast_2d(points_moving)
        if use_transform and (self.transform is not None):
            points_moving = self.transform.transform_points(points_moving)

        grad = numpy.empty((len(points_moving), 3), dtype=float)
        f = numpy.empty(len(points_moving), dtype=float)
        f[:] = numpy.inf
        grad_segment = numpy.empty((len(points_moving), 3), dtype=float)

        for i in xrange(len(self.starts)):
            pos = (i,)
            f_segment, grad_segment = self.segment_metric_grad(points_moving, pos, grad_segment)
            distance_to_keep = f_segment < f
            f[distance_to_keep] = f_segment[distance_to_keep]
            grad[distance_to_keep, :] = grad_segment[distance_to_keep, :]

        return f, grad

    def segment_metric_grad(self, points_moving, pos, grad_segment):
        diff_start = points_moving - self.starts[pos, :]
        diff_end = points_moving - self.ends[pos, :]
        t = ((diff_start) * self.line_vectors[pos, :]).sum(1) / (self.line_norms[pos] ** 2)
        closer_to_start = t <= 0
        closer_to_end = 1 <= t
        inside = ~(closer_to_start + closer_to_end)

        grad_segment[closer_to_start] = diff_start[closer_to_start]
        if any(inside):
            grad_segment[inside] = -(t[inside][:, None] * self.line_vectors[pos, :] - diff_start[inside])
        grad_segment[closer_to_end] = diff_end[closer_to_end]

        f_segment = (grad_segment ** 2).sum(1)
        grad_segment *= 2
        return f_segment, grad_segment

    def segment_hessian_evigendecom(self, points_moving, pos):

        hessian_segment_eval = numpy.empty((len(points_moving), 3))
        hessian_segment_evec = numpy.empty((len(points_moving), 3, 3))

        diff_start = points_moving - self.starts[pos, :]
        diff_end = points_moving - self.ends[pos, :]
        t = ((diff_start) * self.line_vectors[pos, :]).sum(1) / (self.line_norms[pos] ** 2)
        closer_to_start = t <= 0
        closer_to_end = 1 <= t
        not_inside = (closer_to_start + closer_to_end)
        inside = ~not_inside

        v = self.starts[pos]
        w = self.ends[pos]

        if any(not_inside):
            hessian_segment_evec[not_inside, :, :] = 0
            hessian_segment_evec[not_inside, (0, 1, 2), (2, 1, 0)] = 1
            hessian_segment_evec[not_inside, :] = (0, 0, 2)

        if any(inside):
            d1 = (-w[0] + v[0])
            d2 = (-w[2] + v[2])

            if abs(d1) == 0:
                d1 = 1e-10

            if abs(d2) == 0:
                d2 = 1e-10

            hessian_segment_evec[inside, 0, 0] = -(-w[2] + v[2]) / d1
            hessian_segment_evec[inside, 0, 1] = -(-w[1] + v[1]) / d1
            hessian_segment_evec[inside, 0, 2] = (-w[0] + v[0]) / d2
            hessian_segment_evec[inside, 1, 0] = 0
            hessian_segment_evec[inside, 1, 1] = 1
            hessian_segment_evec[inside, 1, 2] = (-w[1] + v[1]) / d2
            hessian_segment_evec[inside, 2, 0] = 1
            hessian_segment_evec[inside, 2, 1] = 0
            hessian_segment_evec[inside, 2, 2] = 1

            hessian_segment_eval[inside, :] = (2., 2., 0.)

        return hessian_segment_eval, hessian_segment_evec


class SmoothSquare(metric.Metric):
    def __init__(self, transform=None, points_moving=None):
        self.transform = transform
        self.points_moving = points_moving
        self.tensors = None
        self.vectors = None

    def metric_gradient(self, points_moving, use_transform=True):
        if use_transform and (self.transform is not None):
            points_moving = self.transform.transform_points(points_moving)

        gradient = numpy.empty((len(points_moving), 3), dtype=float)

        x, y, z = points_moving.T
        x2, y2, z2 = (points_moving ** 2).T
        x3, y3, z3 = (points_moving ** 3).T
        x4, y4, z4 = (points_moving ** 4).T

        f = x4 - x2 + y4 - y2 + z4 - z2
        gradient[:, 0] = 4 * (2 * x3 - x) * f
        gradient[:, 1] = 4 * (2 * y3 - y) * f
        gradient[:, 2] = 4 * (2 * z3 - z) * f

        #m2 = points_moving ** 2
        #m4 = m2 ** 2
        #sqrt_f = (m2.sum(1) - m4.sum(1))
        #f = sqrt_f ** 2
        #grad = 4 * (2 * points_moving ** 3 - points_moving) * sqrt_f[:, None]

        return (f ** 2).sum(), gradient

    def gradient(self, points_moving):
        gradient = numpy.empty((len(points_moving), 3), dtype=float)

        x, y, z = points_moving.T
        x2, y2, z2 = (points_moving ** 2).T
        x3, y3, z3 = (points_moving ** 3).T
        x4, y4, z4 = (points_moving ** 4).T

        f = x4 - x2 + y4 - y2 + z4 - z2
        gradient[:, 0] = 4 * (2 * x3 - x) * f
        gradient[:, 1] = 4 * (2 * y3 - y) * f
        gradient[:, 2] = 4 * (2 * z3 - z) * f

        return gradient

    def hessian(self, points_moving):
        hessian = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T
        x2, y2, z2 = (points_moving ** 2).T
        x3, y3, z3 = (points_moving ** 3).T
        f = x2 * x2 - x2 + y2 * y2 - y2 + z2 * z2 - z2

        two_x3_minus_x = 2 * x3 - x
        two_y3_minus_y = 2 * y3 - y
        two_z3_minus_z = 2 * z3 - z

        hessian[:, 0, 0] = (6 * x2 - 1) * f + 8 * (2 * x3 - x) ** 2
        hessian[:, 0, 1] = 8 * two_x3_minus_x * two_y3_minus_y
        hessian[:, 0, 2] = 8 * two_x3_minus_x * two_z3_minus_z

        hessian[:, 1, 0] = hessian[:, 0, 1]
        hessian[:, 1, 1] = (6 * y2 - 1) * f + 8 * (2 * y3 - y) ** 2
        hessian[:, 1, 2] = 8 * two_y3_minus_y * two_z3_minus_z

        hessian[:, 2, 0] = hessian[:, 0, 2]
        hessian[:, 2, 1] = hessian[:, 1, 2]
        hessian[:, 2, 2] = (6 * z2 - 1) * f + 8 * (2 * z3 - z) ** 2

        return hessian

    def hessian_eigendecomp(self, points_moving, sort=True):
        evals = numpy.empty((len(points_moving), 1, 3), dtype=float)
        evecs = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T
        x = numpy.maximum(x, 1e-100)
        y = numpy.maximum(y, 1e-100)
        z = numpy.maximum(z, 1e-100)

        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2

        sx = numpy.sign(x)
        sz = numpy.sign(z)

        ax = numpy.abs(x)
        az = numpy.abs(z)

        dxyz = numpy.sqrt(x2 + y2 + z2)
        dxy = numpy.sqrt(x2 + y2)

        evecs[:, 0, 0] = sz * x / dxyz
        evecs[:, 1, 0] = sz * y / dxyz
        evecs[:, 2, 0] = az / dxyz

        evecs[:, 0, 1] = -sx * y / dxy
        evecs[:, 1, 1] = ax / dxy
        evecs[:, 2, 1] = 0

        evecs[:, 0, 2] = -z * x / (dxyz * dxy)
        evecs[:, 1, 2] = -y * z / (dxyz * dxy)
        evecs[:, 2, 2] = dxy / dxyz

        evals[:, 0, 0] = 12 * (x2 + y2 + z2) - 4.
        evals[:, 0, 1] = 4. * (x2 + y2 + z2 - 1.)
        evals[:, 0, 2] = evals[:, 0, 1]

        if sort:
            ix = numpy.arange(len(evals))
            evals_order = numpy.argsort(evals, 2).squeeze()[:, ::-1]

            evals_sorted = numpy.c_[
                evals[ix, 0, evals_order[:, 0]],
                evals[ix, 0, evals_order[:, 1]],
                evals[ix, 0, evals_order[:, 2]],
            ]

            evecs_sorted = numpy.empty_like(evecs)
            evecs_sorted[ix, :, 0] = evecs[ix, :, evals_order[:, 0]]
            evecs_sorted[ix, :, 1] = evecs[ix, :, evals_order[:, 1]]
            evecs_sorted[ix, :, 2] = evecs[ix, :, evals_order[:, 2]]
            evals = evals_sorted
            evecs = evecs_sorted

        return evals, evecs

    def valley_surface(self, points_moving):

        evals, evecs = self.hessian_eigendecomp(self, points_moving)

        ridge_surface = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T

        x2 = numpy.maximum(x ** 2, 1e-10)
        y2 = numpy.maximum(y ** 2, 1e-10)
        z2 = numpy.maximum(z ** 2, 1e-10)

        ridge_surface[:, 0, 0] = 1 / (x2 / z2 + 1) ** 2 + 1 / (y2 / x2 + z2 / x2 + 1) ** 2
        ridge_surface[:, 0, 1] = y / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)
        ridge_surface[:, 0, 2] = -x / ((x2 / z2 + 1) ** 2 * z) + z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)

        ridge_surface[:, 1, 0] = y / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)
        ridge_surface[:, 1, 1] = y2 / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)
        ridge_surface[:, 1, 2] = y * z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)

        ridge_surface[:, 2, 0] = -x / ((x2 / z2 + 1) ** 2 * z) + z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x)
        ridge_surface[:, 2, 1] = y * z / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)
        ridge_surface[:, 2, 2] = x2 / ((x2 / z2 + 1) ** 2 * z2) + z2 / ((y2 / x2 + z2 / x2 + 1) ** 2 * x2)

        return ridge_surface


class BumpedCube(metric.Metric):
    def __init__(self, radius, transform=None, points_moving=None):
        self.radius = radius
        self.radius2 = radius ** 2
        self.transform = transform
        self.points_moving = points_moving

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None or self.points_moving is None:
            return ValueError('transform and points_moving must be set for this')

        self.transform.parameter[:] = parameter
        points_moving = numpy.ascontiguousarray(self.transform.transform_points(self.points_moving))

        f, grad = self.metric_gradient(points_moving, use_transform=False)

        jacobian = self.transform.jacobian(points_moving)
        grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

        return f, grad

    def metric_gradient(self, points_moving, use_transform=True):
        if use_transform and (self.transform is not None):
            points_moving = self.transform.transform_points(points_moving)

        f = (((points_moving ** 2).sum(1) - self.radius2) ** 2).mean()
        grad = 2 * (((points_moving ** 2).sum(1) - self.radius2))[:, None] * 2 * points_moving / len(points_moving)

        return f, grad

    def hessian(self, points_moving):
        hessian = numpy.empty((len(points_moving), 3, 3), dtype=float)

        x, y, z = points_moving.T

        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2

        hessian[:, 0, 0] = 4 * (3 * x2 + y2 + z2 - 1)
        hessian[:, 0, 1] = 8 * x * y
        hessian[:, 0, 2] = 8 * x * z

        hessian[:, 1, 0] = hessian[:, 0, 1]
        hessian[:, 1, 1] = 4 * (x2 + 3 * y2 + z2 - 1)
        hessian[:, 1, 2] = 8 * y * z

        hessian[:, 2, 0] = hessian[:, 0, 2]
        hessian[:, 2, 1] = hessian[:, 2, 1]
        hessian[:, 2, 2] = 4 * (x2 + y2 + 3 * z2 - 1)

        return hessian
