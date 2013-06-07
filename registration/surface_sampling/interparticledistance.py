from .. import metric
import numpy


class InterParticleDistanceMeyer(metric.Metric):
    def __init__(self, sigma, transform=None, points_moving=None):
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.transform = transform
        self.points_moving = points_moving

    def metric_gradient(self, points_moving, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        differences = (points_moving[..., None, :] - points_moving[..., :])
        dist = numpy.sqrt((differences ** 2).sum(-1))
        dist[dist == 0] = 1e-10
        differences /= dist[..., None] * self.sigma

        mask = dist > self.sigma
        arg = dist * (numpy.pi / (2. * self.sigma))
        f = 1. / numpy.tan(arg) + arg - numpy.pi / 2.
        f[numpy.diag_indices(f.shape[0])] = 0
        f[mask] = 0

        f = f.mean()

        grad_mat = numpy.pi / 2. * (1 - numpy.sin(arg) ** -2)
        grad_mat[numpy.diag_indices(grad_mat.shape[0])] = 0
        grad_mat[mask] = 0
        differences *= grad_mat[..., None]
        grad = differences.sum(1)

        return f, grad / len(grad)


class InterParticleDistanceKindlmann(metric.Metric):
    def __init__(self, sigma, depth=-0.002, well_width=0.6, transform=None, points_moving=None, tensors=None):
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.depth = depth
        self.well_width = well_width
        self.transform = transform
        self.points_moving = points_moving
        self.tensors = tensors
        self.vectors = None

    def energy_function_derivative(self, distances):
        energy = numpy.empty_like(distances)
        energy_derivative = numpy.empty_like(distances)

        r = distances / self.sigma
        in_radius = r <= 1
        before_well = r <= self.well_width
        after_well = (r > self.well_width) * in_radius

        dbw = r[before_well]
        x = dbw / self.well_width
        energy[before_well] = 1 + (self.depth - 1) * (3 * x - 3 * x ** 2 + x ** 3)
        energy_derivative[before_well] = (self.depth - 1) * (3 - 6 * x + 3 * x ** 2) / self.well_width

        daw = r[after_well]
        x = (daw - self.well_width) / (self.well_width - 1)
        #energy[after_well] = (
        #    (daw - 1) ** 2 *
        #    (3 * self.well_width - 2 * daw - 1) *
        #    self.depth / ((self.well_width - 1) ** 3)
        #)
        energy[after_well] = (
            self.depth *
            (1 - x ** 2 * (2 * x - 3))
        )

        energy_derivative[after_well] = (
            6 * (daw - 1) * (self.well_width - daw) *
            self.depth / (self.well_width - 1) ** 3
        )

        energy[-in_radius] = 0
        energy_derivative[-in_radius] = 0
        return energy, energy_derivative

    def metric_gradient(self, points_moving, tensors=None, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        differences = (points_moving[..., None, :] - points_moving[..., :]).squeeze()

        if tensors is None:
            dist = numpy.sqrt((differences ** 2).sum(-1))
            grad_pos_dist = differences
        else:
            #Assume that the tensors are the inverse of the metric tensor
            inv_tensors = matrix_3x3_inv(tensors)
            differences_weighted = (differences[..., None] * inv_tensors[:, None, :, :]).sum(-2)
            dist = (differences * differences_weighted).sum(-1)
            grad_pos_dist = differences_weighted * 2

            #grad_tensor_dist = (
            #    differences[:, :, None, :],
            #    differences[:, :, :, None]
            #) * 0

        grad_pos_dist[dist > 0] /= .5 * dist[dist > 0][...,  None] * self.sigma

        f, grad_mat = self.energy_function_derivative(dist)

        f = f.sum() / len(dist)
        grad_pos_dist *= grad_mat[..., None]
        grad = grad_pos_dist.sum(1) / len(dist)

        if tensors is None:
            return f, grad
        else:
            #grad_tensor_dist[dist > 0] /= .5 * dist[dist > 0][...,  None, None] * self.sigma
            #grad_tensor_dist *= grad_mat[..., None, None]
            #grad_tensors = grad_pos_dist.sum(1) / len(dist)

            grad_tensors = numpy.zeros_like(tensors)
            return f, grad, grad_tensors

    def metric_gradient_per_point(self, points_moving, use_transform=False):
        if use_transform and self.transform is not None:
            points_moving = self.transform.transform_points(points_moving)

        differences = (points_moving[..., None, :] - points_moving[..., :])
        dist = numpy.sqrt((differences ** 2).sum(-1))
        dist[dist == 0] = 1e-10
        differences /= dist[..., None] * self.sigma

        f, grad_mat = self.energy_function_derivative(dist)

        differences *= grad_mat[..., None]
        grad = differences.sum(1)

        return f.mean(1), grad / len(grad)


def matrix_3x3_inv(A, result=None):
    if result is None:
        result = numpy.empty_like(A)
    invdet = numpy.empty(len(A))
    invdet[:] = (
        + A[:, 0, 0] * (A[:, 1, 1] * A[:, 2, 2] - A[:, 2, 1] * A[:, 1, 2])
        - A[:, 0, 1] * (A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0])
        + A[:, 0, 2] * (A[:, 1, 0] * A[:, 2, 1] - A[:, 1, 1] * A[:, 2, 0])
    )
    invdet **= -1
    result[:, 0, 0] = (A[:, 1, 1] * A[:, 2, 2] - A[:, 2, 1] * A[:, 1, 2]) * invdet
    result[:, 1, 0] = -(A[:, 0, 1] * A[:, 2, 2] - A[:, 0, 2] * A[:, 2, 1]) * invdet
    result[:, 2, 0] = (A[:, 0, 1] * A[:, 1, 2] - A[:, 0, 2] * A[:, 1, 1]) * invdet
    result[:, 0, 1] = -(A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0]) * invdet
    result[:, 1, 1] = (A[:, 0, 0] * A[:, 2, 2] - A[:, 0, 2] * A[:, 2, 0]) * invdet
    result[:, 2, 1] = -(A[:, 0, 0] * A[:, 1, 2] - A[:, 1, 0] * A[:, 0, 2]) * invdet
    result[:, 0, 2] = (A[:, 1, 0] * A[:, 2, 1] - A[:, 2, 0] * A[:, 1, 1]) * invdet
    result[:, 1, 2] = -(A[:, 0, 0] * A[:, 2, 1] - A[:, 2, 0] * A[:, 0, 1]) * invdet
    result[:, 2, 2] = (A[:, 0, 0] * A[:, 1, 1] - A[:, 1, 0] * A[:, 0, 1]) * invdet
    return result
