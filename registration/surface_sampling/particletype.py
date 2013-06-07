import numpy
from .. import metric

#from .._metrics import tensor_patches_inner_product


class LineParticlesFitVectorImage(metric.Metric):
    def __init__(self, edges, displacements, vector_function=None, transform=None):
        self.edges = edges
        self.displacements = displacements
        self.vector_function = vector_function
        self.transform = transform

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform
        self.metric.transform = self._transform

    @property
    def points_moving(self):
        return self._moving

    @points_moving.setter
    def points_moving(self, points_moving):
        self._moving = points_moving
        self.metric.points_moving = self._moving

    def metric_gradient(self, edges=None, displacements=None):
        if (edges is None) or (displacements is None):
            edges = self.edges
            displacements = self.displacements

        vectors = self.vector_function(edges)
        transformed_displacements = self.transform(edges, displacements)

        metric = ((vectors - transformed_displacements) ** 2).sum()

        gradient = numpy.empty((len(edges), 3))

        return metric, gradient

    def metric_gradient_tranform_particle_samples(self, points, displacements, t):
        transformed_points = self.transform_points(points)
        transformed_point_jacobian = self.transform.jacobian(points)

        d1, d2, d3 = displacements.T

        t1 = transformed_points[:, 0]
        t2 = transformed_point_jacobian[:, 0, 0]
        t4 = transformed_point_jacobian[:, 0, 1]
        t6 = transformed_point_jacobian[:, 0, 2]
        t8 = t2 * d1 + t4 * d2 + t6 * d3
        t9 = numpy.abs(t8)
        t10 = t9 * t9
        t11 = transformed_point_jacobian[:, 1, 0]
        t13 = transformed_point_jacobian[:, 1, 1]
        t15 = transformed_point_jacobian[:, 1, 2]
        t17 = t11 * d1 + t13 * d2 + t15 * d3
        t18 = numpy.abs(t17)
        t19 = t18 * t18
        t20 = transformed_point_jacobian[:, 2, 0]
        t22 = transformed_point_jacobian[:, 2, 1]
        t24 = transformed_point_jacobian[:, 2, 2]
        t26 = t20 * d1 + t22 * d2 + t24 * d3
        t27 = numpy.abs(t26)
        t28 = t27 * t27
        t29 = t10 + t19 + t28
        t30 = numpy.sqrt(t29)
        t39 = 1 / t30
        t41 = transformed_points[:, 1]
        t51 = transformed_points[:, 2]
        points = numpy.r_[
            (t1 * t30 + t * t2 * d1 + t * t4 * d2 + t * t6 * d3) * t39,
            (t41 * t30 + t * t11 * d1 + t * t13 * d2 + t * t15 * d3) * t39,
            (t51 * t30 + t * t20 * d1 + t * t22 * d2 + t * t24 * d3) * t39
        ].T
        t61, _ = self.metric.metric_gradient_per_point(points)
        t62 = 1 / t29
        t63 = t8 * t8
        t65 = numpy.abs(t62 * t63)
        t66 = t17 * t17
        t68 = numpy.abs(t62 * t66)
        t69 = t26 * t26
        t71 = numpy.abs(t62 * t69)
        t73 = numpy.sqrt(t65 + t68 + t71)

        return(t61 * t73)

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None:
            return ValueError('transform and points_moving must be set for this')

        self.transform.parameter = parameter
        edges = numpy.ascontiguousarray(self.transform.transform_points(self.edges))
        #displacements = self.displacements.copy()
        displacements = numpy.ascontiguousarray(
            self.transform_vectors(
                self.edges, self.displacements
            )
        )

        f, grad = self.metric_gradient(edges, displacements)

        jacobian = self.transform.jacobian(edges)
        grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

        return f, grad

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def transform_vectors(self, edges, displacements):
        displacement_norm = numpy.sqrt((displacements ** 2).sum(1))
        new_displacements = self.transform.transform_vectors(edges, displacements)
        new_displacement_norm = numpy.sqrt((new_displacements ** 2).sum(1))

        new_displacements *= (displacement_norm / new_displacement_norm)[:, None]

        return new_displacements


class LineParticles(metric.Metric):
    def __init__(self, edges, displacements, samples, metric=None, transform=None):
        self.edges = edges
        self.displacements = displacements
        self.samples = int(samples)
        self.sample_limits = int(numpy.ceil(float(samples + 1) / 2.))
        self.metric = metric
        self.transform = transform

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform):
        self._transform = transform
        self.metric.transform = self._transform

    @property
    def points_moving(self):
        return self._moving

    @points_moving.setter
    def points_moving(self, points_moving):
        self._moving = points_moving
        self.metric.points_moving = self._moving

    def metric_gradient(self, edges=None, displacements=None):
        if (edges is None) or (displacements is None):
            edges = self.edges
            displacements = self.displacements

        N = len(edges)
        points = numpy.vstack((
            edges + displacements * i / float(self.samples)
            for i in xrange(self.samples + 1)
#            for i in xrange(-self.sample_limits, self.sample_limits + 1)
        ))

        values, gradients = self.metric.metric_gradient_per_point(
            points
        )
        values = -numpy.exp(-values)
        gradients *= numpy.exp(-values)[:, None]

        displacements_lengths = numpy.sqrt((displacements ** 2).sum(1))[:, None]

        value = values.reshape(N, self.samples + 1).sum(-1) * displacements_lengths

        gradient = gradients.reshape(self.samples + 1, N, 3).sum(0) * displacements_lengths

        return value.mean(), gradient / len(value)

    def metric_gradient_tranform_particle_samples(self, points, displacements, t):
        transformed_points = self.transform_points(points)
        transformed_point_jacobian = self.transform.jacobian(points)

        d1, d2, d3 = displacements.T

        t1 = transformed_points[:, 0]
        t2 = transformed_point_jacobian[:, 0, 0]
        t4 = transformed_point_jacobian[:, 0, 1]
        t6 = transformed_point_jacobian[:, 0, 2]
        t8 = t2 * d1 + t4 * d2 + t6 * d3
        t9 = numpy.abs(t8)
        t10 = t9 * t9
        t11 = transformed_point_jacobian[:, 1, 0]
        t13 = transformed_point_jacobian[:, 1, 1]
        t15 = transformed_point_jacobian[:, 1, 2]
        t17 = t11 * d1 + t13 * d2 + t15 * d3
        t18 = numpy.abs(t17)
        t19 = t18 * t18
        t20 = transformed_point_jacobian[:, 2, 0]
        t22 = transformed_point_jacobian[:, 2, 1]
        t24 = transformed_point_jacobian[:, 2, 2]
        t26 = t20 * d1 + t22 * d2 + t24 * d3
        t27 = numpy.abs(t26)
        t28 = t27 * t27
        t29 = t10 + t19 + t28
        t30 = numpy.sqrt(t29)
        t39 = 1 / t30
        t41 = transformed_points[:, 1]
        t51 = transformed_points[:, 2]
        points = numpy.r_[
            (t1 * t30 + t * t2 * d1 + t * t4 * d2 + t * t6 * d3) * t39,
            (t41 * t30 + t * t11 * d1 + t * t13 * d2 + t * t15 * d3) * t39,
            (t51 * t30 + t * t20 * d1 + t * t22 * d2 + t * t24 * d3) * t39
        ].T
        t61, _ = self.metric.metric_gradient_per_point(points)
        t62 = 1 / t29
        t63 = t8 * t8
        t65 = numpy.abs(t62 * t63)
        t66 = t17 * t17
        t68 = numpy.abs(t62 * t66)
        t69 = t26 * t26
        t71 = numpy.abs(t62 * t69)
        t73 = numpy.sqrt(t65 + t68 + t71)

        return(t61 * t73)

    def metric_jacobian_transform_parameters(self, parameter):
        if self.transform is None:
            return ValueError('transform and points_moving must be set for this')

        self.transform.parameter = parameter
        edges = numpy.ascontiguousarray(self.transform.transform_points(self.edges))
        #displacements = self.displacements.copy()
        displacements = numpy.ascontiguousarray(
            self.transform_vectors(
                self.edges, self.displacements
            )
        )

        f, grad = self.metric_gradient(edges, displacements)

        jacobian = self.transform.jacobian(edges)
        grad = numpy.asfortranarray((jacobian * grad[:, None, :]).sum(-1))

        return f, grad

    def metric_gradient_transform_parameters(self, parameter):
        f, grad = self.metric_jacobian_transform_parameters(parameter)
        return f, numpy.asfortranarray(grad.mean(0))

    def transform_vectors(self, edges, displacements):
        displacement_norm = numpy.sqrt((displacements ** 2).sum(1))
        new_displacements = self.transform.transform_vectors(edges, displacements)
        new_displacement_norm = numpy.sqrt((new_displacements ** 2).sum(1))

        new_displacements *= (displacement_norm / new_displacement_norm)[:, None]

        return new_displacements


