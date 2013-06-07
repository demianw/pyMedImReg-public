import numpy

from . import model

__all__ = [
    'Regularizer', 'BendingEnergyKernelBasedTransform',
]


class Regularizer(object):
    def __init__(self):
        raise NotImplementedError()

    def metric_gradient_transform_parameters(self, parameter):
        raise NotImplementedError()


class BendingEnergyKernelBasedTransform(Regularizer):
    def __init__(self, transform, weight):
        if hasattr(transform, 'transform'):  # Is it a nested transform
            self.transform = transform.transform
        else:
            self.transform = transform
        self.start = 0
        self.end = None
        if isinstance(self.transform, model.KernelBasedTransformNoBasis):
            pass
        elif isinstance(self.transform, model.KernelBasedTransformNoBasisMovingAnchors):
            self.end = len(self.transform.control_points) * 3
        elif isinstance(self.transform, model.KernelBasedTransform):
            self.start = 12
        elif not isinstance(self.transform, model.KernelBasedTransform):
            raise ValueError("Transform must be an instance of KernelBasedTransform not %s" % (str(type(transform))))

        self.weight = weight

    def metric_gradient_transform_parameters(self, parameter):
        d = self.transform.control_points.shape[1]

        bending_energy = 0
        bending_energy_jacobian = numpy.zeros(len(parameter))
        for i in xrange(d):
            slice_ = slice(self.start, self.end, 3)
            ps = parameter[slice_]
            bending_energy_jacobian[slice_] = 2 * numpy.dot(ps.T, self.transform.K)
            bending_energy += numpy.dot(ps.T, numpy.dot(self.transform.K, ps))

        return bending_energy, bending_energy_jacobian
