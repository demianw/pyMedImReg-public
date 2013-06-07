import numpy


__all__ = ['JoshiLDDMM3DKernel', 'InverseExp3DKernel', 'ThinPlateSpline3DKernel']


class JoshiLDDMM3DKernel(object):
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.sigma_sqrt = numpy.sqrt(sigma)
        self.norm_constant = 2. / numpy.sqrt(sigma * 2 * numpy.pi)

    def __call__(self, radiuses2):
        return self.norm_constant * numpy.exp(- self.sigma_sqrt * numpy.sqrt(radiuses2))

    @property
    def support_radius(self):
        return numpy.inf

    def derivative(self, radiuses2):
        #radiuses[abs(radiuses) < 1e-20] = 1e-20
        return (
            -self.sigma_sqrt * numpy.maximum(radiuses2, 1e-20) ** -.5 *
            self.norm_constant * numpy.exp(- self.sigma_sqrt * numpy.sqrt(radiuses2))
        )


class InverseExp3DKernel(object):
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.sigma2 = sigma ** 2
        self.support = self.sigma * 4
        self.second_derivative_is_multiple = -.5 / self.sigma2

    def __call__(self, radiuses2):
        #radiuses[abs(radiuses) < 1e-20] = 1e-20
        return numpy.exp(- .5 * radiuses2 / self.sigma2)

    @property
    def support_radius(self):
        return numpy.inf

    def derivative(self, radiuses2):
        #radiuses[abs(radiuses) < 1e-20] = 1e-20
        return - .5 / self.sigma2 * numpy.exp(- .5 * radiuses2 / self.sigma2)

    def second_derivative(self, radiuses2):
        return (.5 / self.sigma2) ** 2 * numpy.exp(- .5 * radiuses2 / self.sigma2)


class ThinPlateSpline3DKernel(object):
    def __init__(self, sigma=1):
        self.sigma = sigma
        self.sigma2 = sigma ** 2

    def __call__(self, radiuses2):
        #radiuses[abs(radiuses) < 1e-20] = 1e-20
        return numpy.sqrt(radiuses2 / self.sigma2)

    @property
    def support_radius(self):
        return numpy.inf

    def derivative(self, radiuses2):
        #radiuses[abs(radiuses) < 1e-20] = 1e-20
        return .5 / self.sigma2 * (numpy.maximum(radiuses2, 1e-20) / self.sigma2) ** (-.5)
