import numpy
from scipy import ndimage

__all__ = ['transform_image']


def transform_image(transform, image, points=None):
    if points is None:
        points = numpy.transpose((~numpy.isnan(image)).nonzero())

    new_image = numpy.zeros_like(image)
    transformed_points = transform.transform_points(points)
    new_image[tuple(points.T)] = ndimage.map_coordinates(image, transformed_points.T, order=1)

    return new_image
