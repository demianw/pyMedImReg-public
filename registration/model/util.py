import numpy

__all__ = ['grid_from_bounding_box_and_resolution']

def grid_from_bounding_box_and_resolution(bounding_box, resolution):
    if numpy.isscalar(resolution):
        resolution = numpy.repeat(resolution, len(bounding_box))
    widths = bounding_box[:, 1] - bounding_box[:, 0]
    resolution = numpy.minimum(resolution, widths)

    x, y, z = numpy.mgrid[
        bounding_box[0, 0]:bounding_box[0, 1] + resolution[0]: resolution[0],
        bounding_box[1, 0]:bounding_box[1, 1] + resolution[1]: resolution[1],
        bounding_box[2, 0]:bounding_box[2, 1] + resolution[2]: resolution[2]
    ]

    return numpy.c_[x.ravel(), y.ravel(), z.ravel()].astype(float)
