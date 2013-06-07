import registration as reg

import numpy

points_fixed = numpy.random.randn(100, 3)

transform = reg.Affine()

transform.parameter[:] = transform.identity
transform.parameter[0] = numpy.pi / 8.

points_moving = transform.transform_points(points_fixed)

metric = reg.ExactLandmarkL2(points_moving, points_fixed, transform=None)

opt = reg.ModifiedLBFGS(optimizer_args={'factr': 1, 'pgtol': 1e-10, 'disp': 1})

registration = reg.Registration(
    model=transform,
    metric=metric,
    optimizer=opt
)

registration.register(points_moving)

points_moving_transformed = transform.transform_points(points_moving)

print "Initial Maximum MSE:", ((points_fixed - points_moving) ** 2).max()
print "Registered Maximum MSE:", ((points_fixed - points_moving_transformed) ** 2).max()
