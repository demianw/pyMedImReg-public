import sys
from itertools import izip
import cPickle
from unittest import skip

from numpy import *
from numpy.random import RandomState
from numpy import testing
from scipy import linalg
from .. import Registration, Correlation, RegistrationError, LieAlgebraLogEucSteepestGradientDescent, VectorImageMeanSquares
#from .. import currents
import registration.model.linear
import registration.model.poly_linear

_multiprocess_can_split_ = True
random_ = RandomState(0)

#from polyaffine_block_registration import *
#from tractographyGP import vtkInterface


def generate_arc_tract(N=10, radius=1):
    if radius <= 0:
        raise ValueError("Radius must be greater than 0")
    theta = linspace(0, pi, N)
    tract = radius * c_[sin(theta), cos(theta), linspace(0, 1e-5, N)]

    return tract


def exp_weight(sigma):
    return lambda x: exp(-(x / sigma) ** 2)


def apply_deform_field_from_matrix(matrix, center, tracts,  weight_function=1.):

    if isscalar(weight_function):
        weight_function = exp_weight(weight_function)

    l = linalg.logm(matrix)
    r = asarray(l[:3, :3])
    v = asarray(l[:3, -1]).squeeze()

    deformed_tracts = []
    for tract in tracts:
        centered_tract = tract - center
        weights = sqrt(((tract - center) ** 2).sum(1))
        displacement = (weights * (dot(r, centered_tract.T).T + v).T).T
        deformed_tract = tract + displacement + center

        deformed_tracts.append(deformed_tract)

    return deformed_tracts


def apply_deform_fields_from_matrices(matrices, centers, tracts,  weight_function=1.):

    if isscalar(weight_function):
        weight_function = exp_weight(weight_function)

    displacements = [zeros_like(tract) for tract in tracts]
    weights = [zeros(len(tract)) for tract in tracts]

    for matrix, center in izip(matrices, centers):

        l = linalg.logm(matrix)
        r = asarray(l[:3, :3])
        v = asarray(l[:3, -1]).squeeze()

        for i, tract in enumerate(tracts):
            centered_tract = tract - center
            weight = sqrt(((tract - center) ** 2).sum(1))
            weights[i] += weight
            displacements[i] += (weight * (dot(r, centered_tract.T).T + v).T).T

    deformed_tracts = [tract.copy() for tract in tracts]

    for displacement, weight in izip(displacements, weights):
        weight[weight == 0] = 1
        displacement /= weight[:, None]
        for i, tract in enumerate(deformed_tracts):
            tract += displacement

    return deformed_tracts, displacements


@skip
def test_translation(noise_sigma=3, n_tracts=10):
    tracts = [generate_arc_tract(radius=r) for r in linspace(10, 14, n_tracts)]

    mean_delta = sum(
        ((sqrt((diff(t, axis=0) ** 2).sum(1))).mean() for t in tracts)
    ) / len(tracts)

    fixed = ascontiguousarray(vstack(tracts))
    moving = fixed + (2, 0, 0)
    metric = [Correlation(fixed, mean_delta * 2), Correlation(fixed, mean_delta)]
    m = registration.model.linear.Translation()
    reg = Registration(metric=metric, model=m)

    reg.register(moving)

    testing.assert_array_almost_equal(fixed, reg.model.transform_points(moving), decimal=4)


@skip
def test_deformable(N=25):
    case = 0
    for noise_sigma in (1, 1.5, 2, 3, 4):
        for _ in xrange(N):
            tracts, deformed_tracts = synth_tracts_complex(10, noise_sigma=noise_sigma)

            prefix = 'case_%04d' % case
            cPickle.dump(tracts, open(prefix + '_fixed_tracts.pck', 'wb'))
            cPickle.dump(deformed_tracts, open(prefix + '_deformed_tracts.pck', 'wb'))

            linearly_aligned_points, affine_models = affine_gp(tracts, deformed_tracts)
            for tract in deformed_tracts:
                for model in affine_models:
                    tract[:] = model.transform_points(tract)

            cPickle.dump(affine_models, open(prefix + '_linear_transforms.pck', 'wb'))

            yield deformable_currents, case, tracts, deformed_tracts, noise_sigma
            yield deformable_gp, case, tracts, deformed_tracts, noise_sigma
            yield deformable_currents, case, tracts, deformed_tracts, noise_sigma, True
            yield deformable_gp, case, tracts, deformed_tracts, noise_sigma, True

            case += 1


def deformable_currents(number, tracts, deformed_tracts, noise_sigma, randomize_direction=False, grid_size=60, current_sigma=4):
    mean_delta = sum(
        ((sqrt((diff(t, axis=0) ** 2).sum(1))).mean() for t in tracts)
    ) / len(tracts)

    if randomize_direction:
        fixed_tracts = []
        for tract in deformed_tracts:
            alpha = random_.rand()
            direction = 1 if (alpha > .5) else -1
            fixed_tracts.append(tract[::direction, :])
        moving_tracts = []
        for tract in tracts:
            alpha = random_.rand()
            direction = 1 if (alpha > .5) else -1
            moving_tracts.append(tract[::direction, :])
    else:
        fixed_tracts = deformed_tracts
        moving_tracts = tracts

    fixed_curve_points = ascontiguousarray(vstack(tracts))
    moving_curve_points = ascontiguousarray(vstack(deformed_tracts))

    all_points = vstack((fixed_curve_points, moving_curve_points))
    bounding_box = c_[[
        all_points.min(0) - 10,
        all_points.max(0) + 10
    ]].T
    max_distance = (bounding_box[:, 1] - bounding_box[:, 0]).max()

    current_buffer = zeros((grid_size, grid_size, grid_size, 3))
    fixed = zeros_like(current_buffer)
    moving = zeros_like(current_buffer)

    image_affine = eye(4)

    image_affine[(0, 1, 2), (0, 1, 2)] = (max_distance / grid_size)
    image_affine[:3, -1] = bounding_box[:, 0]

    fixed = currents.curves_current(fixed_tracts, fixed, image_affine, current_sigma)
    moving = currents.curves_current(moving_tracts, moving, image_affine, current_sigma)

    fixed_points = transpose(any(fixed != 0, axis=-1).nonzero())

    metric = [
        VectorImageMeanSquares(
            fixed, moving=moving,
            fixed_points=fixed_points,
            transform=None,
            gradient_operator=currents.gradient,
            interpolator=currents.map_coordinates,
            smooth=mean_delta * max_distance / grid_size * i
        )
        for i in (2, 1, .5)
    ]

    def transform_target(transform, target):
        return currents.transform_image(transform, target, points=fixed_points)

    lie_opt = LieAlgebraLogEucSteepestGradientDescent(None, registration.model.poly_linear.PolyAffine.exp, registration.model.poly_linear.PolyAffine.log, registration.model.poly_linear.PolyAffine.gradient_log, {'disp': True})
    polyaffines = [
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 1, max_distance / 1),
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 2, max_distance / 2),
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 4, max_distance / 4),
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 8, max_distance / 8)
#        model.PolyAffine(bounding_box, mean_delta, mean_delta),
#        model.PolyAffine(bounding_box, mean_delta / 2, mean_delta / 2),
    ]
    reg = Registration(metric=metric, model=polyaffines, optimizer=lie_opt, transform_target=transform_target)

    try:
        reg.register(moving, iprint=1)
    except RegistrationError:
        pass

    new_moving = moving
    for m in reg.model:
        new_moving = transform_target(m, new_moving)
    models = reg.model

    affine_tracts = [image_affine[:3, :3].dot(t.T).T + image_affine[:3, -1] for t in tracts]
    affine_deformed_tracts = [image_affine[:3, :3].dot(t.T).T + image_affine[:3, -1] for t in deformed_tracts]

    fixed_curve_points = vstack(affine_tracts)
    moving_curve_points = vstack(affine_deformed_tracts)

    prefix = 'currents' + ('rnd' if randomize_direction else '') + ' '
    moving_curve_points = report_registration(
        tracts, deformed_tracts, models, number, moving_curve_points, fixed_curve_points, noise_sigma,
        prefix=prefix)
    cPickle.dump(image_affine, open(prefix.strip() + '_%04d_currents_affine.pck' % number, 'wb'))
    testing.assert_array_almost_equal(fixed_curve_points, moving_curve_points, decimal=4)


def report_registration(tracts, deformed_tracts, models, number, moving_curve_points, fixed_curve_points, noise_sigma, prefix=''):
    for m in models:
        moving_curve_points = m.transform_points(moving_curve_points)

    squared_error = ((fixed_curve_points - moving_curve_points) ** 2).sum(1)
    print "Sigma %04f: SSE: %04f +- %04f" % (noise_sigma, squared_error.mean(), squared_error.std())

    clf()
    for tract in tracts:
        plot(tract[:, 0], tract[:, 1], '-', hold=True, c='r')

    for tract in deformed_tracts:
        plot(tract[:, 0], tract[:, 1], '-', hold=True, c='g')
    axis('equal')
    title(prefix + "SSE: %04f +- %04f case %04d" % (
        squared_error.mean(), squared_error.std(), number)
    )
    draw()
    savefig(prefix.strip() + '_registered_case_%04d_%04d.pdf' % (number, 0))

    for i in xrange(len(models)):
        clf()
        for tract in tracts:
            plot(tract[:, 0], tract[:, 1], '-', hold=True, c='r')

        for tract in deformed_tracts:
            for m in models[:i + 1]:
                tract = m.transform_points(tract)
            plot(tract[:, 0], tract[:, 1], '-', hold=True, c='g')
        axis('equal')
        title(prefix + "SSE: %04f +- %04f case %04d " % (
            squared_error.mean(), squared_error.std(), number
        ) + str(models[i].__class__.__name__))
        draw()
        savefig(prefix.strip() + '_registered_case_%04d_%04d.pdf' % (number, i + 1))

    cPickle.dump(models, open(prefix.strip() + '%04d_nonlinear.pck' % number, 'wb'))

    squared_error = ((fixed_curve_points - moving_curve_points) ** 2).sum(1)
    print prefix + "SSE: %04f +- %04f" % (squared_error.mean(), squared_error.std())
    return moving_curve_points


def affine_gp(tracts, deformed_tracts):
    mean_delta = sum(
        ((sqrt((diff(t, axis=0) ** 2).sum(1))).mean() for t in tracts)
    ) / len(tracts)

    fixed = ascontiguousarray(vstack(tracts))
    moving = ascontiguousarray(vstack(deformed_tracts))

    center = fixed.mean(0)

    metric = [
        Correlation(fixed, mean_delta * 2),
        Correlation(fixed, mean_delta),
        Correlation(fixed, mean_delta / 2.),
    ]
    affines = [
        registration.model.linear.Translation(),
        registration.model.linear.Rigid(center),
        registration.model.linear.Affine(center)
    ]

    reg_affine = Registration(metric=metric, model=affines)
    reg_affine.register(moving)
    new_moving = moving

    for m in reg_affine.model:
        new_moving = m.transform_points(new_moving)

    return new_moving, reg_affine.model


def deformable_gp(number, tracts, deformed_tracts, noise_sigma, randomize_direction=False):
    mean_delta = sum(
        ((sqrt((diff(t, axis=0) ** 2).sum(1))).mean() for t in tracts)
    ) / len(tracts)

    if randomize_direction:
        fixed_tracts = []
        for tract in tracts:
            alpha = random_.rand()
            direction = 1 if (alpha > .5) else -1
            fixed_tracts.append(tract[::direction, :])
        moving_tracts = []
        for tract in deformed_tracts:
            alpha = random_.rand()
            direction = 1 if (alpha > .5) else -1
            moving_tracts.append(tract[::direction, :])
    else:
        fixed_tracts = tracts
        moving_tracts = deformed_tracts

    fixed = ascontiguousarray(vstack(fixed_tracts))
    moving = ascontiguousarray(vstack(deformed_tracts))

    all_points = vstack((fixed, moving))
    bounding_box = c_[[
        all_points.min(0) - 5,
        all_points.max(0) + 5
    ]].T
    max_distance = (bounding_box[:, 1] - bounding_box[:, 0]).max()

    metric = [
        Correlation(fixed, mean_delta * 2),
        Correlation(fixed, mean_delta),
        Correlation(fixed, mean_delta / 2.),
    ]

    lie_opt = LieAlgebraLogEucSteepestGradientDescent(None, registration.model.poly_linear.PolyAffine.exp, registration.model.poly_linear.PolyAffine.log, registration.model.poly_linear.PolyAffine.gradient_log, {'disp': True})
    polyaffines = [
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 1, max_distance / 1),
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 2, max_distance / 2),
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 4, max_distance / 4),
        registration.model.poly_linear.PolyAffine(bounding_box, max_distance / 8, max_distance / 8)
#        model.PolyAffine(bounding_box, mean_delta, mean_delta),
#        model.PolyAffine(bounding_box, mean_delta / 2, mean_delta / 2),
    ]
    reg = Registration(metric=metric, model=polyaffines, optimizer=lie_opt)

    try:
        reg.register(ascontiguousarray(moving), iprint=1)
    except RegistrationError:
        pass

    models = reg.model
    moving = report_registration(tracts, deformed_tracts, models, number, moving, fixed, noise_sigma,
                                 prefix='gp' + ('rnd' if randomize_direction else '') + ' ')
    testing.assert_array_almost_equal(fixed, moving, decimal=4)


def synth_tracts(n_tracts, mixing_sigma=2, noise_centers=10, noise_sigma=1):
    tracts = [generate_arc_tract(radius=r) for r in linspace(10, 14, n_tracts)]

    deformed_tracts, displacement, tract_centers = deform_tracts(tracts, n_tracts, noise_sigma, noise_centers, mixing_sigma)
    return tracts, deformed_tracts


def synth_tracts_complex(n_tracts, mixing_sigma=2, noise_centers=10, noise_sigma=1):
    tracts = [generate_arc_tract(radius=r) for r in linspace(10, 14, n_tracts)]

    small_tracts_centers = tracts[-1]

    rotation = registration.model.linear.Rotation()
    rotation.parameter[2] = -pi / 2
    angle_delta = -pi / len(small_tracts_centers)

    for center in small_tracts_centers:
        for r in (2, 2.2, 2.4):
            tracts.append(
                rotation.transform_points(generate_arc_tract(N=10, radius=r))
                + center
            )

        rotation.parameter[2] += angle_delta

    deformed_tracts, displacement, tract_centers = deform_tracts(tracts, n_tracts, noise_sigma, noise_centers, mixing_sigma)
    return tracts, deformed_tracts


def deform_tracts(tracts, n_tracts, noise_sigma, noise_centers, mixing_sigma):
    '''
    We assume that the tracts are ordered
    '''
    center_tract = tracts[len(tracts) / 2]

    for i in xrange(n_tracts):
        deform_matrix = eye(4)
        deformed_tracts = tracts
        matrices = []
        centers = center_tract[::max(1, len(center_tract) / noise_centers)]
        for center in centers:
            displacement = random_.randn(2) / noise_sigma
            deform_matrix[:-2, -1] = displacement
            matrices.append(deform_matrix.copy())

        deformed_tracts, displacement = apply_deform_fields_from_matrices(
            matrices,
            centers,
            deformed_tracts,
            mixing_sigma
        )

        return deformed_tracts, displacement, centers


if __name__ == 'main':
    main()


def main():
    tract = generate_arc_tract()

    N = 25

    if len(sys.argv) > 1:
        noise_sigma = float(sys.argv[1])
    else:
        noise_sigma = 3
    if len(sys.argv) > 2:
        n_tracts = int(sys.argv[2])
    else:
        n_tracts = 10

    if len(sys.argv) > 3:
        noise_centers = int(sys.argv[3])
    else:
        noise_centers = 10

    if len(sys.argv) > 4:
        mixing_sigma = float(sys.argv[4])
    else:
        mixing_sigma = 2

    print "N_tracts %d sigma: %f" % (n_tracts, noise_sigma)

    tract_name = 'tract_%04d.vtk'

    tracts = [generate_arc_tract(radius=r) for r in linspace(10, 14, N)]
    vtkInterface.writeLinesToVtkPolyData('template_tract.vtk', tracts)

    center_tract = tracts[len(tracts) / 2]

    figure(1)
    clf()
    for tract in tracts:
        plot(tract[:, 0], tract[:, 1], '-', hold=True)

    plot(center_tract[:, 0], center_tract[:, 1], 'r', lw=4, hold=True)
    axis('equal')
    draw()

    #Deformation
    for i in xrange(n_tracts):
        deform_matrix = eye(4)
        deformed_tracts = tracts
        matrices = []
        centers = center_tract[::max(1, len(center_tract) / noise_centers)]
        for center in centers:
            displacement = random_.randn(2) / noise_sigma
            deform_matrix[:-2, -1] = displacement
            matrices.append(deform_matrix.copy())

        deformed_tracts, displacement = apply_deform_fields_from_matrices(
            matrices,
            centers,
            deformed_tracts,
            mixing_sigma
        )

        vtkInterface.writeLinesToVtkPolyData(tract_name % i, deformed_tracts)

        figure(2)
        clf()
        for tract in deformed_tracts:
            plot(tract[:, 0], tract[:, 1], '-', hold=True)

        scatter(centers[:, 0], centers[:, 1], hold=True)
        axis('equal')
        draw()
        show()
