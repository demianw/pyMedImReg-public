from .. import optimizer
import numpy
from scipy import optimize

__all__ = [
    'SingleResolutionOptimizerPopulationControl',
    'MultiResolutionOptimizer',
    'MultiResolutionOptimizerPopulationControl'
]


class SingleResolutionOptimizerPopulationControl(optimizer.Optimizer):
    def __init__(self, cost_gradient_function, particle_generated_transform, optimizer_args={}):
        self.bounds = None
        self.cost_gradient_function = cost_gradient_function
        self.transform = particle_generated_transform
        self.optimizer_args = optimizer_args

        if 'pgtol' not in self.optimizer_args:
            self.optimizer_args['pgtol'] = 1e-10
        if 'factr' not in self.optimizer_args:
            self.optimizer_args['factr'] = 1e4
        if 'disp' not in self.optimizer_args:
            self.optimizer_args['disp'] = 1

        if 'energy_tol' not in self.optimizer_args:
            self.optimizer_args['energy_tol'] = 1e-8

        self.non_bfgs_args = [
            'initial_resolution',
            'resolution_update',
            'minimal_resolution',
            'energy_tol',
            'verbose'
        ]

    def optimize(self, points, *args, **kwargs):
        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)
        for non_bfgs_arg in self.non_bfgs_args:
            if non_bfgs_arg in optimizer_args:
                del optimizer_args[non_bfgs_arg]

        if self.bounds is not None:
            optimizer_args['bounds'] = self.bounds

        energy_tol = self.optimizer_args['energy_tol']
        energy_old = numpy.inf
        energy = 0
        self.transforms = []
        self.points = points
        self.actual_points = points.copy()

        self.iteration = 0
        self.active_point_mask = numpy.repeat(True, len(points))
        number_of_particles_changed = False
        while (
            abs(energy_old - energy) > energy_tol or
            number_of_particles_changed
        ):

            self.points_to_work = self.actual_points[self.active_point_mask]
            transform = self.transform(self.points_to_work)

            cgf = self.cost_gradient_function(self.points_to_work, transform)

            if hasattr(self, 'pre_optimize_callback'):
                self.pre_optimize_callback()

            param = cgf.transform.identity.copy()

            r = optimize.fmin_l_bfgs_b(
                cgf.metric_gradient_transform_parameters, param,
                **optimizer_args
            )

            self.actual_points = cgf.transform.transform_points(self.actual_points)

            number_of_particles_changed = False
            if self.active_point_mask.sum() > 1:
                self.active_point_mask, number_of_particles_changed = self.eliminate_points(cgf.metrics[1].metric_gradient, self.actual_points, self.active_point_mask)

            if number_of_particles_changed:
                print "Particles eliminated"
            else:
                energy_old = energy
                energy = r[1]

                self.transforms.append(cgf)

                if hasattr(self, 'post_optimize_callback'):
                    self.pre_optimize_callback()

                print "Energy difference: %0.2e" % abs(energy - energy_old)
            self.iteration += 1

        self.result = self.actual_points

        return self.actual_points, self.active_point_mask

    def eliminate_points(self, metric_gradient, points, mask=None):
        if mask is None:
            mask = numpy.repeat(True, len(points))
        f, _ = metric_gradient(points)
        changed = False
        for i in mask.nonzero()[0]:
            mask[i] = False
            new_f, _ = metric_gradient(points[mask])
            print new_f, f, new_f < f
            if new_f < f:
                f = new_f
                changed = True
            else:
                mask[i] = True

        return mask, changed


class MultiResolutionOptimizer(optimizer.Optimizer):
    def __init__(self, cost_gradient_function, resolution_generated_transform, optimizer_args={}):
        self.bounds = None
        self.cost_gradient_function = cost_gradient_function
        self.resolution_generated_transform = resolution_generated_transform
        self.optimizer_args = optimizer_args

        if 'pgtol' not in self.optimizer_args:
            self.optimizer_args['pgtol'] = 1e-10
        if 'factr' not in self.optimizer_args:
            self.optimizer_args['factr'] = 1e4
        if 'disp' not in self.optimizer_args:
            self.optimizer_args['disp'] = 1

        if 'initial_resolution' not in self.optimizer_args:
            self.optimizer_args['initial_resolution'] = 1.
        if 'resolution_update' not in self.optimizer_args:
            self.optimizer_args['resolution_update'] = 1. / 2.
        if 'minimal_resolution' not in self.optimizer_args:
            self.optimizer_args['minimal_resolution'] = 2. ** (-4.)
        if 'energy_tol' not in self.optimizer_args:
            self.optimizer_args['energy_tol'] = 1e-8

        self.non_bfgs_args = [
            'initial_resolution',
            'resolution_update',
            'minimal_resolution',
            'energy_tol',
            'verbose'
        ]

    def optimize(self, points, *args, **kwargs):
        if 'displacements' in kwargs:
            displacements = kwargs['displacements']
        else:
            displacements = None

        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)
        for non_bfgs_arg in self.non_bfgs_args:
            if non_bfgs_arg in optimizer_args:
                del optimizer_args[non_bfgs_arg]

        if self.bounds is not None:
            optimizer_args['bounds'] = self.bounds

        resolution = self.optimizer_args['initial_resolution']
        resolution_update = self.optimizer_args['resolution_update']
        minimal_resolution = self.optimizer_args['minimal_resolution']
        resolution_changed = True

        energy_tol = self.optimizer_args['energy_tol']
        energy_old = numpy.inf
        energy = 0
        self.transforms = []
        self.points = points
        self.displacements = displacements
        self.actual_points = points.copy()
        if self.displacements is not None:
            self.actual_displacements = displacements.copy()

        self.iteration = 0
        while (
            abs(energy_old - energy) > energy_tol or
            (resolution_changed and (resolution > minimal_resolution))
        ):

            resolution_changed = False

            transform = self.resolution_generated_transform(self.actual_points, resolution)

            if self.displacements is None:
                cgf = self.cost_gradient_function(self.actual_points, transform)
            else:
                cgf = self.cost_gradient_function(self.actual_points, self.actual_displacements, transform)

            if hasattr(self, 'pre_optimize_callback'):
                self.pre_optimize_callback()

            param = cgf.transform.identity.copy()

            r = optimize.fmin_l_bfgs_b(
                cgf.metric_gradient_transform_parameters, param,
                **optimizer_args
            )

            if self.displacements is not None:
                self.actual_displacements = cgf.transform.transform_vectors(self.actual_points, self.actual_displacements)
            self.actual_points = cgf.transform.transform_points(self.actual_points)

            energy_old = energy
            energy = r[1]

            if abs(energy - energy_old) < 1e-6:
                resolution_changed = True
                resolution *= resolution_update

            self.transforms.append(cgf)

            if hasattr(self, 'post_optimize_callback'):
                self.pre_optimize_callback()

            print "Energy difference: %0.2e" % abs(energy - energy_old)
            self.iteration += 1

        self.result = self.actual_points

        return self.actual_points


class MultiResolutionOptimizerPopulationControl(optimizer.Optimizer):
    def __init__(self, cost_gradient_function, resolution_generated_transform, optimizer_args={}):
        self.bounds = None
        self.cost_gradient_function = cost_gradient_function
        self.resolution_generated_transform = resolution_generated_transform
        self.optimizer_args = optimizer_args

        if 'pgtol' not in self.optimizer_args:
            self.optimizer_args['pgtol'] = 1e-10
        if 'factr' not in self.optimizer_args:
            self.optimizer_args['factr'] = 1e4
        if 'disp' not in self.optimizer_args:
            self.optimizer_args['disp'] = 1

        if 'initial_resolution' not in self.optimizer_args:
            self.optimizer_args['initial_resolution'] = 1.
        if 'resolution_update' not in self.optimizer_args:
            self.optimizer_args['resolution_update'] = 1. / 2.
        if 'minimal_resolution' not in self.optimizer_args:
            self.optimizer_args['minimal_resolution'] = 2. ** (-4.)
        if 'energy_tol' not in self.optimizer_args:
            self.optimizer_args['energy_tol'] = 1e-8

        self.non_bfgs_args = [
            'initial_resolution',
            'resolution_update',
            'minimal_resolution',
            'energy_tol',
            'verbose'
        ]

    def optimize(self, points, *args, **kwargs):
        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)
        for non_bfgs_arg in self.non_bfgs_args:
            if non_bfgs_arg in optimizer_args:
                del optimizer_args[non_bfgs_arg]

        if self.bounds is not None:
            optimizer_args['bounds'] = self.bounds

        resolution = self.optimizer_args['initial_resolution']
        resolution_update = self.optimizer_args['resolution_update']
        minimal_resolution = self.optimizer_args['minimal_resolution']
        resolution_changed = True

        energy_tol = self.optimizer_args['energy_tol']
        energy_old = numpy.inf
        energy = 0
        self.energy_record = []
        self.transforms = []
        self.points = points
        self.actual_points = points.copy()

        self.iteration = 0
        self.active_point_mask = numpy.repeat(True, len(points))
        while (
            abs(energy_old - energy) > energy_tol or
            (resolution_changed and (resolution > minimal_resolution))
        ):

            resolution_changed = False

            self.points_to_work = self.actual_points[self.active_point_mask]
            transform = self.resolution_generated_transform(self.points_to_work, resolution)

            cgf = self.cost_gradient_function(self.points_to_work, transform)

            if hasattr(self, 'pre_optimize_callback'):
                self.pre_optimize_callback()

            param = cgf.transform.identity.copy()

            r = optimize.fmin_l_bfgs_b(
                cgf.metric_gradient_transform_parameters, param,
                **optimizer_args
            )

            self.actual_points = cgf.transform.transform_points(self.actual_points)

            changed = False
            if self.active_point_mask.sum() > 1:
                self.active_point_mask, changed = self.eliminate_points(cgf.metrics[1].metric_gradient, self.actual_points, self.active_point_mask)

            if changed:
                print "Particles eliminated"
            else:
                energy_old = energy
                energy = r[1]

                if abs(energy - energy_old) < 1e-6:
                    resolution_changed = True
                    resolution *= resolution_update

                if hasattr(self, 'post_optimize_callback'):
                    self.pre_optimize_callback()

                print "Energy difference: %0.2e" % abs(energy - energy_old)

            self.energy_record.append(energy)
            self.transforms.append(cgf)
            self.iteration += 1

        self.result = self.actual_points

        return self.actual_points, self.active_point_mask

    def eliminate_points(self, metric_gradient, points, mask=None):
        if mask is None:
            mask = numpy.repeat(True, len(points))
        f, _ = metric_gradient(points)
        changed = False
        for i in mask.nonzero()[0]:
            mask[i] = False
            new_f, _ = metric_gradient(points[mask])
            print new_f, f, new_f < f
            if new_f < f:
                f = new_f
                changed = True
            else:
                mask[i] = True

        return mask, changed
