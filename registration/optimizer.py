from warnings import warn

import numpy
from scipy import optimize
try:
    import nlopt
except:
    pass

from . import lbfgsb
#from .metric import RosenGradientProjection as MetricRosenGradientProjection

__all__ = ['Optimizer', 'ScipyLBFGS', 'SteepestGradientDescent', 'LieAlgebraLogEucSteepestGradientDescent', 'ModifiedLBFGS', 'NLOpt']


Inf = numpy.Inf


class Optimizer:

    @property
    def initial():
        return None

    @property
    def result():
        return None

    def optimize():
        raise NotImplementedError()


class ScipyLBFGS(Optimizer):
    def __init__(self, cost_gradient_function=None, optimizer_args={}):
        self.bounds = None
        self.cost_gradient_function = cost_gradient_function
        self.optimizer_args = optimizer_args
        if 'pgtol' not in self.optimizer_args:
            self.optimizer_args['pgtol'] = 1e-10
        if 'factr' not in self.optimizer_args:
            self.optimizer_args['factr'] = 1e4

    def optimize(self, *args, **kwargs):

        if self.initial is None:
            raise ValueError("The property initial must be set before optimizing")

        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)

        if self.bounds is not None:
            optimizer_args['bounds'] = self.bounds

        result, value, info = optimize.fmin_l_bfgs_b(
            self.cost_gradient_function, self.initial, *args, **optimizer_args
        )

        self.info = info
        self.optimized_cost = value

        self.optimized_ok = info['warnflag'] < 2

        if info['warnflag'] == 2:
            self.error_msg = 'Error in the optimization algorithm: %s' % info['task']

        self.result = result.copy()

        return self.optimized_ok


class ModifiedLBFGS(Optimizer):
    def __init__(self, cost_gradient_function=None, optimizer_args={}):
        self.bounds = None
        self.cost_gradient_function = cost_gradient_function
        self.optimizer_args = optimizer_args
        if 'pgtol' not in self.optimizer_args:
            self.optimizer_args['pgtol'] = 1e-10
        if 'factr' not in self.optimizer_args:
            self.optimizer_args['factr'] = 1e4

    def optimize(self, *args, **kwargs):

        if self.initial is None:
            raise ValueError("The property initial must be set before optimizing")

        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)

        if self.bounds is not None:
            optimizer_args['bounds'] = self.bounds

        result, value, info = lbfgsb.fmin_l_bfgs_b(
            self.cost_gradient_function, self.initial, *args, **optimizer_args
        )

        self.info = info
        self.optimized_cost = value

        self.optimized_ok = info['warnflag'] < 2

        if info['warnflag'] == 2:
            self.error_msg = 'Error in the optimization algorithm: %s' % info['task']

        self.result = result

        return self.optimized_ok


#class RosenGradientProjection(Optimizer):
#    '''
#    Rosen Gradient Projection method for Metrics
#    '''
#    def __init__(
#        self,
#        metric_to_optimize, restriction,
#        restrict_to_tangent=True, verbose=False, callback=None,
#        optimizer_args={},
#        proj_optimizer_args={'pgtol': 1e-3, 'factr': 1e7, 'disp': 0, 'maxiter': 5}
#    ):
#        self.bounds = None
#        self.metric = metric_to_optimize
#        self.restriction = restriction
#        self.initial = None
#        self.verbose = verbose
#        self.callback = callback
#        self.restrict_to_tangent = restrict_to_tangent
#
#        if self.restrict_to_tangent:
#            self.gradient_projected_metric = MetricRosenGradientProjection(
#                self.metric,
#                self.restriction,
#                transform=self.metric.transform,
#                moving=self.metric.moving
#            )
#
#            self.metric_to_optimize = self.gradient_projected_metric
#        else:
#            self.metric_to_optimize = self.metric
#
#        self.restriction.transform = self.metric.transform
#        self.restriction.moving = self.metric.moving
#
#        self.optimizer_args = optimizer_args
#        if 'pgtol' not in self.optimizer_args:
#            self.optimizer_args['pgtol'] = 1e-10
#        if 'factr' not in self.optimizer_args:
#            self.optimizer_args['factr'] = 1e4
#
#        self.proj_optimizer_args = {'pgtol': 1e-3, 'factr': 1e7, 'disp': 0, 'maxiter': 5}
#        self.proj_optimizer_args.update(proj_optimizer_args)
#        print self.proj_optimizer_args
#
#    def generate_callback(self):
#        def callback(parameter, **args):
#            if self.verbose:
#                print "Performing projection", self.restriction.metric_gradient_transform_parameters(parameter)[0]
#
#            #self.restriction.transform.parameter[:] = parameter
#            #moving_transformed = self.restriction.transform.transform_points(
#            #    self.restriction.moving
#            #)
#            #self.restriction.start_one_dimensional_search(moving_transformed)
#            res = lbfgsb.fmin_l_bfgs_b(
#                self.restriction.metric_gradient_transform_parameters,
#                parameter,
#                **self.proj_optimizer_args
#            )
#
#            parameter[:] = res[0]
#            #self.restriction.stop_one_dimensional_search()
#            if self.verbose:
#                print '\t', res[1], res[2]['nit']
#
#            if self.callback is not None:
#                self.callback(parameter)
#
#        return callback
#
#    def optimize(self, *args, **kwargs):
#
#        if self.initial is None:
#            self.initial = self.metric.transform.identity
#
#        optimizer_args = {}
#        optimizer_args.update(self.optimizer_args)
#        optimizer_args['callback'] = self.generate_callback()
#
#        if self.verbose:
#            optimizer_args['disp'] = 1
#
#        if self.bounds is not None:
#            optimizer_args['bounds'] = self.bounds
#
#        result, value, info = lbfgsb.fmin_l_bfgs_b(
#            self.metric_to_optimize.metric_gradient_transform_parameters, self.initial,
#            *args, **optimizer_args
#        )
#
#        self.info = info
#        self.optimized_cost = value
#
#        self.optimized_ok = info['warnflag'] < 2
#
#        if info['warnflag'] == 2:
#            self.error_msg = 'Error in the optimization algorithm: %s' % info['task']
#
#        self.result = result
#
#        return self.optimized_ok


class SteepestGradientDescent(Optimizer):
    def __init__(self, cost_gradient_function, optimizer_args={}):
        self.cost_gradient_function = cost_gradient_function
        self.optimizer_args = {
            'ftol': 1e-10,
            'gtol': 1e-10,
            'maxfun': 1e6,
            'initial_step_size': 1000,
            'step_damping_factor': 10,
            'min_step_size': 1e-9,
            'disp': False
        }

    def optimize(self, *args, **kwargs):

        if self.initial is None:
            raise ValueError("The property initial must be set before optimizing")

        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)
        disp = self.optimizer_args['disp']

        step_size = float(self.optimizer_args['initial_step_size'])
        parameter = numpy.asanyarray(self.initial).copy()

        f, grad = self.cost_gradient_function(parameter)
        grad_norm = numpy.sqrt((grad ** 2).sum())
        normalized_gradient = grad / grad_norm

        n_func_eval = 0
        old_f = numpy.inf
        if disp:
            print "n: %d f: %04f" % (n_func_eval, f)

        while (
            numpy.abs(f - old_f) > self.optimizer_args['ftol'] and
            grad_norm > self.optimizer_args['gtol'] and
            n_func_eval < self.optimizer_args['maxfun'] and
            step_size > self.optimizer_args['min_step_size']
        ):
            new_parameter = parameter - step_size * normalized_gradient
            new_f, new_grad = self.cost_gradient_function(new_parameter)

            if new_f < f:
                old_f = f
                f = new_f
                grad = new_grad
                grad_norm = numpy.sqrt((grad ** 2).sum())
                normalized_gradient = grad / grad_norm
                parameter = new_parameter
                n_func_eval += 1
                if disp:
                    print "n: %d f: %04f" % (n_func_eval, f)
            else:
                step_size /= self.optimizer_args['step_damping_factor']
                if disp:
                    print "\tDamping: %f" % step_size

        self.result = parameter
        self.optimized_cost = f
        self.last_gradient = grad

        return True


class LieAlgebraLogEucSteepestGradientDescent(Optimizer):
    def __init__(self, cost_gradient_function, exp, log, gradient_log, optimizer_args={}):
        self.cost_gradient_function = cost_gradient_function
        self.exp = exp
        self.log = log
        self.gradient_log = gradient_log
        self.optimizer_args = {
            'ftol': 1e-10,
            'gtol': 1e-10,
            'maxfun': 1e6,
            'initial_step_size': 1000,
            'step_damping_factor': 10,
            'min_step_size': 1e-9,
            'disp': True
        }

    def optimize(self, *args, **kwargs):

        if self.initial is None:
            raise ValueError("The property initial must be set before optimizing")

        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)
        disp = self.optimizer_args['disp']

        step_size = float(self.optimizer_args['initial_step_size'])
        parameter = numpy.asanyarray(self.initial).copy()

        f, grad = self.cost_gradient_function(parameter)
        grad_norm = numpy.sqrt((grad ** 2).sum())
        normalized_grad = grad / grad_norm

        log_parameter = self.log(parameter)
        log_normalized_gradient = self.gradient_log(normalized_grad)

        n_func_eval = 0
        old_f = numpy.inf
        if disp:
            print "n: %d f: %04f" % (n_func_eval, f)

        while (
            numpy.abs(f - old_f) > self.optimizer_args['ftol'] and
            grad_norm > self.optimizer_args['gtol'] and
            n_func_eval < self.optimizer_args['maxfun'] and
            step_size > self.optimizer_args['min_step_size']
        ):
            new_parameter = self.exp(log_parameter - step_size * log_normalized_gradient)
            valid_step = ~numpy.any(numpy.isnan(new_parameter))

            if valid_step:
                new_f, new_grad = self.cost_gradient_function(new_parameter)

            if valid_step and (new_f < f):
                old_f = f
                f = new_f
                grad = new_grad
                grad_norm = numpy.sqrt((grad ** 2).sum())
                normalized_grad = grad / grad_norm
                parameter = new_parameter

                log_parameter = self.log(parameter)
                log_normalized_gradient = self.gradient_log(normalized_grad)

                n_func_eval += 1
                if disp:
                    print "n: %d f: %04f" % (n_func_eval, f)
            else:
                step_size /= self.optimizer_args['step_damping_factor']
                if disp:
                    print "\tDamping: %g" % step_size

        self.result = parameter
        self.optimized_cost = f
        self.last_gradient = grad

        return True


class RegularizedSteepestGradientDescent(Optimizer):
    def __init__(self, cost_gradient_function, gradient_regularizer, optimizer_args={}):
        self.cost_gradient_function = cost_gradient_function
        self.gradient_regularizer = gradient_regularizer
        self.optimizer_args = {
            'ftol': 1e-10,
            'gtol': 1e-10,
            'maxfun': 1e6,
            'initial_step_size': 1000,
            'step_damping_factor': 10,
            'min_step_size': 1e-9,
            'disp': False
        }

    def optimize(self, *args, **kwargs):

        if self.initial is None:
            raise ValueError("The property initial must be set before optimizing")

        optimizer_args = {}
        optimizer_args.update(self.optimizer_args)
        disp = self.optimizer_args['disp']

        step_size = float(self.optimizer_args['initial_step_size'])
        parameter = numpy.asanyarray(self.initial).copy()

        f, grad = self.cost_gradient_function(parameter)
        grad = self.gradient_regularizer(grad)
        grad_norm = numpy.sqrt((grad ** 2).sum())

        n_func_eval = 0
        old_f = numpy.inf
        if disp:
            print "n: %d f: %04f" % (n_func_eval, f)

        while (
            numpy.abs(f - old_f) > self.optimizer_args['ftol'] and
            grad_norm > self.optimizer_args['gtol'] and
            n_func_eval < self.optimizer_args['maxfun'] and
            step_size > self.optimizer_args['min_step_size']
        ):
            new_parameter = parameter - step_size * grad / grad_norm
            new_f, new_grad = self.cost_gradient_function(new_parameter)

            if new_f < f:
                old_f = f
                f = new_f
                grad = self.gradient_regularizer(new_grad)
                grad_norm = numpy.sqrt((grad ** 2).sum())
                parameter = new_parameter
                n_func_eval += 1
                if disp:
                    print "n: %d f: %04f" % (n_func_eval, f)
            else:
                step_size /= self.optimizer_args['step_damping_factor']
                if disp:
                    print "\tDamping: %f" % step_size

        self.result = parameter
        self.optimized_cost = f
        self.last_gradient = grad

        return True


class NLOpt(Optimizer):
    def __init__(
        self,
        optimize_algorithm='LD_MMA',
        cost_gradient_function=None,
        equality_constraints=None,
        inequality_constraints=None,
        vector_inequality_constraints=None,
        local_optimize_algorithm='LD_MMA',
        callback=None,
        optimizer_args={'ftol_abs': 1e-6, 'ftol_rel': 1e-6},
        local_optimizer_args={'ftol_abs': 1e-6, 'ftol_rel': 1e-6},
        verbose=True
    ):
        self.cost_gradient_function = cost_gradient_function
        self.optimize_algorithm = optimize_algorithm
        self.local_optimize_algorithm = local_optimize_algorithm

        self.optimizer_args = optimizer_args
        self.local_optimizer_args = local_optimizer_args

        self.callback = callback
        self.equality_constraints = equality_constraints
        self.inequality_constraints = inequality_constraints
        self.vector_inequality_constraints = vector_inequality_constraints
        self.bounds = None

        self.verbose = verbose

    def _prepare_nlopt(self):
        self._cost_function = self._generate_cost_function(
            self.cost_gradient_function, callback=self.callback, method=False
        )
        self.opt = nlopt.opt(
            getattr(nlopt, self.optimize_algorithm),
            self.parameter_number
        )

        for k, v in self.optimizer_args.iteritems():
            try:
                if k == 'maxeval':
                    v = int(v)
                else:
                    v = float(v)
                getattr(self.opt, 'set_' + k)(v)
                if self.verbose:
                    print "\tOptimizer attribute %s set to %g" % (k, v)
            except AttributeError:
                warning = 'Attribute %s not in optimizer' % k
                warn(warning)
        if self.bounds is not None:
            self.opt.set_lower_bounds(self.bounds[:, 0])
            self.opt.set_upper_bounds(self.bounds[:, 1])
        self.opt.set_min_objective(self._cost_function)

        if (
            (self.equality_constraints is not None and len(self.equality_constraints) > 0) or
            (self.inequality_constraints is not None and len(self.inequality_constraints) > 0) or
            (self.vector_inequality_constraints is not None and len(self.vector_inequality_constraints) > 0)
        ):
            self.local_optimize_algorithm = self.local_optimize_algorithm
            self.local_optimizer_args = self.local_optimizer_args
            self.local_opt = nlopt.opt(
                getattr(nlopt, self.local_optimize_algorithm),
                self.parameter_number
            )
            for k, v in self.local_optimizer_args.iteritems():
                getattr(self.local_opt, 'set_' + k)(v)

            self.opt.set_local_optimizer(self.local_opt)

            if self.equality_constraints is not None:

                self.equality_constraint_functions = []
                for i, eq_constraint in enumerate(self.equality_constraints):
                    fun = eq_constraint[0]
                    tol = eq_constraint[1]
                    if len(eq_constraint) > 2:
                        inner_callback = eq_constraint[2]
                    else:
                        inner_callback = None

                    _eq_constraint = self._generate_cost_function(
                        fun, callback=inner_callback, method=False
                    )

                    self.equality_constraint_functions.append((_eq_constraint, tol))
                    self.opt.add_equality_constraint(_eq_constraint, tol)
                    if self.verbose:
                        print "Added equality constraint %d tolerance %f" % (i, tol)

            if self.inequality_constraints is not None:
                self.inequality_constraint_functions = []
                for i, ineq_constraint in enumerate(self.inequality_constraints):
                    fun = ineq_constraint[0]
                    tol = ineq_constraint[1]
                    if len(ineq_constraint) > 2:
                        inner_callback = ineq_constraint[2]
                    else:
                        inner_callback = None

                    _ineq_constraint = self._generate_cost_function(
                        fun, callback=inner_callback, method=False
                    )

                    self.inequality_constraint_functions.append((_ineq_constraint, tol))
                    self.opt.add_inequality_constraint(_ineq_constraint, tol)
                    if self.verbose:
                        print "Added inequality constraint %d tolerance %f" % (i, tol)

            if self.vector_inequality_constraints is not None:
                self.vector_inequality_constraint_functions = []
                for i, vec_ineq_constraint in enumerate(self.vector_inequality_constraints):
                    fun = vec_ineq_constraint[0]
                    tol = vec_ineq_constraint[1]
                    if len(vec_ineq_constraint) > 2:
                        inner_callback = vec_ineq_constraint[2]
                    else:
                        inner_callback = None

                    _ineq_constraint = self._generate_vector_cost_function(
                        fun, callback=inner_callback, method=False
                    )

                    self.vector_inequality_constraint_functions.append((_ineq_constraint, tol))
                    self.opt.add_inequality_mconstraint(_ineq_constraint, tol)
                    if self.verbose:
                        print "Added vector inequality constraint %d, dimensions %d tolerance %f" % (i, len(tol), tol.mean())

        else:
            self.local_opt = None

    def _generate_cost_function(self, cost_gradient_function, callback=None, method=True):
        if method:
            if callback is None:
                def method_cost_function_no_callback(self, x, grad, *args):
                    if len(grad) == 0:
                        f, _ = cost_gradient_function(x)
                    else:
                        f, grad[:] = cost_gradient_function(x)

                    return f

                cost_function = method_cost_function_no_callback
            else:
                def method_cost_function_callback(self, x, grad, *args):
                    if len(grad) == 0:
                        f, _ = cost_gradient_function(x)
                    else:
                        f, grad[:] = cost_gradient_function(x)

                    callback(f, grad)

                    return f

                cost_function = method_cost_function_callback
        else:
            if callback is None:
                def cost_function_no_callback(x, grad, *args):
                    if len(grad) == 0:
                        f, _ = cost_gradient_function(x)
                    else:
                        f, grad[:] = cost_gradient_function(x)

                    return f

                cost_function = cost_function_no_callback
            else:
                def cost_function_callback(x, grad, *args):
                    if len(grad) == 0:
                        f, _ = cost_gradient_function(x)
                    else:
                        f, grad[:] = cost_gradient_function(x)

                    callback(f, grad)

                    return f

                cost_function = cost_function_callback
        return cost_function

    def _generate_vector_cost_function(self, cost_gradient_function, callback=None, method=True):
        if method:
            if callback is None:
                def method_cost_function_no_callback(self, result, x, grad):
                    if len(grad) == 0:
                        result[:], _ = cost_gradient_function(x)
                    else:
                        result[:], grad[:] = cost_gradient_function(x)

                    return result[:]

                cost_function = method_cost_function_no_callback
            else:
                def method_cost_function_callback(self, result, x, grad):
                    if len(grad) == 0:
                        result[:], _ = cost_gradient_function(x)
                    else:
                        result[:], grad[:] = cost_gradient_function(x)

                    callback(result, x, grad)

                    return result

                cost_function = method_cost_function_callback
        else:
            if callback is None:
                def cost_function_no_callback(result, x, grad):
                    if len(grad) == 0:
                        result[:], _ = cost_gradient_function(x)
                    else:
                        result[:], grad[:] = cost_gradient_function(x)

                    return result

                cost_function = cost_function_no_callback
            else:
                def cost_function_callback(result, x, grad, *args):
                    if len(grad) == 0:
                        result[:], _ = cost_gradient_function(x)
                    else:
                        result[:], grad[:] = cost_gradient_function(x)

                    callback(result, x, grad)

                    return result

                cost_function = cost_function_callback
        return cost_function

    def optimize(self, *args, **kwargs):
        self.parameter_number = len(self.initial)
        self._prepare_nlopt()

        for k, v in kwargs.iteritems():
            getattr(self.opt, 'set_' + k)(v)
            if self.local_opt is not None:
                getattr(self.local_opt, 'set_' + k)(v)

        self.result = self.opt.optimize(self.initial)
        self.optimized_cost = self.opt.last_optimum_value()
        self.optimize_result = self.opt.last_optimize_result()

        if self.verbose:
            if self.optimize_result == nlopt.SUCCESS:
                msg = "Succesful termination"
            elif self.optimize_result == nlopt.STOPVAL_REACHED:
                msg = "Stop value reached"
            elif self.optimize_result == nlopt.FTOL_REACHED:
                msg = "Function change tolerance (ftol) reached"
            elif self.optimize_result == nlopt.XTOL_REACHED:
                msg = "Parameter changed tolerance (xtol) reached"
            elif self.optimize_result == nlopt.MAXEVAL_REACHED:
                msg = "Maximum number of evaluations reached"
            elif self.optimize_result == nlopt.MAXTIME_REACHED:
                msg = "Maximum time reached"

            print "Succesful termination: %s value: %g" % (msg, self.optimized_cost)

        return self.result
