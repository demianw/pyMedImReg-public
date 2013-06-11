from collections import Iterable
from itertools import izip

import numpy
from .optimizer import ScipyLBFGS


__all__ = ['Registration', 'RegistrationError']


def assert_iterable(obj):
    if not isinstance(obj, Iterable):
        return (obj,)
    else:
        return obj


class Registration(object):
    def __init__(
            self, model=None, metric=None, regularizer=None, optimizer=None,
            transform_target=None, paired=False, regularizer_constraint=None
    ):
        metric = assert_iterable(metric)
        model = assert_iterable(model)
        regularizer = assert_iterable(regularizer)

        if len(regularizer) != len(model):
            if len(regularizer) == 1:
                regularizer = (regularizer[0],) * len(model)
            else:
                raise ValueError(
                    "The number of regularizers must be 1 "
                    "or equal to the number of models"
                )

        self.regularizer_constraint = regularizer_constraint
        self.paired = paired

        if self.paired and (len(model) != len(metric)):
            raise ValueError("Paired mode requires the same number of models and metrics")

        self.model = model
        self.metric = metric
        self.regularizer = regularizer

        if optimizer is None:
            self.optimizer = ScipyLBFGS(None)
        else:
            self.optimizer = optimizer

        if transform_target is None:
            self.transform_target = lambda transform, target: transform.transform_points(target)
        else:
            self.transform_target = transform_target

    def register(self, points, vectors=None, tensors=None, return_all=False, **kwargs):

        if not self.paired:
            vectors, tensors, points = self.register_paired(points, vectors, tensors)
        else:
            self.register_sequential(vectors, tensors, points)

    def register_sequential(self, vectors, tensors, points):
        for step_num, model, regularizer, metric in izip(
                xrange(len(self.model)), self.model,
                self.regularizer, self.metric
        ):
            if not hasattr(model, 'initial'):
                initial = model.identity
            else:
                initial = model.initial

            self.optimizer.initial = initial
            self.optimizer.bounds = model.bounds
            self.result = []
            initial = self.registration_inner_loop(
                points, vectors, tensors,
                model, regularizer, metric,
                step_num, step_num, initial
            )

            model.registration_optimizer_result = self.optimizer.result
            points = model.transform_points(points)
            if vectors is not None:
                vectors = model.transform_vectors(points, vectors)
            if tensors is not None:
                tensors = model.transform_tensors(points, tensors)

    def register_paired(self, points, vectors, tensors):
        for model_num, model, regularizer in izip(
                xrange(len(self.model)), self.model, self.regularizer
        ):
            if not hasattr(model, 'initial'):
                initial = model.identity
            else:
                initial = model.initial

            self.optimizer.initial = initial
            self.optimizer.bounds = model.bounds
            self.result = []
            for metric_num, metric in enumerate(self.metric):
                initial = self.registration_inner_loop(
                    points, vectors, tensors,
                    model, regularizer, metric,
                    model_num, metric_num, initial
                )

            model.registration_optimizer_result = self.optimizer.result
            points = model.transform_points(points)
            if vectors is not None:
                vectors = model.transform_vectors(points, vectors)
            if tensors is not None:
                tensors = model.transform_tensors(points, tensors)
        return vectors, tensors, points

    def registration_inner_loop(self, points, vectors, tensors, model, regularizer, metric, model_num, metric_num, initial):
        metric.points_moving = numpy.ascontiguousarray(points)
        if vectors is not None:
            metric.vectors = numpy.ascontiguousarray(vectors)
        else:
            vectors = None
        if tensors is not None:
            metric.tensors = numpy.ascontiguousarray(tensors)
        else:
            tensors = None
        metric.transform = model
        print "\nStart optimization: model %d (%s) metric %d (%s) \n\t Number of parameters: %d, initial value %g initial norm difference %g" % (
            model_num, str(type(model)), metric_num, str(type(metric)), len(model.identity),
            metric.metric_gradient_transform_parameters(initial)[0],
            numpy.sqrt(((model.identity - initial) ** 2).sum())
        )

        if regularizer is not None:
            if self.regularizer_constraint is None:
                def cost_gradient(*args, **kwargs):
                    f, g = metric.metric_gradient_transform_parameters(*args, **kwargs)
                    fr, gr = regularizer.metric_gradient_transform_parameters(*args, **kwargs)
                    return f + fr, g + gr
                self.optimizer.cost_gradient_function = cost_gradient
            else:
                self.optimizer.clear_constraints()
                self.optimizer.add_constraint(regularizer.metric_gradient_transform_parameters, self.regularizer_constraint)
        else:
            self.optimizer.cost_gradient_function = metric.metric_gradient_transform_parameters
        self.optimizer.initial = initial
        self.optimizer.optimize()

        initial = self.optimizer.result
        self.result.append(self.optimizer.result)
        print "\nEnd optimization: ", self.optimizer.optimized_cost
        return initial


class RegistrationError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)
