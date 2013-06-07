"""
The :mod:`registration.metric` module gathers
metrics to quantify the similarity between two objects.
"""

from .basis import *
from .densities import *
from .landmark import *

__all__ = [
    'Metric', 'AdditiveMetric', 'Correlation', 'CorrelationWithTensorFeatures', 'ExactLandmarkL2',
    'ImageMeanSquares', 'VectorImageMeanSquares', 'RosenGradientProjection', 'SquaredDifference', 'Correlation',
]
