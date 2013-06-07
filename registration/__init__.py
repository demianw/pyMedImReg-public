"""
Registration module for Python
==================================

Registration is a Python module to serve as an implementation of various
medical image registration algorithms.

"""
__version__ = '0.1'

from .model import *
from .metric import *
from .regularizer import *
from .registration import *
from .optimizer import *
from .image_tools import *

import model
import metric
import registration
import regularizer
import optimizer
import image_tools
import util
