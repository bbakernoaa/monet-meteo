"""
Monet Meteo - A comprehensive meteorological library for atmospheric sciences.

This package provides tools for atmospheric calculations including thermodynamic variables,
derived parameters, dynamic calculations, unit conversions, and statistical operations.
"""

try:
    # Try to get version from setuptools_scm
    from ._version import version as __version__
except ImportError:
    # Fallback to hardcoded version
    __version__ = "0.0.1"

__author__ = "NOAA Air Resources Laboratory"

from . import constants, derived, dynamics, io, models, statistical, thermodynamics, units
from .derived import *
from .dynamics import *
from .models import *
from .statistical import *

# Import main functions for easy access
from .thermodynamics import *
from .units import *
