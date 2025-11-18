"""
Monet Meteo - A comprehensive meteorological library for atmospheric sciences.

This package provides tools for atmospheric calculations including thermodynamic variables,
derived parameters, dynamic calculations, unit conversions, interpolations, and coordinate
transformations.
"""

__version__ = "0.0.1"
__author__ = "NOAA Air Resources Laboratory"

from . import thermodynamics
from . import derived
from . import dynamics
from . import statistical
from . import constants
from . import models
from . import units
from . import interpolation
from . import coordinates
from . import io

# Import main functions for easy access
from .thermodynamics import *
from .derived import *
from .dynamics import *
from .models import *
from .units import *
from .interpolation import *
from .coordinates import *