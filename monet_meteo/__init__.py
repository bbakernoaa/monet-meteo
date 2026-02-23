"""
Monet Meteo - A comprehensive meteorological library for atmospheric sciences.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.1"

__author__ = "NOAA Air Resources Laboratory"

from . import thermodynamics as thermodynamics  # noqa: F401
from . import derived as derived  # noqa: F401
from . import dynamics as dynamics  # noqa: F401
from . import statistical as statistical  # noqa: F401
from . import constants as constants  # noqa: F401
from . import models as models  # noqa: F401
from . import units as units  # noqa: F401
from . import io as io  # noqa: F401
from . import micrometeo as micrometeo  # noqa: F401
from . import fv3 as fv3  # noqa: F401
from . import airquality as airquality  # noqa: F401

# Import main functions for easy access
# We use explicit imports to satisfy mypy and resolve conflicts
from .units import (  # noqa: F401
    pressure,
    temperature,
    distance,
    wind_speed,
    concentration,
)
from .thermodynamics import *  # noqa: F403
from .derived import *  # noqa: F403
from .dynamics import *  # noqa: F403
from .models import *  # noqa: F403

# For micrometeo and statistical, we explicitly import names to avoid conflicts
# and choose the most appropriate one for the top level.
from .micrometeo import (  # noqa: F401
    friction_velocity,
    obukhov_length,
    monin_obukhov_stability,
    richardson_bulk,
    richardson_gradient,
    bowen_ratio,
    psi_h,
    psi_m,
    aerodynamic_resistance,
    sensible_heat_flux,
    latent_heat_flux,
    surface_energy_balance,
    turbulence_intensity,
    sun_angles,
)

from .statistical import (  # noqa: F401
    bulk_richardson_number,
    monin_obukhov_length,
    stability_parameter,
    psi_momentum,
    psi_heat,
    momentum_flux,
    turbulence_kinetic_energy,
    standard_deviation,
    correlation_coefficient,
    covariance,
    obukhov_stability_parameter,
    friction_velocity_from_wind,
    atmospheric_boundary_layer_height,
    xarray_bulk_richardson_number,
    xarray_monin_obukhov_length,
    xarray_surface_energy_balance,
    xarray_turbulent_fluxes_from_similarity,
)

from .fv3 import *  # noqa: F403
from .airquality import *  # noqa: F403
