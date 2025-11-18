"""
Derived parameters module for atmospheric science.

This module provides functions for calculating derived meteorological parameters including:
- Heat index
- Wind chill
- Lifting condensation level
- Wet bulb temperature
- Dew point temperature
"""

# Import all derived parameter functions
from .derived_calculations import (
    heat_index,
    wind_chill,
    lifting_condensation_level,
    wet_bulb_temperature,
    dewpoint_temperature,
    saturation_vapor_pressure,
    actual_vapor_pressure
)

__all__ = [
    'heat_index',
    'wind_chill',
    'lifting_condensation_level',
    'wet_bulb_temperature',
    'dewpoint_temperature',
    'saturation_vapor_pressure',
    'actual_vapor_pressure'
]