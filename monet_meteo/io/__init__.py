"""
IO utilities for atmospheric data.

This module provides functions for reading, writing, and processing atmospheric data
with xarray integration.
"""

# Import xarray integration functions
try:
    from .xarray_integration import (
        xr_convert_pressure,
        xr_convert_temperature,
        xr_pressure_to_altitude,
        xr_altitude_to_pressure,
        xr_interpolate_vertical,
        xr_interpolate_temperature_pressure,
        xr_interpolate_wind_components,
        xr_calculate_distance,
        xr_convert_vertical_coord,
        xr_interpolate_with_dask,
        add_coordinate_metadata,
        validate_coordinate_system
    )
except ImportError:
    # If xarray_integration is not available, skip importing
    pass

# Import other IO-related functions as needed