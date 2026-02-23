"""
IO utilities for atmospheric data.

This module provides functions for reading, writing, and processing atmospheric data
with xarray integration.
"""

# Import xarray integration functions
try:
    from .xarray_integration import (
        xr_convert_pressure as xr_convert_pressure,  # noqa: F401
        xr_convert_temperature as xr_convert_temperature,  # noqa: F401
        xr_calculate_distance as xr_calculate_distance,  # noqa: F401
        xr_convert_vertical_coord as xr_convert_vertical_coord,  # noqa: F401
        add_coordinate_metadata as add_coordinate_metadata,  # noqa: F401
        validate_coordinate_system as validate_coordinate_system,  # noqa: F401
    )
except ImportError:
    # If xarray_integration is not available, skip importing
    pass

# Import other IO-related functions as needed
