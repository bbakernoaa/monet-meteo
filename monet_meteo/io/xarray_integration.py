"""
Xarray integration for atmospheric calculations.

This module provides xarray-compatible wrappers for various atmospheric calculations
including unit conversions and statistical operations.
"""

import numpy as np
import xarray as xr
from typing import Union, Optional, Tuple
from ..units import (
    pressure, temperature, distance, wind_speed, mixing_ratio, concentration
)


def xr_convert_pressure(
    data: xr.DataArray,
    from_unit: str,
    to_unit: str
) -> xr.DataArray:
    """
    Xarray wrapper for pressure unit conversion using Pint.
    
    Parameters
    ----------
    data : xarray.DataArray
        Pressure data to convert
    from_unit : str
        Source unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')
    to_unit : str
        Target unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')
    
    Returns
    -------
    xarray.DataArray
        Converted pressure data
    """
    return pressure(data, from_unit, to_unit)


def xr_convert_temperature(
    data: xr.DataArray,
    from_unit: str,
    to_unit: str
) -> xr.DataArray:
    """
    Xarray wrapper for temperature unit conversion using Pint.
    
    Parameters
    ----------
    data : xarray.DataArray
        Temperature data to convert
    from_unit : str
        Source unit ('K', 'C', 'F')
    to_unit : str
        Target unit ('K', 'C', 'F')
    
    Returns
    -------
    xarray.DataArray
        Converted temperature data
    """
    return temperature(data, from_unit, to_unit)


def xr_calculate_distance(
    lat1: xr.DataArray,
    lon1: xr.DataArray,
    lat2: xr.DataArray,
    lon2: xr.DataArray,
    method: str = 'haversine'
) -> xr.DataArray:
    """
    Calculate distance between two geographic points using haversine formula.
    
    Parameters
    ----------
    lat1, lon1 : xarray.DataArray
        Latitude and longitude of first point in degrees
    lat2, lon2 : xarray.DataArray
        Latitude and longitude of second point in degrees
    method : str, optional
        Method for distance calculation ('haversine' or 'vincenty')
    
    Returns
    -------
    xarray.DataArray
        Distance in meters
    """
    # Haversine formula implementation
    R = 6371000  # Earth radius in meters
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    distance_m = R * c
    
    result = xr.DataArray(
        distance_m,
        coords=lat1.coords,
        dims=lat1.dims,
        name="distance"
    )
    
    # Preserve metadata
    result.attrs = {
        "long_name": "Distance",
        "units": "m",
        "standard_name": "distance"
    }
    
    return result


def xr_convert_vertical_coord(
    data: xr.DataArray,
    from_coord: str,
    to_coord: str,
    surface_pressure: Optional[xr.DataArray] = None,
    reference_pressure: float = 1000.0
) -> xr.DataArray:
    """
    Convert between different vertical coordinate systems.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input data with vertical coordinate
    from_coord : str
        Source coordinate system ('pressure', 'altitude', 'sigma', 'z')
    to_coord : str
        Target coordinate system ('pressure', 'altitude', 'sigma', 'z')
    surface_pressure : xarray.DataArray, optional
        Surface pressure for sigma coordinate conversion
    reference_pressure : float, optional
        Reference pressure for sigma coordinate conversion
    
    Returns
    -------
    xarray.DataArray
        Data with converted vertical coordinate
    """
    # Basic coordinate conversion implementations
    if from_coord.lower() == 'pressure' and to_coord.lower() == 'altitude':
        # Simplified pressure to altitude conversion using barometric formula
        p0 = 101325.0  # Sea level pressure (Pa)
        T0 = 288.15    # Sea level temperature (K)
        L = 0.0065     # Temperature lapse rate (K/m)
        g = 9.80665    # Gravitational acceleration (m/s²)
        M = 0.0289644  # Molar mass of dry air (kg/mol)
        R = 8.31432    # Universal gas constant (J/(mol·K))
        
        altitude = (T0 / L) * (1 - (data / p0)**((R * L) / (M * g)))
        
    elif from_coord.lower() == 'altitude' and to_coord.lower() == 'pressure':
        # Simplified altitude to pressure conversion
        p0 = 101325.0
        T0 = 288.15
        L = 0.0065
        g = 9.80665
        M = 0.0289644
        R = 8.31432
        
        pressure = p0 * (1 - (L * data) / T0)**((M * g) / (R * L))
        
    elif from_coord.lower() == 'sigma' and to_coord.lower() == 'pressure':
        if surface_pressure is None:
            raise ValueError("surface_pressure is required for sigma to pressure conversion")
        pressure = surface_pressure * data
        
    elif from_coord.lower() == 'pressure' and to_coord.lower() == 'sigma':
        if surface_pressure is None:
            raise ValueError("surface_pressure is required for pressure to sigma conversion")
        sigma = data / surface_pressure
        
    else:
        raise NotImplementedError(f"Conversion from {from_coord} to {to_coord} not implemented")
    
    result = xr.DataArray(
        eval(to_coord),
        coords=data.coords,
        dims=data.dims,
        name=data.name
    )
    
    # Preserve metadata
    result.attrs = data.attrs.copy()
    
    return result


# Additional xarray integration utilities
def add_coordinate_metadata(
    ds: xr.Dataset,
    coord_system: str = 'cartesian'
) -> xr.Dataset:
    """
    Add standard metadata to coordinates in a dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    coord_system : str, optional
        Coordinate system ('cartesian', 'geographic', 'pressure')
    
    Returns
    -------
    xarray.Dataset
        Dataset with updated coordinate metadata
    """
    ds_out = ds.copy()
    
    if coord_system.lower() == 'geographic':
        if 'lat' in ds_out.coords:
            ds_out.coords['lat'].attrs.update({
                'standard_name': 'latitude',
                'long_name': 'Latitude',
                'units': 'degrees_north',
                'axis': 'Y'
            })
        if 'lon' in ds_out.coords:
            ds_out.coords['lon'].attrs.update({
                'standard_name': 'longitude',
                'long_name': 'Longitude',
                'units': 'degrees_east',
                'axis': 'X'
            })
    elif coord_system.lower() == 'pressure':
        if 'pressure' in ds_out.coords:
            ds_out.coords['pressure'].attrs.update({
                'standard_name': 'air_pressure',
                'long_name': 'Pressure',
                'units': 'Pa',
                'positive': 'down'
            })
    elif coord_system.lower() == 'cartesian':
        if 'x' in ds_out.coords:
            ds_out.coords['x'].attrs.update({
                'standard_name': 'projection_x_coordinate',
                'long_name': 'X-coordinate in Cartesian system',
                'units': 'm'
            })
        if 'y' in ds_out.coords:
            ds_out.coords['y'].attrs.update({
                'standard_name': 'projection_y_coordinate',
                'long_name': 'Y-coordinate in Cartesian system',
                'units': 'm'
            })
    
    return ds_out


def validate_coordinate_system(
    ds: xr.Dataset,
    required_coords: list
) -> bool:
    """
    Validate that a dataset has the required coordinates.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    required_coords : list
        List of required coordinate names
    
    Returns
    -------
    bool
        True if all required coordinates are present
    """
    for coord in required_coords:
        if coord not in ds.coords and coord not in ds.dims:
            return False
    return True