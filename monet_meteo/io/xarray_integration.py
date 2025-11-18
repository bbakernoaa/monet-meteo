"""
Xarray integration for atmospheric calculations.

This module provides xarray-compatible wrappers for various atmospheric calculations
including unit conversions, interpolations, and coordinate transformations.
"""

import numpy as np
import xarray as xr
from typing import Union, Optional, Tuple
from ..units import (
    convert_pressure, convert_temperature, convert_distance, 
    convert_wind_speed, convert_mixing_ratio, convert_specific_humidity, 
    convert_concentration
)
from ..interpolation import (
    pressure_to_altitude, altitude_to_pressure, interpolate_vertical, 
    interpolate_horizontal, interpolate_3d, interpolate_temperature_pressure,
    interpolate_wind_components, interpolate_with_dask
)
from ..coordinates import (
    latlon_to_cartesian, cartesian_to_latlon, rotate_wind_components,
    pressure_to_sigma, sigma_to_pressure, calculate_grid_spacing,
    calculate_distance, bearing, destination_point, convert_vertical_coord
)


def xr_convert_pressure(
    data: xr.DataArray,
    from_unit: str,
    to_unit: str
) -> xr.DataArray:
    """
    Xarray wrapper for pressure unit conversion.
    
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
    result = xr.apply_ufunc(
        convert_pressure,
        data,
        from_unit,
        to_unit,
        input_core_dims=[[], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Preserve metadata
    result.name = data.name
    result.attrs = data.attrs.copy()
    result.attrs["units"] = to_unit
    
    # Copy coordinates
    result = result.assign_coords(data.coords)
    
    return result


def xr_convert_temperature(
    data: xr.DataArray,
    from_unit: str,
    to_unit: str
) -> xr.DataArray:
    """
    Xarray wrapper for temperature unit conversion.
    
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
    result = xr.apply_ufunc(
        convert_temperature,
        data,
        from_unit,
        to_unit,
        input_core_dims=[[], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Preserve metadata
    result.name = data.name
    result.attrs = data.attrs.copy()
    result.attrs["units"] = to_unit
    
    # Copy coordinates
    result = result.assign_coords(data.coords)
    
    return result


def xr_pressure_to_altitude(
    pressure: xr.DataArray,
    method: str = 'standard',
    t0: float = 288.15,
    p0: float = 101325.0
) -> xr.DataArray:
    """
    Xarray wrapper for pressure to altitude conversion.
    
    Parameters
    ----------
    pressure : xarray.DataArray
        Atmospheric pressure (Pa)
    method : str, optional
        Method to use ('standard', 'hypsometric', 'barometric')
    t0 : float, optional
        Reference temperature (K), default is 28.15 K
    p0 : float, optional
        Reference pressure (Pa), default is 101325.0 Pa
    
    Returns
    -------
    xarray.DataArray
        Altitude (m)
    """
    result = xr.apply_ufunc(
        pressure_to_altitude,
        pressure,
        method,
        t0,
        p0,
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Preserve metadata
    result.name = "altitude"
    result.attrs = {
        "long_name": "Altitude",
        "units": "m",
        "standard_name": "height"
    }
    
    # Copy coordinates from pressure array
    result = result.assign_coords(pressure.coords)
    
    return result


def xr_altitude_to_pressure(
    altitude: xr.DataArray,
    method: str = 'standard',
    t0: float = 288.15,
    p0: float = 101325.0
) -> xr.DataArray:
    """
    Xarray wrapper for altitude to pressure conversion.
    
    Parameters
    ----------
    altitude : xarray.DataArray
        Altitude (m)
    method : str, optional
        Method to use ('standard', 'hypsometric', 'barometric')
    t0 : float, optional
        Reference temperature (K), default is 28.15 K
    p0 : float, optional
        Reference pressure (Pa), default is 101325.0 Pa
    
    Returns
    -------
    xarray.DataArray
        Atmospheric pressure (Pa)
    """
    result = xr.apply_ufunc(
        altitude_to_pressure,
        altitude,
        method,
        t0,
        p0,
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Preserve metadata
    result.name = "pressure"
    result.attrs = {
        "long_name": "Atmospheric Pressure",
        "units": "Pa",
        "standard_name": "air_pressure"
    }
    
    # Copy coordinates from altitude array
    result = result.assign_coords(altitude.coords)
    
    return result


def xr_interpolate_vertical(
    data: xr.DataArray,
    old_levels: Union[np.ndarray, xr.DataArray],
    new_levels: Union[np.ndarray, xr.DataArray],
    axis: int = -1,
    method: str = 'linear',
    bounds_error: bool = False,
    fill_value: Union[float, str] = np.nan
) -> xr.DataArray:
    """
    Xarray wrapper for vertical interpolation.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input data array to interpolate
    old_levels : numpy.ndarray or xarray.DataArray
        Original coordinate levels (pressure or altitude)
    new_levels : numpy.ndarray or xarray.DataArray
        New coordinate levels for interpolation
    axis : int, optional
        Axis along which to interpolate (default is -1, the last axis)
    method : str, optional
        Interpolation method ('linear', 'cubic', 'log')
    bounds_error : bool, optional
        Whether to raise error for out-of-bounds values
    fill_value : float or str, optional
        Value to use for out-of-bounds values
    
    Returns
    -------
    xarray.DataArray
        Interpolated data array
    """
    # Determine the dimension name for the interpolation axis
    dim_name = data.dims[axis]
    new_dim_name = f"new_{dim_name}"
    
    # Apply the interpolation function element-wise
    result = xr.apply_ufunc(
        interpolate_vertical,
        data,
        old_levels,
        new_levels,
        axis,
        method,
        bounds_error,
        fill_value,
        input_core_dims=[[dim_name], [dim_name], [], [], [], []],
        output_core_dims=[[new_dim_name]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype]
    )
    
    # Preserve metadata
    result.name = data.name
    result.attrs = data.attrs.copy()
    
    # Assign new coordinate values
    result = result.assign_coords({new_dim_name: new_levels})
    
    return result


def xr_interpolate_temperature_pressure(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    new_pressure_levels: Union[np.ndarray, xr.DataArray],
    method: str = 'linear'
) -> xr.DataArray:
    """
    Xarray wrapper for temperature interpolation to new pressure levels.
    
    Parameters
    ----------
    temperature : xarray.DataArray
        Temperature data
    pressure : xarray.DataArray
        Pressure levels corresponding to temperature
    new_pressure_levels : numpy.ndarray or xarray.DataArray
        New pressure levels for interpolation
    method : str, optional
        Interpolation method ('linear', 'log', 'cubic')
    
    Returns
    -------
    xarray.DataArray
        Interpolated temperature at new pressure levels
    """
    # Apply the interpolation function element-wise
    result = xr.apply_ufunc(
        interpolate_temperature_pressure,
        temperature,
        pressure,
        new_pressure_levels,
        method,
        input_core_dims=[["level"], ["level"], ["new_level"], []],
        output_core_dims=[["new_level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[temperature.dtype]
    )
    
    # Preserve metadata
    result.name = temperature.name
    result.attrs = temperature.attrs.copy()
    
    # Assign new coordinate values
    result = result.assign_coords({"new_level": new_pressure_levels})
    
    return result


def xr_interpolate_wind_components(
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    old_levels: Union[np.ndarray, xr.DataArray],
    new_levels: Union[np.ndarray, xr.DataArray],
    method: str = 'linear'
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Xarray wrapper for interpolating wind components to new vertical levels.
    
    Parameters
    ----------
    u_wind : xarray.DataArray
        Eastward wind component
    v_wind : xarray.DataArray
        Northward wind component
    old_levels : numpy.ndarray or xarray.DataArray
        Original coordinate levels
    new_levels : numpy.ndarray or xarray.DataArray
        New coordinate levels for interpolation
    method : str, optional
        Interpolation method ('linear', 'log', 'cubic')
    
    Returns
    -------
    tuple of xarray.DataArray
        Interpolated u and v wind components
    """
    # Apply the interpolation function element-wise for u component
    u_interp = xr.apply_ufunc(
        interpolate_wind_components,
        u_wind,
        v_wind,
        old_levels,
        new_levels,
        method,
        input_core_dims=[["level"], ["level"], ["level"], ["new_level"], []],
        output_core_dims=[["new_level"], ["new_level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[u_wind.dtype, v_wind.dtype]
    )
    
    # The function returns both u and v, so we need to separate them
    # For now, let's call each component separately
    u_result = xr.apply_ufunc(
        interpolate_vertical,
        u_wind,
        old_levels,
        new_levels,
        -1,  # axis
        method,
        False,  # bounds_error
        np.nan,  # fill_value
        input_core_dims=[["level"], ["level"], ["new_level"], [], [], []],
        output_core_dims=[["new_level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[u_wind.dtype]
    )
    
    v_result = xr.apply_ufunc(
        interpolate_vertical,
        v_wind,
        old_levels,
        new_levels,
        -1,  # axis
        method,
        False,  # bounds_error
        np.nan,  # fill_value
        input_core_dims=[["level"], ["level"], ["new_level"], [], [], []],
        output_core_dims=[["new_level"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[v_wind.dtype]
    )
    
    # Preserve metadata for u component
    u_result.name = u_wind.name if u_wind.name else "eastward_wind"
    u_result.attrs = u_wind.attrs.copy()
    u_result = u_result.assign_coords({"new_level": new_levels})
    
    # Preserve metadata for v component
    v_result.name = v_wind.name if v_wind.name else "northward_wind"
    v_result.attrs = v_wind.attrs.copy()
    v_result = v_result.assign_coords({"new_level": new_levels})
    
    return u_result, v_result


def xr_calculate_distance(
    lat1: xr.DataArray,
    lon1: xr.DataArray,
    lat2: xr.DataArray,
    lon2: xr.DataArray,
    method: str = 'haversine'
) -> xr.DataArray:
    """
    Xarray wrapper for calculating distance between two points.
    
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
    result = xr.apply_ufunc(
        calculate_distance,
        lat1, lon1, lat2, lon2, method,
        input_core_dims=[[], [], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Preserve metadata
    result.name = "distance"
    result.attrs = {
        "long_name": "Distance",
        "units": "m",
        "standard_name": "distance"
    }
    
    # Copy coordinates from lat1 array
    result = result.assign_coords(lat1.coords)
    
    return result


def xr_convert_vertical_coord(
    data: xr.DataArray,
    from_coord: str,
    to_coord: str,
    surface_pressure: Optional[xr.DataArray] = None,
    reference_pressure: float = 1000.0
) -> xr.DataArray:
    """
    Xarray wrapper for converting between different vertical coordinate systems.
    
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
    # This function is more complex and may require special handling
    # For now, we'll implement a simplified version
    result = convert_vertical_coord(data, from_coord, to_coord, surface_pressure, reference_pressure)
    
    return result


def xr_interpolate_with_dask(
    data: xr.DataArray,
    coords: Union[np.ndarray, xr.DataArray],
    new_coords: Union[np.ndarray, xr.DataArray],
    method: str = 'linear',
    chunk_size: Optional[int] = None
) -> xr.DataArray:
    """
    Xarray wrapper for interpolation with dask support.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input data with dask arrays
    coords : numpy.ndarray or xarray.DataArray
        Original coordinate values
    new_coords : numpy.ndarray or xarray.DataArray
        New coordinate values for interpolation
    method : str, optional
        Interpolation method
    chunk_size : int, optional
        Size of chunks for dask processing
    
    Returns
    -------
    xarray.DataArray
        Interpolated data with dask support
    """
    # Apply the interpolation function with dask support
    result = interpolate_with_dask(data, coords, new_coords, method, chunk_size)
    
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