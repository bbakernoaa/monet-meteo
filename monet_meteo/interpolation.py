"""
Interpolation methods for atmospheric data.

This module provides functions for interpolating atmospheric data in both vertical
and horizontal dimensions, including pressure/altitude conversions, temperature/
pressure interpolations, and coordinate transformations with xarray/dask support.
"""

import numpy as np
from typing import Union, Optional, Tuple
import xarray as xr
from scipy import interpolate
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
import warnings

# Import constants
from .constants import R_d, g


def pressure_to_altitude(
    pressure: Union[float, np.ndarray, xr.DataArray],
    method: str = 'standard',
    t0: float = 288.15,
    p0: float = 101325.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert pressure to altitude using various methods.
    
    Parameters
    ----------
    pressure : float, numpy.ndarray, or xarray.DataArray
        Atmospheric pressure (Pa)
    method : str, optional
        Method to use ('standard', 'hypsometric', 'barometric')
    t0 : float, optional
        Reference temperature (K), default is 288.15 K
    p0 : float, optional
        Reference pressure (Pa), default is 101325.0 Pa
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Altitude (m)
    """
    if method.lower() == 'standard':
        # Standard atmosphere approximation
        # Altitude = (1 - (p/p0)^(R*g/T0)) * (T0/L) where L is lapse rate
        # Using simplified form: h = (1 - (p/p0)^(R_d/g)) * (T0/0.0065)
        altitude = (1 - (pressure / p0) ** (R_d / g)) * (t0 / 0.065)
    elif method.lower() == 'hypsometric':
        # Hypsometric equation: h = (R_d/g) * T_avg * ln(p0/p)
        # Assuming constant average temperature
        altitude = (R_d / g) * t0 * np.log(p0 / pressure)
    elif method.lower() == 'barometric':
        # Barometric formula: h = (R_d*T0/g) * ln(p0/p)
        altitude = (R_d * t0 / g) * np.log(p0 / pressure)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'hypsometric', or 'barometric'")
    
    return altitude


def altitude_to_pressure(
    altitude: Union[float, np.ndarray, xr.DataArray],
    method: str = 'standard',
    t0: float = 288.15,
    p0: float = 101325.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert altitude to pressure using various methods.
    
    Parameters
    ----------
    altitude : float, numpy.ndarray, or xarray.DataArray
        Altitude (m)
    method : str, optional
        Method to use ('standard', 'hypsometric', 'barometric')
    t0 : float, optional
        Reference temperature (K), default is 28.15 K
    p0 : float, optional
        Reference pressure (Pa), default is 101325.0 Pa
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Atmospheric pressure (Pa)
    """
    if method.lower() == 'standard':
        # Standard atmosphere approximation
        pressure = p0 * (1 - (0.0065 * altitude) / t0) ** (g / (R_d * 0.0065))
    elif method.lower() == 'hypsometric':
        # Hypsometric equation: p = p0 * exp(-g*h/(R_d*T_avg))
        pressure = p0 * np.exp(-g * altitude / (R_d * t0))
    elif method.lower() == 'barometric':
        # Barometric formula: p = p0 * exp(-g*h/(R_d*T0))
        pressure = p0 * np.exp(-g * altitude / (R_d * t0))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'standard', 'hypsometric', or 'barometric'")
    
    return pressure


def interpolate_vertical(
    data: Union[np.ndarray, xr.DataArray],
    old_levels: Union[np.ndarray, xr.DataArray],
    new_levels: Union[np.ndarray, xr.DataArray],
    axis: int = -1,
    method: str = 'linear',
    bounds_error: bool = False,
    fill_value: Union[float, str] = np.nan
) -> Union[np.ndarray, xr.DataArray]:
    """
    Interpolate atmospheric data vertically between pressure/altitude levels.
    
    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
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
    numpy.ndarray or xarray.DataArray
        Interpolated data array
    """
    if isinstance(data, xr.DataArray):
        # Handle xarray DataArray
        if method.lower() == 'log':
            # Logarithmic interpolation (useful for pressure coordinates)
            old_levels_log = np.log(old_levels)
            new_levels_log = np.log(new_levels)
            
            # Create interpolator function
            interpolator = RegularGridInterpolator(
                (old_levels_log,), 
                data.values, 
                method='linear', 
                bounds_error=bounds_error, 
                fill_value=fill_value
            )
            
            # Interpolate
            points = np.expand_dims(new_levels_log, axis=1)
            result_values = interpolator(points)
            
            # Create new DataArray with interpolated values
            result = xr.DataArray(
                result_values,
                dims=data.dims,
                coords={**{data.dims[axis]: new_levels}},
                attrs=data.attrs
            )
            return result
        else:
            # Regular interpolation
            interpolator = RegularGridInterpolator(
                (old_levels,), 
                data.values, 
                method=method, 
                bounds_error=bounds_error, 
                fill_value=fill_value
            )
            
            # Prepare points for interpolation
            points = np.expand_dims(new_levels, axis=1)
            result_values = interpolator(points)
            
            # Create new DataArray with interpolated values
            result = xr.DataArray(
                result_values,
                dims=data.dims,
                coords={**{data.dims[axis]: new_levels}},
                attrs=data.attrs
            )
            return result
    else:
        # Handle numpy arrays
        if method.lower() == 'log':
            # Logarithmic interpolation
            old_levels_log = np.log(old_levels)
            new_levels_log = np.log(new_levels)
            
            # Create interpolation function
            f = interp1d(
                old_levels_log, 
                data, 
                axis=axis, 
                kind='linear', 
                bounds_error=bounds_error, 
                fill_value=fill_value
            )
            
            # Interpolate
            result = f(new_levels_log)
        else:
            # Regular interpolation
            f = interp1d(
                old_levels, 
                data, 
                axis=axis, 
                kind=method, 
                bounds_error=bounds_error, 
                fill_value=fill_value
            )
            
            # Interpolate
            result = f(new_levels)
        
        return result


def interpolate_horizontal(
    data: Union[np.ndarray, xr.DataArray],
    old_coords: Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]],
    new_coords: Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]],
    method: str = 'linear',
    bounds_error: bool = False,
    fill_value: Union[float, str] = np.nan
) -> Union[np.ndarray, xr.DataArray]:
    """
    Interpolate atmospheric data horizontally between grid points.
    
    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Input 2D data array to interpolate
    old_coords : tuple of numpy.ndarray or xarray.DataArray
        Original coordinate arrays (x, y)
    new_coords : tuple of numpy.ndarray or xarray.DataArray
        New coordinate arrays (x, y) for interpolation
    method : str, optional
        Interpolation method ('linear', 'cubic', 'nearest')
    bounds_error : bool, optional
        Whether to raise error for out-of-bounds values
    fill_value : float or str, optional
        Value to use for out-of-bounds values
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Interpolated data array
    """
    if isinstance(data, xr.DataArray):
        # Handle xarray DataArray
        interpolator = RegularGridInterpolator(
            old_coords,
            data.values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value
        )
        
        # Create meshgrid for new coordinates
        new_x, new_y = np.meshgrid(new_coords[0], new_coords[1], indexing='ij')
        points = np.column_stack((new_x.ravel(), new_y.ravel()))
        
        # Interpolate
        result_values = interpolator(points)
        result_values = result_values.reshape(new_x.shape)
        
        # Create new DataArray with interpolated values
        result = xr.DataArray(
            result_values,
            dims=data.dims,
            coords={data.dims[0]: new_coords[0], data.dims[1]: new_coords[1]},
            attrs=data.attrs
        )
        return result
    else:
        # Handle numpy arrays
        if method.lower() == 'linear':
            interpolator = interpolate.LinearNDInterpolator(
                np.column_stack((old_coords[0].ravel(), old_coords[1].ravel())),
                data.ravel(),
                fill_value=fill_value
            )
            new_x, new_y = np.meshgrid(new_coords[0], new_coords[1], indexing='ij')
            points = np.column_stack((new_x.ravel(), new_y.ravel()))
            result = interpolator(points).reshape(new_x.shape)
        else:
            # For other methods, use griddata
            interpolator = interpolate.griddata(
                np.column_stack((old_coords[0].ravel(), old_coords[1].ravel())),
                data.ravel(),
                (new_coords[0][:, None], new_coords[1][None, :]),
                method=method,
                fill_value=fill_value
            )
            result = interpolator
        
        return result


def interpolate_3d(
    data: Union[np.ndarray, xr.DataArray],
    old_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    new_coords: Tuple[np.ndarray, np.ndarray, np.ndarray],
    method: str = 'linear',
    bounds_error: bool = False,
    fill_value: Union[float, str] = np.nan
) -> Union[np.ndarray, xr.DataArray]:
    """
    Interpolate atmospheric data in 3D space (e.g., lat, lon, pressure).
    
    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Input 3D data array to interpolate
    old_coords : tuple of numpy.ndarray
        Original coordinate arrays (x, y, z)
    new_coords : tuple of numpy.ndarray
        New coordinate arrays (x, y, z) for interpolation
    method : str, optional
        Interpolation method ('linear', 'cubic')
    bounds_error : bool, optional
        Whether to raise error for out-of-bounds values
    fill_value : float or str, optional
        Value to use for out-of-bounds values
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Interpolated 3D data array
    """
    if isinstance(data, xr.DataArray):
        # Handle xarray DataArray
        interpolator = RegularGridInterpolator(
            old_coords,
            data.values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value
        )
        
        # Create meshgrid for new coordinates
        new_x, new_y, new_z = np.meshgrid(
            new_coords[0], new_coords[1], new_coords[2], 
            indexing='ij'
        )
        points = np.column_stack((
            new_x.ravel(), 
            new_y.ravel(), 
            new_z.ravel()
        ))
        
        # Interpolate
        result_values = interpolator(points)
        result_values = result_values.reshape(new_x.shape)
        
        # Create new DataArray with interpolated values
        result = xr.DataArray(
            result_values,
            dims=data.dims,
            coords={
                data.dims[0]: new_coords[0], 
                data.dims[1]: new_coords[1], 
                data.dims[2]: new_coords[2]
            },
            attrs=data.attrs
        )
        return result
    else:
        # Handle numpy arrays
        interpolator = RegularGridInterpolator(
            old_coords,
            data,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value
        )
        
        # Create meshgrid for new coordinates
        new_x, new_y, new_z = np.meshgrid(
            new_coords[0], new_coords[1], new_coords[2], 
            indexing='ij'
        )
        points = np.column_stack((
            new_x.ravel(), 
            new_y.ravel(), 
            new_z.ravel()
        ))
        
        # Interpolate
        result = interpolator(points)
        result = result.reshape(new_x.shape)
        
        return result


def interpolate_temperature_pressure(
    temperature: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    new_pressure_levels: Union[np.ndarray, xr.DataArray],
    method: str = 'linear'
) -> Union[np.ndarray, xr.DataArray]:
    """
    Interpolate temperature to new pressure levels.
    
    Parameters
    ----------
    temperature : numpy.ndarray or xarray.DataArray
        Temperature data
    pressure : numpy.ndarray or xarray.DataArray
        Pressure levels corresponding to temperature
    new_pressure_levels : numpy.ndarray or xarray.DataArray
        New pressure levels for interpolation
    method : str, optional
        Interpolation method ('linear', 'log', 'cubic')
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Interpolated temperature at new pressure levels
    """
    return interpolate_vertical(temperature, pressure, new_pressure_levels, method=method)


def interpolate_wind_components(
    u_wind: Union[np.ndarray, xr.DataArray],
    v_wind: Union[np.ndarray, xr.DataArray],
    old_levels: Union[np.ndarray, xr.DataArray],
    new_levels: Union[np.ndarray, xr.DataArray],
    method: str = 'linear'
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Interpolate wind components to new vertical levels.
    
    Parameters
    ----------
    u_wind : numpy.ndarray or xarray.DataArray
        Eastward wind component
    v_wind : numpy.ndarray or xarray.DataArray
        Northward wind component
    old_levels : numpy.ndarray or xarray.DataArray
        Original coordinate levels
    new_levels : numpy.ndarray or xarray.DataArray
        New coordinate levels for interpolation
    method : str, optional
        Interpolation method ('linear', 'log', 'cubic')
    
    Returns
    -------
    tuple of numpy.ndarray or xarray.DataArray
        Interpolated u and v wind components
    """
    u_interp = interpolate_vertical(u_wind, old_levels, new_levels, method=method)
    v_interp = interpolate_vertical(v_wind, old_levels, new_levels, method=method)
    
    return u_interp, v_interp


def remap_coordinates(
    data: xr.DataArray,
    target_grid: xr.Dataset,
    method: str = 'bilinear',
    **kwargs
) -> xr.DataArray:
    """
    Remap data from one coordinate system to another using xarray.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input data with coordinates to remap
    target_grid : xarray.Dataset
        Target grid specification
    method : str, optional
        Remapping method ('bilinear', 'nearest', 'conservative')
    **kwargs
        Additional arguments for the remapping function
    
    Returns
    -------
    xarray.DataArray
        Remapped data on the target grid
    """
    try:
        # If xESMF is available, use it for sophisticated remapping
        import xesmf as xe
        
        if method == 'bilinear':
            regridder = xe.Regridder(data, target_grid, 'bilinear', **kwargs)
        elif method == 'nearest':
            regridder = xe.Regridder(data, target_grid, 'nearest_s2d', **kwargs)
        elif method == 'conservative':
            regridder = xe.Regridder(data, target_grid, 'conservative', **kwargs)
        else:
            raise ValueError(f"Unknown remapping method: {method}")
        
        remapped_data = regridder(data)
        return remapped_data
    except ImportError:
        # Fallback to basic interpolation
        warnings.warn("xESMF not available, using basic interpolation")
        
        # Basic interpolation along lat/lon dimensions
        if 'lat' in data.coords and 'lon' in data.coords:
            # Assume lat/lon coordinates exist
            lat_new = target_grid['lat'] if 'lat' in target_grid else target_grid.coords['lat']
            lon_new = target_grid['lon'] if 'lon' in target_grid else target_grid.coords['lon']
            
            # Interpolate to new lat/lon grid
            result = data.interp(lat=lat_new, lon=lon_new, method='linear')
            return result
        else:
            raise ValueError("Coordinate remapping requires lat/lon coordinates")


def interpolate_with_dask(
    data: xr.DataArray,
    coords: Union[np.ndarray, xr.DataArray],
    new_coords: Union[np.ndarray, xr.DataArray],
    method: str = 'linear',
    chunk_size: Optional[int] = None
) -> xr.DataArray:
    """
    Perform interpolation with dask support for large datasets.
    
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
    import dask.array as da
    
    def interp_chunk(chunk, old_coords, new_coords, method):
        """Apply interpolation to a single chunk."""
        if chunk.ndim == 1:
            f = interp1d(old_coords, chunk, kind=method, fill_value='extrapolate')
            return f(new_coords)
        else:
            # For multi-dimensional chunks, interpolate along the last axis
            result = np.empty((chunk.shape[0], len(new_coords)))
            for i in range(chunk.shape[0]):
                f = interp1d(old_coords, chunk[i, :], kind=method, fill_value='extrapolate')
                result[i, :] = f(new_coords)
            return result
    
    # Apply the interpolation function to each chunk
    if chunk_size is not None:
        data = data.chunk({'time': chunk_size})  # Example for time dimension
    
    # Use map_blocks to apply interpolation to each chunk
    result = xr.apply_ufunc(
        lambda x: interp_chunk(x, coords, new_coords, method),
        data,
        input_core_dims=[['level']],
        output_core_dims=[['new_level']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={
            'output_sizes': {'new_level': len(new_coords)},
        }
    )
    
    # Add the new coordinate to the result
    result = result.assign_coords(new_level=new_coords)
    
    return result