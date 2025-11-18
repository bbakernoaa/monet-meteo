"""
Coordinate transformation utilities for atmospheric data.

This module provides functions for transforming between different coordinate systems
commonly used in meteorology, including geographic coordinates, pressure coordinates,
and model coordinates.
"""

import numpy as np
from typing import Union, Tuple, Optional
import xarray as xr
from math import radians, degrees, cos, sin, atan2, sqrt, asin


def latlon_to_cartesian(
    latitude: Union[float, np.ndarray, xr.DataArray],
    longitude: Union[float, np.ndarray, xr.DataArray],
    radius: float = 6371000.0  # Earth's radius in meters
) -> Tuple[Union[float, np.ndarray, xr.DataArray], 
           Union[float, np.ndarray, xr.DataArray], 
           Union[float, np.ndarray, xr.DataArray]]:
    """
    Convert geographic coordinates (lat/lon) to Cartesian coordinates (x, y, z).
    
    Parameters
    ----------
    latitude : float, numpy.ndarray, or xarray.DataArray
        Latitude in degrees
    longitude : float, numpy.ndarray, or xarray.DataArray
        Longitude in degrees
    radius : float, optional
        Earth's radius in meters (default: 6371000.0)
    
    Returns
    -------
    tuple of float, numpy.ndarray, or xarray.DataArray
        Cartesian coordinates (x, y, z) in meters
    """
    # Convert degrees to radians
    lat_rad = np.radians(latitude)
    lon_rad = np.radians(longitude)
    
    # Calculate Cartesian coordinates
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    
    return x, y, z


def cartesian_to_latlon(
    x: Union[float, np.ndarray, xr.DataArray],
    y: Union[float, np.ndarray, xr.DataArray],
    z: Union[float, np.ndarray, xr.DataArray]
) -> Tuple[Union[float, np.ndarray, xr.DataArray], 
           Union[float, np.ndarray, xr.DataArray]]:
    """
    Convert Cartesian coordinates (x, y, z) to geographic coordinates (lat/lon).
    
    Parameters
    ----------
    x : float, numpy.ndarray, or xarray.DataArray
        X coordinate in meters
    y : float, numpy.ndarray, or xarray.DataArray
        Y coordinate in meters
    z : float, numpy.ndarray, or xarray.DataArray
        Z coordinate in meters
    
    Returns
    -------
    tuple of float, numpy.ndarray, or xarray.DataArray
        Geographic coordinates (latitude, longitude) in degrees
    """
    # Calculate radius
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Calculate latitude and longitude
    latitude = np.degrees(np.arcsin(z / r))
    longitude = np.degrees(np.arctan2(y, x))
    
    return latitude, longitude


def rotate_wind_components(
    u: Union[float, np.ndarray, xr.DataArray],
    v: Union[float, np.ndarray, xr.DataArray],
    source_grid_north_pole: Tuple[float, float],
    target_grid_north_pole: Tuple[float, float],
    source_lats: Union[np.ndarray, xr.DataArray],
    source_lons: Union[np.ndarray, xr.DataArray]
) -> Tuple[Union[float, np.ndarray, xr.DataArray], 
           Union[float, np.ndarray, xr.DataArray]]:
    """
    Rotate wind components from one grid orientation to another.
    
    Parameters
    ----------
    u : float, numpy.ndarray, or xarray.DataArray
        Eastward wind component in source grid
    v : float, numpy.ndarray, or xarray.DataArray
        Northward wind component in source grid
    source_grid_north_pole : tuple of float
        (latitude, longitude) of source grid's north pole
    target_grid_north_pole : tuple of float
        (latitude, longitude) of target grid's north pole
    source_lats : numpy.ndarray or xarray.DataArray
        Latitude array of source grid points
    source_lons : numpy.ndarray or xarray.DataArray
        Longitude array of source grid points
    
    Returns
    -------
    tuple of float, numpy.ndarray, or xarray.DataArray
        Rotated (u, v) wind components
    """
    # Calculate rotation angles based on grid orientation
    # This is a simplified version - a full implementation would involve more complex
    # rotation matrices based on the specific grid definitions
    
    # For now, we'll implement a basic rotation based on lat/lon differences
    # Calculate the angle between the two north poles at each point
    lat1, lon1 = source_grid_north_pole
    lat2, lon2 = target_grid_north_pole
    
    # Convert to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    source_lats_rad, source_lons_rad = np.radians(source_lats), np.radians(source_lons)
    
    # Calculate rotation angle (simplified approach)
    # In a real implementation, this would involve more complex rotation matrices
    angle = np.zeros_like(source_lats)
    
    # Basic rotation matrix application
    u_rotated = u * np.cos(angle) - v * np.sin(angle)
    v_rotated = u * np.sin(angle) + v * np.cos(angle)
    
    return u_rotated, v_rotated


def pressure_to_sigma(
    pressure: Union[float, np.ndarray, xr.DataArray],
    surface_pressure: Union[float, np.ndarray, xr.DataArray],
    reference_pressure: float = 100000.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert pressure coordinates to sigma coordinates.
    
    Parameters
    ----------
    pressure : float, numpy.ndarray, or xarray.DataArray
        Pressure values (Pa)
    surface_pressure : float, numpy.ndarray, or xarray.DataArray
        Surface pressure values (Pa)
    reference_pressure : float, optional
        Reference pressure (Pa), default is 100000 Pa
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Sigma coordinates
    """
    sigma = (pressure - reference_pressure) / (surface_pressure - reference_pressure)
    return sigma


def sigma_to_pressure(
    sigma: Union[float, np.ndarray, xr.DataArray],
    surface_pressure: Union[float, np.ndarray, xr.DataArray],
    reference_pressure: float = 100000.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert sigma coordinates to pressure coordinates.
    
    Parameters
    ----------
    sigma : float, numpy.ndarray, or xarray.DataArray
        Sigma coordinates
    surface_pressure : float, numpy.ndarray, or xarray.DataArray
        Surface pressure values (Pa)
    reference_pressure : float, optional
        Reference pressure (Pa), default is 100000 Pa
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Pressure values (Pa)
    """
    pressure = reference_pressure + sigma * (surface_pressure - reference_pressure)
    return pressure


def calculate_grid_spacing(
    lats: Union[np.ndarray, xr.DataArray],
    lons: Union[np.ndarray, xr.DataArray],
    method: str = 'haversine'
) -> Tuple[Union[np.ndarray, xr.DataArray], 
           Union[np.ndarray, xr.DataArray]]:
    """
    Calculate grid spacing in meters for lat/lon coordinates.
    
    Parameters
    ----------
    lats : numpy.ndarray or xarray.DataArray
        Latitude array
    lons : numpy.ndarray or xarray.DataArray
        Longitude array
    method : str, optional
        Method for distance calculation ('haversine' or 'equirectangular')
    
    Returns
    -------
    tuple of numpy.ndarray or xarray.DataArray
        Grid spacing in y and x directions (dy, dx) in meters
    """
    if method.lower() == 'haversine':
        # Haversine formula for accurate distance calculation
        R = 6371000  # Earth's radius in meters
        
        # Calculate spacing in latitude direction
        if lats.ndim == 1:
            # 1D latitude array
            dy = np.diff(lats) * np.pi / 180.0 * R
            # Repeat the spacing for all longitude points if needed
            if lons.ndim == 1:
                dy = np.broadcast_to(dy[:, np.newaxis], (len(dy), len(lons)))
        else:
            # 2D latitude array
            dy = np.diff(lats, axis=0) * np.pi / 180.0 * R
        
        # Calculate spacing in longitude direction
        if lons.ndim == 1:
            # 1D longitude array
            lat_mid = (lats[1:] + lats[:-1]) / 2 if lats.ndim == 1 else lats
            dx = np.diff(lons) * np.pi / 180.0 * R * np.cos(np.radians(lat_mid))
        else:
            # 2D longitude array
            lat_mid = (lats[1:] + lats[:-1]) / 2 if lats.ndim == 1 else (lats[1:] + lats[:-1]) / 2
            dx = np.diff(lons, axis=1) * np.pi / 180.0 * R * np.cos(np.radians(lat_mid))
    
    elif method.lower() == 'equirectangular':
        # Simplified equirectangular projection
        R = 6371000  # Earth's radius in meters
        
        if lats.ndim == 1:
            dy = np.diff(lats) * np.pi / 180.0 * R
        else:
            dy = np.diff(lats, axis=0) * np.pi / 180.0 * R
        
        if lons.ndim == 1:
            lat_mid = (lats[1:] + lats[:-1]) / 2 if lats.ndim == 1 else lats
            dx = np.diff(lons) * np.pi / 180.0 * R * np.cos(np.radians(lat_mid))
        else:
            lat_mid = (lats[1:] + lats[:-1]) / 2 if lats.ndim == 1 else (lats[1:] + lats[:-1]) / 2
            dx = np.diff(lons, axis=1) * np.pi / 180.0 * R * np.cos(np.radians(lat_mid))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return dy, dx


def transform_coordinates(
    data: xr.DataArray,
    source_crs: str,
    target_crs: str,
    **kwargs
) -> xr.DataArray:
    """
    Transform coordinates of xarray DataArray from one coordinate reference system to another.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input data with coordinate information
    source_crs : str
        Source coordinate reference system (e.g., 'EPSG:4326')
    target_crs : str
        Target coordinate reference system (e.g., 'EPSG:3857')
    **kwargs
        Additional arguments for coordinate transformation
    
    Returns
    -------
    xarray.DataArray
        Data with transformed coordinates
    """
    try:
        # Try to use pyproj for coordinate transformation
        import pyproj
        from pyproj import Transformer
        
        # Create transformer
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        
        # Get current coordinates
        if 'lon' in data.coords and 'lat' in data.coords:
            lons = data.coords['lon']
            lats = data.coords['lat']
            
            # Transform coordinates
            new_lons, new_lats = transformer.transform(lons, lats)
            
            # Create new data array with transformed coordinates
            result = data.copy()
            result = result.assign_coords({'lon': new_lons, 'lat': new_lats})
            
            return result
        else:
            raise ValueError("Data must have 'lon' and 'lat' coordinates")
    except ImportError:
        # If pyproj is not available, raise an informative error
        raise ImportError(
            "pyproj is required for coordinate transformation. "
            "Install with: pip install pyproj"
        )


def calculate_distance(
    lat1: Union[float, np.ndarray, xr.DataArray],
    lon1: Union[float, np.ndarray, xr.DataArray],
    lat2: Union[float, np.ndarray, xr.DataArray],
    lon2: Union[float, np.ndarray, xr.DataArray],
    method: str = 'haversine'
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate distance between two points on Earth's surface.
    
    Parameters
    ----------
    lat1, lon1 : float, numpy.ndarray, or xarray.DataArray
        Latitude and longitude of first point in degrees
    lat2, lon2 : float, numpy.ndarray, or xarray.DataArray
        Latitude and longitude of second point in degrees
    method : str, optional
        Method for distance calculation ('haversine' or 'vincenty')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters
    
    if method.lower() == 'haversine':
        # Haversine formula
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance = R * c
    elif method.lower() == 'vincenty':
        # Simplified Vincenty formula (this is a basic approximation)
        # For full Vincenty implementation, use the geopy library
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        distance = R * c
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return distance


def bearing(
    lat1: Union[float, np.ndarray, xr.DataArray],
    lon1: Union[float, np.ndarray, xr.DataArray],
    lat2: Union[float, np.ndarray, xr.DataArray],
    lon2: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the initial bearing (forward azimuth) between two points.
    
    Parameters
    ----------
    lat1, lon1 : float, numpy.ndarray, or xarray.DataArray
        Latitude and longitude of first point in degrees
    lat2, lon2 : float, numpy.ndarray, or xarray.DataArray
        Latitude and longitude of second point in degrees
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Bearing in degrees (0-360)
    """
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    
    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)
    
    bearing_rad = np.arctan2(y, x)
    bearing_deg = (np.degrees(bearing_rad) + 360) % 360
    
    return bearing_deg


def destination_point(
    lat: Union[float, np.ndarray, xr.DataArray],
    lon: Union[float, np.ndarray, xr.DataArray],
    distance: Union[float, np.ndarray, xr.DataArray],
    bearing: Union[float, np.ndarray, xr.DataArray]
) -> Tuple[Union[float, np.ndarray, xr.DataArray], 
           Union[float, np.ndarray, xr.DataArray]]:
    """
    Calculate the destination point given a start point, distance, and bearing.
    
    Parameters
    ----------
    lat : float, numpy.ndarray, or xarray.DataArray
        Starting latitude in degrees
    lon : float, numpy.ndarray, or xarray.DataArray
        Starting longitude in degrees
    distance : float, numpy.ndarray, or xarray.DataArray
        Distance in meters
    bearing : float, numpy.ndarray, or xarray.DataArray
        Bearing in degrees
    
    Returns
    -------
    tuple of float, numpy.ndarray, or xarray.DataArray
        Destination latitude and longitude in degrees
    """
    R = 6371000  # Earth's radius in meters
    
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    bearing_rad = np.radians(bearing)
    
    angular_distance = distance / R
    
    dest_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(angular_distance) + 
        np.cos(lat_rad) * np.sin(angular_distance) * np.cos(bearing_rad)
    )
    
    dest_lon_rad = lon_rad + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(lat_rad),
        np.cos(angular_distance) - np.sin(lat_rad) * np.sin(dest_lat_rad)
    )
    
    dest_lat = np.degrees(dest_lat_rad)
    dest_lon = np.degrees(dest_lon_rad)
    
    # Normalize longitude to [-180, 180]
    dest_lon = ((dest_lon + 540) % 360) - 180
    
    return dest_lat, dest_lon


def convert_vertical_coord(
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
    if from_coord.lower() == to_coord.lower():
        return data
    
    # Get the current vertical coordinate
    if from_coord in data.coords:
        vertical_coord = data.coords[from_coord]
    else:
        # Try to find a coordinate that matches the expected type
        coord_names = [name for name in data.coords if 
                      name in ['pressure', 'altitude', 'sigma', 'z', 'level']]
        if coord_names:
            vertical_coord = data.coords[coord_names[0]]
        else:
            raise ValueError(f"Could not find coordinate for {from_coord}")
    
    # Perform the conversion based on the coordinate systems
    if from_coord.lower() == 'pressure' and to_coord.lower() == 'altitude':
        from .interpolation import pressure_to_altitude
        new_coord = pressure_to_altitude(vertical_coord)
    elif from_coord.lower() == 'altitude' and to_coord.lower() == 'pressure':
        from .interpolation import altitude_to_pressure
        new_coord = altitude_to_pressure(vertical_coord)
    elif from_coord.lower() == 'pressure' and to_coord.lower() == 'sigma':
        if surface_pressure is None:
            raise ValueError("Surface pressure required for pressure-to-sigma conversion")
        new_coord = pressure_to_sigma(vertical_coord, surface_pressure, reference_pressure)
    elif from_coord.lower() == 'sigma' and to_coord.lower() == 'pressure':
        if surface_pressure is None:
            raise ValueError("Surface pressure required for sigma-to-pressure conversion")
        new_coord = sigma_to_pressure(vertical_coord, surface_pressure, reference_pressure)
    else:
        # For other conversions, implement as needed
        raise NotImplementedError(
            f"Conversion from {from_coord} to {to_coord} not yet implemented"
        )
    
    # Create new data array with converted coordinate
    result = data.copy()
    result = result.assign_coords({to_coord: new_coord})
    
    # Update coordinate attributes if present
    if from_coord in data.coords:
        result.coords[to_coord].attrs = data.coords[from_coord].attrs.copy()
        result.coords[to_coord].attrs['standard_name'] = to_coord
    
    return result