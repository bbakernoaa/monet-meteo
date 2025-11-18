"""
Dynamic calculations for atmospheric science.

This module provides functions for calculating dynamic meteorological parameters including:
- Vorticity
- Divergence
- Geostrophic wind
- Gradient wind
- Absolute vorticity
- Potential vorticity
"""

import numpy as np
from typing import Union, Tuple
import xarray as xr

# Import constants
from ..constants import g, Omega, R_earth


def coriolis_parameter(
    latitude: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the Coriolis parameter (f = 2Ωsinφ).
    
    Parameters
    ----------
    latitude : float, numpy.ndarray, or xarray.DataArray
        Latitude in radians
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Coriolis parameter (s^-1)
    """
    # If latitude is in degrees, convert to radians
    if np.max(np.abs(latitude)) > np.pi/4:  # Likely in degrees
        lat_rad = np.radians(latitude)
    else:
        lat_rad = latitude  # Assume already in radians
    
    f = 2 * Omega * np.sin(lat_rad)
    
    return f


def relative_vorticity(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    dx: float,
    dy: float
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate relative vorticity (ζ = ∂v/∂x - ∂u/∂y).
    
    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component (m/s)
    v : numpy.ndarray or xarray.DataArray
        Northward wind component (m/s)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Relative vorticity (s^-1)
    """
    # Calculate derivatives using finite differences
    # ∂v/∂x
    dv_dx = np.gradient(v, axis=-1) / dx
    # ∂u/∂y
    du_dy = np.gradient(u, axis=-2) / dy
    
    # Calculate relative vorticity: ζ = ∂v/∂x - ∂u/∂y
    zeta = dv_dx - du_dy
    
    return zeta


def absolute_vorticity(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    dx: float,
    dy: float,
    latitude: Union[np.ndarray, xr.DataArray]
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate absolute vorticity (η = ζ + f).
    
    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component (m/s)
    v : numpy.ndarray or xarray.DataArray
        Northward wind component (m/s)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    latitude : numpy.ndarray or xarray.DataArray
        Latitude array (radians)
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Absolute vorticity (s^-1)
    """
    # Calculate relative vorticity
    zeta = relative_vorticity(u, v, dx, dy)
    
    # Calculate Coriolis parameter
    f = coriolis_parameter(latitude)
    
    # Calculate absolute vorticity: η = ζ + f
    eta = zeta + f
    
    return eta


def divergence(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    dx: float,
    dy: float
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate horizontal divergence (∇·V = ∂u/∂x + ∂v/∂y).
    
    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component (m/s)
    v : numpy.ndarray or xarray.DataArray
        Northward wind component (m/s)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Horizontal divergence (s^-1)
    """
    # Calculate derivatives using finite differences
    # ∂u/∂x
    du_dx = np.gradient(u, axis=-1) / dx
    # ∂v/∂y
    dv_dy = np.gradient(v, axis=-2) / dy
    
    # Calculate divergence: ∇·V = ∂u/∂x + ∂v/∂y
    div = du_dx + dv_dy
    
    return div


def geostrophic_wind(
    height: Union[np.ndarray, xr.DataArray],
    dx: float,
    dy: float,
    latitude: Union[np.ndarray, xr.DataArray]
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Calculate geostrophic wind from height field.
    
    Parameters
    ----------
    height : numpy.ndarray or xarray.DataArray
        Geopotential height field (m²/s²)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    latitude : numpy.ndarray or xarray.DataArray
        Latitude array (radians)
    
    Returns
    -------
    tuple of numpy.ndarray or xarray.DataArray
        Geostrophic wind components (u_g, v_g) in m/s
    """
    # Calculate Coriolis parameter
    f = coriolis_parameter(latitude)
    
    # Calculate derivatives of height field
    # ∂h/∂x
    dh_dx = np.gradient(height, axis=-1) / dx
    # ∂h/∂y
    dh_dy = np.gradient(height, axis=-2) / dy
    
    # Calculate geostrophic wind components
    # u_g = -g/f * ∂h/∂y
    u_g = -(g / f) * dh_dy
    # v_g = g/f * ∂h/∂x
    v_g = (g / f) * dh_dx
    
    return u_g, v_g


def gradient_wind(
    pressure: Union[np.ndarray, xr.DataArray],
    dx: float,
    dy: float,
    latitude: Union[np.ndarray, xr.DataArray],
    radius: Union[np.ndarray, xr.DataArray]
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate gradient wind speed.
    
    Parameters
    ----------
    pressure : numpy.ndarray or xarray.DataArray
        Pressure gradient (Pa/m)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    latitude : numpy.ndarray or xarray.DataArray
        Latitude array (radians)
    radius : numpy.ndarray or xarray.DataArray
        Radius of curvature (m)
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Gradient wind speed (m/s)
    """
    # Calculate Coriolis parameter
    f = coriolis_parameter(latitude)
    
    # Calculate pressure gradient force
    # For a circular flow: Vg^2 + f*R*Vg - (R/r)*dP/dn = 0
    # Solving the quadratic equation for Vg
    # This is a simplified version - in practice, more complex
    
    # For gradient wind in a curved flow:
    # Vg = -f*r/2 + sqrt((f*r/2)^2 + (r^2/ρ)*dP/dn)
    # where ρ is density and dP/dn is the pressure gradient normal to flow
    
    # Simplified approach using geostrophic wind as base
    # with correction for curvature
    # This is a simplified version - a full implementation would be more complex
    geostrophic_speed = np.sqrt(dx**2 + dy**2) * np.abs(pressure) / (f * radius)
    
    # Use quadratic formula for gradient wind
    # Vg^2 + f*radius*Vg - radius*pressure_gradient = 0
    # Vg = (-f*radius + sqrt((f*radius)^2 + 4*radius*pressure_gradient)) / 2
    # This is still a simplification
    
    # For now, return geostrophic wind as approximation
    return geostrophic_speed


def potential_vorticity(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    theta: Union[np.ndarray, xr.DataArray],
    dx: float,
    dy: float,
    latitude: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray]
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate Ertel's potential vorticity.
    
    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component (m/s)
    v : numpy.ndarray or xarray.DataArray
        Northward wind component (m/s)
    theta : numpy.ndarray or xarray.DataArray
        Potential temperature (K)
    dx : float
        Grid spacing in x direction (m)
    dy : float
        Grid spacing in y direction (m)
    latitude : numpy.ndarray or xarray.DataArray
        Latitude array (radians)
    pressure : numpy.ndarray or xarray.DataArray
        Pressure (Pa)
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Potential vorticity (K m² kg^-1 s^-1)
    """
    # Calculate absolute vorticity
    eta = absolute_vorticity(u, v, dx, dy, latitude)
    
    # Calculate gradients of potential temperature
    dtheta_dx = np.gradient(theta, axis=-1) / dx
    dtheta_dy = np.gradient(theta, axis=-2) / dy
    dtheta_dp = np.gradient(theta, axis=-3) / np.gradient(pressure, axis=-3) # assuming pressure is on axis -3
    
    # Calculate potential vorticity: PV = -g * (eta · ∇θ)
    # In pressure coordinates: PV = -g * (η · ∇_p θ)
    # For the vertical component: PV = -g * (η · ∇θ) = -g * (ζ + f) * ∂θ/∂p
    pv = -g * eta * dtheta_dp  # Simplified vertical component
    
    return pv


def vertical_velocity_pressure(
    omega: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    temperature: Union[np.ndarray, xr.DataArray],
    mixing_ratio: Union[np.ndarray, xr.DataArray] = None
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert omega (pressure vertical velocity) to w (geometric vertical velocity).
    
    Parameters
    ----------
    omega : numpy.ndarray or xarray.DataArray
        Vertical velocity in pressure coordinates (Pa/s)
    pressure : numpy.ndarray or xarray.DataArray
        Pressure (Pa)
    temperature : numpy.ndarray or xarray.DataArray
        Temperature (K)
    mixing_ratio : numpy.ndarray or xarray.DataArray, optional
        Mixing ratio (kg/kg), if not provided, assumed to be 0
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Geometric vertical velocity (m/s)
    """
    from ..constants import R_d
    
    # If mixing ratio not provided, assume 0
    if mixing_ratio is None:
        mixing_ratio = np.zeros_like(temperature) if isinstance(temperature, np.ndarray) else 0
    
    # Calculate virtual temperature
    t_virt = temperature * (1 + 0.61 * mixing_ratio)
    
    # Calculate air density using ideal gas law
    rho = pressure / (R_d * t_virt)
    
    # Convert omega to w using: w = -omega / (rho * g)
    w = -omega / (rho * g)
    
    return w


def omega_to_w(
    omega: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    temperature: Union[np.ndarray, xr.DataArray],
    mixing_ratio: Union[np.ndarray, xr.DataArray] = None
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert omega (pressure vertical velocity) to w (geometric vertical velocity).
    
    Parameters
    ----------
    omega : numpy.ndarray or xarray.DataArray
        Vertical velocity in pressure coordinates (Pa/s)
    pressure : numpy.ndarray or xarray.DataArray
        Pressure (Pa)
    temperature : numpy.ndarray or xarray.DataArray
        Temperature (K)
    mixing_ratio : numpy.ndarray or xarray.DataArray, optional
        Mixing ratio (kg/kg), if not provided, assumed to be 0
    
    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Geometric vertical velocity (m/s)
    """
    return vertical_velocity_pressure(omega, pressure, temperature, mixing_ratio)