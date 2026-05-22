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

from typing import Tuple, Union

import numpy as np
import xarray as xr

# Import constants
from ..constants import Omega, g


def coriolis_parameter(latitude: Union[float, np.ndarray, xr.DataArray]) -> Union[float, np.ndarray, xr.DataArray]:
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
    if np.max(np.abs(latitude)) > np.pi / 4:  # Likely in degrees
        lat_rad = np.radians(latitude)
    else:
        lat_rad = latitude  # Assume already in radians

    f = 2 * Omega * np.sin(lat_rad)

    return f


def relative_vorticity(
    u: Union[np.ndarray, xr.DataArray], v: Union[np.ndarray, xr.DataArray], dx: float, dy: float
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
    u: Union[np.ndarray, xr.DataArray], v: Union[np.ndarray, xr.DataArray], dx: float, dy: float, latitude: Union[np.ndarray, xr.DataArray]
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


def divergence(u: Union[np.ndarray, xr.DataArray], v: Union[np.ndarray, xr.DataArray], dx: float, dy: float) -> Union[np.ndarray, xr.DataArray]:
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
    height: Union[np.ndarray, xr.DataArray], dx: float, dy: float, latitude: Union[np.ndarray, xr.DataArray]
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
    radius: Union[np.ndarray, xr.DataArray],
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
    pressure: Union[np.ndarray, xr.DataArray],
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
    np.gradient(theta, axis=-1) / dx
    np.gradient(theta, axis=-2) / dy
    dtheta_dp = np.gradient(theta, axis=-3) / np.gradient(pressure, axis=-3)  # assuming pressure is on axis -3

    # Calculate potential vorticity: PV = -g * (eta · ∇θ)
    # In pressure coordinates: PV = -g * (η · ∇_p θ)
    # For the vertical component: PV = -g * (η · ∇θ) = -g * (ζ + f) * ∂θ/∂p
    pv = -g * eta * dtheta_dp  # Simplified vertical component

    return pv


def vertical_velocity_pressure(
    omega: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    temperature: Union[np.ndarray, xr.DataArray],
    mixing_ratio: Union[np.ndarray, xr.DataArray] = None,
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
    mixing_ratio: Union[np.ndarray, xr.DataArray] = None,
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


def bunkers_storm_motion(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    heights: Union[np.ndarray, xr.DataArray],
    latitude: Union[float, np.ndarray, xr.DataArray],
) -> Tuple[Union[np.ndarray, xr.DataArray], Union[np.ndarray, xr.DataArray]]:
    """
    Calculate the Bunkers storm motion vector.

    Based on UPP's CALHEL.f which uses the dynamic method (Bunkers et al. 1998).
    It computes estimated storm motion 7.5 m/s to the right of the 0-6 km mean
    wind, constrained along a line perpendicular to the 0-6 km mean vertical
    wind shear vector.

    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component (m/s). Assumed to have vertical dimension at axis -3.
    v : numpy.ndarray or xarray.DataArray
        Northward wind component (m/s). Assumed to have vertical dimension at axis -3.
    heights : numpy.ndarray or xarray.DataArray
        Heights AGL (m). Same shape as u, v.
    latitude : float, numpy.ndarray, or xarray.DataArray
        Latitude (degrees).

    Returns
    -------
    tuple of numpy.ndarray or xarray.DataArray
        Estimated storm motion (u_st, v_st) in m/s.
    """
    # 0-6 km mean wind
    mask_0_6 = (heights >= 0) & (heights <= 6000)
    u_mean6 = np.where(mask_0_6, u, np.nan)
    v_mean6 = np.where(mask_0_6, v, np.nan)
    u_mean6 = np.nanmean(u_mean6, axis=-3)
    v_mean6 = np.nanmean(v_mean6, axis=-3)

    # 0-0.5 km mean wind
    mask_0_05 = (heights >= 0) & (heights <= 500)
    u_mean05 = np.where(mask_0_05, u, np.nan)
    v_mean05 = np.where(mask_0_05, v, np.nan)
    u_mean05 = np.nanmean(u_mean05, axis=-3)
    v_mean05 = np.nanmean(v_mean05, axis=-3)

    # 5.5-6.0 km mean wind
    mask_55_6 = (heights >= 5500) & (heights <= 6000)
    u_mean55_6 = np.where(mask_55_6, u, np.nan)
    v_mean55_6 = np.where(mask_55_6, v, np.nan)
    u_mean55_6 = np.nanmean(u_mean55_6, axis=-3)
    v_mean55_6 = np.nanmean(v_mean55_6, axis=-3)

    # Shear vector
    u_shr6 = u_mean55_6 - u_mean05
    v_shr6 = v_mean55_6 - v_mean05

    denom = np.sqrt(u_shr6**2 + v_shr6**2)
    denom = np.where(denom == 0, np.nan, denom)

    # Determine hemisphere
    if isinstance(latitude, (xr.DataArray, np.ndarray)):
        is_northern = latitude >= 0
    else:
        is_northern = latitude >= 0

    # Storm motion (Right Mover)
    if isinstance(u_mean6, xr.DataArray):
        u_st = xr.where(is_northern, u_mean6 + (7.5 * v_shr6 / denom), u_mean6 - (7.5 * v_shr6 / denom))
        v_st = xr.where(is_northern, v_mean6 - (7.5 * u_shr6 / denom), v_mean6 + (7.5 * u_shr6 / denom))
    else:
        u_st = np.where(is_northern, u_mean6 + (7.5 * v_shr6 / denom), u_mean6 - (7.5 * v_shr6 / denom))
        v_st = np.where(is_northern, v_mean6 - (7.5 * u_shr6 / denom), v_mean6 + (7.5 * u_shr6 / denom))

    return u_st, v_st


def storm_relative_helicity(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    heights: Union[np.ndarray, xr.DataArray],
    u_st: Union[np.ndarray, xr.DataArray],
    v_st: Union[np.ndarray, xr.DataArray],
    depth: float = 3000.0,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate storm-relative helicity (SRH).

    Based on UPP's CALHEL.f logic.

    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component (m/s).
    v : numpy.ndarray or xarray.DataArray
        Northward wind component (m/s).
    heights : numpy.ndarray or xarray.DataArray
        Heights AGL (m).
    u_st : float, numpy.ndarray, or xarray.DataArray
        U component of storm motion (m/s).
    v_st : float, numpy.ndarray, or xarray.DataArray
        V component of storm motion (m/s).
    depth : float, optional
        Depth over which to compute SRH (m). Default is 3000.0 (0-3 km SRH).

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Storm-relative helicity (m^2/s^2).
    """
    # Filter by depth

    # Get layers
    # We assume heights are sorted vertically
    # Using trapezoidal integration: SRH = sum( (v - v_st)*du - (u - u_st)*dv )

    du = np.diff(u, axis=-3)
    dv = np.diff(v, axis=-3)

    # Mid-point values
    u_mid = (u[..., 1:, :, :] + u[..., :-1, :, :]) / 2.0
    v_mid = (v[..., 1:, :, :] + v[..., :-1, :, :]) / 2.0

    mask_mid = heights[..., 1:, :, :] <= depth

    term1 = (v_mid - v_st) * du
    term2 = (u_mid - u_st) * dv

    srh_layers = term1 - term2
    srh = np.nansum(np.where(mask_mid, srh_layers, 0), axis=-3)

    return srh


def moisture_convergence(
    u: Union[np.ndarray, xr.DataArray], v: Union[np.ndarray, xr.DataArray], specific_humidity: Union[np.ndarray, xr.DataArray], dx: float, dy: float
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate horizontal moisture convergence.

    Based on UPP's CALMCVG.f: QCNVG = - ( ∂(u*q)/∂x + ∂(v*q)/∂y )

    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component (m/s).
    v : numpy.ndarray or xarray.DataArray
        Northward wind component (m/s).
    specific_humidity : numpy.ndarray or xarray.DataArray
        Specific humidity (kg/kg).
    dx : float
        Grid spacing in x direction (m).
    dy : float
        Grid spacing in y direction (m).

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Moisture convergence (kg/kg/s).
    """
    uq = u * specific_humidity
    vq = v * specific_humidity

    duq_dx = np.gradient(uq, axis=-1) / dx
    dvq_dy = np.gradient(vq, axis=-2) / dy

    mconv = -(duq_dx + dvq_dy)

    return mconv
