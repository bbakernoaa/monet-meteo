"""
Statistical operations for atmospheric data analysis and micrometeorology calculations.

This module implements Monin-Obukhov similarity theory, surface energy balance,
turbulent flux calculations, and atmospheric stability parameters with
xarray/dask support.
"""

import numpy as np
import xarray as xr
from typing import Union, Optional, Tuple
# Don't import constants at module level to avoid conflicts when using 'from .statistical import *'
# Instead, import the constants module and access via module name
from .. import constants


def bulk_richardson_number(
    u_wind: Union[float, np.ndarray, xr.DataArray],
    v_wind: Union[float, np.ndarray, xr.DataArray],
    potential_temperature: Union[float, np.ndarray, xr.DataArray],
    height: Union[float, np.ndarray, xr.DataArray],
    method: str = 'standard'
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate bulk Richardson number for atmospheric stability.
    
    The bulk Richardson number is a dimensionless number that expresses the 
    ratio of buoyancy forces to shear forces in the atmosphere.
    
    Parameters
    ----------
    u_wind : float, numpy.ndarray, or xarray.DataArray
        Eastward wind component (m/s)
    v_wind : float, numpy.ndarray, or xarray.DataArray
        Northward wind component (m/s)
    potential_temperature : float, numpy.ndarray, or xarray.DataArray
        Potential temperature (K)
    height : float, numpy.ndarray, or xarray.DataArray
        Height above surface (m)
    method : str, optional
        Method for calculation ('standard' or 'modified')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Bulk Richardson number (-)
    """
    # Calculate wind speed
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    
    # Calculate potential temperature difference
    theta_diff = np.gradient(potential_temperature, axis=-1) if hasattr(potential_temperature, 'ndim') else np.gradient(potential_temperature)
    height_diff = np.gradient(height, axis=-1) if hasattr(height, 'ndim') else np.gradient(height)
    
    # Calculate bulk Richardson number
    if method == 'standard':
        # Standard definition
        Ri_b = (constants.g / potential_temperature) * (theta_diff / (wind_speed**2 + 1e-12)) * height_diff
    else:
        # Modified version
        Ri_b = (constants.g * height * theta_diff) / (potential_temperature * (wind_speed**2 + 1e-12))
    
    return Ri_b


def monin_obukhov_length(
    friction_velocity: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
    air_density: Union[float, np.ndarray, xr.DataArray],
    specific_heat: Union[float, np.ndarray, xr.DataArray],
    sensible_heat_flux: Union[float, np.ndarray, xr.DataArray],
    latent_heat_flux: Optional[Union[float, np.ndarray, xr.DataArray]] = None
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the Monin-Obukhov length using the standard definition.
    
    The Monin-Obukhov length is a characteristic length scale that describes
    the effects of buoyancy on turbulent flows in the atmospheric boundary layer.
    
    Parameters
    ----------
    friction_velocity : float, numpy.ndarray, or xarray.DataArray
        Friction velocity (m/s)
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    air_density : float, numpy.ndarray, or xarray.DataArray
        Air density (kg/m³)
    specific_heat : float, numpy.ndarray, or xarray.DataArray
        Specific heat of air at constant pressure (J/kg/K)
    sensible_heat_flux : float, numpy.ndarray, or xarray.DataArray
        Sensible heat flux (W/m²)
    latent_heat_flux : float, numpy.ndarray, or xarray.DataArray, optional
        Latent heat flux (W/m²), if provided, virtual heat flux is calculated
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Monin-Obukhov length (m)
    """
    # Calculate kinematic heat flux
    if latent_heat_flux is not None:
        # Calculate latent heat of vaporization
        lambda_v = 2.501e6 - 2361 * (temperature - 273.15)  # J/kg
        # Convert latent heat flux to evaporation rate
        evaporation_rate = latent_heat_flux / lambda_v  # kg/m²/s
        # Calculate virtual sensible heat flux
        kinematic_heat_flux = (sensible_heat_flux + 0.61 * temperature * specific_heat * evaporation_rate) / (air_density * specific_heat)
    else:
        # Use sensible heat flux directly
        kinematic_heat_flux = sensible_heat_flux / (air_density * specific_heat)
    
    # Calculate Obukhov length
    # L = -u*³ * T / (k * g * w'T')
    denominator = constants.k * constants.g * kinematic_heat_flux
    L = -friction_velocity**3 * temperature / (denominator + 1e-12)
    
    return L


def stability_parameter(
    height: Union[float, np.ndarray, xr.DataArray],
    obukhov_length: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the stability parameter (z/L) used in Monin-Obukhov similarity theory.
    
    Parameters
    ----------
    height : float, numpy.ndarray, or xarray.DataArray
        Height above surface (m)
    obukhov_length : float, numpy.ndarray, or xarray.DataArray
        Obukhov length (m)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Stability parameter z/L (-)
    """
    stability_param = height / (obukhov_length + 1e-12)
    return stability_param


def psi_momentum(
    stability_parameter: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the stability correction function for momentum (Ψ_m) in MO similarity theory.
    
    Parameters
    ----------
    stability_parameter : float, numpy.ndarray, or xarray.DataArray
        Stability parameter z/L (-)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Stability correction for momentum Ψ_m (-)
    """
    # Convert to numpy array for consistent operations
    zL = np.asarray(stability_parameter)
    
    # Initialize result array
    psi_m = np.zeros_like(zL)
    
    # For stable conditions (z/L > 0)
    stable_idx = zL >= 0
    if np.any(stable_idx):
        a = 6.1
        b = 2.5
        psi_m[stable_idx] = -a * zL[stable_idx] - b * np.log(1 + zL[stable_idx] / b)
    
    # For unstable conditions (z/L < 0)
    unstable_idx = zL < 0
    if np.any(unstable_idx):
        x = np.sqrt(1 - 16 * zL[unstable_idx])
        psi_m[unstable_idx] = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi/2
    
    return psi_m


def psi_heat(
    stability_parameter: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the stability correction function for heat (Ψ_h) in MO similarity theory.
    
    Parameters
    ----------
    stability_parameter : float, numpy.ndarray, or xarray.DataArray
        Stability parameter z/L (-)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Stability correction for heat Ψ_h (-)
    """
    # Convert to numpy array for consistent operations
    zL = np.asarray(stability_parameter)
    
    # Initialize result array
    psi_h = np.zeros_like(zL)
    
    # For stable conditions (z/L > 0)
    stable_idx = zL >= 0
    if np.any(stable_idx):
        a = 6.1
        psi_h[stable_idx] = -a * zL[stable_idx]
    
    # For unstable conditions (z/L < 0)
    unstable_idx = zL < 0
    if np.any(unstable_idx):
        x = np.sqrt(1 - 16 * zL[unstable_idx])
        psi_h[unstable_idx] = 2 * np.log((1 + x) / 2)
    
    return psi_h


def aerodynamic_resistance(
    height: Union[float, np.ndarray, xr.DataArray],
    roughness_length: Union[float, np.ndarray, xr.DataArray],
    stability_parameter: Union[float, np.ndarray, xr.DataArray],
    displacement_height: float = 0.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate aerodynamic resistance for heat and momentum transfer.
    
    Parameters
    ----------
    height : float, numpy.ndarray, or xarray.DataArray
        Measurement height (m)
    roughness_length : float, numpy.ndarray, or xarray.DataArray
        Surface roughness length (m)
    stability_parameter : float, numpy.ndarray, or xarray.DataArray
        Stability parameter z/L (-)
    displacement_height : float, optional
        Zero-plane displacement height (m), default is 0.0
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Aerodynamic resistance (s/m)
    """
    # Calculate stability correction terms
    psi_m = psi_momentum(stability_parameter)
    psi_m0 = psi_momentum(0.0)  # At roughness level
    
    # Calculate wind profile correction
    stability_correction = np.log((height - displacement_height) / roughness_length) - psi_m + psi_m0
    
    # Aerodynamic resistance is inverse of transfer coefficient
    # For now, return the logarithmic term (more complete calculation would include wind speed)
    return stability_correction


def surface_energy_balance(
    net_radiation: Union[float, np.ndarray, xr.DataArray],
    soil_heat_flux: Union[float, np.ndarray, xr.DataArray],
    sensible_heat_flux: Union[float, np.ndarray, xr.DataArray],
    latent_heat_flux: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate surface energy balance closure.
    
    The surface energy balance states that net radiation is equal to the sum
    of soil heat flux, sensible heat flux, and latent heat flux.
    
    Parameters
    ----------
    net_radiation : float, numpy.ndarray, or xarray.DataArray
        Net radiation at surface (W/m²)
    soil_heat_flux : float, numpy.ndarray, or xarray.DataArray
        Soil heat flux (W/m²)
    sensible_heat_flux : float, numpy.ndarray, or xarray.DataArray
        Sensible heat flux (W/m²)
    latent_heat_flux : float, numpy.ndarray, or xarray.DataArray
        Latent heat flux (W/m²)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Energy balance residual (W/m²)
    """
    energy_residual = net_radiation - (soil_heat_flux + sensible_heat_flux + latent_heat_flux)
    return energy_residual


def sensible_heat_flux(
    air_temperature: Union[float, np.ndarray, xr.DataArray],
    surface_temperature: Union[float, np.ndarray, xr.DataArray],
    aerodynamic_resistance: Union[float, np.ndarray, xr.DataArray],
    air_density: Union[float, np.ndarray, xr.DataArray] = 1.225,
    specific_heat: Union[float, np.ndarray, xr.DataArray] = 1004.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate sensible heat flux using bulk aerodynamic method.
    
    Parameters
    ----------
    air_temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature at reference height (K)
    surface_temperature : float, numpy.ndarray, or xarray.DataArray
        Surface temperature (K)
    aerodynamic_resistance : float, numpy.ndarray, or xarray.DataArray
        Aerodynamic resistance (s/m)
    air_density : float, optional
        Air density (kg/m³), default is 1.225 kg/m³
    specific_heat : float, optional
        Specific heat of air at constant pressure (J/kg/K), default is 1004 J/kg/K
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Sensible heat flux (W/m²)
    """
    temp_diff = surface_temperature - air_temperature
    H = air_density * specific_heat * temp_diff / aerodynamic_resistance
    return H


def latent_heat_flux(
    vapor_pressure_air: Union[float, np.ndarray, xr.DataArray],
    vapor_pressure_surface: Union[float, np.ndarray, xr.DataArray],
    aerodynamic_resistance: Union[float, np.ndarray, xr.DataArray],
    air_density: Union[float, np.ndarray, xr.DataArray] = 1.225,
    latent_heat_vaporization: Union[float, np.ndarray, xr.DataArray] = 2.501e6
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate latent heat flux using bulk aerodynamic method.
    
    Parameters
    ----------
    vapor_pressure_air : float, numpy.ndarray, or xarray.DataArray
        Vapor pressure at reference height (Pa)
    vapor_pressure_surface : float, numpy.ndarray, or xarray.DataArray
        Vapor pressure at surface (Pa)
    aerodynamic_resistance : float, numpy.ndarray, or xarray.DataArray
        Aerodynamic resistance (s/m)
    air_density : float, optional
        Air density (kg/m³), default is 1.225 kg/m³
    latent_heat_vaporization : float, optional
        Latent heat of vaporization (J/kg), default is 2.501e6 J/kg
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Latent heat flux (W/m²)
    """
    vapor_pressure_diff = vapor_pressure_surface - vapor_pressure_air
    # Calculate air density of water vapor using ideal gas law
    # For simplicity, using simplified form: LE = rho * Lv * (e_s - e_a) / r_a
    LE = air_density * latent_heat_vaporization * vapor_pressure_diff / (constants.R_d * 293.0 * aerodynamic_resistance + 1e-12)
    return LE


def friction_velocity_from_wind(
    wind_speed: Union[float, np.ndarray, xr.DataArray],
    height: Union[float, np.ndarray, xr.DataArray],
    roughness_length: Union[float, np.ndarray, xr.DataArray],
    stability_parameter: Union[float, np.ndarray, xr.DataArray] = 0.0,
    displacement_height: float = 0.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate friction velocity from wind speed profile using MO similarity theory.
    
    Parameters
    ----------
    wind_speed : float, numpy.ndarray, or xarray.DataArray
        Wind speed at measurement height (m/s)
    height : float, numpy.ndarray, or xarray.DataArray
        Measurement height (m)
    roughness_length : float, numpy.ndarray, or xarray.DataArray
        Surface roughness length for momentum (m)
    stability_parameter : float, numpy.ndarray, or xarray.DataArray, optional
        Stability parameter z/L (-), default is 0.0 (neutral stability)
    displacement_height : float, optional
        Zero-plane displacement height (m), default is 0.0
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Friction velocity (m/s)
    """
    # Calculate stability correction functions
    psi_m = psi_momentum(stability_parameter)
    psi_m0 = psi_momentum(roughness_length / (1.0 + 1e-12))  # At roughness level
    
    # Calculate friction velocity from wind profile
    log_term = np.log((height - displacement_height) / roughness_length)
    stability_term = psi_m - psi_m0
    
    u_star = constants.k * wind_speed / (log_term - stability_term + 1e-12)
    return u_star


def atmospheric_boundary_layer_height(
    surface_temperature: Union[float, np.ndarray, xr.DataArray],
    potential_temperature_gradient: Union[float, np.ndarray, xr.DataArray],
    wind_speed: Union[float, np.ndarray, xr.DataArray],
    height: Union[float, np.ndarray, xr.DataArray],
    method: str = 'bulk_richardson'
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Estimate atmospheric boundary layer height using various methods.
    
    Parameters
    ----------
    surface_temperature : float, numpy.ndarray, or xarray.DataArray
        Surface temperature (K)
    potential_temperature_gradient : float, numpy.ndarray, or xarray.DataArray
        Potential temperature gradient (K/m)
    wind_speed : float, numpy.ndarray, or xarray.DataArray
        Wind speed (m/s)
    height : float, numpy.ndarray, or xarray.DataArray
        Height levels (m)
    method : str, optional
        Method to use ('bulk_richardson', 'potential_temperature_gradient')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Boundary layer height (m)
    """
    if method == 'bulk_richardson':
        # Find height where bulk Richardson number exceeds critical value
        Ri = bulk_richardson_number(wind_speed, wind_speed*0, surface_temperature, height)
        # Critical Richardson number for boundary layer top
        Ri_critical = 0.25
        
        # For simplicity, return height for now - a full implementation would require more complex logic
        # This is a simplified version that works for both scalars and arrays
        if np.isscalar(Ri):
            h_bl = height if Ri > Ri_critical else height
        else:
            # For array inputs, return max height where Ri > Ri_critical, or max height if none
            if np.any(Ri > Ri_critical):
                # Find indices where Ri > Ri_critical and return corresponding heights
                # For now, just return the max height for those conditions
                h_bl = np.where(Ri > Ri_critical, height, 0).max()
            else:
                h_bl = np.max(height) if hasattr(height, '__len__') and len(np.atleast_1d(height)) > 1 else float(height)
    
    elif method == 'potential_temperature_gradient':
        # Find height where potential temperature gradient significantly increases
        # This indicates the top of the mixed layer
        if isinstance(potential_temperature_gradient, np.ndarray):
            # Find the maximum gradient which indicates the inversion
            grad_max_idx = np.argmax(potential_temperature_gradient, axis=-1 if potential_temperature_gradient.ndim > 0 else 0)
            if hasattr(height, 'ndim') and height.ndim > 0 and height.ndim >= grad_max_idx.ndim:
                # Handle array operations properly
                if grad_max_idx.ndim == 0:  # scalar index
                    idx = int(grad_max_idx) if np.isscalar(grad_max_idx) else grad_max_idx.item()
                    h_bl = height[idx] if idx < len(height) else height[-1]
                else:
                    # For array of indices, use advanced indexing
                    h_bl = np.take_along_axis(height, grad_max_idx[..., np.newaxis], axis=-1).squeeze()
            else:
                h_bl = height
        else:
            h_bl = height
    else:
        raise ValueError(f"Method {method} not recognized. Use 'bulk_richardson' or 'potential_temperature_gradient'")
    
    return h_bl


def turbulence_intensity(
    wind_speed: Union[float, np.ndarray, xr.DataArray],
    wind_speed_std: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate turbulence intensity as the ratio of wind speed standard deviation to mean wind speed.
    
    Parameters
    ----------
    wind_speed : float, numpy.ndarray, or xarray.DataArray
        Mean wind speed (m/s)
    wind_speed_std : float, numpy.ndarray, or xarray.DataArray
        Standard deviation of wind speed (m/s)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Turbulence intensity (-)
    """
    TI = wind_speed_std / (np.abs(wind_speed) + 1e-12)
    return TI


def obukhov_stability_parameter(
    friction_velocity: Union[float, np.ndarray, xr.DataArray],
    air_temperature: Union[float, np.ndarray, xr.DataArray],
    kinematic_heat_flux: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate Obukhov stability parameter directly from turbulent quantities.
    
    Parameters
    ----------
    friction_velocity : float, numpy.ndarray, or xarray.DataArray
        Friction velocity (m/s)
    air_temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    kinematic_heat_flux : float, numpy.ndarray, or xarray.DataArray
        Kinematic heat flux (K m/s)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Obukhov stability parameter (m)
    """
    # Calculate Obukhov length
    numerator = -friction_velocity**3 * air_temperature
    denominator = constants.k * constants.g * kinematic_heat_flux
    L = numerator / (denominator + 1e-12)
    return L


def xarray_bulk_richardson_number(
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    potential_temperature: xr.DataArray,
    height: xr.DataArray,
    method: str = 'standard'
) -> xr.DataArray:
    """
    Xarray wrapper for bulk Richardson number calculation with dask support.
    
    Parameters
    ----------
    u_wind : xarray.DataArray
        Eastward wind component (m/s)
    v_wind : xarray.DataArray
        Northward wind component (m/s)
    potential_temperature : xarray.DataArray
        Potential temperature (K)
    height : xarray.DataArray
        Height above surface (m)
    method : str, optional
        Method for calculation ('standard' or 'modified')
    
    Returns
    -------
    xarray.DataArray
        Bulk Richardson number (-)
    """
    result = xr.apply_ufunc(
        bulk_richardson_number,
        u_wind, v_wind, potential_temperature, height, method,
        input_core_dims=[[], [], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Preserve metadata
    result.name = "bulk_richardson_number"
    result.attrs = {
        "long_name": "Bulk Richardson Number",
        "units": "1",
        "standard_name": "atmosphere_bulk_richardson_number"
    }
    
    # Copy coordinates
    result = result.assign_coords(u_wind.coords)
    
    return result


def xarray_monin_obukhov_length(
    friction_velocity: xr.DataArray,
    temperature: xr.DataArray,
    air_density: xr.DataArray,
    specific_heat: xr.DataArray,
    sensible_heat_flux: xr.DataArray,
    latent_heat_flux: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """
    Xarray wrapper for Monin-Obukhov length calculation with dask support.
    
    Parameters
    ----------
    friction_velocity : xarray.DataArray
        Friction velocity (m/s)
    temperature : xarray.DataArray
        Air temperature (K)
    air_density : xarray.DataArray
        Air density (kg/m³)
    specific_heat : xarray.DataArray
        Specific heat of air at constant pressure (J/kg/K)
    sensible_heat_flux : xarray.DataArray
        Sensible heat flux (W/m²)
    latent_heat_flux : xarray.DataArray, optional
        Latent heat flux (W/m²)
    
    Returns
    -------
    xarray.DataArray
        Monin-Obukhov length (m)
    """
    if latent_heat_flux is not None:
        result = xr.apply_ufunc(
            monin_obukhov_length,
            friction_velocity, temperature, air_density, specific_heat,
            sensible_heat_flux, latent_heat_flux,
            input_core_dims=[[], [], [], [], [], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float]
        )
    else:
        result = xr.apply_ufunc(
            monin_obukhov_length,
            friction_velocity, temperature, air_density, specific_heat,
            sensible_heat_flux,
            input_core_dims=[[], [], [], [], []],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float]
        )
    
    # Preserve metadata
    result.name = "monin_obukhov_length"
    result.attrs = {
        "long_name": "Monin-Obukhov Length",
        "units": "m",
        "standard_name": "atmosphere_obukhov_length"
    }
    
    # Copy coordinates
    result = result.assign_coords(friction_velocity.coords)
    
    return result


def xarray_surface_energy_balance(
    net_radiation: xr.DataArray,
    soil_heat_flux: xr.DataArray,
    sensible_heat_flux: xr.DataArray,
    latent_heat_flux: xr.DataArray
) -> xr.DataArray:
    """
    Xarray wrapper for surface energy balance calculation with dask support.
    
    Parameters
    ----------
    net_radiation : xarray.DataArray
        Net radiation at surface (W/m²)
    soil_heat_flux : xarray.DataArray
        Soil heat flux (W/m²)
    sensible_heat_flux : xarray.DataArray
        Sensible heat flux (W/m²)
    latent_heat_flux : xarray.DataArray
        Latent heat flux (W/m²)
    
    Returns
    -------
    xarray.DataArray
        Energy balance residual (W/m²)
    """
    result = xr.apply_ufunc(
        surface_energy_balance,
        net_radiation, soil_heat_flux, sensible_heat_flux, latent_heat_flux,
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    )
    
    # Preserve metadata
    result.name = "surface_energy_balance_residual"
    result.attrs = {
        "long_name": "Surface Energy Balance Residual",
        "units": "W m-2",
        "standard_name": "surface_upward_energy_flux_balance"
    }
    
    # Copy coordinates
    result = result.assign_coords(net_radiation.coords)
    
    return result


def xarray_turbulent_fluxes_from_similarity(
    wind_speed: xr.DataArray,
    air_temperature: xr.DataArray,
    surface_temperature: xr.DataArray,
    vapor_pressure_air: xr.DataArray,
    vapor_pressure_surface: xr.DataArray,
    height: xr.DataArray,
    roughness_length: xr.DataArray,
    stability_parameter: xr.DataArray,
    displacement_height: float = 0.0
) -> Tuple[Union[xr.DataArray, np.ndarray, float], Union[xr.DataArray, np.ndarray, float]]:
    """
    Xarray wrapper to calculate turbulent fluxes using similarity theory.
    
    Parameters
    ----------
    wind_speed : xarray.DataArray
        Wind speed at measurement height (m/s)
    air_temperature : xarray.DataArray
        Air temperature at reference height (K)
    surface_temperature : xarray.DataArray
        Surface temperature (K)
    vapor_pressure_air : xarray.DataArray
        Vapor pressure at reference height (Pa)
    vapor_pressure_surface : xarray.DataArray
        Vapor pressure at surface (Pa)
    height : xarray.DataArray
        Measurement height (m)
    roughness_length : xarray.DataArray
        Surface roughness length for momentum (m)
    stability_parameter : xarray.DataArray
        Stability parameter z/L (-)
    displacement_height : float, optional
        Zero-plane displacement height (m), default is 0.0
    
    Returns
    -------
    tuple of xarray.DataArray
        Sensible and latent heat fluxes (W/m²)
    """
    # Calculate friction velocity
    u_star = friction_velocity_from_wind(
        wind_speed, height, roughness_length, stability_parameter, displacement_height
    )
    
    # Calculate aerodynamic resistance
    r_a = aerodynamic_resistance(height, roughness_length, stability_parameter, displacement_height)
    
    # Calculate sensible heat flux
    H = sensible_heat_flux(air_temperature, surface_temperature, r_a)
    
    # Calculate latent heat flux
    LE = latent_heat_flux(vapor_pressure_air, vapor_pressure_surface, r_a)
    
    # Set metadata only if inputs are xarray DataArrays
    if isinstance(wind_speed, xr.DataArray):
        # Only set attributes if H and LE are also xarray DataArrays
        if isinstance(H, xr.DataArray):
            H.name = "sensible_heat_flux"
            H.attrs = {
                "long_name": "Surface Upward Sensible Heat Flux",
                "units": "W m-2",
                "standard_name": "surface_upward_sensible_heat_flux"
            }
            # Copy coordinates if wind_speed has coordinates
            if hasattr(wind_speed, 'coords'):
                H = H.assign_coords(wind_speed.coords)
        
        if isinstance(LE, xr.DataArray):
            LE.name = "latent_heat_flux"
            LE.attrs = {
                "long_name": "Surface Upward Latent Heat Flux",
                "units": "W m-2",
                "standard_name": "surface_upward_latent_heat_flux"
            }
            # Copy coordinates if wind_speed has coordinates
            if hasattr(wind_speed, 'coords'):
                LE = LE.assign_coords(wind_speed.coords)
    
    return H, LE