"""
Statistical operations for atmospheric data analysis and micrometeorology calculations.

This module implements Monin-Obukhov similarity theory, surface energy balance,
turbulent flux calculations, and atmospheric stability parameters with
xarray/dask support.
"""

from typing import Optional, Union

import numpy as np
import xarray as xr

# Don't import constants at module level to avoid conflicts when using 'from .statistical import *'
# Instead, import the constants module and access via module name
from .. import constants


def bulk_richardson_number(
    u_wind: Union[float, np.ndarray, xr.DataArray],
    v_wind: Union[float, np.ndarray, xr.DataArray],
    potential_temperature: Union[float, np.ndarray, xr.DataArray],
    height: Union[float, np.ndarray, xr.DataArray],
    method: str = "standard",
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate bulk Richardson number for atmospheric stability.
    """
    # Calculate wind speed
    wind_speed = np.sqrt(u_wind**2 + v_wind**2)

    # Calculate potential temperature difference
    theta_diff = (
        np.gradient(potential_temperature, axis=-1)
        if (hasattr(potential_temperature, "ndim") and potential_temperature.ndim > 0)
        else np.gradient(potential_temperature)
    )
    height_diff = np.gradient(height, axis=-1) if (hasattr(height, "ndim") and height.ndim > 0) else np.gradient(height)

    # Calculate bulk Richardson number
    if method == "standard":
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
    latent_heat_flux: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the Monin-Obukhov length.
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
    denominator = constants.k * constants.g * kinematic_heat_flux
    L = -(friction_velocity**3) * temperature / (denominator + 1e-12)

    return L


# Alias for backward compatibility
obukhov_length = monin_obukhov_length


def stability_parameter(
    height: Union[float, np.ndarray, xr.DataArray], obukhov_length: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the stability parameter (z/L).
    """
    return height / (obukhov_length + 1e-12)


def psi_momentum(stability_parameter: Union[float, np.ndarray, xr.DataArray]) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Stability correction function for momentum.
    """
    zL = np.asarray(stability_parameter)
    psi_m = np.zeros_like(zL)
    stable_idx = zL >= 0
    if np.any(stable_idx):
        a = 6.1
        b = 2.5
        psi_m[stable_idx] = -a * zL[stable_idx] - b * np.log(1 + zL[stable_idx] / b)
    unstable_idx = zL < 0
    if np.any(unstable_idx):
        x = np.sqrt(1 - 16 * zL[unstable_idx])
        psi_m[unstable_idx] = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + np.pi / 2
    return psi_m


def psi_heat(stability_parameter: Union[float, np.ndarray, xr.DataArray]) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Stability correction function for heat.
    """
    zL = np.asarray(stability_parameter)
    psi_h = np.zeros_like(zL)
    stable_idx = zL >= 0
    if np.any(stable_idx):
        a = 6.1
        psi_h[stable_idx] = -a * zL[stable_idx]
    unstable_idx = zL < 0
    if np.any(unstable_idx):
        x = np.sqrt(1 - 16 * zL[unstable_idx])
        psi_h[unstable_idx] = 2 * np.log((1 + x) / 2)
    return psi_h


def aerodynamic_resistance(
    height: Union[float, np.ndarray, xr.DataArray],
    roughness_length: Union[float, np.ndarray, xr.DataArray],
    stability_parameter: Union[float, np.ndarray, xr.DataArray],
    displacement_height: float = 0.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate aerodynamic resistance.
    """
    psi_m = psi_momentum(stability_parameter)
    psi_m0 = psi_momentum(0.0)
    return np.log((height - displacement_height) / roughness_length) - psi_m + psi_m0


def surface_energy_balance(
    net_radiation: Union[float, np.ndarray, xr.DataArray],
    soil_heat_flux: Union[float, np.ndarray, xr.DataArray],
    sensible_heat_flux: Union[float, np.ndarray, xr.DataArray],
    latent_heat_flux: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate surface energy balance closure.
    """
    return net_radiation - (soil_heat_flux + sensible_heat_flux + latent_heat_flux)


def sensible_heat_flux(
    air_temperature: Union[float, np.ndarray, xr.DataArray],
    surface_temperature: Union[float, np.ndarray, xr.DataArray],
    aerodynamic_resistance: Union[float, np.ndarray, xr.DataArray],
    air_density: Union[float, np.ndarray, xr.DataArray] = 1.225,
    specific_heat: Union[float, np.ndarray, xr.DataArray] = 1004.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate sensible heat flux.
    """
    return air_density * specific_heat * (surface_temperature - air_temperature) / aerodynamic_resistance


def latent_heat_flux(
    vapor_pressure_air: Union[float, np.ndarray, xr.DataArray],
    vapor_pressure_surface: Union[float, np.ndarray, xr.DataArray],
    aerodynamic_resistance: Union[float, np.ndarray, xr.DataArray],
    air_density: Union[float, np.ndarray, xr.DataArray] = 1.225,
    latent_heat_vaporization: Union[float, np.ndarray, xr.DataArray] = 2.501e6,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate latent heat flux.
    """
    return (
        air_density
        * latent_heat_vaporization
        * (vapor_pressure_surface - vapor_pressure_air)
        / (constants.R_d * 293.0 * aerodynamic_resistance + 1e-12)
    )


def friction_velocity_from_wind(
    wind_speed: Union[float, np.ndarray, xr.DataArray],
    height: Union[float, np.ndarray, xr.DataArray],
    roughness_length: Union[float, np.ndarray, xr.DataArray],
    stability_parameter: Union[float, np.ndarray, xr.DataArray] = 0.0,
    displacement_height: float = 0.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate friction velocity from wind speed.
    """
    psi_m = psi_momentum(stability_parameter)
    psi_m0 = psi_momentum(roughness_length / (height + 1e-12))
    log_term = np.log((height - displacement_height) / roughness_length)
    return constants.k * wind_speed / (log_term - psi_m + psi_m0 + 1e-12)


# Alias for backward compatibility
friction_velocity = friction_velocity_from_wind


# The following functions are used by the test suite but were missing implementions or aliases
def momentum_flux(u_prime, w_prime, air_density):
    return -air_density * np.mean(u_prime * w_prime)


def turbulence_kinetic_energy(u_prime, v_prime, w_prime):
    return 0.5 * (np.mean(u_prime**2) + np.mean(v_prime**2) + np.mean(w_prime**2))


def standard_deviation(data):
    if len(data) == 0:
        return np.nan
    if len(data) == 1:
        return 0.0
    return np.std(data, ddof=1)


def correlation_coefficient(x, y):
    if len(x) == 0:
        return np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(x, y)[0, 1]
    return c


def covariance(x, y):
    if len(x) == 0:
        return np.nan
    return np.cov(x, y, ddof=1)[0, 1]


def atmospheric_boundary_layer_height(surface_temperature, potential_temperature_gradient, wind_speed, height, method="bulk_richardson"):
    return np.max(height)  # Placeholder


def turbulence_intensity(wind_speed, wind_speed_std):
    return wind_speed_std / (np.abs(wind_speed) + 1e-12)


def obukhov_stability_parameter(friction_velocity, air_temperature, kinematic_heat_flux):
    return -(friction_velocity**3) * air_temperature / (constants.k * constants.g * kinematic_heat_flux + 1e-12)


def xarray_bulk_richardson_number(u_wind, v_wind, potential_temperature, height, method="standard"):
    return xr.apply_ufunc(bulk_richardson_number, u_wind, v_wind, potential_temperature, height, method)


def xarray_monin_obukhov_length(friction_velocity, temperature, air_density, specific_heat, sensible_heat_flux, latent_heat_flux=None):
    return xr.apply_ufunc(monin_obukhov_length, friction_velocity, temperature, air_density, specific_heat, sensible_heat_flux, latent_heat_flux)


def xarray_surface_energy_balance(net_radiation, soil_heat_flux, sensible_heat_flux, latent_heat_flux):
    return xr.apply_ufunc(surface_energy_balance, net_radiation, soil_heat_flux, sensible_heat_flux, latent_heat_flux)


def xarray_turbulent_fluxes_from_similarity(
    wind_speed,
    air_temperature,
    surface_temperature,
    vapor_pressure_air,
    vapor_pressure_surface,
    height,
    roughness_length,
    stability_parameter,
    displacement_height=0.0,
):
    r_a = aerodynamic_resistance(height, roughness_length, stability_parameter, displacement_height)
    H = sensible_heat_flux(air_temperature, surface_temperature, r_a)
    LE = latent_heat_flux(vapor_pressure_air, vapor_pressure_surface, r_a)
    return H, LE
