"""
Thermodynamic calculations for atmospheric science.

This module provides functions for calculating thermodynamic variables including:
- Potential temperature
- Equivalent potential temperature
- Virtual temperature
- Saturation vapor pressure
- Mixing ratio
- Lapse rates
"""

import numpy as np
from typing import Union, Optional
import xarray as xr

# Import constants
from ..constants import R_d, R_v, c_pd, c_pv, g, epsilon, p0, L_v0


def potential_temperature(
    pressure: Union[float, np.ndarray, xr.DataArray], 
    temperature: Union[float, np.ndarray, xr.DataArray],
    p0: float = 1000.0
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate potential temperature using the Poisson equation.
    
    Parameters
    ----------
    pressure : float, numpy.ndarray, or xarray.DataArray
        Total atmospheric pressure (hPa or mb)
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    p0 : float, optional
        Reference pressure (hPa), default is 1000.0
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Potential temperature (K)
    """
    # Convert pressure to Pa if needed (assuming input is in hPa)
    if np.max(pressure) < 2000:  # Likely in hPa
        pressure_pa = pressure * 100
    else:
        pressure_pa = pressure
        
    # Reference pressure in Pa
    p0_pa = p0 * 100
    
    # Calculate potential temperature
    theta = temperature * (p0_pa / pressure_pa) ** (R_d / c_pd)
    
    return theta


def virtual_temperature(
    temperature: Union[float, np.ndarray, xr.DataArray],
    mixing_ratio: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate virtual temperature.
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    mixing_ratio : float, numpy.ndarray, or xarray.DataArray
        Mixing ratio (kg/kg)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Virtual temperature (K)
    """
    t_virt = temperature * (1 + (R_v / R_d - 1) * mixing_ratio)
    
    return t_virt


def saturation_vapor_pressure(
    temperature: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate saturation vapor pressure using the Clausius-Clapeyron equation.
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Saturation vapor pressure (Pa)
    """
    # Convert from K to C for the formula
    t_celsius = temperature - 273.15
    
    # Bolton (1980) formula for saturation vapor pressure over water
    e_s = 61.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))
    
    return e_s


def mixing_ratio(
    vapor_pressure: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate mixing ratio.
    
    Parameters
    ----------
    vapor_pressure : float, numpy.ndarray, or xarray.DataArray
        Vapor pressure (Pa)
    pressure : float, numpy.ndarray, or xarray.DataArray
        Total pressure (Pa)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Mixing ratio (kg/kg)
    """
    # Calculate mixing ratio
    r = epsilon * vapor_pressure / (pressure - vapor_pressure)
    
    return r


def relative_humidity(
    vapor_pressure: Union[float, np.ndarray, xr.DataArray],
    saturation_vapor_pressure: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate relative humidity.
    
    Parameters
    ----------
    vapor_pressure : float, numpy.ndarray, or xarray.DataArray
        Actual vapor pressure (Pa)
    saturation_vapor_pressure : float, numpy.ndarray, or xarray.DataArray
        Saturation vapor pressure (Pa)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Relative humidity (dimensionless, 0-1)
    """
    rh = vapor_pressure / saturation_vapor_pressure
    
    # Ensure RH is between 0 and 1
    if isinstance(rh, (np.ndarray, xr.DataArray)):
        rh = np.clip(rh, 0, 1)
    else:
        rh = min(max(rh, 0), 1)
        
    return rh


def dewpoint_from_relative_humidity(
    temperature: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate dewpoint temperature from temperature and relative humidity.
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    relative_humidity : float, numpy.ndarray, or xarray.DataArray
        Relative humidity (dimensionless, 0-1)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Dewpoint temperature (K)
    """
    # Convert to Celsius for calculation
    t_c = temperature - 273.15
    
    # Calculate saturation vapor pressure
    e_s = saturation_vapor_pressure(temperature)
    
    # Calculate actual vapor pressure
    e = relative_humidity * e_s
    
    # Calculate dewpoint using inverse of Clausius-Clapeyron equation
    # Bolton (1980) formula
    t_d_c = (243.5 * np.log(e / 611.2)) / (17.67 - np.log(e / 611.2))
    
    # Convert back to Kelvin
    t_d = t_d_c + 273.15
    
    return t_d


def equivalent_potential_temperature(
    pressure: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
    mixing_ratio_val: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate equivalent potential temperature.
    
    Parameters
    ----------
    pressure : float, numpy.ndarray, or xarray.DataArray
        Total atmospheric pressure (Pa)
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    mixing_ratio_val : float, numpy.ndarray, or xarray.DataArray
        Mixing ratio (kg/kg)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Equivalent potential temperature (K)
    """
    # Calculate saturation vapor pressure
    e_s = saturation_vapor_pressure(temperature)
    
    # Calculate saturation mixing ratio
    rs = mixing_ratio(e_s, pressure)
    
    # Bolton (1980) formula for equivalent potential temperature
    theta_e = (temperature * (100000.0 / pressure) ** (R_d / c_pd) *
               np.exp((3036.0 / temperature - 1.78) * mixing_ratio_val * (1 + 0.448 * mixing_ratio_val)))
    
    return theta_e


def wet_bulb_temperature(
    temperature: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate wet bulb temperature using Stull (2011) approximation.
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    pressure : float, numpy.ndarray, or xarray.DataArray
        Total atmospheric pressure (Pa)
    relative_humidity : float, numpy.ndarray, or xarray.DataArray
        Relative humidity (dimensionless, 0-1)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Wet bulb temperature (K)
    """
    # Convert temperature to Celsius for calculation
    t_c = temperature - 273.15
    
    # Convert pressure to hPa
    p_hpa = pressure / 100.0
    
    # Calculate vapor pressure
    e_s = saturation_vapor_pressure(temperature)
    e = relative_humidity * e_s / 100.0  # Convert from Pa to hPa
    
    # Stull (2011) approximation for wet bulb temperature
    tw_c = t_c * np.arctan(0.151977 * np.sqrt(relative_humidity * 100 + 8.313659)) + \
           np.arctan(t_c + relative_humidity * 100) - \
           np.arctan(relative_humidity * 100 - 1.676331) + \
           0.00391838 * (relative_humidity * 100) ** (3/2) * \
           np.arctan(0.023101 * (relative_humidity * 100)) - \
           4.686035
    
    # Convert back to Kelvin
    tw_k = tw_c + 273.15
    
    return tw_k


def dry_lapse_rate() -> float:
    """
    Calculate dry adiabatic lapse rate.
    
    Returns
    -------
    float
        Dry adiabatic lapse rate (K/m)
    """
    return g / c_pd


def moist_lapse_rate(
    temperature: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate moist adiabatic lapse rate.
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    pressure : float, numpy.ndarray, or xarray.DataArray
        Atmospheric pressure (Pa)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Moist adiabatic lapse rate (K/m)
    """
    # Calculate saturation vapor pressure
    e_s = saturation_vapor_pressure(temperature)
    
    # Calculate saturation mixing ratio
    rs = mixing_ratio(e_s, pressure)
    
    # Calculate the moist adiabatic lapse rate
    numerator = (g / c_pd) * (1 + (L_v0 * rs) / (R_d * temperature))
    denominator = 1 + (L_v0**2 * rs * epsilon) / (c_pd * R_d * temperature**2)
    
    gamma_m = numerator / denominator
    
    return gamma_m


def lifting_condensation_level(
    temperature: Union[float, np.ndarray, xr.DataArray],
    dewpoint: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the lifting condensation level (LCL).
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K)
    dewpoint : float, numpy.ndarray, or xarray.DataArray
        Dewpoint temperature (K)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Lifting condensation level height (m)
    """
    # Convert to Celsius for calculation
    t_c = temperature - 273.15
    td_c = dewpoint - 273.15
    
    # Calculate LCL temperature (K)
    t_lcl = t_c - (t_c - td_c) / 3.0 + 273.15
    
    # Calculate LCL height (m) using approximate formula
    # LCL height ≈ 125 * (T - Td) where T and Td are in Celsius
    lcl_height = 125.0 * (t_c - td_c)
    
    # Ensure positive heights
    if isinstance(lcl_height, (np.ndarray, xr.DataArray)):
        lcl_height = np.maximum(lcl_height, 0)
    else:
        lcl_height = max(lcl_height, 0)
    
    return lcl_height