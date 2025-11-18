"""
Derived meteorological parameter calculations.

This module provides functions for calculating derived meteorological parameters including:
- Heat index
- Wind chill
- Lifting condensation level
- Wet bulb temperature
- Dew point temperature
"""

import numpy as np
from typing import Union
import xarray as xr

# Import constants
from ..constants import R_d, R_v, c_pd, g, epsilon


def heat_index(
    temperature: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the heat index using the Rothfusz regression.
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (°F)
    relative_humidity : float, numpy.ndarray, or xarray.DataArray
        Relative humidity (0-100 scale)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Heat index (°F)
    """
    # Ensure temperature is in Fahrenheit
    # If in Kelvin, convert to Fahrenheit
    if np.max(temperature) > 273.15 * 2:  # Likely in Kelvin
        temp_f = (temperature - 273.15) * 9/5 + 32
    else:
        temp_f = temperature  # Assume already in Fahrenheit
    
    # Ensure relative humidity is in 0-100 scale
    if np.max(relative_humidity) <= 1:  # Likely in 0-1 scale
        rh = relative_humidity * 100
    else:
        rh = relative_humidity  # Assume already in 0-100 scale
    
    # Calculate heat index using Rothfusz regression
    # Only valid for temp_f >= 80°F and rh >= 40%
    hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (rh * 0.094))
    
    # For conditions where heat index is more complex
    mask = (temp_f >= 80) & (rh >= 40)
    
    if np.any(mask):
        # More accurate formula for high temperature and humidity
        hi_complex = (
            -42.379 +
            2.04901523 * temp_f +
            10.14333127 * rh -
            0.22475541 * temp_f * rh -
            0.0683783 * temp_f**2 -
            0.05481717 * rh**2 +
            0.0122874 * temp_f**2 * rh +
            0.00085282 * temp_f * rh**2 -
            0.0000199 * temp_f**2 * rh**2
        )
        
        # Adjust for RH < 13% and temp_f between 80-112
        adjust1 = ((13 - rh) / 4) ** 0.5
        adjust2 = (17 - np.abs(temp_f - 95)) / 17
        hi_complex = hi_complex - adjust1 * adjust2
        
        # Adjust for RH > 85% and temp_f between 80-87
        adjust3 = (rh - 85) / 10
        adjust4 = (87 - temp_f) / 5
        hi_complex = hi_complex + adjust3 * adjust4
        
        # Use the more complex formula where appropriate
        if isinstance(hi, np.ndarray):
            hi = np.where(mask, hi_complex, hi)
        elif isinstance(hi, xr.DataArray):
            hi = xr.where(mask, hi_complex, hi)
        else:
            hi = hi_complex if mask else hi
    
    return hi


def wind_chill(
    temperature: Union[float, np.ndarray, xr.DataArray],
    wind_speed: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate the wind chill temperature.
    
    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (°F)
    wind_speed : float, numpy.ndarray, or xarray.DataArray
        Wind speed (mph)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Wind chill temperature (°F)
    """
    # Ensure temperature is in Fahrenheit
    if np.max(temperature) > 273.15 * 2:  # Likely in Kelvin
        temp_f = (temperature - 273.15) * 9/5 + 32
    else:
        temp_f = temperature  # Assume already in Fahrenheit
    
    # Ensure wind speed is in mph
    if np.max(wind_speed) > 10:  # Likely in m/s, convert to mph
        wind_mph = wind_speed * 2.23694
    else:
        wind_mph = wind_speed  # Assume already in mph
    
    # Wind chill is only defined for T <= 50°F and wind_speed >= 3 mph
    mask = (temp_f <= 50) & (wind_mph >= 3)
    
    # Calculate wind chill using the new formula (2001)
    wc = 35.74 + 0.6215 * temp_f - 35.75 * (wind_mph**0.16) + 0.4275 * temp_f * (wind_mph**0.16)
    
    # Where wind chill is not defined, return the actual temperature
    if isinstance(wc, np.ndarray):
        result = np.where(mask, wc, temp_f)
    elif isinstance(wc, xr.DataArray):
        result = xr.where(mask, wc, temp_f)
    else:
        result = wc if mask else temp_f
    
    return result


def dewpoint_temperature(
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
        Relative humidity (0-1 scale)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Dewpoint temperature (K)
    """
    # Ensure relative humidity is in 0-1 scale
    if np.max(relative_humidity) > 1:  # Likely in 0-100 scale
        rh = relative_humidity / 100
    else:
        rh = relative_humidity  # Assume already in 0-1 scale
    
    # Convert temperature to Celsius for calculation
    t_c = temperature - 273.15
    
    # Calculate dewpoint using Magnus formula
    a = 17.27
    b = 237.7
    alpha = ((a * t_c) / (b + t_c)) + np.log(rh)
    t_d_c = (b * alpha) / (a - alpha)
    
    # Convert back to Kelvin
    t_d = t_d_c + 273.15
    
    # Handle case where RH = 0 (log(0) is undefined)
    if isinstance(t_d, np.ndarray):
        t_d = np.where(rh == 0, -273.15, t_d)  # -Infinity in Kelvin
    elif isinstance(t_d, xr.DataArray):
        t_d = xr.where(rh == 0, -273.15, t_d)
    elif rh == 0:
        t_d = -273.15
    
    return t_d


def actual_vapor_pressure(
    dewpoint: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate actual vapor pressure from dewpoint temperature.
    
    Parameters
    ----------
    dewpoint : float, numpy.ndarray, or xarray.DataArray
        Dewpoint temperature (K)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Actual vapor pressure (Pa)
    """
    # Convert dewpoint to Celsius
    t_d_c = dewpoint - 273.15
    
    # Calculate actual vapor pressure using the Clausius-Clapeyron equation
    # Bolton (1980) formula
    e = 611.2 * np.exp(17.67 * t_d_c / (t_d_c + 243.5))
    
    return e


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
    e_s = 611.2 * np.exp(17.67 * t_celsius / (t_celsius + 243.5))
    
    return e_s


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
    
    # Calculate LCL height (m) using the exact formula
    # LCL height ≈ 125 * (T - Td) where T and Td are in Celsius
    lcl_height = 125.0 * (t_c - td_c)
    
    # Ensure positive heights
    if isinstance(lcl_height, (np.ndarray, xr.DataArray)):
        lcl_height = np.maximum(lcl_height, 0)
    else:
        lcl_height = max(lcl_height, 0)
    
    return lcl_height


def wet_bulb_temperature(
    temperature: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate wet bulb temperature using Stull (201) approximation.
    
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
    
    # Ensure relative humidity is in 0-100 scale for the formula
    if np.max(relative_humidity) <= 1:  # Likely in 0-1 scale
        rh_percent = relative_humidity * 100
    else:
        rh_percent = relative_humidity  # Assume already in 0-100 scale
    
    # Stull (201) approximation for wet bulb temperature in Celsius
    tw_c = t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) + \
           np.arctan(t_c + rh_percent) - \
           np.arctan(rh_percent - 1.676331) + \
           0.00391838 * rh_percent ** (3/2) * \
           np.arctan(0.023101 * rh_percent) - \
           4.686035
    
    # Convert back to Kelvin
    tw_k = tw_c + 273.15
    
    return tw_k