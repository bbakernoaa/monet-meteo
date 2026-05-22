"""
Derived meteorological parameter calculations.

This module provides functions for calculating derived meteorological parameters including:
- Heat index
- Wind chill
- Lifting condensation level
- Wet bulb temperature
- Dew point temperature
"""

from typing import Union

import numpy as np
import xarray as xr

# Import constants
from ..constants import R_d, g


def heat_index(
    temperature: Union[float, np.ndarray, xr.DataArray], relative_humidity: Union[float, np.ndarray, xr.DataArray]
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
        temp_f = (temperature - 273.15) * 9 / 5 + 32
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
            -42.379
            + 2.04901523 * temp_f
            + 10.14333127 * rh
            - 0.22475541 * temp_f * rh
            - 0.00683783 * temp_f**2
            - 0.05481717 * rh**2
            + 0.00122874 * temp_f**2 * rh
            + 0.00085282 * temp_f * rh**2
            - 0.00000199 * temp_f**2 * rh**2
        )

        # Adjust for RH < 13% and temp_f between 80-112
        mask1 = (rh < 13) & (temp_f >= 80) & (temp_f <= 112)
        if np.any(mask1):
            adjust1 = ((13 - rh) / 4) ** 0.5
            adjust2 = (17 - np.abs(temp_f - 95)) / 17
            hi_complex = np.where(mask1, hi_complex - adjust1 * adjust2, hi_complex)

        # Adjust for RH > 85% and temp_f between 80-87
        mask2 = (rh > 85) & (temp_f >= 80) & (temp_f <= 87)
        if np.any(mask2):
            adjust3 = (rh - 85) / 10
            adjust4 = (87 - temp_f) / 5
            hi_complex = np.where(mask2, hi_complex + adjust3 * adjust4, hi_complex)

        # Use the more complex formula where appropriate
        if isinstance(hi, np.ndarray):
            hi = np.where(mask, hi_complex, hi)
        elif isinstance(hi, xr.DataArray):
            hi = xr.where(mask, hi_complex, hi)
        else:
            hi = hi_complex if mask else hi

    return hi


def wind_chill(
    temperature: Union[float, np.ndarray, xr.DataArray], wind_speed: Union[float, np.ndarray, xr.DataArray]
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
        temp_f = (temperature - 273.15) * 9 / 5 + 32
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
    temperature: Union[float, np.ndarray, xr.DataArray], relative_humidity: Union[float, np.ndarray, xr.DataArray]
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


def actual_vapor_pressure(dewpoint: Union[float, np.ndarray, xr.DataArray]) -> Union[float, np.ndarray, xr.DataArray]:
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


def saturation_vapor_pressure(temperature: Union[float, np.ndarray, xr.DataArray]) -> Union[float, np.ndarray, xr.DataArray]:
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
    temperature: Union[float, np.ndarray, xr.DataArray], dewpoint: Union[float, np.ndarray, xr.DataArray]
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
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
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
    tw_c = (
        t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659))
        + np.arctan(t_c + rh_percent)
        - np.arctan(rh_percent - 1.676331)
        + 0.00391838 * rh_percent ** (3 / 2) * np.arctan(0.023101 * rh_percent)
        - 4.686035
    )

    # Convert back to Kelvin
    tw_k = tw_c + 273.15

    return tw_k


def wind_gust_diagnostic(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    heights: Union[np.ndarray, xr.DataArray],
    pbl_height: Union[float, np.ndarray, xr.DataArray],
    u10: Union[float, np.ndarray, xr.DataArray],
    v10: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate surface wind gust diagnostic.

    Based on UPP's CALGUST.f logic. It mixes down momentum from the PBL height
    to the surface, with a scaling factor that depends on the height.

    Parameters
    ----------
    u : numpy.ndarray or xarray.DataArray
        Eastward wind component profile (m/s).
    v : numpy.ndarray or xarray.DataArray
        Northward wind component profile (m/s).
    heights : numpy.ndarray or xarray.DataArray
        Heights AGL (m).
    pbl_height : float, numpy.ndarray, or xarray.DataArray
        Planetary Boundary Layer height (m).
    u10 : float, numpy.ndarray, or xarray.DataArray
        10m U wind component (m/s).
    v10 : float, numpy.ndarray, or xarray.DataArray
        10m V wind component (m/s).

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Surface wind gust speed (m/s).
    """
    sfc_wind = np.sqrt(u10**2 + v10**2)

    # Wind speed profile
    wind_profile = np.sqrt(u**2 + v**2)

    # Find wind at PBL height
    # For simplicity, we'll take the max wind within the PBL as a conservative estimate
    # similar to the loop in CALGUST.f for RAP/GFS/FV3
    mask_pbl = (heights >= 0) & (heights <= pbl_height)

    # Scaling factor from CALGUST.f: DELWIND = DELWIND * (1.0 - MIN(0.5, DZ/2000.))
    # where DZ is height above ground.
    scaling = 1.0 - np.minimum(0.5, heights / 2000.0)

    del_wind = (wind_profile - sfc_wind) * scaling
    gust_profile = sfc_wind + del_wind

    # Maximum gust within the PBL
    gust = np.nanmax(np.where(mask_pbl, gust_profile, -np.inf), axis=-3)

    # Ensure gust is at least the surface wind
    if isinstance(gust, xr.DataArray):
        gust = xr.where(gust < sfc_wind, sfc_wind, gust)
    else:
        gust = np.where(gust < sfc_wind, sfc_wind, gust)

    return gust


def visibility_diagnostic(
    temperature: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
    specific_humidity: Union[float, np.ndarray, xr.DataArray],
    cloud_water: Union[float, np.ndarray, xr.DataArray],
    rain_water: Union[float, np.ndarray, xr.DataArray],
    cloud_ice: Union[float, np.ndarray, xr.DataArray],
    snow_water: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate horizontal visibility.

    Based on UPP's CALVIS.f logic using extinction coefficients for different
    hydrometeors.

    Parameters
    ----------
    temperature : float, numpy.ndarray, or xarray.DataArray
        Air temperature (K).
    pressure : float, numpy.ndarray, or xarray.DataArray
        Air pressure (Pa).
    specific_humidity : float, numpy.ndarray, or xarray.DataArray
        Specific humidity (kg/kg).
    cloud_water : float, numpy.ndarray, or xarray.DataArray
        Cloud water mixing ratio (kg/kg).
    rain_water : float, numpy.ndarray, or xarray.DataArray
        Rain water mixing ratio (kg/kg).
    cloud_ice : float, numpy.ndarray, or xarray.DataArray
        Cloud ice mixing ratio (kg/kg).
    snow_water : float, numpy.ndarray, or xarray.DataArray
        Snow mixing ratio (kg/kg).

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Visibility (m).
    """
    # Virtual temperature and air density
    tv = temperature * (1.0 + 0.608 * specific_humidity)
    rho_air = pressure / (R_d * tv)

    # Volume of air per unit mass of dry air approx.
    # UPP uses: VOVERMD = (1.+QV)/RHOAIR + (QCLW+QRAIN)/RHOWAT + (QCLICE+QSNOW)/RHOICE
    # But for beta calculation, mass concentration C (g/m^3) is needed.
    # C = mixing_ratio * rho_air * 1000.0

    c_lc = np.maximum(0, cloud_water * rho_air * 1000.0)
    c_lp = np.maximum(0, rain_water * rho_air * 1000.0)
    c_fc = np.maximum(0, cloud_ice * rho_air * 1000.0)
    c_fp = np.maximum(0, snow_water * rho_air * 1000.0)

    # Extinction coefficients beta (km^-1)
    beta = 144.7 * c_lc**0.88 + 2.24 * c_lp**0.75 + 327.8 * c_fc**1.0 + 10.36 * c_fp**0.7776 + 1e-10

    # Visibility (km)
    # vis = -ln(0.02) / beta
    const1 = -np.log(0.02)
    vis_km = const1 / beta

    # Limit visibility to 24.135 km as in UPP
    vis_km = np.minimum(24.135, vis_km)

    return vis_km * 1000.0  # Return in meters


def sea_level_pressure_diagnostic(
    temperature_700: Union[float, np.ndarray, xr.DataArray],
    pressure_sfc: Union[float, np.ndarray, xr.DataArray],
    geopotential_sfc: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate Sea Level Pressure (SLP) reduced from surface pressure.

    Based on UPP's MAPSSLP.f logic using a constant lapse rate and 700 hPa
    temperature to estimate effective surface temperature.

    Parameters
    ----------
    temperature_700 : float, numpy.ndarray, or xarray.DataArray
        Temperature at 700 hPa (K).
    pressure_sfc : float, numpy.ndarray, or xarray.DataArray
        Surface pressure (Pa).
    geopotential_sfc : float, numpy.ndarray, or xarray.DataArray
        Surface geopotential (m^2/s^2).

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Sea Level Pressure (Pa).
    """
    lapses = 0.0065  # K/m
    rog = R_d / g
    expo = rog * lapses
    expinv = 1.0 / expo

    # Estimate surface temperature from 700 hPa temp
    # 70000 Pa is 700 hPa
    t_sfc_eff = temperature_700 * (pressure_sfc / 70000.0) ** expo

    # Reduction formula
    # PSLP = PSFC * ((TSFC + LAPSES * ZSFC) / TSFC)**EXPINV
    # ZSFC = geopotential_sfc / g

    # Using geopotential directly: LAPSES * FIS / g
    slp = pressure_sfc * ((t_sfc_eff + lapses * geopotential_sfc / g) / t_sfc_eff) ** expinv

    return slp
