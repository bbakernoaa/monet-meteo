"""
Derived meteorological parameters.
Includes heat index, wind chill, etc.
"""

import numpy as np
import xarray as xr
from typing import Union, Optional, Any
from ..thermodynamics import thermodynamic_calculations as thermo


def _update_history(obj: Any, msg: str) -> Any:
    """Update history attribute."""
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
    return obj


def heat_index(
    temperature: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate Heat Index."""
    if isinstance(temperature, xr.DataArray):
        T_k = xr.where(temperature < 150, temperature + 273.15, temperature)
        T_f = (T_k - 273.15) * 9 / 5 + 32
        RH = xr.where(
            relative_humidity <= 1.0, relative_humidity * 100.0, relative_humidity
        )
    else:
        T_val = np.asanyarray(temperature)
        T_k = np.where(T_val < 150, T_val + 273.15, T_val)
        T_f = (T_k - 273.15) * 9 / 5 + 32
        RH_val = np.asanyarray(relative_humidity)
        RH = np.where(RH_val <= 1.0, RH_val * 100.0, RH_val)

    # Simple formula
    HI_simple = 0.5 * (T_f + 61.0 + ((T_f - 68.0) * 1.2) + (RH * 0.094))

    # Full regression
    HI_f = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * RH
        - 0.22475541 * T_f * RH
        - 0.00683783 * T_f**2
        - 0.05481717 * RH**2
        + 0.00122874 * T_f**2 * RH
        + 0.00085282 * T_f * RH**2
        - 0.00000199 * T_f**2 * RH**2
    )

    # Adjustments
    adj1 = ((13.0 - RH) / 4.0) * np.sqrt(
        np.maximum(0, (17.0 - np.abs(T_f - 95.0)) / 17.0)
    )
    adj2 = ((RH - 85.0) / 10.0) * ((87.0 - T_f) / 5.0)

    if isinstance(T_f, xr.DataArray):
        HI_reg = xr.where((RH < 13) & (T_f >= 80) & (T_f <= 112), HI_f - adj1, HI_f)
        HI_reg = xr.where((RH > 85) & (T_f >= 80) & (T_f <= 87), HI_reg + adj2, HI_reg)

        # decision
        res_f = xr.where(T_f >= 80, HI_reg, T_f)
        # Ensure HI >= T
        res_f = xr.where(res_f < T_f, T_f, res_f)
    else:
        HI_reg = np.where((RH < 13) & (T_f >= 80) & (T_f <= 112), HI_f - adj1, HI_f)
        HI_reg = np.where((RH > 85) & (T_f >= 80) & (T_f <= 87), HI_reg + adj2, HI_reg)
        res_f = np.where(T_f >= 80, HI_reg, T_f)
        res_f = np.maximum(res_f, T_f)

    if isinstance(temperature, xr.DataArray):
        res_k = (res_f - 32) * 5 / 9 + 273.15
        res = xr.where(temperature < 150, res_k - 273.15, res_k)
        return _update_history(res, "Calculated heat_index.")
    else:
        T_val = np.asanyarray(temperature)
        res_k = (res_f - 32) * 5 / 9 + 273.15
        return np.where(T_val < 150, res_k - 273.15, res_k)


def wind_chill(
    temperature: Union[float, np.ndarray, xr.DataArray],
    wind_speed: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate Wind Chill Index."""
    if isinstance(temperature, xr.DataArray):
        T_k = xr.where(temperature < 150, temperature + 273.15, temperature)
        T_f = (T_k - 273.15) * 9 / 5 + 32
        V_mph = wind_speed * 2.23694
    else:
        T_val = np.asanyarray(temperature)
        T_k = np.where(T_val < 150, T_val + 273.15, T_val)
        T_f = (T_k - 273.15) * 9 / 5 + 32
        V_mph = np.asanyarray(wind_speed) * 2.23694

    WC_f = 35.74 + 0.6215 * T_f - 35.75 * V_mph**0.16 + 0.4275 * T_f * V_mph**0.16

    if isinstance(T_f, xr.DataArray):
        WC_f = xr.where((T_f <= 50) & (V_mph > 3), WC_f, T_f)
    else:
        WC_f = np.where((T_f <= 50) & (V_mph > 3), WC_f, T_f)

    if isinstance(temperature, xr.DataArray):
        res_k = (WC_f - 32) * 5 / 9 + 273.15
        res = xr.where(temperature < 150, res_k - 273.15, res_k)
        return _update_history(res, "Calculated wind_chill.")
    else:
        T_val = np.asanyarray(temperature)
        res_k = (WC_f - 32) * 5 / 9 + 273.15
        return np.where(T_val < 150, res_k - 273.15, res_k)


def dewpoint(
    temperature: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate dewpoint."""
    res_k = thermo.dewpoint_from_relative_humidity(temperature, relative_humidity)
    if isinstance(temperature, xr.DataArray):
        return xr.where(temperature < 150, res_k - 273.15, res_k)
    else:
        T_val = np.asanyarray(temperature)
        return np.where(T_val < 150, res_k - 273.15, res_k)


def dewpoint_temperature(
    temperature: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Alias for dewpoint."""
    return dewpoint(temperature, relative_humidity)


def mixing_ratio(
    pressure: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate mixing ratio."""
    es = thermo.saturation_vapor_pressure(temperature)
    if isinstance(relative_humidity, xr.DataArray):
        rh_frac = xr.where(
            relative_humidity > 1.0, relative_humidity / 100.0, relative_humidity
        )
    else:
        rh_val = np.asanyarray(relative_humidity)
        rh_frac = np.where(rh_val > 1.0, rh_val / 100.0, rh_val)

    e = rh_frac * es
    return thermo.mixing_ratio(e, pressure)


def lifting_condensation_level(
    temperature: Union[float, np.ndarray, xr.DataArray],
    dewpoint: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate LCL height."""
    return thermo.lifting_condensation_level(temperature, dewpoint)


def wet_bulb_temperature(
    temperature: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate wet bulb temperature."""
    return thermo.wet_bulb_temperature(temperature, pressure, relative_humidity)


def saturation_vapor_pressure(
    temperature: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate saturation vapor pressure."""
    return thermo.saturation_vapor_pressure(temperature)


def actual_vapor_pressure(
    temperature: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate actual vapor pressure."""
    es = thermo.saturation_vapor_pressure(temperature)
    if isinstance(relative_humidity, xr.DataArray):
        rh_frac = xr.where(
            relative_humidity > 1.0, relative_humidity / 100.0, relative_humidity
        )
    else:
        rh_val = np.asanyarray(relative_humidity)
        rh_frac = np.where(rh_val > 1.0, rh_val / 100.0, rh_val)
    return rh_frac * es
