"""
Unit conversion utilities for meteorological applications using Pint.

This module provides functions for converting between various meteorological units
leveraging the Pint library for robust unit handling and dimensional analysis.
"""

from typing import Optional, Union

import numpy as np
import pint
import xarray as xr

# Create a unit registry with meteorological units
ureg = pint.UnitRegistry()

# Add common meteorological units to the registry
ureg.define("millibar = 100 * pascal = mb")
ureg.define("torr = 133.322 * pascal = mmHg")
ureg.define("inHg = 3386.39 * pascal")
ureg.define("atmosphere = 101325 * pascal = atm")
ureg.define("knot = 0.514444 * meter / second = kt")
ureg.define("mile = 1609.34 * meter = mi")
ureg.define("nautical_mile = 1852 * meter = nm")
ureg.define("micrometer = 1e-6 * meter = um")
ureg.define("ppm = parts_per_million = 1e-6")
ureg.define("ppb = parts_per_billion = 1e-9")
ureg.define("ppt = parts_per_trillion = 1e-12")
ureg.define("ug_per_m3 = microgram / meter**3")
ureg.define("ng_per_m3 = nanogram / meter**3")


# Convenience functions for meteorological units
def pressure(
    value: Union[float, np.ndarray, xr.DataArray], unit: str, to_unit: Optional[str] = None
) -> Union[float, np.ndarray, xr.DataArray, pint.Quantity]:
    """
    Convert pressure between different units using Pint.
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")

    if isinstance(value, xr.DataArray):
        data_values = value.values
    else:
        data_values = np.asarray(value)

    quantity = data_values * ureg(unit)
    if to_unit is not None:
        converted = quantity.to(to_unit)
        result = converted.magnitude
        if isinstance(value, xr.DataArray):
            return xr.DataArray(result, coords=value.coords, dims=value.dims, attrs={"units": to_unit})
        return result
    return quantity


def temperature(
    value: Union[float, np.ndarray, xr.DataArray], unit: str, to_unit: Optional[str] = None
) -> Union[float, np.ndarray, xr.DataArray, pint.Quantity]:
    """
    Convert temperature between different units using Pint.
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")

    if isinstance(value, xr.DataArray):
        data_values = value.values
    else:
        data_values = np.asarray(value)

    # Explicit mapping to avoid Pint ambiguity with 'C' as Coulomb
    unit_map = {"C": "degC", "F": "degF", "K": "kelvin", "R": "rankine"}
    src_unit = unit_map.get(unit.upper(), unit)
    dst_unit = unit_map.get(to_unit.upper() if to_unit else None, to_unit)

    quantity = ureg.Quantity(data_values, ureg(src_unit))
    if dst_unit is not None:
        converted = quantity.to(dst_unit)
        result = converted.magnitude
        if isinstance(value, xr.DataArray):
            return xr.DataArray(result, coords=value.coords, dims=value.dims, attrs={"units": to_unit})
        return result
    return quantity


def wind_speed(
    value: Union[float, np.ndarray, xr.DataArray], unit: str, to_unit: Optional[str] = None
) -> Union[float, np.ndarray, xr.DataArray, pint.Quantity]:
    """
    Convert wind speed between different units using Pint.
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")

    if isinstance(value, xr.DataArray):
        data_values = value.values
    else:
        data_values = np.asarray(value)

    quantity = data_values * ureg(unit)
    if to_unit is not None:
        converted = quantity.to(to_unit)
        result = converted.magnitude
        if isinstance(value, xr.DataArray):
            return xr.DataArray(result, coords=value.coords, dims=value.dims, attrs={"units": to_unit})
        return result
    return quantity


def mixing_ratio(
    value: Union[float, np.ndarray, xr.DataArray], unit: str, to_unit: Optional[str] = None
) -> Union[float, np.ndarray, xr.DataArray, pint.Quantity]:
    """
    Convert mixing ratio between different units using Pint.
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")

    if isinstance(value, xr.DataArray):
        data_values = value.values
    else:
        data_values = np.asarray(value)

    quantity = data_values * ureg(unit)
    if to_unit is not None:
        converted = quantity.to(to_unit)
        result = converted.magnitude
        if isinstance(value, xr.DataArray):
            return xr.DataArray(result, coords=value.coords, dims=value.dims, attrs={"units": to_unit})
        return result
    return quantity


# Aliases for backward compatibility
def celsius_to_kelvin(t_c):
    return temperature(t_c, "C", "K")


def kelvin_to_celsius(t_k):
    return temperature(t_k, "K", "C")


def fahrenheit_to_celsius(t_f):
    return temperature(t_f, "F", "C")


def celsius_to_fahrenheit(t_c):
    return temperature(t_c, "C", "F")


def fahrenheit_to_kelvin(t_f):
    return temperature(t_f, "F", "K")


def kelvin_to_fahrenheit(t_k):
    return temperature(t_k, "K", "F")


def meters_per_second_to_knots(v_ms):
    return wind_speed(v_ms, "m/s", "knots")


def knots_to_meters_per_second(v_kt):
    return wind_speed(v_kt, "knots", "m/s")


def miles_per_hour_to_meters_per_second(v_mph):
    return wind_speed(v_mph, "mph", "m/s")


def meters_per_second_to_miles_per_hour(v_ms):
    return wind_speed(v_ms, "m/s", "mph")


def hpa_to_pa(p_hpa):
    return pressure(p_hpa, "hPa", "Pa")


def pa_to_hpa(p_pa):
    return pressure(p_pa, "Pa", "hPa")


def mb_to_pa(p_mb):
    return pressure(p_mb, "mb", "Pa")


def pa_to_mb(p_pa):
    return pressure(p_pa, "Pa", "mb")
