"""
Unit conversion utilities for meteorological applications.

This module provides functions for converting between various meteorological units
including pressure, temperature, distance, wind speed, and other common atmospheric
measurements.
"""

import numpy as np
from typing import Union
import xarray as xr

# Conversion constants
PA_TO_HPA = 0.01
HPA_TO_PA = 100.0
KELVIN_OFFSET = 273.15
MPS_TO_KNOTS = 1.94384
KNOTS_TO_MPS = 0.514444
MPS_TO_KMH = 3.6
KMH_TO_MPS = 0.277778
M_TO_KM = 0.001
KM_TO_M = 1000.0
M_TO_FT = 3.28084
FT_TO_M = 0.3048
PA_TO_MMHG = 0.00750062
MMHG_TO_PA = 133.322
PA_TO_INHG = 0.0002953
INHG_TO_PA = 3386.39


def convert_pressure(
    value: Union[float, np.ndarray, xr.DataArray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert pressure between different units.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Pressure value(s) to convert
    from_unit : str
        Source unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')
    to_unit : str
        Target unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Converted pressure value(s)
    """
    # Input validation
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Unit parameters must be strings")
    
    if from_unit.lower() not in ['pa', 'hpa', 'mb', 'mmhg', 'inhg', 'atm']:
        raise ValueError(f"Unsupported pressure unit: {from_unit}. "
                        f"Supported units: 'Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm'")
    
    if to_unit.lower() not in ['pa', 'hpa', 'mb', 'mmhg', 'inhg', 'atm']:
        raise ValueError(f"Unsupported pressure unit: {to_unit}. "
                        f"Supported units: 'Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm'")
    
    # Validate value
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Pressure cannot be negative")
    elif isinstance(value, np.ndarray):
        if np.any(value < 0):
            raise ValueError("Pressure cannot be negative")
    elif isinstance(value, xr.DataArray):
        if np.any(value.values < 0):
            raise ValueError("Pressure cannot be negative")
    
    # Convert from_unit to Pa
    if from_unit.lower() in ['hpa', 'mb']:
        value_pa = value * HPA_TO_PA
    elif from_unit.lower() == 'mmhg':
        value_pa = value / PA_TO_MMHG
    elif from_unit.lower() == 'inhg':
        value_pa = value / PA_TO_INHG
    elif from_unit.lower() == 'atm':
        value_pa = value * 101325.0  # 1 atm = 101325 Pa
    elif from_unit.lower() == 'pa':
        value_pa = value
    else:
        raise ValueError(f"Unsupported pressure unit: {from_unit}")
    
    # Convert Pa to to_unit
    if to_unit.lower() in ['hpa', 'mb']:
        return value_pa * PA_TO_HPA
    elif to_unit.lower() == 'mmhg':
        return value_pa * PA_TO_MMHG
    elif to_unit.lower() == 'inhg':
        return value_pa * PA_TO_INHG
    elif to_unit.lower() == 'atm':
        return value_pa / 101325.0
    elif to_unit.lower() == 'pa':
        return value_pa
    else:
        raise ValueError(f"Unsupported pressure unit: {to_unit}")


def convert_temperature(
    value: Union[float, np.ndarray, xr.DataArray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert temperature between different units.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Temperature value(s) to convert
    from_unit : str
        Source unit ('K', 'C', 'F')
    to_unit : str
        Target unit ('K', 'C', 'F')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Converted temperature value(s)
    """
    # Input validation
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Unit parameters must be strings")
    
    if from_unit.upper() not in ['K', 'C', 'F']:
        raise ValueError(f"Unsupported temperature unit: {from_unit}. "
                        f"Supported units: 'K', 'C', 'F'")
    
    if to_unit.upper() not in ['K', 'C', 'F']:
        raise ValueError(f"Unsupported temperature unit: {to_unit}. "
                        f"Supported units: 'K', 'C', 'F'")
    
    # Validate value - check for physically impossible temperatures
    if isinstance(value, (int, float)):
        if from_unit.upper() == 'K' and value < 0:
            raise ValueError("Temperature in Kelvin cannot be negative")
        elif from_unit.upper() == 'C' and value < -273.15:
            raise ValueError("Temperature in Celsius cannot be below absolute zero (-273.15°C)")
        elif from_unit.upper() == 'F' and value < -459.67:
            raise ValueError("Temperature in Fahrenheit cannot be below absolute zero (-459.67°F)")
    elif isinstance(value, np.ndarray):
        if from_unit.upper() == 'K' and np.any(value < 0):
            raise ValueError("Temperature in Kelvin cannot be negative")
        elif from_unit.upper() == 'C' and np.any(value < -273.15):
            raise ValueError("Temperature in Celsius cannot be below absolute zero (-273.15°C)")
        elif from_unit.upper() == 'F' and np.any(value < -459.67):
            raise ValueError("Temperature in Fahrenheit cannot be below absolute zero (-459.67°F)")
    elif isinstance(value, xr.DataArray):
        if from_unit.upper() == 'K' and np.any(value.values < 0):
            raise ValueError("Temperature in Kelvin cannot be negative")
        elif from_unit.upper() == 'C' and np.any(value.values < -273.15):
            raise ValueError("Temperature in Celsius cannot be below absolute zero (-273.15°C)")
        elif from_unit.upper() == 'F' and np.any(value.values < -459.67):
            raise ValueError("Temperature in Fahrenheit cannot be below absolute zero (-459.67°F)")
    
    # Convert to Kelvin first
    if from_unit.upper() == 'C':
        temp_k = value + KELVIN_OFFSET
    elif from_unit.upper() == 'F':
        temp_k = (value - 32) * 5/9 + KELVIN_OFFSET
    elif from_unit.upper() == 'K':
        temp_k = value
    else:
        raise ValueError(f"Unsupported temperature unit: {from_unit}")
    
    # Convert Kelvin to target unit
    if to_unit.upper() == 'C':
        return temp_k - KELVIN_OFFSET
    elif to_unit.upper() == 'F':
        return (temp_k - KELVIN_OFFSET) * 9/5 + 32
    elif to_unit.upper() == 'K':
        return temp_k
    else:
        raise ValueError(f"Unsupported temperature unit: {to_unit}")


def convert_distance(
    value: Union[float, np.ndarray, xr.DataArray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert distance between different units.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Distance value(s) to convert
    from_unit : str
        Source unit ('m', 'km', 'ft', 'mi', 'nm')
    to_unit : str
        Target unit ('m', 'km', 'ft', 'mi', 'nm')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Converted distance value(s)
    """
    # Input validation
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Unit parameters must be strings")
    
    if from_unit.lower() not in ['m', 'km', 'ft', 'mi', 'nm']:
        raise ValueError(f"Unsupported distance unit: {from_unit}. "
                        f"Supported units: 'm', 'km', 'ft', 'mi', 'nm'")
    
    if to_unit.lower() not in ['m', 'km', 'ft', 'mi', 'nm']:
        raise ValueError(f"Unsupported distance unit: {to_unit}. "
                        f"Supported units: 'm', 'km', 'ft', 'mi', 'nm'")
    
    # Validate value - distance cannot be negative
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Distance cannot be negative")
    elif isinstance(value, np.ndarray):
        if np.any(value < 0):
            raise ValueError("Distance cannot be negative")
    elif isinstance(value, xr.DataArray):
        if np.any(value.values < 0):
            raise ValueError("Distance cannot be negative")
    
    # Convert to meters first
    if from_unit.lower() == 'km':
        value_m = value * KM_TO_M
    elif from_unit.lower() == 'ft':
        value_m = value * FT_TO_M
    elif from_unit.lower() == 'mi':
        value_m = value * 1609.34  # 1 mile = 1609.34 meters
    elif from_unit.lower() == 'nm':
        value_m = value * 1852.0  # 1 nautical mile = 1852 meters
    elif from_unit.lower() == 'm':
        value_m = value
    else:
        raise ValueError(f"Unsupported distance unit: {from_unit}")
    
    # Convert meters to target unit
    if to_unit.lower() == 'km':
        return value_m * M_TO_KM
    elif to_unit.lower() == 'ft':
        return value_m * M_TO_FT
    elif to_unit.lower() == 'mi':
        return value_m / 1609.34
    elif to_unit.lower() == 'nm':
        return value_m / 1852.0
    elif to_unit.lower() == 'm':
        return value_m
    else:
        raise ValueError(f"Unsupported distance unit: {to_unit}")


def convert_wind_speed(
    value: Union[float, np.ndarray, xr.DataArray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert wind speed between different units.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Wind speed value(s) to convert
    from_unit : str
        Source unit ('m/s', 'knots', 'km/h', 'mph')
    to_unit : str
        Target unit ('m/s', 'knots', 'km/h', 'mph')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Converted wind speed value(s)
    """
    # Input validation
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Unit parameters must be strings")
    
    if from_unit.lower() not in ['m/s', 'mps', 'knots', 'kt', 'km/h', 'kmh', 'mph']:
        raise ValueError(f"Unsupported wind speed unit: {from_unit}. "
                        f"Supported units: 'm/s', 'mps', 'knots', 'kt', 'km/h', 'kmh', 'mph'")
    
    if to_unit.lower() not in ['m/s', 'mps', 'knots', 'kt', 'km/h', 'kmh', 'mph']:
        raise ValueError(f"Unsupported wind speed unit: {to_unit}. "
                        f"Supported units: 'm/s', 'mps', 'knots', 'kt', 'km/h', 'kmh', 'mph'")
    
    # Validate value - wind speed cannot be negative
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Wind speed cannot be negative")
    elif isinstance(value, np.ndarray):
        if np.any(value < 0):
            raise ValueError("Wind speed cannot be negative")
    elif isinstance(value, xr.DataArray):
        if np.any(value.values < 0):
            raise ValueError("Wind speed cannot be negative")
    
    # Convert to m/s first
    if from_unit.lower() in ['knots', 'kt']:
        value_mps = value * KNOTS_TO_MPS
    elif from_unit.lower() in ['km/h', 'kmh']:
        value_mps = value * KMH_TO_MPS
    elif from_unit.lower() in ['mph']:
        value_mps = value * 0.4704  # 1 mph = 0.4704 m/s
    elif from_unit.lower() in ['m/s', 'mps']:
        value_mps = value
    else:
        raise ValueError(f"Unsupported wind speed unit: {from_unit}")
    
    # Convert m/s to target unit
    if to_unit.lower() in ['knots', 'kt']:
        return value_mps * MPS_TO_KNOTS
    elif to_unit.lower() in ['km/h', 'kmh']:
        return value_mps * MPS_TO_KMH
    elif to_unit.lower() in ['mph']:
        return value_mps / 0.44704
    elif to_unit.lower() in ['m/s', 'mps']:
        return value_mps
    else:
        raise ValueError(f"Unsupported wind speed unit: {to_unit}")


def convert_mixing_ratio(
    value: Union[float, np.ndarray, xr.DataArray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert mixing ratio between different units.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Mixing ratio value(s) to convert
    from_unit : str
        Source unit ('kg/kg', 'g/kg', 'ppm', 'ppt')
    to_unit : str
        Target unit ('kg/kg', 'g/kg', 'ppm', 'ppt')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Converted mixing ratio value(s)
    """
    # Input validation
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Unit parameters must be strings")
    
    if from_unit.lower() not in ['kg/kg', 'g/kg', 'ppm', 'ppt']:
        raise ValueError(f"Unsupported mixing ratio unit: {from_unit}. "
                        f"Supported units: 'kg/kg', 'g/kg', 'ppm', 'ppt'")
    
    if to_unit.lower() not in ['kg/kg', 'g/kg', 'ppm', 'ppt']:
        raise ValueError(f"Unsupported mixing ratio unit: {to_unit}. "
                        f"Supported units: 'kg/kg', 'g/kg', 'ppm', 'ppt'")
    
    # Validate value - mixing ratio should be non-negative
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Mixing ratio cannot be negative")
    elif isinstance(value, np.ndarray):
        if np.any(value < 0):
            raise ValueError("Mixing ratio cannot be negative")
    elif isinstance(value, xr.DataArray):
        if np.any(value.values < 0):
            raise ValueError("Mixing ratio cannot be negative")
    
    # Convert to kg/kg first
    if from_unit.lower() == 'g/kg':
        value_kgkg = value * 0.01  # g/kg to kg/kg
    elif from_unit.lower() == 'ppm':
        value_kgkg = value * 1e-6   # ppm to kg/kg
    elif from_unit.lower() == 'ppt':
        value_kgkg = value * 1e-12  # ppt to kg/kg
    elif from_unit.lower() == 'kg/kg':
        value_kgkg = value
    else:
        raise ValueError(f"Unsupported mixing ratio unit: {from_unit}")
    
    # Convert kg/kg to target unit
    if to_unit.lower() == 'g/kg':
        return value_kgkg / 0.001
    elif to_unit.lower() == 'ppm':
        return value_kgkg / 1e-6
    elif to_unit.lower() == 'ppt':
        return value_kgkg / 1e-12
    elif to_unit.lower() == 'kg/kg':
        return value_kgkg
    else:
        raise ValueError(f"Unsupported mixing ratio unit: {to_unit}")


def convert_specific_humidity(
    value: Union[float, np.ndarray, xr.DataArray],
    from_unit: str,
    to_unit: str
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert specific humidity between different units.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Specific humidity value(s) to convert
    from_unit : str
        Source unit ('kg/kg', 'g/kg', 'g/m3', 'mg/m3')
    to_unit : str
        Target unit ('kg/kg', 'g/kg', 'g/m3', 'mg/m3')
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Converted specific humidity value(s)
    """
    # Input validation
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Unit parameters must be strings")
    
    if from_unit.lower() not in ['kg/kg', 'g/kg', 'g/m3', 'mg/m3']:
        raise ValueError(f"Unsupported specific humidity unit: {from_unit}. "
                        f"Supported units: 'kg/kg', 'g/kg', 'g/m3', 'mg/m3'")
    
    if to_unit.lower() not in ['kg/kg', 'g/kg', 'g/m3', 'mg/m3']:
        raise ValueError(f"Unsupported specific humidity unit: {to_unit}. "
                        f"Supported units: 'kg/kg', 'g/kg', 'g/m3', 'mg/m3'")
    
    # Validate value - specific humidity should be non-negative
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Specific humidity cannot be negative")
    elif isinstance(value, np.ndarray):
        if np.any(value < 0):
            raise ValueError("Specific humidity cannot be negative")
    elif isinstance(value, xr.DataArray):
        if np.any(value.values < 0):
            raise ValueError("Specific humidity cannot be negative")
    
    # Convert to kg/kg first
    if from_unit.lower() == 'g/kg':
        value_kgkg = value * 0.001  # g/kg to kg/kg
    elif from_unit.lower() == 'g/m3':
        # Note: conversion from g/m3 to kg/kg requires density information
        # For simplicity, we'll treat g/m3 as equivalent to kg/kg for this function
        # In real applications, you'd need to account for air density
        value_kgkg = value * 0.001  # g/m3 to kg/kg (approximation)
    elif from_unit.lower() == 'mg/m3':
        value_kgkg = value * 1e-6   # mg/m3 to kg/kg
    elif from_unit.lower() == 'kg/kg':
        value_kgkg = value
    else:
        raise ValueError(f"Unsupported specific humidity unit: {from_unit}")
    
    # Convert kg/kg to target unit
    if to_unit.lower() == 'g/kg':
        return value_kgkg / 0.01
    elif to_unit.lower() == 'g/m3':
        return value_kgkg / 0.001 # Approximation
    elif to_unit.lower() == 'mg/m3':
        return value_kgkg / 1e-6
    elif to_unit.lower() == 'kg/kg':
        return value_kgkg
    else:
        raise ValueError(f"Unsupported specific humidity unit: {to_unit}")


def convert_concentration(
    value: Union[float, np.ndarray, xr.DataArray],
    from_unit: str,
    to_unit: str,
    molecular_weight: float = 28.97  # Default for dry air in g/mol
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert concentration between different units.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Concentration value(s) to convert
    from_unit : str
        Source unit ('ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol')
    to_unit : str
        Target unit ('ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol')
    molecular_weight : float, optional
        Molecular weight of the species in g/mol (default for dry air)
    
    Returns
    -------
    float, numpy.ndarray, or xarray.DataArray
        Converted concentration value(s)
    """
    # Input validation
    if not isinstance(from_unit, str) or not isinstance(to_unit, str):
        raise TypeError("Unit parameters must be strings")
    
    if from_unit.lower() not in ['ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol']:
        raise ValueError(f"Unsupported concentration unit: {from_unit}. "
                        f"Supported units: 'ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol'")
    
    if to_unit.lower() not in ['ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol']:
        raise ValueError(f"Unsupported concentration unit: {to_unit}. "
                        f"Supported units: 'ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol'")
    
    # Validate value - concentration should be non-negative
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("Concentration cannot be negative")
    elif isinstance(value, np.ndarray):
        if np.any(value < 0):
            raise ValueError("Concentration cannot be negative")
    elif isinstance(value, xr.DataArray):
        if np.any(value.values < 0):
            raise ValueError("Concentration cannot be negative")
    
    # Convert to mol/mol first (this is a simplified conversion)
    # Real conversion would require temperature and pressure for gas density
    if from_unit.lower() in ['ppm']:
        value_molmol = value * 1e-6
    elif from_unit.lower() in ['ppb']:
        value_molmol = value * 1e-9
    elif from_unit.lower() in ['ppt']:
        value_molmol = value * 1e-12
    elif from_unit.lower() in ['ug/m3']:
        # Approximate conversion - would need temperature and pressure for accuracy
        value_molmol = value * 1e-6  # Simplified
    elif from_unit.lower() in ['ng/m3']:
        value_molmol = value * 1e-9  # Simplified
    elif from_unit.lower() in ['mol/mol']:
        value_molmol = value
    else:
        raise ValueError(f"Unsupported concentration unit: {from_unit}")
    
    # Convert mol/mol to target unit
    if to_unit.lower() in ['ppm']:
        return value_molmol / 1e-6
    elif to_unit.lower() in ['ppb']:
        return value_molmol / 1e-9
    elif to_unit.lower() in ['ppt']:
        return value_molmol / 1e-12
    elif to_unit.lower() in ['ug/m3']:
        return value_molmol / 1e-6  # Simplified
    elif to_unit.lower() in ['ng/m3']:
        return value_molmol / 1e-9  # Simplified
    elif to_unit.lower() in ['mol/mol']:
        return value_molmol
    else:
        raise ValueError(f"Unsupported concentration unit: {to_unit}")