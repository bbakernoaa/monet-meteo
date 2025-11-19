"""
Unit conversion utilities for meteorological applications using Pint.

This module provides functions for converting between various meteorological units
leveraging the Pint library for robust unit handling and dimensional analysis.
"""

import numpy as np
import xarray as xr
from typing import Union
import pint

# Create a unit registry with meteorological units
ureg = pint.UnitRegistry()

# Add common meteorological units to the registry
ureg.define('millibar = 100 * pascal = mb')
ureg.define('torr = 133.322 * pascal = mmHg')
ureg.define('inHg = 3386.39 * pascal')
ureg.define('atmosphere = 101325 * pascal = atm')
ureg.define('knot = 0.514444 * meter / second = kt')
ureg.define('mile = 1609.34 * meter = mi')
ureg.define('nautical_mile = 1852 * meter = nm')
ureg.define('micrometer = 1e-6 * meter = um')
ureg.define('ppm = parts_per_million = 1e-6')
ureg.define('ppb = parts_per_billion = 1e-9')
ureg.define('ppt = parts_per_trillion = 1e-12')
ureg.define('ug_per_m3 = microgram / meter**3')
ureg.define('ng_per_m3 = nanogram / meter**3')

# Convenience functions for meteorological units
def pressure(value: Union[float, np.ndarray, xr.DataArray], 
            unit: str, 
            to_unit: Union[str, None] = None) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert pressure between different units using Pint.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Pressure value(s) to convert
    unit : str
        Source unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')
    to_unit : str, optional
        Target unit. If None, return as pint quantity
        
    Returns
    -------
    float, numpy.ndarray, xarray.DataArray, or pint.Quantity
        Converted pressure value(s)
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")
    
    # Convert to numpy array if needed
    if isinstance(value, xr.DataArray):
        data_array = value
        data_values = value.values
    else:
        data_array = None
        data_values = np.asarray(value)
    
    # Create pint quantity and convert
    quantity = data_values * ureg(unit)
    if to_unit is not None:
        converted = quantity.to(to_unit)
        result = converted.magnitude
    else:
        return quantity
    
    # Return appropriate type
    if data_array is not None:
        return xr.DataArray(result, 
                          coords=data_array.coords, 
                          dims=data_array.dims,
                          attrs={'units': to_unit})
    else:
        return result


def temperature(value: Union[float, np.ndarray, xr.DataArray], 
               unit: str, 
               to_unit: Union[str, None] = None) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert temperature between different units using Pint.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Temperature value(s) to convert
    unit : str
        Source unit ('K', 'C', 'F', 'R')
    to_unit : str, optional
        Target unit. If None, return as pint quantity
        
    Returns
    -------
    float, numpy.ndarray, xarray.DataArray, or pint.Quantity
        Converted temperature value(s)
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")
    
    # Convert to numpy array if needed
    if isinstance(value, xr.DataArray):
        data_array = value
        data_values = value.values
    else:
        data_array = None
        data_values = np.asarray(value)
    
    # Handle temperature conversion with proper offset handling
    if unit.upper() == 'C':
        temp_k = data_values + 273.15
    elif unit.upper() == 'F':
        temp_k = (data_values - 32) * 5/9 + 273.15
    elif unit.upper() == 'R':
        temp_k = data_values * 5/9
    elif unit.upper() == 'K':
        temp_k = data_values
    else:
        raise ValueError(f"Unsupported temperature unit: {unit}")
    
    if to_unit is None:
        return temp_k * ureg.K
    else:
        # Convert to target units
        if to_unit.upper() == 'C':
            result = temp_k - 273.15
        elif to_unit.upper() == 'F':
            result = (temp_k - 273.15) * 9/5 + 32
        elif to_unit.upper() == 'R':
            result = temp_k * 9/5
        elif to_unit.upper() == 'K':
            result = temp_k
        else:
            raise ValueError(f"Unsupported target unit: {to_unit}")
        
        # Return appropriate type
        if data_array is not None:
            return xr.DataArray(result, 
                              coords=data_array.coords, 
                              dims=data_array.dims,
                              attrs={'units': to_unit})
        else:
            return result


def distance(value: Union[float, np.ndarray, xr.DataArray], 
            unit: str, 
            to_unit: Union[str, None] = None) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert distance between different units using Pint.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Distance value(s) to convert
    unit : str
        Source unit ('m', 'km', 'ft', 'mi', 'nm', 'cm', 'mm')
    to_unit : str, optional
        Target unit. If None, return as pint quantity
        
    Returns
    -------
    float, numpy.ndarray, xarray.DataArray, or pint.Quantity
        Converted distance value(s)
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")
    
    # Convert to numpy array if needed
    if isinstance(value, xr.DataArray):
        data_array = value
        data_values = value.values
    else:
        data_array = None
        data_values = np.asarray(value)
    
    # Create pint quantity and convert
    quantity = data_values * ureg(unit)
    if to_unit is not None:
        converted = quantity.to(to_unit)
        result = converted.magnitude
    else:
        return quantity
    
    # Return appropriate type
    if data_array is not None:
        return xr.DataArray(result, 
                          coords=data_array.coords, 
                          dims=data_array.dims,
                          attrs={'units': to_unit})
    else:
        return result


def wind_speed(value: Union[float, np.ndarray, xr.DataArray], 
               unit: str, 
               to_unit: Union[str, None] = None) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert wind speed between different units using Pint.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Wind speed value(s) to convert
    unit : str
        Source unit ('m/s', 'knots', 'km/h', 'mph', 'ft/s')
    to_unit : str, optional
        Target unit. If None, return as pint quantity
        
    Returns
    -------
    float, numpy.ndarray, xarray.DataArray, or pint.Quantity
        Converted wind speed value(s)
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")
    
    # Convert to numpy array if needed
    if isinstance(value, xr.DataArray):
        data_array = value
        data_values = value.values
    else:
        data_array = None
        data_values = np.asarray(value)
    
    # Parse unit string for pint
    pint_unit = unit.lower()
    if pint_unit in ['m/s', 'mps']:
        pint_unit = 'meter/second'
    elif pint_unit in ['km/h', 'kmh']:
        pint_unit = 'kilometer/hour'
    elif pint_unit in ['mph']:
        pint_unit = 'mile/hour'
    elif pint_unit in ['knots', 'kt']:
        pint_unit = 'knot'
    elif pint_unit in ['ft/s', 'fps']:
        pint_unit = 'foot/second'
    else:
        raise ValueError(f"Unsupported wind speed unit: {unit}")
    
    # Create pint quantity and convert
    quantity = data_values * ureg(pint_unit)
    if to_unit is not None:
        # Parse target unit for pint
        target_pint_unit = to_unit.lower()
        if target_pint_unit in ['m/s', 'mps']:
            target_pint_unit = 'meter/second'
        elif target_pint_unit in ['km/h', 'kmh']:
            target_pint_unit = 'kilometer/hour'
        elif target_pint_unit in ['mph']:
            target_pint_unit = 'mile/hour'
        elif target_pint_unit in ['knots', 'kt']:
            target_pint_unit = 'knot'
        elif target_pint_unit in ['ft/s', 'fps']:
            target_pint_unit = 'foot/second'
        else:
            raise ValueError(f"Unsupported target unit: {to_unit}")
        
        converted = quantity.to(target_pint_unit)
        result = converted.magnitude
    else:
        return quantity
    
    # Return appropriate type
    if data_array is not None:
        return xr.DataArray(result, 
                          coords=data_array.coords, 
                          dims=data_array.dims,
                          attrs={'units': to_unit})
    else:
        return result


def mixing_ratio(value: Union[float, np.ndarray, xr.DataArray], 
                unit: str, 
                to_unit: Union[str, None] = None) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert mixing ratio between different units using Pint.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Mixing ratio value(s) to convert
    unit : str
        Source unit ('kg/kg', 'g/kg', 'ppm', 'ppt', 'ppb')
    to_unit : str, optional
        Target unit. If None, return as pint quantity
        
    Returns
    -------
    float, numpy.ndarray, xarray.DataArray, or pint.Quantity
        Converted mixing ratio value(s)
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")
    
    # Convert to numpy array if needed
    if isinstance(value, xr.DataArray):
        data_array = value
        data_values = value.values
    else:
        data_array = None
        data_values = np.asarray(value)
    
    # Parse unit string for pint
    pint_unit = unit.lower()
    if pint_unit in ['kg/kg']:
        pint_unit = 'kilogram/kilogram'
    elif pint_unit in ['g/kg']:
        pint_unit = 'gram/kilogram'
    elif pint_unit in ['ppm']:
        pint_unit = 'parts_per_million'
    elif pint_unit in ['ppb']:
        pint_unit = 'parts_per_billion'
    elif pint_unit in ['ppt']:
        pint_unit = 'parts_per_trillion'
    else:
        raise ValueError(f"Unsupported mixing ratio unit: {unit}")
    
    # Create pint quantity and convert
    quantity = data_values * ureg(pint_unit)
    if to_unit is not None:
        # Parse target unit for pint
        target_pint_unit = to_unit.lower()
        if target_pint_unit in ['kg/kg']:
            target_pint_unit = 'kilogram/kilogram'
        elif target_pint_unit in ['g/kg']:
            target_pint_unit = 'gram/kilogram'
        elif target_pint_unit in ['ppm']:
            target_pint_unit = 'parts_per_million'
        elif target_pint_unit in ['ppb']:
            target_pint_unit = 'parts_per_billion'
        elif target_pint_unit in ['ppt']:
            target_pint_unit = 'parts_per_trillion'
        else:
            raise ValueError(f"Unsupported target unit: {to_unit}")
        
        converted = quantity.to(target_pint_unit)
        result = converted.magnitude
    else:
        return quantity
    
    # Return appropriate type
    if data_array is not None:
        return xr.DataArray(result, 
                          coords=data_array.coords, 
                          dims=data_array.dims,
                          attrs={'units': to_unit})
    else:
        return result


def concentration(value: Union[float, np.ndarray, xr.DataArray], 
                  unit: str, 
                  to_unit: Union[str, None] = None) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Convert concentration between different units using Pint.
    
    Parameters
    ----------
    value : float, numpy.ndarray, or xarray.DataArray
        Concentration value(s) to convert
    unit : str
        Source unit ('ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol')
    to_unit : str, optional
        Target unit. If None, return as pint quantity
        
    Returns
    -------
    float, numpy.ndarray, xarray.DataArray, or pint.Quantity
        Converted concentration value(s)
    """
    if not isinstance(value, (int, float, np.ndarray, xr.DataArray)):
        raise TypeError("Value must be numeric or array-like")
    
    # Convert to numpy array if needed
    if isinstance(value, xr.DataArray):
        data_array = value
        data_values = value.values
    else:
        data_array = None
        data_values = np.asarray(value)
    
    # Parse unit string for pint
    pint_unit = unit.lower()
    if pint_unit in ['ppm']:
        pint_unit = 'parts_per_million'
    elif pint_unit in ['ppb']:
        pint_unit = 'parts_per_billion'
    elif pint_unit in ['ppt']:
        pint_unit = 'parts_per_trillion'
    elif pint_unit in ['ug/m3']:
        pint_unit = 'microgram/meter**3'
    elif pint_unit in ['ng/m3']:
        pint_unit = 'nanogram/meter**3'
    elif pint_unit in ['mol/mol']:
        pint_unit = 'dimensionless'
    else:
        raise ValueError(f"Unsupported concentration unit: {unit}")
    
    # Create pint quantity and convert
    quantity = data_values * ureg(pint_unit)
    if to_unit is not None:
        # Parse target unit for pint
        target_pint_unit = to_unit.lower()
        if target_pint_unit in ['ppm']:
            target_pint_unit = 'parts_per_million'
        elif target_pint_unit in ['ppb']:
            target_pint_unit = 'parts_per_billion'
        elif target_pint_unit in ['ppt']:
            target_pint_unit = 'parts_per_trillion'
        elif target_pint_unit in ['ug/m3']:
            target_pint_unit = 'microgram/meter**3'
        elif target_pint_unit in ['ng/m3']:
            target_pint_unit = 'nanogram/meter**3'
        elif target_pint_unit in ['mol/mol']:
            target_pint_unit = 'dimensionless'
        else:
            raise ValueError(f"Unsupported target unit: {to_unit}")
        
        converted = quantity.to(target_pint_unit)
        result = converted.magnitude
    else:
        return quantity
    
    # Return appropriate type
    if data_array is not None:
        return xr.DataArray(result, 
                          coords=data_array.coords, 
                          dims=data_array.dims,
                          attrs={'units': to_unit})
    else:
        return result


# Legacy function names for backward compatibility
def convert_pressure(value, from_unit, to_unit):
    """Legacy alias for pressure() function."""
    return pressure(value, from_unit, to_unit)

def convert_temperature(value, from_unit, to_unit):
    """Legacy alias for temperature() function."""
    return temperature(value, from_unit, to_unit)

def convert_distance(value, from_unit, to_unit):
    """Legacy alias for distance() function."""
    return distance(value, from_unit, to_unit)

def convert_wind_speed(value, from_unit, to_unit):
    """Legacy alias for wind_speed() function."""
    return wind_speed(value, from_unit, to_unit)

def convert_mixing_ratio(value, from_unit, to_unit):
    """Legacy alias for mixing_ratio() function."""
    return mixing_ratio(value, from_unit, to_unit)

def convert_specific_humidity(value, from_unit, to_unit):
    """Specific humidity conversion - treat same as mixing ratio for now."""
    return mixing_ratio(value, from_unit, to_unit)

def convert_concentration(value, from_unit, to_unit, molecular_weight=28.97):
    """Legacy alias for concentration() function."""
    return concentration(value, from_unit, to_unit)