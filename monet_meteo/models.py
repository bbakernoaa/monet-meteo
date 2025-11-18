"""
Data models for atmospheric profiles and related structures.

This module defines data structures and models for representing atmospheric
profiles and related meteorological data.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np
import xarray as xr


@dataclass
class AtmosphericProfile:
    """
    A data class representing a complete atmospheric profile.
    
    Attributes
    ----------
    pressure : array-like
        Pressure levels (Pa)
    temperature : array-like
        Temperature profile (K)
    potential_temperature : array-like, optional
        Potential temperature profile (K)
    equivalent_potential_temperature : array-like, optional
        Equivalent potential temperature profile (K)
    virtual_temperature : array-like, optional
        Virtual temperature profile (K)
    u_wind : array-like, optional
        Eastward wind component (m/s)
    v_wind : array-like, optional
        Northward wind component (m/s)
    mixing_ratio : array-like, optional
        Water vapor mixing ratio (kg/kg)
    relative_humidity : array-like, optional
        Relative humidity (0-1)
    height : array-like, optional
        Height levels (m)
    """
    pressure: Union[np.ndarray, xr.DataArray]
    temperature: Union[np.ndarray, xr.DataArray]
    potential_temperature: Optional[Union[np.ndarray, xr.DataArray]] = None
    equivalent_potential_temperature: Optional[Union[np.ndarray, xr.DataArray]] = None
    virtual_temperature: Optional[Union[np.ndarray, xr.DataArray]] = None
    u_wind: Optional[Union[np.ndarray, xr.DataArray]] = None
    v_wind: Optional[Union[np.ndarray, xr.DataArray]] = None
    mixing_ratio: Optional[Union[np.ndarray, xr.DataArray]] = None
    relative_humidity: Optional[Union[np.ndarray, xr.DataArray]] = None
    height: Optional[Union[np.ndarray, xr.DataArray]] = None
    
    def __post_init__(self):
        """Validate the atmospheric profile after initialization."""
        # Check that pressure and temperature have the same shape
        if hasattr(self.pressure, 'shape') and hasattr(self.temperature, 'shape'):
            if self.pressure.shape != self.temperature.shape:
                raise ValueError("Pressure and temperature arrays must have the same shape")
        elif hasattr(self.pressure, '__len__') and hasattr(self.temperature, '__len__'):
            if len(self.pressure) != len(self.temperature):
                raise ValueError("Pressure and temperature arrays must have the same length")
    
    def calculate_thermodynamic_properties(self):
        """
        Calculate thermodynamic properties if not already provided.
        
        This method calculates potential temperature, equivalent potential temperature,
        and virtual temperature if they are not already provided in the profile.
        """
        from .thermodynamics.thermodynamic_calculations import (
            potential_temperature,
            equivalent_potential_temperature,
            virtual_temperature
        )
        
        # Calculate potential temperature if not provided
        if self.potential_temperature is None:
            self.potential_temperature = potential_temperature(
                self.pressure, self.temperature
            )
        
        # Calculate virtual temperature if not provided and mixing ratio is available
        if self.virtual_temperature is None and self.mixing_ratio is not None:
            self.virtual_temperature = virtual_temperature(
                self.temperature, self.mixing_ratio
            )
        
        # Calculate equivalent potential temperature if not provided and mixing ratio is available
        if self.equivalent_potential_temperature is None and self.mixing_ratio is not None:
            self.equivalent_potential_temperature = equivalent_potential_temperature(
                self.pressure, self.temperature, self.mixing_ratio
            )


@dataclass
class WindProfile:
    """
    A data class representing a wind profile.
    
    Attributes
    ----------
    height : array-like
        Height levels (m)
    u_wind : array-like
        Eastward wind component (m/s)
    v_wind : array-like
        Northward wind component (m/s)
    """
    height: Union[np.ndarray, xr.DataArray]
    u_wind: Union[np.ndarray, xr.DataArray]
    v_wind: Union[np.ndarray, xr.DataArray]
    
    def __post_init__(self):
        """Validate the wind profile after initialization."""
        # Check that all arrays have the same shape
        shapes = []
        for arr in [self.height, self.u_wind, self.v_wind]:
            if hasattr(arr, 'shape'):
                shapes.append(arr.shape)
            elif hasattr(arr, '__len__'):
                shapes.append((len(arr),))
        
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("Height, u_wind, and v_wind arrays must have the same shape")
    
    def wind_speed(self) -> Union[np.ndarray, xr.DataArray]:
        """Calculate wind speed from u and v components."""
        return np.sqrt(self.u_wind**2 + self.v_wind**2)
    
    def wind_direction(self) -> Union[np.ndarray, xr.DataArray]:
        """Calculate wind direction from u and v components (in degrees)."""
        direction = (270 - np.degrees(np.arctan2(self.v_wind, self.u_wind))) % 360
        return direction


@dataclass
class ThermodynamicProfile:
    """
    A data class representing thermodynamic properties of the atmosphere.
    
    Attributes
    ----------
    pressure : array-like
        Pressure levels (Pa)
    temperature : array-like
        Temperature profile (K)
    dewpoint : array-like, optional
        Dewpoint temperature profile (K)
    mixing_ratio : array-like, optional
        Water vapor mixing ratio (kg/kg)
    relative_humidity : array-like, optional
        Relative humidity (0-1)
    """
    pressure: Union[np.ndarray, xr.DataArray]
    temperature: Union[np.ndarray, xr.DataArray]
    dewpoint: Optional[Union[np.ndarray, xr.DataArray]] = None
    mixing_ratio: Optional[Union[np.ndarray, xr.DataArray]] = None
    relative_humidity: Optional[Union[np.ndarray, xr.DataArray]] = None
    
    def __post_init__(self):
        """Validate the thermodynamic profile after initialization."""
        # Check that pressure and temperature have the same shape
        if hasattr(self.pressure, 'shape') and hasattr(self.temperature, 'shape'):
            if self.pressure.shape != self.temperature.shape:
                raise ValueError("Pressure and temperature arrays must have the same shape")
        elif hasattr(self.pressure, '__len__') and hasattr(self.temperature, '__len__'):
            if len(self.pressure) != len(self.temperature):
                raise ValueError("Pressure and temperature arrays must have the same length")
        
        # Ensure that at least one of dewpoint, mixing_ratio, or relative_humidity is provided
        # for a complete thermodynamic profile
        provided = sum(x is not None for x in [self.dewpoint, self.mixing_ratio, self.relative_humidity])
        if provided == 0:
            # We can work with just temperature and pressure, but more properties are needed
            # for full thermodynamic calculations
            pass
        elif provided == 3:
            # All three provided, check consistency
            pass
        # If only one or two are provided, that's acceptable


@dataclass
class DerivedParameters:
    """
    A data class representing derived meteorological parameters.
    
    Attributes
    ----------
    heat_index : array-like, optional
        Heat index values (°F)
    wind_chill : array-like, optional
        Wind chill temperatures (°F)
    lcl_height : array-like, optional
        Lifting condensation level height (m)
    wet_bulb_temp : array-like, optional
        Wet bulb temperature (K)
    """
    heat_index: Optional[Union[np.ndarray, xr.DataArray]] = None
    wind_chill: Optional[Union[np.ndarray, xr.DataArray]] = None
    lcl_height: Optional[Union[np.ndarray, xr.DataArray]] = None
    wet_bulb_temp: Optional[Union[np.ndarray, xr.DataArray]] = None
    
    def __post_init__(self):
        """Validate the derived parameters after initialization."""
        # All parameters are optional, so no strict validation required
        pass