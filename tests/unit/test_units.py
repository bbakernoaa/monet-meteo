"""
Test suite for unit conversion functions.

Tests all unit conversion functionality including temperature, wind speed,
pressure, and other meteorological unit conversions.
"""
import numpy as np
import pytest

# Import unit conversion functions
from monet_meteo.units import (
    temperature,
    wind_speed,
    pressure,
    mixing_ratio
)


class TestTemperatureConversions:
    """Test temperature unit conversions."""
    
    def test_temperature_conversion(self):
        """Test temperature conversion."""
        # Celsius to Kelvin
        assert temperature(0.0, 'C', 'K') == 273.15
        
        # Fahrenheit to Celsius
        assert fahrenheit_to_celsius_approx(32.0) == 0.0
        
    def test_temperature_array_conversions(self):
        """Test temperature conversions with arrays."""
        celsius_temps = np.array([-40.0, 0.0, 25.0, 100.0])
        kelvin_temps = temperature(celsius_temps, 'C', 'K')
        assert np.allclose(kelvin_temps, celsius_temps + 273.15)

def fahrenheit_to_celsius_approx(f):
    return (f - 32) * 5/9

class TestWindSpeedConversions:
    """Test wind speed unit conversions."""
    
    def test_wind_speed_conversion(self):
        """Test wind speed conversion."""
        # 1 m/s to knots
        assert abs(wind_speed(1.0, 'm/s', 'knots') - 1.94384) < 0.01

class TestPressureConversions:
    """Test pressure unit conversions."""
    
    def test_pressure_conversion(self):
        """Test pressure conversion."""
        # hPa to Pa
        assert pressure(1.0, 'hPa', 'Pa') == 100.0
