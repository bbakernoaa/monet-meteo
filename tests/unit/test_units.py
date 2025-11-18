"""
Test suite for unit conversion functions.

Tests all unit conversion functionality including temperature, wind speed,
pressure, and other meteorological unit conversions.
"""
import numpy as np
import pytest

# Import unit conversion functions
from monet_meteo.units import (
    celsius_to_kelvin,
    kelvin_to_celsius,
    fahrenheit_to_celsius,
    celsius_to_fahrenheit,
    fahrenheit_to_kelvin,
    kelvin_to_fahrenheit,
    meters_per_second_to_knots,
    knots_to_meters_per_second,
    miles_per_hour_to_meters_per_second,
    meters_per_second_to_miles_per_hour,
    hpa_to_pa,
    pa_to_hpa,
    mb_to_pa,
    pa_to_mb
)


class TestTemperatureConversions:
    """Test temperature unit conversions."""
    
    def test_celsius_to_kelvin(self):
        """Test Celsius to Kelvin conversion."""
        # Absolute zero
        assert celsius_to_kelvin(-273.15) == 0.0
        
        # Freezing point of water
        assert celsius_to_kelvin(0.0) == 273.15
        
        # Boiling point of water
        assert celsius_to_kelvin(100.0) == 373.15
        
        # Standard room temperature
        assert abs(celsius_to_kelvin(25.0) - 298.15) < 1e-10
    
    def test_kelvin_to_celsius(self):
        """Test Kelvin to Celsius conversion."""
        # Absolute zero
        assert kelvin_to_celsius(0.0) == -273.15
        
        # Freezing point of water
        assert kelvin_to_celsius(273.15) == 0.0
        
        # Boiling point of water
        assert kelvin_to_celsius(373.15) == 100.0
        
        # Standard room temperature
        assert abs(kelvin_to_celsius(298.15) - 25.0) < 1e-10
    
    def test_fahrenheit_to_celsius(self):
        """Test Fahrenheit to Celsius conversion."""
        # Absolute zero
        assert abs(fahrenheit_to_celsius(-459.67) - (-273.15)) < 0.01
        
        # Freezing point of water
        assert fahrenheit_to_celsius(32.0) == 0.0
        
        # Boiling point of water
        assert fahrenheit_to_celsius(212.0) == 100.0
        
        # Body temperature
        assert abs(fahrenheit_to_celsius(98.6) - 37.0) < 0.1
    
    def test_celsius_to_fahrenheit(self):
        """Test Celsius to Fahrenheit conversion."""
        # Absolute zero
        assert abs(celsius_to_fahrenheit(-273.15) - (-459.67)) < 0.01
        
        # Freezing point of water
        assert celsius_to_fahrenheit(0.0) == 32.0
        
        # Boiling point of water
        assert celsius_to_fahrenheit(100.0) == 212.0
        
        # Body temperature
        assert abs(celsius_to_fahrenheit(37.0) - 98.6) < 0.1
    
    def test_fahrenheit_to_kelvin(self):
        """Test Fahrenheit to Kelvin conversion."""
        # Absolute zero
        assert abs(fahrenheit_to_kelvin(-459.67) - 0.0) < 0.01
        
        # Freezing point of water
        assert abs(fahrenheit_to_kelvin(32.0) - 273.15) < 1e-10
        
        # Boiling point of water
        assert abs(fahrenheit_to_kelvin(212.0) - 373.15) < 1e-10
    
    def test_kelvin_to_fahrenheit(self):
        """Test Kelvin to Fahrenheit conversion."""
        # Absolute zero
        assert abs(kelvin_to_fahrenheit(0.0) - (-459.67)) < 0.01
        
        # Freezing point of water
        assert abs(kelvin_to_fahrenheit(273.15) - 32.0) < 1e-10
        
        # Boiling point of water
        assert abs(kelvin_to_fahrenheit(373.15) - 212.0) < 1e-10
    
    def test_temperature_array_conversions(self):
        """Test temperature conversions with arrays."""
        celsius_temps = np.array([-40.0, 0.0, 25.0, 100.0])
        
        # Test round-trip conversion
        kelvin_temps = celsius_to_kelvin(celsius_temps)
        converted_back = kelvin_to_celsius(kelvin_temps)
        
        assert np.allclose(converted_back, celsius_temps, rtol=1e-10)
        
        # Test Fahrenheit conversions
        fahrenheit_temps = celsius_to_fahrenheit(celsius_temps)
        f_to_c_back = fahrenheit_to_celsius(fahrenheit_temps)
        
        assert np.allclose(f_to_c_back, celsius_temps, rtol=1e-10)


class TestWindSpeedConversions:
    """Test wind speed unit conversions."""
    
    def test_meters_per_second_to_knots(self):
        """Test m/s to knots conversion."""
        # 1 m/s = 1.94384 knots
        assert abs(meters_per_second_to_knots(1.0) - 1.94384) < 0.001
        
        # 10 m/s
        assert abs(meters_per_second_to_knots(10.0) - 19.4384) < 0.001
        
        # 50 m/s (hurricane force)
        assert abs(meters_per_second_to_knots(50.0) - 97.192) < 0.001
    
    def test_knots_to_meters_per_second(self):
        """Test knots to m/s conversion."""
        # 1 knot = 0.514444 m/s
        assert abs(knots_to_meters_per_second(1.0) - 0.514444) < 0.001
        
        # 10 knots
        assert abs(knots_to_meters_per_second(10.0) - 5.14444) < 0.001
        
        # 50 knots
        assert abs(knots_to_meters_per_second(50.0) - 25.7222) < 0.001
    
    def test_miles_per_hour_to_meters_per_second(self):
        """Test mph to m/s conversion."""
        # 1 mph = 0.44704 m/s
        assert abs(miles_per_hour_to_meters_per_second(1.0) - 0.44704) < 0.001
        
        # 60 mph
        assert abs(miles_per_hour_to_meters_per_second(60.0) - 26.8224) < 0.001
        
        # 100 mph
        assert abs(miles_per_hour_to_meters_per_second(100.0) - 44.704) < 0.001
    
    def test_meters_per_second_to_miles_per_hour(self):
        """Test m/s to mph conversion."""
        # 1 m/s = 2.23694 mph
        assert abs(meters_per_second_to_miles_per_hour(1.0) - 2.23694) < 0.001
        
        # 10 m/s
        assert abs(meters_per_second_to_miles_per_hour(10.0) - 22.3694) < 0.001
        
        # 50 m/s
        assert abs(meters_per_second_to_miles_per_hour(50.0) - 111.847) < 0.001
    
    def test_wind_speed_round_trip(self):
        """Test round-trip wind speed conversions."""
        # Test m/s -> knots -> m/s
        original = 15.5
        knots = meters_per_second_to_knots(original)
        back_to_ms = knots_to_meters_per_second(knots)
        
        assert abs(back_to_ms - original) < 1e-6
        
        # Test m/s -> mph -> m/s
        mph = meters_per_second_to_miles_per_hour(original)
        back_to_ms_2 = miles_per_hour_to_meters_per_second(mph)
        
        assert abs(back_to_ms_2 - original) < 1e-6
    
    def test_wind_speed_array_conversions(self):
        """Test wind speed conversions with arrays."""
        wind_speeds = np.array([0.0, 5.0, 10.0, 20.0, 50.0])  # m/s
        
        # Convert to knots and back
        knots = meters_per_second_to_knots(wind_speeds)
        back_to_ms = knots_to_meters_per_second(knots)
        
        assert np.allclose(back_to_ms, wind_speeds, rtol=1e-10)


class TestPressureConversions:
    """Test pressure unit conversions."""
    
    def test_hpa_to_pa(self):
        """Test hPa to Pa conversion."""
        # Standard atmospheric pressure
        assert hpa_to_pa(1013.25) == 101325.0
        
        # 1 hPa = 100 Pa
        assert hpa_to_pa(1.0) == 100.0
        
        # 1000 hPa
        assert hpa_to_pa(1000.0) == 100000.0
    
    def test_pa_to_hpa(self):
        """Test Pa to hPa conversion."""
        # Standard atmospheric pressure
        assert pa_to_hpa(101325.0) == 1013.25
        
        # 100 Pa = 1 hPa
        assert pa_to_hpa(100.0) == 1.0
        
        # 100000 Pa
        assert pa_to_hpa(100000.0) == 1000.0
    
    def test_mb_to_pa(self):
        """Test mb to Pa conversion."""
        # mb and hPa are equivalent
        assert mb_to_pa(1013.25) == 101325.0
        assert mb_to_pa(1.0) == 100.0
        assert mb_to_pa(1000.0) == 100000.0
    
    def test_pa_to_mb(self):
        """Test Pa to mb conversion."""
        # mb and hPa are equivalent
        assert pa_to_mb(101325.0) == 1013.25
        assert pa_to_mb(100.0) == 1.0
        assert pa_to_mb(100000.0) == 1000.0
    
    def test_pressure_round_trip(self):
        """Test round-trip pressure conversions."""
        original = 1013.25  # hPa
        
        # hPa -> Pa -> hPa
        pa = hpa_to_pa(original)
        back_to_hpa = pa_to_hpa(pa)
        
        assert abs(back_to_hpa - original) < 1e-10
        
        # mb -> Pa -> mb
        pa_2 = mb_to_pa(original)
        back_to_mb = pa_to_mb(pa_2)
        
        assert abs(back_to_mb - original) < 1e-10
    
    def test_pressure_array_conversions(self):
        """Test pressure conversions with arrays."""
        pressures = np.array([900.0, 950.0, 1000.0, 1013.25, 1050.0, 1100.0])  # hPa
        
        # Convert to Pa and back
        pa = hpa_to_pa(pressures)
        back_to_hpa = pa_to_hpa(pa)
        
        assert np.allclose(back_to_hpa, pressures, rtol=1e-10)


class TestUnitConversionEdgeCases:
    """Test edge cases and extreme values in unit conversions."""
    
    def test_temperature_extreme_values(self):
        """Test temperature conversions at extreme values."""
        # Very cold temperatures
        assert celsius_to_kelvin(-273.15) == 0.0
        assert kelvin_to_celsius(0.0) == -273.15
        
        # Very hot temperatures
        hot_c = 1000.0
        hot_k = celsius_to_kelvin(hot_c)
        assert abs(hot_k - 1273.15) < 1e-10
        
        # Test with arrays containing extreme values
        extreme_temps = np.array([-273.15, -40.0, 0.0, 25.0, 1000.0])
        kelvin_temps = celsius_to_kelvin(extreme_temps)
        back_to_c = kelvin_to_celsius(kelvin_temps)
        
        assert np.allclose(back_to_c, extreme_temps, rtol=1e-10)
    
    def test_wind_speed_edge_cases(self):
        """Test wind speed conversions at edge cases."""
        # Zero wind
        assert meters_per_second_to_knots(0.0) == 0.0
        assert knots_to_meters_per_second(0.0) == 0.0
        
        # Very small wind speeds
        assert meters_per_second_to_knots(0.001) > 0.0
        assert meters_per_second_to_knots(0.001) < 0.01
        
        # Very large wind speeds (hurricane +)
        hurricane_plus = 100.0  # m/s
        knots = meters_per_second_to_knots(hurricane_plus)
        assert knots > 190.0  # Should be very high
        
        # Test with arrays
        wind_speeds = np.array([0.0, 0.1, 50.0, 100.0])
        knots_array = meters_per_second_to_knots(wind_speeds)
        back_to_ms = knots_to_meters_per_second(knots_array)
        
        assert np.allclose(back_to_ms, wind_speeds, rtol=1e-10)
    
    def test_pressure_edge_cases(self):
        """Test pressure conversions at edge cases."""
        # Standard pressure
        assert hpa_to_pa(1013.25) == 101325.0
        
        # Very low pressure (hurricane)
        assert hpa_to_pa(900.0) == 90000.0
        
        # Very high pressure
        assert hpa_to_pa(1100.0) == 110000.0
        
        # Test with arrays
        pressures = np.array([800.0, 900.0, 1013.25, 1100.0, 1200.0])
        pa_array = hpa_to_pa(pressures)
        back_to_hpa = pa_to_hpa(pa_array)
        
        assert np.allclose(back_to_hpa, pressures, rtol=1e-10)


class TestUnitConversionConsistency:
    """Test consistency between different conversion paths."""
    
    def test_temperature_consistency(self):
        """Test consistency between temperature conversion paths."""
        # Test C -> F -> K -> C
        c1 = 25.0
        f = celsius_to_fahrenheit(c1)
        k = fahrenheit_to_kelvin(f)
        c2 = kelvin_to_celsius(k)
        
        assert abs(c2 - c1) < 1e-10
        
        # Test K -> C -> F -> K
        k1 = 300.0
        c = kelvin_to_celsius(k1)
        f = celsius_to_fahrenheit(c)
        k2 = fahrenheit_to_kelvin(f)
        
        assert abs(k2 - k1) < 1e-10
    
    def test_wind_speed_consistency(self):
        """Test consistency between wind speed conversion paths."""
        # Test m/s -> mph -> knots -> m/s
        ms1 = 10.0
        mph = meters_per_second_to_miles_per_hour(ms1)
        knots = miles_per_hour_to_meters_per_second(mph)
        knots = meters_per_second_to_knots(knots)
        ms2 = knots_to_meters_per_second(knots)
        
        # This should be consistent (though with some rounding)
        assert abs(ms2 - ms1) < 0.1
    
    def test_pressure_consistency(self):
        """Test consistency between pressure conversion systems."""
        # hPa and mb should be equivalent
        hpa_value = 1013.25
        
        pa_from_hpa = hpa_to_pa(hpa_value)
        pa_from_mb = mb_to_pa(hpa_value)
        
        assert pa_from_hpa == pa_from_mb
        
        hpa_back = pa_to_hpa(pa_from_hpa)
        mb_back = pa_to_mb(pa_from_mb)
        
        assert hpa_back == mb_back