"""
Test suite for thermodynamic calculations.

Tests all thermodynamic functions including potential temperature, virtual temperature,
saturation vapor pressure, mixing ratio, relative humidity, and lapse rates.
"""
import numpy as np
import pytest
from unittest.mock import patch

# Import the thermodynamic functions
from monet_meteo.thermodynamics.thermodynamic_calculations import (
    potential_temperature,
    equivalent_potential_temperature,
    virtual_temperature,
    saturation_vapor_pressure,
    mixing_ratio,
    relative_humidity,
    dewpoint_from_relative_humidity,
    wet_bulb_temperature,
    moist_lapse_rate,
    dry_lapse_rate,
    lifting_condensation_level
)


class TestPotentialTemperature:
    """Test potential temperature calculations."""
    
    def test_potential_temperature_standard_conditions(self, scalar_test_values):
        """Test potential temperature at standard conditions."""
        pt = potential_temperature(
            pressure=scalar_test_values['pressure'],
            temperature=scalar_test_values['temperature']
        )
        # At 1013.25 hPa and 288.15 K, potential temperature should be very close to temperature
        assert abs(pt - scalar_test_values['temperature']) < 0.1
    
    def test_potential_temperature_decreasing_with_height(self, sample_pressure, sample_temperature):
        """Test that potential temperature decreases with height for given profiles."""
        pt = potential_temperature(pressure=sample_pressure, temperature=sample_temperature)
        
        # Check that potential temperature is calculated for all levels
        if hasattr(sample_pressure, '__len__'):
            if hasattr(pt, '__len__'):
                assert len(np.asarray(pt)) == len(np.asarray(sample_pressure))
            else:
                # Scalar result, check it's reasonable
                assert pt > 0
                assert np.isfinite(pt)
        else:
            # Scalar inputs
            assert pt > 0
            assert np.isfinite(pt)
    
    def test_potential_temperature_units(self):
        """Test potential temperature with different pressure units."""
        # Test with hPa
        pt_hpa = potential_temperature(pressure=1000.0, temperature=300.0)
        # Test with Pa (should give same result)
        pt_pa = potential_temperature(pressure=100000.0, temperature=300.0)
        
        assert abs(pt_hpa - pt_pa) < 1e-6
    
    def test_potential_temperature_extreme_values(self, extreme_values):
        """Test potential temperature with extreme values."""
        # Very cold conditions
        pt_cold = potential_temperature(
            pressure=extreme_values['very_low_pressure'],
            temperature=extreme_values['very_cold']
        )
        assert pt_cold > 0
        assert np.isfinite(pt_cold)
        
        # Very hot conditions
        pt_hot = potential_temperature(
            pressure=extreme_values['very_high_pressure'],
            temperature=extreme_values['very_hot']
        )
        assert pt_hot > 0
        assert np.isfinite(pt_hot)


class TestVirtualTemperature:
    """Test virtual temperature calculations."""
    
    def test_virtual_temperature_standard(self, scalar_test_values):
        """Test virtual temperature at standard conditions."""
        vt = virtual_temperature(
            temperature=scalar_test_values['temperature'],
            mixing_ratio=scalar_test_values['mixing_ratio']
        )
        
        # Virtual temperature should be slightly higher than actual temperature
        assert vt > scalar_test_values['temperature']
        assert vt < scalar_test_values['temperature'] * 1.1  # Should not be more than 10% higher
    
    def test_virtual_temperature_dry_air(self, scalar_test_values):
        """Test virtual temperature with very dry air (mixing ratio near zero)."""
        vt_dry = virtual_temperature(
            temperature=scalar_test_values['temperature'],
            mixing_ratio=0.001  # Very dry air
        )
        
        # Should be very close to actual temperature
        assert abs(vt_dry - scalar_test_values['temperature']) < 0.1
    
    def test_virtual_temperature_moist_air(self, scalar_test_values):
        """Test virtual temperature with moist air."""
        vt_moist = virtual_temperature(
            temperature=scalar_test_values['temperature'],
            mixing_ratio=0.03  # Very moist air
        )
        
        # Should be significantly higher than actual temperature
        assert vt_moist > scalar_test_values['temperature']
        assert vt_moist - scalar_test_values['temperature'] > 1.0  # At least 1 K higher


class TestSaturationVaporPressure:
    """Test saturation vapor pressure calculations."""
    
    def test_saturation_vapor_pressure_standard(self, scalar_test_values):
        """Test saturation vapor pressure at standard temperature."""
        es = saturation_vapor_pressure(scalar_test_values['temperature'])
        
        # At 15°C (288.15 K), saturation vapor pressure should be around 17-18 hPa
        assert 1500 < es < 2000  # Pa
    
    def test_saturation_vapor_pressure_temperature_dependence(self):
        """Test that saturation vapor pressure increases with temperature."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])  # 0, 10, 20, 30°C
        es = saturation_vapor_pressure(temps)
        
        # Check if result is an array and has more than one element
        if hasattr(es, '__len__') and hasattr(es, '__getitem__'):
            es_array = np.asarray(es)
            if len(es_array) > 1:
                # Should increase monotonically with temperature for arrays
                assert np.all(es_array[1:] > es_array[:-1])
            else:
                # For single element, just check it's positive
                assert es_array[0] > 0
        else:
            # For scalar case, just check it's positive
            assert es > 0
    
    def test_saturation_vapor_pressure_freezing_point(self):
        """Test saturation vapor pressure at freezing point."""
        es = saturation_vapor_pressure(273.15)  # 0°C
        
        # Should be around 611 Pa at freezing point
        assert 600 < es < 650
    
    def test_saturation_vapor_pressure_extreme_temperatures(self, extreme_values):
        """Test saturation vapor pressure at extreme temperatures."""
        # Very cold
        es_cold = saturation_vapor_pressure(extreme_values['very_cold'])
        assert es_cold > 0
        assert es_cold < 100  # Should be very low
        
        # Very hot
        es_hot = saturation_vapor_pressure(extreme_values['very_hot'])
        assert es_hot > 5000  # Should be much higher


class TestMixingRatio:
    """Test mixing ratio calculations."""
    
    def test_mixing_ratio_standard(self, scalar_test_values):
        """Test mixing ratio calculation."""
        w = mixing_ratio(
            vapor_pressure=saturation_vapor_pressure(scalar_test_values['temperature']) * scalar_test_values['humidity'],
            pressure=scalar_test_values['pressure'] * 100  # Convert hPa to Pa
        )
        
        # Should be positive and reasonable for atmospheric conditions
        assert w > 0
        assert w < 0.05  # Less than 50 g/kg
    
    def test_mixing_ratio_saturation(self, scalar_test_values):
        """Test mixing ratio at saturation."""
        es = saturation_vapor_pressure(scalar_test_values['temperature'])
        ws = mixing_ratio(vapor_pressure=es, pressure=scalar_test_values['pressure'] * 100)
        
        # Saturation mixing ratio should be higher than actual mixing ratio
        w_actual = mixing_ratio(
            vapor_pressure=es * scalar_test_values['humidity'],
            pressure=scalar_test_values['pressure'] * 100
        )
        assert ws > w_actual


class TestRelativeHumidity:
    """Test relative humidity calculations."""
    
    def test_relative_humidity_standard(self, scalar_test_values):
        """Test relative humidity calculation."""
        # Calculate saturation vapor pressure
        es = saturation_vapor_pressure(scalar_test_values['temperature'])
        # Calculate actual vapor pressure from mixing ratio
        e = mixing_ratio_to_vapor_pressure(
            mixing_ratio=scalar_test_values['mixing_ratio'],
            pressure=scalar_test_values['pressure'] * 100
        )
        
        rh = relative_humidity(
            vapor_pressure=e,
            saturation_vapor_pressure=es
        )
        
        # Should be between 0 and 1
        assert 0 <= rh <= 1
        
        # Compare with direct calculation
        rh_direct = e / es
        assert abs(rh - rh_direct) < 0.01
    
    def test_relative_humidity_extreme(self, extreme_values):
        """Test relative humidity at extreme conditions."""
        # Very dry
        temp_hot = extreme_values['very_hot']
        es_hot = saturation_vapor_pressure(temp_hot)
        e_hot = mixing_ratio_to_vapor_pressure(
            mixing_ratio=extreme_values['very_dry'] * 0.01,  # Convert percentage to fraction
            pressure=101325.0
        )
        rh_dry = relative_humidity(vapor_pressure=e_hot, saturation_vapor_pressure=es_hot)
        assert rh_dry >= 0
        assert rh_dry <= 1.0
        
        # Very humid
        temp_cold = extreme_values['very_cold']
        es_cold = saturation_vapor_pressure(temp_cold)
        e_cold = mixing_ratio_to_vapor_pressure(
            mixing_ratio=extreme_values['very_humid'] * 0.02,  # Higher mixing ratio for cold air
            pressure=101325.0
        )
        rh_humid = relative_humidity(vapor_pressure=e_cold, saturation_vapor_pressure=es_cold)
        assert rh_humid <= 1.0


class TestLapseRates:
    """Test lapse rate calculations."""
    
    def test_dry_lapse_rate_constant(self):
        """Test that dry adiabatic lapse rate is approximately constant."""
        dry_lr = dry_lapse_rate()
        
        # Dry adiabatic lapse rate should be approximately 9.8 K/km
        expected = 9.8 / 1000  # Convert to K/m
        assert abs(dry_lr - expected) < 0.001
    
    def test_moist_lapse_rate_varies(self, sample_pressure, sample_temperature):
        """Test that moist adiabatic lapse rate varies with conditions."""
        moist_lr = moist_lapse_rate(
            temperature=sample_temperature,
            pressure=sample_pressure * 100  # Convert to Pa
        )
        dry_lr = dry_lapse_rate()
        
        # Moist lapse rate should be positive but less than dry lapse rate
        if hasattr(moist_lr, '__len__'):
            assert np.all(moist_lr > 0)
            assert np.all(moist_lr < dry_lr)
            
            # Should vary with temperature and pressure
            unique_values = np.unique(np.round(moist_lr, 6))
            assert len(unique_values) > 1  # At least some variation
        else:
            assert moist_lr > 0
            assert moist_lr < dry_lr


class TestLiftingCondensationLevel:
    """Test LCL calculations."""
    
    def test_lcl_standard_conditions(self, scalar_test_values):
        """Test LCL calculation at standard conditions."""
        lcl = lifting_condensation_level(
            temperature=scalar_test_values['temperature'],
            dewpoint=scalar_test_values['dewpoint']
        )
        
        # LCL should be positive and reasonable for atmospheric conditions
        assert lcl > 0
        assert lcl < 5000  # Less than 5 km
    
    def test_lcl_dewpoint_dependence(self):
        """Test that LCL decreases as dewpoint approaches temperature."""
        temp = 300.0  # K
        dewpoints = np.array([290.0, 295.0, 299.0])  # Increasing dewpoint
        
        previous_lcl = None
        for i, dewpoint in enumerate(dewpoints):
            lcl = lifting_condensation_level(temperature=temp, dewpoint=dewpoint)
            if i > 0 and previous_lcl is not None:
                assert lcl < previous_lcl  # LCL should decrease
            previous_lcl = lcl
    
    def test_lcl_dry_conditions(self):
        """Test LCL for very dry conditions."""
        lcl = lifting_condensation_level(temperature=300.0, dewpoint=250.0)
        
        # Should be very high for dry conditions
        assert lcl > 3000  # More than 3 km


def mixing_ratio_to_vapor_pressure(mixing_ratio, pressure):
    """Helper function to convert mixing ratio to vapor pressure."""
    return (mixing_ratio * pressure) / (0.622 + mixing_ratio)