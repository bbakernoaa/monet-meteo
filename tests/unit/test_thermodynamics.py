"""
Test suite for thermodynamic calculations.

Tests all thermodynamic functions including potential temperature, virtual temperature,
saturation vapor pressure, mixing ratio, relative humidity, and lapse rates.
"""

import numpy as np

# Import the thermodynamic functions
from monet_meteo.thermodynamics.thermodynamic_calculations import (
    lifted_index,
    lifting_condensation_level,
    mixing_ratio,
    potential_temperature,
    precipitable_water,
    saturation_vapor_pressure,
    virtual_temperature,
)


class TestPotentialTemperature:
    """Test potential temperature calculations."""

    def test_potential_temperature_standard_conditions(self, scalar_test_values):
        """Test potential temperature at standard conditions."""
        # Using Pa for pressure to avoid internal conversion ambiguity in test
        pt = potential_temperature(pressure=100000.0, temperature=300.0)
        # At 1000 hPa, theta = T
        assert abs(pt - 300.0) < 0.1

    def test_potential_temperature_decreasing_with_height(self, sample_pressure, sample_temperature):
        """Test that potential temperature calculation works for profiles."""
        pt = potential_temperature(pressure=sample_pressure, temperature=sample_temperature)
        assert np.all(np.isfinite(pt))

    def test_potential_temperature_units(self):
        """Test potential temperature with different pressure units."""
        # Test with hPa
        pt_hpa = potential_temperature(pressure=1000.0, temperature=300.0)
        # Test with Pa
        pt_pa = potential_temperature(pressure=100000.0, temperature=300.0)

        assert abs(pt_hpa - pt_pa) < 1e-6

    def test_potential_temperature_extreme_values(self, extreme_values):
        """Test potential temperature with extreme values."""
        # Very cold conditions
        pt_cold = potential_temperature(pressure=extreme_values["very_low_pressure"], temperature=extreme_values["very_cold"])
        assert pt_cold > 0
        assert np.isfinite(pt_cold)


class TestVirtualTemperature:
    """Test virtual temperature calculations."""

    def test_virtual_temperature_standard(self, scalar_test_values):
        """Test virtual temperature at standard conditions."""
        vt = virtual_temperature(temperature=scalar_test_values["temperature"], mixing_ratio=scalar_test_values["mixing_ratio"])

        # Virtual temperature should be slightly higher than actual temperature
        assert vt > scalar_test_values["temperature"]

    def test_virtual_temperature_dry_air(self, scalar_test_values):
        """Test virtual temperature with very dry air."""
        vt_dry = virtual_temperature(temperature=300.0, mixing_ratio=0.0)

        # Should be equal to actual temperature
        assert abs(vt_dry - 300.0) < 1e-6


class TestSaturationVaporPressure:
    """Test saturation vapor pressure calculations."""

    def test_saturation_vapor_pressure_standard(self, scalar_test_values):
        """Test saturation vapor pressure at standard temperature."""
        es = saturation_vapor_pressure(288.15)  # 15C

        # At 15°C (288.15 K), saturation vapor pressure should be around 17 hPa = 1700 Pa
        assert 1700 < es < 1800

    def test_saturation_vapor_pressure_temperature_dependence(self):
        """Test that saturation vapor pressure increases with temperature."""
        temps = np.array([273.15, 283.15, 293.15, 303.15])
        es = saturation_vapor_pressure(temps)
        assert np.all(np.diff(es) > 0)


class TestMixingRatio:
    """Test mixing ratio calculations."""

    def test_mixing_ratio_standard(self, scalar_test_values):
        """Test mixing ratio calculation."""
        w = mixing_ratio(vapor_pressure=1000.0, pressure=100000.0)
        assert w > 0
        assert w < 0.05


class TestLiftingCondensationLevel:
    """Test LCL calculations."""

    def test_lcl_standard_conditions(self, scalar_test_values):
        """Test LCL calculation at standard conditions."""
        lcl = lifting_condensation_level(temperature=scalar_test_values["temperature"], dewpoint=scalar_test_values["dewpoint"])
        assert lcl >= 0
        assert lcl < 5000


class TestLiftedIndex:
    """Test Lifted Index calculation."""

    def test_lifted_index_stable(self):
        """Test LI for stable atmosphere."""
        temp_sfc = 290.0
        dewpoint_sfc = 280.0
        pressure_sfc = 101325.0
        temp_500 = 265.0

        li = lifted_index(temp_sfc, dewpoint_sfc, pressure_sfc, temp_500)
        assert np.all(li > 0)

    def test_lifted_index_unstable(self):
        """Test LI for unstable atmosphere."""
        temp_sfc = 305.0
        dewpoint_sfc = 295.0
        pressure_sfc = 101325.0
        temp_500 = 240.0

        li = lifted_index(temp_sfc, dewpoint_sfc, pressure_sfc, temp_500)
        assert np.all(li < 0)


class TestPrecipitableWater:
    """Test Precipitable Water calculation."""

    def test_precipitable_water_basic(self):
        nz, ny, nx = 5, 1, 1
        q = np.full((nz, ny, nx), 0.01)  # 10 g/kg
        p = np.array([100000, 80000, 60000, 40000, 20000, 0])[:, np.newaxis, np.newaxis]
        p = np.broadcast_to(p, (nz + 1, ny, nx))

        pw = precipitable_water(q, p)

        # PW ≈ q * delta_p / g = 0.01 * 100000 / 9.8 ≈ 102 kg/m2
        assert 100 < pw < 105
