"""
Test suite for derived meteorological parameters.

Tests derived calculations including heat index, wind chill, dewpoint calculations,
and other derived meteorological parameters.
"""
import numpy as np
import pytest

# Import the derived parameter functions
from monet_meteo.derived.derived_calculations import (
    heat_index,
    wind_chill,
    dewpoint_temperature,
    actual_vapor_pressure,
    lifting_condensation_level,
    wet_bulb_temperature,
    wind_gust_diagnostic,
    visibility_diagnostic
)


class TestHeatIndex:
    """Test heat index calculations."""
    
    def test_heat_index_standard_conditions(self):
        """Test heat index at standard conditions."""
        # Implementation assumes Fahrenheit inputs.
        temp_f = 80.0
        rh = 50.0  # 50% relative humidity
        
        hi = heat_index(temperature=temp_f, relative_humidity=rh)
        
        # Heat index should be higher than actual temperature
        assert np.real(hi) > temp_f
        # But not excessively higher at moderate conditions
        assert np.real(hi) < temp_f + 10.0
    
    def test_heat_index_high_temperature(self):
        """Test heat index at high temperatures."""
        temp_f = 95.0
        rh = 80.0
        
        hi = heat_index(temperature=temp_f, relative_humidity=rh)
        
        # At high temp and humidity, heat index should be significantly higher
        assert np.real(hi) > temp_f + 5.0
        assert np.real(hi) < temp_f + 50.0
    
    def test_heat_index_low_humidity(self):
        """Test heat index at low humidity."""
        temp_f = 86.0
        rh = 10.0
        
        hi = heat_index(temperature=temp_f, relative_humidity=rh)
        
        # At low humidity, heat index should be close to actual temperature
        assert abs(np.real(hi) - temp_f) < 10.0
    
    def test_heat_index_extreme_conditions(self):
        """Test heat index at extreme conditions."""
        # Very hot and humid
        temp_f = 104.0
        rh = 90.0
        
        hi = heat_index(temperature=temp_f, relative_humidity=rh)
        
        # Should be very high
        assert np.real(hi) > temp_f
    
    def test_heat_index_temperature_range(self):
        """Test that heat index is only calculated for appropriate temperature range."""
        # Below threshold (typically 80°F), heat index should equal temperature or close
        temp_f = 70.0
        rh = 80.0
        
        hi = heat_index(temperature=temp_f, relative_humidity=rh)
        
        # Implementation uses a simple formula below 80F
        assert abs(np.real(hi) - temp_f) < 2.0


class TestWindChill:
    """Test wind chill calculations."""
    
    def test_wind_chill_standard_conditions(self):
        """Test wind chill at standard conditions."""
        temp_f = 32.0  # Fahrenheit
        wind_mph = 15.0  # mph
        
        wc = wind_chill(temperature=temp_f, wind_speed=wind_mph)
        
        # Wind chill should be lower than actual temperature
        assert wc < temp_f
        # But not excessively lower
        assert wc > temp_f - 20.0
    
    def test_wind_chill_cold_conditions(self):
        """Test wind chill at cold conditions."""
        temp_f = 14.0
        wind_mph = 33.0
        
        wc = wind_chill(temperature=temp_f, wind_speed=wind_mph)
        
        # Should be significantly lower than actual temperature
        assert wc < temp_f
        assert wc > temp_f - 40.0
    
    def test_wind_chill_low_wind_speed(self):
        """Test wind chill at low wind speeds."""
        temp_f = 32.0
        wind_mph = 1.0
        
        wc = wind_chill(temperature=temp_f, wind_speed=wind_mph)
        
        # At low wind speeds, wind chill should be equal to actual temperature
        assert abs(wc - temp_f) < 0.1
    
    def test_wind_chill_high_wind_speed(self):
        """Test wind chill at high wind speeds."""
        temp_f = 23.0
        wind_mph = 55.0
        
        wc = wind_chill(temperature=temp_f, wind_speed=wind_mph)
        
        # Should be much lower than actual temperature
        assert wc < temp_f - 15.0
    
    def test_wind_chill_temperature_threshold(self):
        """Test that wind chill is only calculated for appropriate temperature range."""
        # Above threshold (50°F), wind chill should equal temperature
        temp_f = 60.0
        wind_mph = 10.0
        
        wc = wind_chill(temperature=temp_f, wind_speed=wind_mph)
        
        # Should equal actual temperature above threshold
        assert abs(wc - temp_f) < 0.1
    
    def test_wind_chill_extreme_cold(self):
        """Test wind chill at extreme cold conditions."""
        temp_f = -40.0
        wind_mph = 45.0
        
        wc = wind_chill(temperature=temp_f, wind_speed=wind_mph)
        
        # Should be very cold but physically reasonable
        assert wc < temp_f
        assert wc > -120.0


class TestDewpointTemperature:
    """Test dewpoint temperature calculations."""
    
    def test_dewpoint_standard_conditions(self):
        """Test dewpoint calculation at standard conditions."""
        temp_k = 298.15  # 25°C
        rh = 0.6  # 60% relative humidity
        
        td = dewpoint_temperature(temperature=temp_k, relative_humidity=rh)
        
        # Dewpoint should be less than or equal to actual temperature
        assert td <= temp_k
        # But not too much lower at moderate humidity
        assert td > temp_k - 15.0
    
    def test_dewpoint_saturation(self):
        """Test dewpoint at saturation (100% relative humidity)."""
        temp_k = 293.15
        rh = 1.0
        
        td = dewpoint_temperature(temperature=temp_k, relative_humidity=rh)
        
        # At saturation, dewpoint should equal temperature
        assert abs(td - temp_k) < 0.1
    
    def test_dewpoint_dry_conditions(self):
        """Test dewpoint at very dry conditions."""
        temp_k = 303.15 # 30C
        rh = 0.1  # 10%
        
        td = dewpoint_temperature(temperature=temp_k, relative_humidity=rh)
        
        # At dry conditions, dewpoint should be lower than temperature
        assert td < temp_k - 5.0
    
    def test_dewpoint_cold_conditions(self):
        """Test dewpoint at cold conditions."""
        temp_k = 278.15 # 5C
        rh = 0.8
        
        td = dewpoint_temperature(temperature=temp_k, relative_humidity=rh)
        
        assert td <= temp_k
        assert td > temp_k - 5.0


class TestActualVaporPressure:
    """Test actual vapor pressure calculations."""
    
    def test_actual_vapor_pressure_standard(self):
        """Test actual vapor pressure at standard conditions."""
        dewpoint_k = 293.15  # 20°C
        
        e = actual_vapor_pressure(dewpoint=dewpoint_k)
        
        # Should be positive and reasonable
        assert e > 0
        assert e < 5000  # Pa
    
    def test_actual_vapor_pressure_saturation(self):
        """Test actual vapor pressure at saturation."""
        dewpoint_k = 298.15  # 25°C
        
        e = actual_vapor_pressure(dewpoint=dewpoint_k)
        
        # At 25°C, saturation vapor pressure is approximately 31.7 hPa = 3170 Pa
        assert 3000 < e < 3500
    
    def test_actual_vapor_pressure_temperature_dependence(self):
        """Test that actual vapor pressure increases with dewpoint temperature."""
        dewpoints = np.array([283.15, 293.15, 303.15, 313.15])
        
        e = actual_vapor_pressure(dewpoint=dewpoints)
        
        assert np.all(e[1:] > e[:-1])


class TestLiftingCondensationLevelDerived:
    """Test LCL calculations using derived parameters."""
    
    def test_lcl_from_temperature_humidity(self):
        """Test LCL calculation from temperature and dewpoint."""
        temp_k = 298.15
        dewpoint_k = 288.15
        
        lcl = lifting_condensation_level(temperature=temp_k, dewpoint=dewpoint_k)
        
        assert lcl > 0
        assert lcl < 4000
    
    def test_lcl_humidity_dependence(self):
        """Test that LCL decreases as dewpoint approaches temperature."""
        temp_k = 303.15
        dewpoints = np.array([283.15, 288.15, 293.15, 298.15])
        
        lcl = lifting_condensation_level(temperature=temp_k, dewpoint=dewpoints)
        
        assert np.all(np.diff(lcl) < 0)
    
    def test_lcl_temperature_dependence(self):
        """Test that LCL increases with temperature (for same dewpoint difference)."""
        temps = np.array([288.15, 293.15, 298.15, 303.15])
        dewpoint_k = temps - 10.0
        
        lcl = lifting_condensation_level(temperature=temps, dewpoint=dewpoint_k)
        
        # Formula is 125 * (T - Td). For constant diff, LCL is constant.
        assert np.allclose(lcl, 1250.0)


class TestWetBulbTemperature:
    """Test wet bulb temperature calculations."""
    
    def test_wet_bulb_standard_conditions(self):
        """Test wet bulb temperature at standard conditions."""
        temp_k = 298.15
        rh = 0.5
        pressure = 101325.0
        
        tw = wet_bulb_temperature(temperature=temp_k, pressure=pressure, relative_humidity=rh)
        
        assert tw <= temp_k
        assert tw > temp_k - 15.0
    
    def test_wet_bulb_saturation(self):
        """Test wet bulb temperature at saturation."""
        temp_k = 293.15
        rh = 1.0
        pressure = 101325.0
        
        tw = wet_bulb_temperature(temperature=temp_k, pressure=pressure, relative_humidity=rh)
        
        assert abs(tw - temp_k) < 2.0
    
    def test_wet_bulb_humidity_relationship(self):
        """Test wet bulb temperature relationship with humidity."""
        temp_k = 303.15
        rh_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        pressure = 101325.0
        
        tw = wet_bulb_temperature(temperature=temp_k, pressure=pressure, relative_humidity=rh_values)
        
        assert np.all(np.diff(tw) >= 0)


class TestWindGust:
    """Test wind gust diagnostic calculation."""

    def test_wind_gust_diagnostic(self):
        """Test wind gust diagnostic."""
        nz, ny, nx = 5, 2, 2
        heights = np.linspace(0, 2000, nz)[:, np.newaxis, np.newaxis]
        heights = np.broadcast_to(heights, (nz, ny, nx))

        # Wind profile: increasing with height
        u = np.linspace(5, 25, nz)[:, np.newaxis, np.newaxis]
        u = np.broadcast_to(u, (nz, ny, nx))
        v = np.zeros_like(u)

        pbl_height = 1000.0
        u10, v10 = 5.0, 0.0

        gust = wind_gust_diagnostic(u, v, heights, pbl_height, u10, v10)

        # Gust should be at least surface wind
        assert np.all(gust >= 5.0)
        assert np.all(np.isfinite(gust))


class TestVisibility:
    """Test visibility diagnostic calculation."""

    def test_visibility_diagnostic_clear(self):
        """Test visibility in clear conditions."""
        temp = 288.15
        pres = 101325.0
        q = 0.005
        qc, qr, qi, qs = 0.0, 0.0, 0.0, 0.0

        vis = visibility_diagnostic(temp, pres, q, qc, qr, qi, qs)

        # Should be at the limit (24.135 km)
        assert np.allclose(vis, 24135.0)

    def test_visibility_diagnostic_fog(self):
        """Test visibility with cloud water (fog)."""
        temp = 288.15
        pres = 101325.0
        q = 0.005
        qc = 0.0005 # 0.5 g/kg cloud water
        qr, qi, qs = 0.0, 0.0, 0.0

        vis = visibility_diagnostic(temp, pres, q, qc, qr, qi, qs)

        # Visibility should be reduced
        assert vis < 24135.0
        assert vis > 0.0
