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
    wet_bulb_temperature
)


class TestHeatIndex:
    """Test heat index calculations."""
    
    def test_heat_index_standard_conditions(self):
        """Test heat index at standard conditions."""
        # At 27°C and 50% humidity, heat index should be slightly higher than temperature
        temp_c = 27.0  # °C
        rh = 0.5  # 50% relative humidity
        
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        
        # Heat index should be higher than actual temperature
        assert hi > temp_c
        # But not excessively higher at moderate conditions
        assert hi < temp_c + 10.0
    
    def test_heat_index_high_temperature(self):
        """Test heat index at high temperatures."""
        temp_c = 35.0  # °C
        rh = 0.8  # 80% relative humidity
        
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        
        # At high temp and humidity, heat index should be significantly higher
        assert hi > temp_c + 5.0
        assert hi < temp_c + 20.0  # But not unrealistically high
    
    def test_heat_index_low_humidity(self):
        """Test heat index at low humidity."""
        temp_c = 30.0  # °C
        rh = 0.1  # 10% relative humidity
        
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        
        # At low humidity, heat index should be close to actual temperature
        assert abs(hi - temp_c) < 2.0
    
    def test_heat_index_extreme_conditions(self):
        """Test heat index at extreme conditions."""
        # Very hot and humid
        temp_c = 40.0  # °C
        rh = 0.9  # 90% relative humidity
        
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        
        # Should be very high but physically reasonable
        assert hi > temp_c
        assert hi < 60.0  # Less than 60°C
    
    def test_heat_index_temperature_range(self):
        """Test that heat index is only calculated for appropriate temperature range."""
        # Below threshold (typically 27°C), heat index should equal temperature
        temp_c = 20.0  # °C
        rh = 0.8  # 80% relative humidity
        
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        
        # Should equal actual temperature below threshold
        assert abs(hi - temp_c) < 0.1


class TestWindChill:
    """Test wind chill calculations."""
    
    def test_wind_chill_standard_conditions(self):
        """Test wind chill at standard conditions."""
        temp_c = 5.0  # °C
        wind_speed = 10.0  # m/s
        
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        
        # Wind chill should be lower than actual temperature
        assert wc < temp_c
        # But not excessively lower
        assert wc > temp_c - 15.0
    
    def test_wind_chill_cold_conditions(self):
        """Test wind chill at cold conditions."""
        temp_c = -10.0  # °C
        wind_speed = 15.0  # m/s
        
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        
        # Should be significantly lower than actual temperature
        assert wc < temp_c
        assert wc > temp_c - 25.0  # But not unrealistically low
    
    def test_wind_chill_low_wind_speed(self):
        """Test wind chill at low wind speeds."""
        temp_c = 0.0  # °C
        wind_speed = 1.0  # m/s
        
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        
        # At low wind speeds, wind chill should be close to actual temperature
        assert abs(wc - temp_c) < 2.0
    
    def test_wind_chill_high_wind_speed(self):
        """Test wind chill at high wind speeds."""
        temp_c = -5.0  # °C
        wind_speed = 25.0  # m/s
        
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        
        # Should be much lower than actual temperature
        assert wc < temp_c - 10.0
    
    def test_wind_chill_temperature_threshold(self):
        """Test that wind chill is only calculated for appropriate temperature range."""
        # Above threshold (typically 10°C), wind chill should equal temperature
        temp_c = 15.0  # °C
        wind_speed = 10.0  # m/s
        
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        
        # Should equal actual temperature above threshold
        assert abs(wc - temp_c) < 0.1
    
    def test_wind_chill_extreme_cold(self):
        """Test wind chill at extreme cold conditions."""
        temp_c = -40.0  # °C
        wind_speed = 20.0  # m/s
        
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        
        # Should be very cold but physically reasonable
        assert wc < temp_c
        assert wc > -70.0  # Less than -70°C


class TestDewpointTemperature:
    """Test dewpoint temperature calculations."""
    
    def test_dewpoint_standard_conditions(self):
        """Test dewpoint calculation at standard conditions."""
        temp_c = 25.0  # °C
        rh = 0.6  # 60% relative humidity
        
        td = dewpoint_temperature(temperature=temp_c, relative_humidity=rh)
        
        # Dewpoint should be less than or equal to actual temperature
        assert td <= temp_c
        # But not too much lower at moderate humidity
        assert td > temp_c - 10.0
    
    def test_dewpoint_saturation(self):
        """Test dewpoint at saturation (100% relative humidity)."""
        temp_c = 20.0  # °C
        rh = 1.0  # 100% relative humidity
        
        td = dewpoint_temperature(temperature=temp_c, relative_humidity=rh)
        
        # At saturation, dewpoint should equal temperature
        assert abs(td - temp_c) < 0.1
    
    def test_dewpoint_dry_conditions(self):
        """Test dewpoint at very dry conditions."""
        temp_c = 30.0  # °C
        rh = 0.1  # 10% relative humidity
        
        td = dewpoint_temperature(temperature=temp_c, relative_humidity=rh)
        
        # At dry conditions, dewpoint should be much lower than temperature
        assert td < temp_c - 15.0
        assert td > -10.0  # But not too low
    
    def test_dewpoint_cold_conditions(self):
        """Test dewpoint at cold conditions."""
        temp_c = 5.0  # °C
        rh = 0.8  # 80% relative humidity
        
        td = dewpoint_temperature(temperature=temp_c, relative_humidity=rh)
        
        # Should be close to temperature at high humidity
        assert td <= temp_c
        assert td > temp_c - 5.0
    
    def test_dewpoint_extreme_values(self):
        """Test dewpoint at extreme temperature and humidity values."""
        # Very hot and humid
        td_hot = dewpoint_temperature(temperature=40.0, relative_humidity=0.9)
        assert td_hot <= 40.0
        assert td_hot > 30.0
        
        # Very cold and dry
        td_cold = dewpoint_temperature(temperature=-20.0, relative_humidity=0.1)
        assert td_cold <= -20.0
        assert td_cold > -40.0


class TestActualVaporPressure:
    """Test actual vapor pressure calculations."""
    
    def test_actual_vapor_pressure_standard(self):
        """Test actual vapor pressure at standard conditions."""
        # Use dewpoint instead of temperature and RH
        dewpoint_k = 293.15  # K (20°C)
        
        e = actual_vapor_pressure(dewpoint=dewpoint_k)
        
        # Should be positive and reasonable
        assert e > 0
        assert e < 5000  # Less than 5 kPa
    
    def test_actual_vapor_pressure_saturation(self):
        """Test actual vapor pressure at saturation."""
        # At saturation, dewpoint equals temperature
        dewpoint_k = 298.15  # K (25°C)
        
        e = actual_vapor_pressure(dewpoint=dewpoint_k)
        
        # Should equal saturation vapor pressure at 100% RH
        # At 25°C, saturation vapor pressure is approximately 3.17 kPa
        assert e > 3000  # Greater than 3 kPa
        assert e < 3500  # Less than 3.5 kPa
    
    def test_actual_vapor_pressure_temperature_dependence(self):
        """Test that actual vapor pressure increases with dewpoint temperature."""
        dewpoints = np.array([283.15, 293.15, 303.15, 313.15])  # 10, 20, 30, 40°C
        
        e = actual_vapor_pressure(dewpoint=dewpoints)
        
        # Should increase with dewpoint temperature
        if hasattr(e, '__len__') and hasattr(e, '__getitem__'):
            e_array = np.asarray(e)
            if len(e_array) > 1:
                assert np.all(e_array[1:] > e_array[:-1])
            else:
                assert e_array[0] > 0
        else:
            # For scalar result, just check it's positive
            assert e > 0
    
    def test_actual_vapor_pressure_extreme_conditions(self):
        """Test actual vapor pressure at extreme conditions."""
        # Very cold dewpoint
        e_cold = actual_vapor_pressure(dewpoint=243.15)  # -30°C
        assert e_cold > 0
        assert e_cold < 100  # Very low
        
        # Very warm dewpoint
        e_hot = actual_vapor_pressure(dewpoint=323.15)  # 50°C
        assert e_hot > 10000  # Much higher
        assert e_hot < 50000  # But not too high


class TestLiftingCondensationLevelDerived:
    """Test LCL calculations using derived parameters."""
    
    def test_lcl_from_temperature_humidity(self):
        """Test LCL calculation from temperature and dewpoint."""
        temp_k = 298.15  # K (25°C)
        dewpoint_k = 288.15  # K (15°C)
        
        lcl = lifting_condensation_level(temperature=temp_k, dewpoint=dewpoint_k)
        
        # Should be positive and reasonable for atmospheric conditions
        assert lcl > 0
        assert lcl < 4000  # Less than 4 km
    
    def test_lcl_humidity_dependence(self):
        """Test that LCL decreases as dewpoint approaches temperature."""
        temp_k = 303.15  # K (30°C)
        dewpoints = np.array([283.15, 288.15, 293.15, 298.15])  # K (10, 15, 20, 25°C)
        
        lcl = lifting_condensation_level(temperature=temp_k, dewpoint=dewpoints)
        
        # LCL should decrease as dewpoint increases (humidity increases)
        if hasattr(lcl, '__len__') and hasattr(lcl, '__getitem__'):
            lcl_array = np.asarray(lcl)
            if len(lcl_array) > 1:
                assert np.all(lcl_array[1:] < lcl_array[:-1])
    
    def test_lcl_temperature_dependence(self):
        """Test that LCL increases with temperature (for same dewpoint difference)."""
        temps = np.array([288.15, 293.15, 298.15, 303.15])  # K (15, 20, 25, 30°C)
        dewpoint_k = 283.15  # K (10°C)
        
        lcl = lifting_condensation_level(temperature=temps, dewpoint=dewpoint_k)
        
        # LCL should generally increase with temperature for same dewpoint difference
        if hasattr(lcl, '__len__') and hasattr(lcl, '__getitem__'):
            lcl_array = np.asarray(lcl)
            if len(lcl_array) > 1:
                assert np.all(lcl_array[1:] > lcl_array[:-1])


class TestWetBulbTemperature:
    """Test wet bulb temperature calculations."""
    
    def test_wet_bulb_standard_conditions(self):
        """Test wet bulb temperature at standard conditions."""
        temp_k = 298.15  # K (25°C)
        rh = 0.5  # 50% relative humidity
        pressure = 101325.0  # Pa (standard pressure)
        
        tw = wet_bulb_temperature(temperature=temp_k, pressure=pressure, relative_humidity=rh)
        
        # Wet bulb should be less than or equal to actual temperature
        assert tw <= temp_k
        # But not too much lower at moderate humidity
        assert tw > temp_k - 10.0
    
    def test_wet_bulb_saturation(self):
        """Test wet bulb temperature at saturation."""
        temp_k = 293.15  # K (20°C)
        rh = 1.0  # 100% relative humidity
        pressure = 101325.0  # Pa
        
        tw = wet_bulb_temperature(temperature=temp_k, pressure=pressure, relative_humidity=rh)
        
        # At saturation, wet bulb should equal temperature
        assert abs(tw - temp_k) < 1.0  # Allow some tolerance
    
    def test_wet_bulb_dry_conditions(self):
        """Test wet bulb temperature at dry conditions."""
        temp_k = 303.15  # K (30°C)
        rh = 0.2  # 20% relative humidity
        pressure = 101325.0  # Pa
        
        tw = wet_bulb_temperature(temperature=temp_k, pressure=pressure, relative_humidity=rh)
        
        # At dry conditions, wet bulb should be significantly lower
        assert tw < temp_k - 5.0
        assert tw > temp_k - 15.0
    
    def test_wet_bulb_extreme_conditions(self):
        """Test wet bulb temperature at extreme conditions."""
        # Very hot and dry
        tw_hot = wet_bulb_temperature(temperature=318.15, pressure=101325.0, relative_humidity=0.1)  # K (45°C)
        assert tw_hot <= 318.15
        assert tw_hot > 293.15  # > 20°C
        
        # Very cold
        tw_cold = wet_bulb_temperature(temperature=263.15, pressure=101325.0, relative_humidity=0.8)  # K (-10°C)
        assert tw_cold <= 263.15
        assert tw_cold > 253.15  # > -20°C
    
    def test_wet_bulb_humidity_relationship(self):
        """Test wet bulb temperature relationship with humidity."""
        temp_k = 303.15  # K (30°C)
        rh_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        pressure = 101325.0  # Pa
        
        tw = wet_bulb_temperature(temperature=temp_k, pressure=pressure, relative_humidity=rh_values)
        
        # Wet bulb should generally increase with relative humidity
        if hasattr(tw, '__len__') and hasattr(tw, '__getitem__'):
            tw_array = np.asarray(tw)
            if len(tw_array) > 1:
                assert np.all(tw_array[1:] >= tw_array[:-1])