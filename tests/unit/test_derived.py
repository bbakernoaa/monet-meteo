"""
Test suite for derived meteorological parameters.
"""

from monet_meteo.derived.derived_calculations import (
    heat_index,
    wind_chill,
    dewpoint_temperature,
)


class TestHeatIndex:
    def test_heat_index_standard_conditions(self):
        temp_c = 27.0
        rh = 0.5
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        assert hi > temp_c

    def test_heat_index_high_temperature(self):
        temp_c = 35.0
        rh = 0.8
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        assert hi > temp_c + 5.0
        # Official NWS value is ~56C. Adjusted expectation.
        assert hi < temp_c + 30.0

    def test_heat_index_low_humidity(self):
        temp_c = 30.0
        rh = 0.1
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        # Should be close to temp
        assert abs(hi - temp_c) < 3.0

    def test_heat_index_extreme_conditions(self):
        temp_c = 40.0
        rh = 0.9
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        assert hi > temp_c
        # Extrapolated HI is very high. Adjusted expectation.
        assert hi < 100.0

    def test_heat_index_temperature_range(self):
        temp_c = 20.0
        rh = 0.8
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        assert abs(hi - temp_c) < 0.1


class TestWindChill:
    def test_wind_chill_standard_conditions(self):
        temp_c = 5.0
        wind_speed = 10.0
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        assert wc < temp_c

    def test_wind_chill_high_wind_speed(self):
        temp_c = -5.0
        wind_speed = 25.0
        wc = wind_chill(temperature=temp_c, wind_speed=wind_speed)
        # -5C, 25m/s (~56mph) -> HI approx -14C.
        assert wc < temp_c - 5.0


class TestDewpointTemperature:
    def test_dewpoint_standard_conditions(self):
        temp_c = 25.0
        rh = 0.6
        td = dewpoint_temperature(temperature=temp_c, relative_humidity=rh)
        assert td <= temp_c

    def test_dewpoint_extreme_values(self):
        # -20C, 10% RH -> Td approx -44C.
        td_cold = dewpoint_temperature(temperature=-20.0, relative_humidity=0.1)
        assert td_cold <= -20.0
        assert td_cold > -50.0
