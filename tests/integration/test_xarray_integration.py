"""
Integration tests for xarray and Dask compatibility.
"""

import numpy as np
import pytest
import xarray as xr

from monet_meteo.thermodynamics.thermodynamic_calculations import (
    potential_temperature,
    virtual_temperature,
    saturation_vapor_pressure,
    mixing_ratio,
)
from monet_meteo.derived.derived_calculations import (
    heat_index,
)


class TestXarrayIntegration:
    @pytest.fixture
    def sample_xarray_data(self):
        pressure_levels = np.array([1000.0, 925.0, 850.0, 700.0, 500.0])
        latitudes = np.linspace(-90, 90, 10)
        longitudes = np.linspace(-180, 180, 20)
        temperature_data = 280.0 + 20.0 * np.random.randn(5, 10, 20)
        pressure_data = pressure_levels[:, np.newaxis, np.newaxis] * (
            1 + 0.1 * np.random.randn(5, 10, 20)
        )
        humidity_data = np.random.uniform(0.1, 0.9, (5, 10, 20))
        u_wind_data = 10.0 * np.random.randn(5, 10, 20)
        v_wind_data = 5.0 * np.random.randn(5, 10, 20)
        coords = {
            "pressure": pressure_levels,
            "latitude": latitudes,
            "longitude": longitudes,
        }
        temperature = xr.DataArray(
            temperature_data,
            coords=coords,
            dims=["pressure", "latitude", "longitude"],
            attrs={"units": "K", "long_name": "Air Temperature"},
        )
        pressure = xr.DataArray(
            pressure_data,
            coords=coords,
            dims=["pressure", "latitude", "longitude"],
            attrs={"units": "Pa", "long_name": "Atmospheric Pressure"},
        )
        humidity = xr.DataArray(
            humidity_data,
            coords=coords,
            dims=["pressure", "latitude", "longitude"],
            attrs={"units": "1", "long_name": "Relative Humidity"},
        )
        return {
            "temperature": temperature,
            "pressure": pressure,
            "humidity": humidity,
        }

    def test_potential_temperature_xarray(self, sample_xarray_data):
        temp = sample_xarray_data["temperature"]
        pressure = sample_xarray_data["pressure"]
        theta = potential_temperature(pressure=pressure, temperature=temp)
        assert isinstance(theta, xr.DataArray)
        assert theta.dims == temp.dims
        assert np.all(np.isfinite(theta))
        assert "units" in theta.attrs

    def test_virtual_temperature_xarray(self, sample_xarray_data):
        temp = sample_xarray_data["temperature"]
        humidity = sample_xarray_data["humidity"]
        pressure = sample_xarray_data["pressure"]
        mixing_ratio_val = mixing_ratio(
            vapor_pressure=humidity * saturation_vapor_pressure(temp), pressure=pressure
        )
        t_virtual = virtual_temperature(temperature=temp, mixing_ratio=mixing_ratio_val)
        assert isinstance(t_virtual, xr.DataArray)
        assert t_virtual.dims == temp.dims
        # Virtual temp >= actual temp
        assert np.all(t_virtual.values >= temp.values - 1e-5)

    def test_heat_index_xarray(self, sample_xarray_data):
        temp_c = sample_xarray_data["temperature"] - 273.15
        rh = sample_xarray_data["humidity"] * 100
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        assert isinstance(hi, xr.DataArray)
        assert hi.dims == temp_c.dims
        warm_mask = (temp_c > 27.0) & (rh > 40.0)
        if np.any(warm_mask):
            # Use values to avoid NaN comparison issues in np.all
            hi_vals = hi.values[warm_mask.values]
            tc_vals = temp_c.values[warm_mask.values]
            assert np.all(hi_vals >= tc_vals - 0.1)


class TestCoordinatePreservation:
    def test_attribute_preservation(self):
        temp_data = np.array([280.0, 290.0, 300.0])
        pressure_data = np.array([100000.0, 85000.0, 70000.0])
        temp = xr.DataArray(
            temp_data,
            dims=["level"],
            attrs={"units": "K", "standard_name": "air_temperature"},
        )
        pressure = xr.DataArray(
            pressure_data,
            dims=["level"],
            attrs={"units": "Pa", "standard_name": "air_pressure"},
        )
        theta = potential_temperature(pressure=pressure, temperature=temp)
        assert "units" in theta.attrs
        assert theta.attrs["units"] == "K"
        assert "standard_name" in theta.attrs
        assert "potential_temperature" in theta.attrs["standard_name"]
