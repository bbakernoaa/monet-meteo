import numpy as np
import pytest
import xarray as xr
import dask.array as da
from monet_meteo.airquality import (
    aqi_us_epa,
    aqhi_canada,
    eaqi_europe,
)


def test_aqi_us_epa_values():
    # Test known breakpoints for PM2.5
    conc = np.array([0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4])
    expected = np.array([0, 50, 100, 150, 200, 300, 400, 500])
    res = aqi_us_epa(conc, "pm25")
    np.testing.assert_allclose(res, expected, atol=0.1)

    # Test intermediate value
    res_interp = aqi_us_epa(np.array([6.0]), "pm25")
    assert res_interp[0] == 25.0


def test_aqhi_canada_values():
    # Test with zero concentrations
    res = aqhi_canada(0.0, 0.0, 0.0)
    assert res == 0.0

    # Test with some values (results should be > 0)
    res = aqhi_canada(30.0, 15.0, 10.0)
    assert res > 0


def test_eaqi_europe_values():
    # Test PM2.5 bands: [0, 10, 20, 25, 50, 75, 800] -> levels [1, 2, 3, 4, 5, 6]
    conc = np.array([5, 15, 22, 30, 60, 100])
    expected = np.array([1, 2, 3, 4, 5, 6])
    res = eaqi_europe(conc, "pm25")
    np.testing.assert_array_equal(res, expected)


@pytest.mark.parametrize("func", [aqi_us_epa, eaqi_europe])
def test_aq_backend_agnostic(func):
    # Test with NumPy
    conc_np = np.array([10.0, 20.0, 30.0])
    if func == aqi_us_epa:
        res_np = func(conc_np, "pm25")
    else:
        res_np = func(conc_np, "pm25")

    # Test with Dask
    conc_da = da.from_array(conc_np, chunks=2)
    res_da = func(conc_da, "pm25")

    assert isinstance(res_da, da.Array)
    np.testing.assert_allclose(res_np, res_da.compute())

    # Test with Xarray (NumPy-backed)
    conc_xr = xr.DataArray(conc_np, dims="x", name="conc")
    res_xr = func(conc_xr, "pm25")

    assert isinstance(res_xr, xr.DataArray)
    np.testing.assert_allclose(res_np, res_xr.values)
    assert "history" in res_xr.attrs

    # Test with Xarray (Dask-backed) - The "Double-Check" test
    conc_xr_lazy = conc_xr.chunk({"x": 2})
    res_xr_lazy = func(conc_xr_lazy, "pm25")

    assert isinstance(res_xr_lazy.data, da.Array)
    np.testing.assert_allclose(res_xr.values, res_xr_lazy.compute().values)


def test_aqhi_canada_backend_agnostic():
    o3_np = np.array([30.0, 40.0])
    no2_np = np.array([10.0, 20.0])
    pm25_np = np.array([5.0, 15.0])

    res_np = aqhi_canada(o3_np, no2_np, pm25_np)

    # Xarray lazy
    o3_xr = xr.DataArray(o3_np, dims="t").chunk(1)
    no2_xr = xr.DataArray(no2_np, dims="t").chunk(1)
    pm25_xr = xr.DataArray(pm25_np, dims="t").chunk(1)

    res_xr_lazy = aqhi_canada(o3_xr, no2_xr, pm25_xr)
    assert isinstance(res_xr_lazy.data, da.Array)
    np.testing.assert_allclose(res_np, res_xr_lazy.compute().values)
