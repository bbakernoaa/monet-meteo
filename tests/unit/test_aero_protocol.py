"""
Test suite for Aero Protocol compliance (NumPy vs Dask).
"""

import numpy as np
import pytest
import xarray as xr
import dask.array as da
from monet_meteo.thermodynamics import (
    potential_temperature,
    virtual_temperature,
    saturation_vapor_pressure,
)
from monet_meteo.dynamics import (
    geostrophic_wind,
)

@pytest.mark.parametrize("func", [
    (potential_temperature, {"pressure": 101325.0, "temperature": 300.0}),
    (virtual_temperature, {"temperature": 300.0, "mixing_ratio": 0.01}),
    (saturation_vapor_pressure, {"temperature": 300.0}),
])
def test_thermo_aero_protocol(func):
    """Verify thermodynamics functions follow Aero Protocol."""
    f, kwargs = func
    # Create input arrays
    np_kwargs = {k: np.array([v] * 4) if isinstance(v, float) else v for k, v in kwargs.items()}

    # NumPy result
    res_np = f(**np_kwargs)

    # Xarray Eager (NumPy)
    xr_kwargs = {k: xr.DataArray(v, dims="x") if isinstance(v, np.ndarray) else v for k, v in np_kwargs.items()}
    res_xr_eager = f(**xr_kwargs)

    assert isinstance(res_xr_eager, xr.DataArray)
    np.testing.assert_allclose(res_np, res_xr_eager.values)

    # Xarray Lazy (Dask)
    xr_lazy_kwargs = {k: v.chunk({"x": 2}) if isinstance(v, xr.DataArray) else v for k, v in xr_kwargs.items()}
    res_xr_lazy = f(**xr_lazy_kwargs)

    assert isinstance(res_xr_lazy.data, da.Array)
    np.testing.assert_allclose(res_np, res_xr_lazy.compute().values)


def test_dynamics_geostrophic_aero_protocol():
    """Verify dynamics functions follow Aero Protocol."""
    # Setup data
    lon = np.linspace(0, 360, 20)
    lat = np.linspace(-90, 90, 10)
    h_data = np.random.uniform(5000, 5500, (10, 20))

    h = xr.DataArray(h_data, coords=[lat, lon], dims=["lat", "lon"])

    # Eager
    ug, vg = geostrophic_wind(h, latitude=h.lat)

    # Lazy
    h_lazy = h.chunk({"lat": 5, "lon": 10})
    ug_lazy, vg_lazy = geostrophic_wind(h_lazy, latitude=h_lazy.lat)

    assert isinstance(ug_lazy.data, da.Array)
    assert isinstance(vg_lazy.data, da.Array)

    np.testing.assert_allclose(ug.values, ug_lazy.compute().values)
    np.testing.assert_allclose(vg.values, vg_lazy.compute().values)
