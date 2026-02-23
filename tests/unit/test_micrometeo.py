"""
Test suite for micro-meteorology calculations.
"""

import numpy as np
import xarray as xr
import dask.array as da

from monet_meteo.micrometeo.micrometeo_calculations import (
    psi_m,
    friction_velocity,
    obukhov_length,
    richardson_bulk,
)
from monet_meteo.micrometeo.solar import sun_angles


def test_double_check_psi_m():
    """Verify psi_m gives same result for NumPy and Dask."""
    zol_values = np.linspace(-2, 2, 20)

    # NumPy
    res_np = psi_m(zol_values)

    # Dask (via xarray)
    zol_xr = xr.DataArray(da.from_array(zol_values, chunks=5), dims="x")
    res_xr = psi_m(zol_xr)

    assert isinstance(res_xr.data, da.Array)
    res_dask = res_xr.compute()

    np.testing.assert_allclose(res_np, res_dask.values)
    assert "Calculated psi_m" in res_xr.attrs["history"]


def test_double_check_friction_velocity():
    """Verify friction_velocity gives same result for NumPy and Dask."""
    u = np.array([5.0, 10.0])
    z = np.array([10.0, 10.0])
    L = np.array([-100.0, 100.0])
    z0m = 0.1

    # NumPy
    res_np = friction_velocity(u, z, L, z0m)

    # Dask
    u_xr = xr.DataArray(da.from_array(u, chunks=1), dims="x")
    z_xr = xr.DataArray(da.from_array(z, chunks=1), dims="x")
    L_xr = xr.DataArray(da.from_array(L, chunks=1), dims="x")

    res_xr = friction_velocity(u_xr, z_xr, L_xr, z0m)
    res_dask = res_xr.compute()

    np.testing.assert_allclose(res_np, res_dask.values)


def test_double_check_obukhov_length():
    """Verify obukhov_length gives same result for NumPy and Dask."""
    ustar = np.array([0.3, 0.5])
    temp = np.array([290.0, 300.0])
    h = np.array([50.0, -10.0])
    rho = 1.2

    # NumPy
    res_np = obukhov_length(ustar, temp, h, rho)

    # Dask
    ustar_xr = xr.DataArray(da.from_array(ustar, chunks=1), dims="x")
    temp_xr = xr.DataArray(da.from_array(temp, chunks=1), dims="x")
    h_xr = xr.DataArray(da.from_array(h, chunks=1), dims="x")

    res_xr = obukhov_length(ustar_xr, temp_xr, h_xr, rho)
    res_dask = res_xr.compute()

    np.testing.assert_allclose(res_np, res_dask.values)


def test_sun_angles():
    """Test sun angles calculation."""
    lat, lon, stdlon = 40.0, -80.0, -75.0
    doy = 180
    ftime = 12.0

    sza, saa = sun_angles(lat, lon, stdlon, doy, ftime)

    assert 0 <= sza <= 180
    assert 0 <= saa <= 360

    # Dask check
    lat_xr = xr.DataArray(da.from_array([lat], chunks=1), dims="x")
    sza_xr, saa_xr = sun_angles(lat_xr, lon, stdlon, doy, ftime)

    assert isinstance(sza_xr.data, da.Array)
    np.testing.assert_allclose(sza, sza_xr.compute().values[0])


def test_richardson_bulk():
    """Test bulk Richardson number."""
    t_top, t_bot = 300.0, 298.0
    u_top, u_bot = 10.0, 5.0
    z_top, z_bot = 20.0, 2.0

    ri = richardson_bulk(t_top, t_bot, u_top, u_bot, z_top, z_bot)
    assert ri > 0  # Stable

    # Dask check
    t_top_xr = xr.DataArray(da.from_array([t_top], chunks=1), dims="x")
    ri_xr = richardson_bulk(t_top_xr, t_bot, u_top, u_bot, z_top, z_bot)
    np.testing.assert_allclose(ri, ri_xr.compute().values[0])
