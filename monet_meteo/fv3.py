"""
Functions for FV3-based models (e.g., GEOS, UFS, HRRR-FV3).
Includes vertical coordinate handling and interpolation.
"""

import numpy as np
import xarray as xr
from typing import Union


def _update_history(obj, msg):
    """Update history attribute if the object is an xarray DataArray."""
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
    return obj


def calculate_fv3_pressure(
    ak: Union[np.ndarray, xr.DataArray],
    bk: Union[np.ndarray, xr.DataArray],
    ps: Union[np.ndarray, xr.DataArray],
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate pressure at interfaces for FV3-based hybrid sigma-pressure coordinates.

    P = ak + bk * ps
    """
    if isinstance(ps, xr.DataArray):
        p = ak + bk * ps
        return _update_history(p, "Calculated FV3 interface pressures.")
    else:
        ak_res = np.asarray(ak).reshape((-1,) + (1,) * ps.ndim)
        bk_res = np.asarray(bk).reshape((-1,) + (1,) * ps.ndim)
        return ak_res + bk_res * ps


def fv3_pressure_to_midpoints(
    p_interfaces: Union[np.ndarray, xr.DataArray],
    dim: str = "lev",
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate pressure at layer midpoints from interface pressures.

    Pm = (P_top + P_bottom) / 2
    """
    if isinstance(p_interfaces, xr.DataArray):
        p_mid_da = (
            p_interfaces.isel({dim: slice(None, -1)}).drop_vars(dim)
            + p_interfaces.isel({dim: slice(1, None)}).drop_vars(dim)
        ) / 2.0
        return _update_history(p_mid_da, "Calculated FV3 midpoint pressures.")
    else:
        return (p_interfaces[:-1] + p_interfaces[1:]) / 2.0


def interpolate_to_pressure_levels(
    data: xr.DataArray,
    pressure: xr.DataArray,
    target_levels: Union[np.ndarray, list],
    dim: str = "lev",
) -> xr.DataArray:
    """
    Interpolate FV3 data from hybrid levels to fixed pressure levels.
    """

    def _interp_1d(d, p, levels):
        if p[0] > p[-1]:
            return np.interp(levels, p[::-1], d[::-1])
        return np.interp(levels, p, d)

    res = xr.apply_ufunc(
        _interp_1d,
        data,
        pressure,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[["plev"]],
        exclude_dims=set((dim,)),
        kwargs={"levels": target_levels},
        dask="parallelized",
        output_dtypes=[data.dtype],
        dask_gufunc_kwargs={"output_sizes": {"plev": len(target_levels)}},
    )

    res = res.assign_coords(plev=target_levels)
    return _update_history(
        res, f"Interpolated to {len(target_levels)} pressure levels."
    )
