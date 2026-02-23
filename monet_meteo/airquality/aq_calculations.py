"""
Air Quality specialized calculations for atmospheric science.
Includes calculations for AQI (USA, Canada, Europe).
"""

import numpy as np
import xarray as xr
from typing import Union, Dict, List
from ..constants import R_d


def _update_history(obj, msg):
    """Update history attribute if the object is an xarray DataArray."""
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
    return obj


def total_column_mass(
    mixing_ratio: xr.DataArray,
    pressure: xr.DataArray,
    dim: str = "lev",
) -> xr.DataArray:
    """
    Calculate the total column mass of a species.

    Column Mass = ∫ (q / g) dp
    """
    dp = pressure.diff(dim=dim)
    q_avg = (
        mixing_ratio.isel({dim: slice(None, -1)})
        + mixing_ratio.isel({dim: slice(1, None)})
    ) / 2.0

    column = (q_avg * np.abs(dp) / 9.80665).sum(dim=dim)
    return _update_history(column, f"Calculated total_column_mass along {dim}.")


def mixing_ratio_to_concentration(
    mixing_ratio: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    temperature: Union[np.ndarray, xr.DataArray],
    molecular_weight: float = 28.96,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert mass mixing ratio (kg/kg) to mass concentration (kg/m^3).
    """
    rho = pressure / (R_d * temperature)
    conc = mixing_ratio * rho
    if isinstance(conc, xr.DataArray):
        return _update_history(conc, "Converted mixing_ratio to concentration.")
    return conc


def concentration_to_mixing_ratio(
    concentration: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    temperature: Union[np.ndarray, xr.DataArray],
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert mass concentration (kg/m^3) to mass mixing ratio (kg/kg).
    """
    rho = pressure / (R_d * temperature)
    q = concentration / rho
    if isinstance(q, xr.DataArray):
        return _update_history(q, "Converted concentration to mixing_ratio.")
    return q


def extinction_coefficient_rh(
    base_extinction: Union[np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
    f_rh_type: str = "kasten",
) -> Union[np.ndarray, xr.DataArray]:
    """
    Apply hygroscopic growth to aerosol extinction coefficient.
    """
    if f_rh_type == "kasten":
        gamma = 0.6
        f_rh = (1.0 - np.minimum(relative_humidity, 0.99)) ** (-gamma)
    else:
        f_rh = 1.0 + 0.1 * relative_humidity

    res = base_extinction * f_rh
    if isinstance(res, xr.DataArray):
        return _update_history(
            res, f"Applied hygroscopic growth ({f_rh_type}) to extinction."
        )
    return res


def _piecewise_linear_numpy(c, breakpoints_c, breakpoints_i):
    """Core logic for piecewise linear AQI on numpy-like arrays."""
    aqi = np.zeros_like(c, dtype=float)
    for i in range(len(breakpoints_c) - 1):
        c_low, c_high = breakpoints_c[i], breakpoints_c[i + 1]
        i_low, i_high = breakpoints_i[i], breakpoints_i[i + 1]
        mask = (c >= c_low) & (c <= c_high)
        val = (i_high - i_low) / (c_high - c_low) * (c - c_low) + i_low
        aqi = np.where(mask, val, aqi)
    aqi = np.where(c > breakpoints_c[-1], float(breakpoints_i[-1]), aqi)
    return aqi


def aqi_us_epa(
    concentration: Union[np.ndarray, xr.DataArray],
    pollutant: str,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the US EPA Air Quality Index (AQI).
    """
    # Standard EPA breakpoints
    breakpoints: Dict[str, Dict[str, List[float]]] = {
        "pm25": {
            "c": [0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4],
            "i": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0],
        },
        "pm10": {
            "c": [0, 54, 154, 254, 354, 424, 504, 604],
            "i": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0],
        },
        "o3_8h": {
            "c": [0, 0.054, 0.070, 0.085, 0.105, 0.200],
            "i": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0],
        },
        "o3_1h": {
            "c": [0.125, 0.164, 0.204, 0.404, 0.504, 0.604],
            "i": [101.0, 150.0, 200.0, 300.0, 400.0, 500.0],
        },
        "no2": {
            "c": [0, 53, 100, 360, 649, 1249, 1649, 2049],
            "i": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0],
        },
        "so2": {
            "c": [0, 35, 75, 185, 304, 604, 804, 1004],
            "i": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0],
        },
        "co": {
            "c": [0, 4.4, 9.4, 12.4, 15.4, 30.4, 40.4, 50.4],
            "i": [0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0],
        },
    }

    p = pollutant.lower().replace(".", "")
    if p not in breakpoints:
        raise ValueError(f"Unsupported pollutant: {pollutant}")

    bp = breakpoints[p]

    if isinstance(concentration, xr.DataArray):
        res = xr.apply_ufunc(
            _piecewise_linear_numpy,
            concentration,
            kwargs={"breakpoints_c": bp["c"], "breakpoints_i": bp["i"]},
            dask="parallelized",
            output_dtypes=[float],
        )
        return _update_history(res, f"Calculated US EPA AQI for {pollutant}.")
    else:
        try:
            import dask.array as da

            if isinstance(concentration, da.Array):
                return concentration.map_blocks(
                    _piecewise_linear_numpy,
                    breakpoints_c=bp["c"],
                    breakpoints_i=bp["i"],
                    dtype=float,
                )
        except ImportError:
            pass
        return _piecewise_linear_numpy(concentration, bp["c"], bp["i"])


def aqhi_canada(
    o3: Union[np.ndarray, xr.DataArray],
    no2: Union[np.ndarray, xr.DataArray],
    pm25: Union[np.ndarray, xr.DataArray],
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the Canadian Air Quality Health Index (AQHI).
    """
    term1 = np.exp(0.000871 * o3) - 1
    term2 = np.exp(0.000537 * no2) - 1
    term3 = np.exp(0.000487 * pm25) - 1

    aqhi = (10.0 / 10.4) * 100.0 * (term1 + term2 + term3)

    if isinstance(aqhi, xr.DataArray):
        return _update_history(aqhi, "Calculated Canadian AQHI.")
    return aqhi


def _eaqi_europe_numpy(c, bp_c, levels):
    """Core logic for European EAQI level on numpy-like arrays."""
    res = np.zeros_like(c, dtype=int)
    for i in range(len(levels)):
        c_low, c_high = bp_c[i], bp_c[i + 1]
        mask = (c >= c_low) & (c < c_high)
        res = np.where(mask, levels[i], res)
    res = np.where(c >= bp_c[-1], 6, res)
    return res


def eaqi_europe(
    concentration: Union[np.ndarray, xr.DataArray],
    pollutant: str,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Calculate the European Air Quality Index (EAQI) level.
    """
    # EEA bands
    bands: Dict[str, List[float]] = {
        "pm25": [0, 10, 20, 25, 50, 75, 800],
        "pm10": [0, 20, 40, 50, 100, 150, 1200],
        "no2": [0, 40, 90, 120, 230, 340, 1000],
        "o3": [0, 50, 100, 130, 240, 380, 800],
        "so2": [0, 100, 200, 350, 500, 750, 1250],
    }

    p = pollutant.lower().replace(".", "")
    if p not in bands:
        raise ValueError(f"Unsupported pollutant: {pollutant}")

    bp_c = bands[p]
    levels = [1, 2, 3, 4, 5, 6]

    if isinstance(concentration, xr.DataArray):
        res = xr.apply_ufunc(
            _eaqi_europe_numpy,
            concentration,
            kwargs={"bp_c": bp_c, "levels": levels},
            dask="parallelized",
            output_dtypes=[int],
        )
        return _update_history(res, f"Calculated European EAQI level for {pollutant}.")
    else:
        try:
            import dask.array as da

            if isinstance(concentration, da.Array):
                return concentration.map_blocks(
                    _eaqi_europe_numpy, bp_c=bp_c, levels=levels, dtype=int
                )
        except ImportError:
            pass
        return _eaqi_europe_numpy(concentration, bp_c, levels)
