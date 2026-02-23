"""
Solar angle calculations for meteorology.
"""

import numpy as np
import xarray as xr
from typing import Union, Tuple


def _update_history(obj, msg):
    """Update history attribute if the object is an xarray DataArray."""
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
    elif isinstance(obj, tuple) and any(isinstance(o, xr.DataArray) for o in obj):
        for o in obj:
            if isinstance(o, xr.DataArray):
                history = o.attrs.get("history", "")
                o.attrs["history"] = (history + "\n" if history else "") + msg
    return obj


def sun_angles(
    lat: Union[float, np.ndarray, xr.DataArray],
    lon: Union[float, np.ndarray, xr.DataArray],
    stdlon: Union[float, np.ndarray, xr.DataArray],
    doy: Union[int, np.ndarray, xr.DataArray],
    ftime: Union[float, np.ndarray, xr.DataArray],
) -> Tuple[
    Union[float, np.ndarray, xr.DataArray], Union[float, np.ndarray, xr.DataArray]
]:
    """
    Calculate the Sun Zenith and Azimuth Angles (SZA & SAA).

    Parameters
    ----------
    lat : float, numpy.ndarray, or xarray.DataArray
        Latitude of the site (degrees).
    lon : float, numpy.ndarray, or xarray.DataArray
        Longitude of the site (degrees).
    stdlon : float, numpy.ndarray, or xarray.DataArray
        Central longitude of the time zone (degrees).
    doy : int, numpy.ndarray, or xarray.DataArray
        Day of year (1-366).
    ftime : float, numpy.ndarray, or xarray.DataArray
        Time of measurement (decimal hours).

    Returns
    -------
    sza : float, numpy.ndarray, or xarray.DataArray
        Sun Zenith Angle (degrees).
    saa : float, numpy.ndarray, or xarray.DataArray
        Sun Azimuth Angle (degrees).
    """
    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    eot = (
        0.258 * np.cos(declination)
        - 7.416 * np.sin(declination)
        - 3.648 * np.cos(2.0 * declination)
        - 9.228 * np.sin(2.0 * declination)
    )
    lc = (stdlon - lon) / 15.0
    time_corr = (-eot / 60.0) + lc
    solar_time = ftime - time_corr

    # Get the hour angle
    w = (solar_time - 12.0) * 15.0

    # Get solar elevation angle
    sin_theta = np.cos(np.radians(w)) * np.cos(declination) * np.cos(
        np.radians(lat)
    ) + np.sin(declination) * np.sin(np.radians(lat))

    # Clip sin_theta to [-1, 1] to avoid nan in arcsin
    if isinstance(sin_theta, xr.DataArray):
        sin_theta = sin_theta.clip(-1, 1)
    elif isinstance(sin_theta, np.ndarray):
        sin_theta = np.clip(sin_theta, -1, 1)
    else:
        sin_theta = max(min(sin_theta, 1), -1)

    sun_elev = np.arcsin(sin_theta)

    # Get solar zenith angle
    sza_rad = np.pi / 2.0 - sun_elev
    sza = np.degrees(sza_rad)

    # Get solar azimuth angle
    cos_phi = (
        np.sin(declination) * np.cos(np.radians(lat))
        - np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat))
    ) / np.cos(sun_elev)

    # Clip cos_phi
    if isinstance(cos_phi, xr.DataArray):
        cos_phi = cos_phi.clip(-1, 1)
    elif isinstance(cos_phi, np.ndarray):
        cos_phi = np.clip(cos_phi, -1, 1)
    else:
        cos_phi = max(min(cos_phi, 1), -1)

    saa_raw = np.degrees(np.arccos(cos_phi))

    if isinstance(w, xr.DataArray):
        saa = xr.where(w <= 0.0, saa_raw, 360.0 - saa_raw)
    elif isinstance(w, np.ndarray):
        saa = np.where(w <= 0.0, saa_raw, 360.0 - saa_raw)
    else:
        saa = saa_raw if w <= 0.0 else 360.0 - saa_raw

    res = (sza, saa)
    return _update_history(res, "Calculated sun_angles.")
