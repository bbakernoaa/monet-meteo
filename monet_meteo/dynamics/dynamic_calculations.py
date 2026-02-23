"""
Dynamic calculations for atmospheric science.
Implementations of geostrophic wind, gradient wind, etc.
"""

import numpy as np
import xarray as xr
from typing import Union, Tuple, Optional, Any

# Import constants
from ..constants import f0_default, omega


def _update_history(obj: Any, msg: str) -> Any:
    """Update history attribute."""
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
    return obj


def coriolis_parameter(
    latitude: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate Coriolis parameter."""
    phi = np.deg2rad(latitude)
    f = 2 * omega * np.sin(phi)
    if isinstance(f, xr.DataArray):
        return _update_history(f, "Calculated coriolis_parameter.")
    return f


def relative_vorticity(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    dx: float = 1.0,
    dy: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate relative vorticity."""
    if isinstance(u, xr.DataArray):
        dims = u.dims
        y_dim, x_dim = dims[-2], dims[-1]
        dv_dx = v.differentiate(x_dim) / dx
        du_dy = u.differentiate(y_dim) / dy
        vort = dv_dx - du_dy
        return _update_history(vort, "Calculated relative vorticity.")
    else:
        u_arr = np.asanyarray(u)
        v_arr = np.asanyarray(v)
        dv_dx = np.gradient(v_arr, axis=-1) / dx
        du_dy = np.gradient(u_arr, axis=-2) / dy
        return dv_dx - du_dy


def absolute_vorticity(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    latitude: Union[float, np.ndarray, xr.DataArray],
    dx: float = 1.0,
    dy: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate absolute vorticity."""
    zeta = relative_vorticity(u, v, dx, dy)
    f = coriolis_parameter(latitude)
    abs_vort = zeta + f
    if isinstance(abs_vort, xr.DataArray):
        return _update_history(abs_vort, "Calculated absolute vorticity.")
    return abs_vort


def geostrophic_wind(
    height: Union[np.ndarray, xr.DataArray],
    dx: float = 1.0,
    dy: float = 1.0,
    latitude: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
    f: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
) -> Tuple[Any, Any]:
    """Calculate geostrophic wind."""
    g = 9.80665
    if f is None:
        if latitude is None:
            f = f0_default
        else:
            if isinstance(latitude, (xr.DataArray, np.ndarray)):
                phi = latitude
            else:
                phi = latitude if latitude < 2 * np.pi else np.deg2rad(latitude)
            f = 2 * omega * np.sin(phi)

    if isinstance(height, xr.DataArray):
        dims = height.dims
        y_dim, x_dim = dims[-2], dims[-1]
        dh_dy = height.differentiate(y_dim) / dy
        dh_dx = height.differentiate(x_dim) / dx
        ug = -(g / f) * dh_dy
        vg = (g / f) * dh_dx
        return _update_history(ug, "Calculated geostrophic_wind (u)."), _update_history(
            vg, "Calculated geostrophic_wind (v)."
        )
    else:
        h_arr = np.asanyarray(height)
        dh_dy = np.gradient(h_arr, axis=-2) / dy
        dh_dx = np.gradient(h_arr, axis=-1) / dx
        ug = -(g / f) * dh_dy
        vg = (g / f) * dh_dx
        return ug, vg


def gradient_wind(
    radius: Union[float, np.ndarray, xr.DataArray],
    pressure_gradient: Union[float, np.ndarray, xr.DataArray],
    density: Union[float, np.ndarray, xr.DataArray],
    f: Optional[Union[float, np.ndarray, xr.DataArray]] = None,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate gradient wind speed.
    V = -fR/2 + (fR/2) * sqrt(1 + 4/(f^2 R rho) * (-dP/dR))
    Wait, let's use the other form: V = -fR/2 + sqrt((fR/2)^2 - R/rho * dP/dR)
    """
    if f is None:
        f = f0_default

    term1 = f * radius / 2.0
    # Equation: V^2/R + fV + (1/rho) dP/dR = 0
    # V = [-f +/- sqrt(f^2 - 4(1/R)(1/rho dP/dR))] / (2/R)
    # V = -fR/2 + sqrt((fR/2)^2 - (R/rho) * dP/dR)
    term2 = (radius / density) * pressure_gradient
    v_grad = -term1 + np.sqrt(term1**2 - term2)

    if isinstance(v_grad, xr.DataArray):
        return _update_history(v_grad, "Calculated gradient_wind.")
    return v_grad


def advection(
    scalar: Union[np.ndarray, xr.DataArray],
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    dx: float = 1.0,
    dy: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate horizontal advection."""
    if isinstance(scalar, xr.DataArray):
        dims = scalar.dims
        y_dim, x_dim = dims[-2], dims[-1]
        ds_dx = scalar.differentiate(x_dim) / dx
        ds_dy = scalar.differentiate(y_dim) / dy
        adv = -(u * ds_dx + v * ds_dy)
        return _update_history(adv, "Calculated horizontal advection.")
    else:
        s_arr = np.asanyarray(scalar)
        ds_dy = np.gradient(s_arr, axis=-2) / dy
        ds_dx = np.gradient(s_arr, axis=-1) / dx
        return -(u * ds_dx + v * ds_dy)


def divergence(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    dx: float = 1.0,
    dy: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate horizontal divergence."""
    if isinstance(u, xr.DataArray):
        dims = u.dims
        y_dim, x_dim = dims[-2], dims[-1]
        du_dx = u.differentiate(x_dim) / dx
        dv_dy = v.differentiate(y_dim) / dy
        div = du_dx + dv_dy
        return _update_history(div, "Calculated horizontal divergence.")
    else:
        u_arr = np.asanyarray(u)
        v_arr = np.asanyarray(v)
        du_dx = np.gradient(u_arr, axis=-1) / dx
        dv_dy = np.gradient(v_arr, axis=-2) / dy
        return du_dx + dv_dy


def potential_vorticity(
    u: Union[np.ndarray, xr.DataArray],
    v: Union[np.ndarray, xr.DataArray],
    latitude: Union[float, np.ndarray, xr.DataArray],
    potential_temperature: Union[np.ndarray, xr.DataArray],
    pressure: Union[np.ndarray, xr.DataArray],
    dx: float = 1.0,
    dy: float = 1.0,
) -> Union[np.ndarray, xr.DataArray]:
    """Calculate potential vorticity."""
    g = 9.80665
    abs_vort = absolute_vorticity(u, v, latitude, dx, dy)

    if isinstance(potential_temperature, xr.DataArray):
        v_dim = "pressure" if "pressure" in potential_temperature.dims else "lev"
        dtheta_dp = potential_temperature.differentiate(v_dim)
        p_val = np.asanyarray(pressure)
        if np.any(p_val < 2000):
            dtheta_dp = dtheta_dp / 100.0
        pv = -g * abs_vort * dtheta_dp
        return _update_history(pv, "Calculated potential_vorticity.")
    else:
        pt_arr = np.asanyarray(potential_temperature)
        p_arr = np.asanyarray(pressure)
        dtheta_dp = np.gradient(pt_arr, p_arr, axis=0)
        return -g * abs_vort * dtheta_dp


def vertical_velocity_pressure(
    divergence_val: xr.DataArray,
    pressure: xr.DataArray,
    surface_omega: float = 0.0,
) -> xr.DataArray:
    """Calculate vertical velocity."""
    v_dim = "pressure" if "pressure" in divergence_val.dims else "lev"
    omega_val = surface_omega + divergence_val.cumulative_integrate(v_dim)
    return _update_history(omega_val, "Calculated vertical_velocity_pressure.")


def omega_to_w(
    omega_val: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
    mixing_ratio: Union[float, np.ndarray, xr.DataArray] = 0.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Convert omega to w."""
    g = 9.80665
    R_d = 287.04
    tv = temperature * (1 + 0.61 * mixing_ratio)
    rho = pressure / (R_d * tv)
    w = -omega_val / (rho * g)
    if isinstance(w, xr.DataArray):
        return _update_history(w, "Converted omega to w.")
    return w
