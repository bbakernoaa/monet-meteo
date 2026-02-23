"""
Micro-meteorology calculations for atmospheric science.
"""

import numpy as np
import xarray as xr
from typing import Union

# Import constants
from ..constants import k, g, c_pd


def _update_history(obj, msg):
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
    return obj


def _apply_micromet_ufunc(func, *args, **kwargs):
    return xr.apply_ufunc(
        func,
        *args,
        input_core_dims=[[]] * len(args),
        output_core_dims=[[]],
        dask="parallelized",
        output_dtypes=[float],
        kwargs=kwargs,
    )


def psi_m(
    zoL: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate the adiabatic correction factor for momentum (Psi_M)."""

    def _psi_m_np(zol_val):
        zol = np.asarray(zol_val)
        psi = np.zeros_like(zol, dtype=float)
        # Stable
        s = zol >= 0
        if np.any(s):
            a, b = 6.1, 2.5
            psi[s] = -a * np.log(zol[s] + (1.0 + zol[s] ** b) ** (1.0 / b))
        # Unstable
        u = zol < 0
        if np.any(u):
            y = -zol[u]
            a, b = 0.33, 0.41
            x = (y / a) ** (1 / 3)
            psi_0 = -np.log(a) + np.sqrt(3) * b * (a ** (1 / 3)) * np.pi / 6.0
            y_min = np.minimum(y, b**-3)
            psi[u] = (
                np.log(a + y_min)
                - 3.0 * b * y_min ** (1 / 3)
                + (b * a ** (1 / 3)) / 2.0 * np.log((1.0 + x) ** 2 / (1.0 - x + x**2))
                + np.sqrt(3)
                * b
                * a ** (1 / 3)
                * np.arctan((2.0 * x - 1.0) / np.sqrt(3))
                + psi_0
            )
        return psi

    if isinstance(zoL, xr.DataArray):
        res = _apply_micromet_ufunc(_psi_m_np, zoL)
        return _update_history(res, "Calculated psi_m.")
    return _psi_m_np(zoL)


def psi_h(
    zoL: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate the adiabatic correction factor for heat (Psi_H)."""

    def _psi_h_np(zol_val):
        zol = np.asarray(zol_val)
        psi = np.zeros_like(zol, dtype=float)
        # Stable
        s = zol >= 0
        if np.any(s):
            a, b = 6.1, 2.5
            psi[s] = -a * np.log(zol[s] + (1.0 + zol[s] ** b) ** (1.0 / b))
        # Unstable
        u = zol < 0
        if np.any(u):
            y = -zol[u]
            c, d, n = 0.33, 0.057, 0.78
            psi[u] = ((1.0 - d) / n) * np.log((c + y**n) / c)
        return psi

    if isinstance(zoL, xr.DataArray):
        res = _apply_micromet_ufunc(_psi_h_np, zoL)
        return _update_history(res, "Calculated psi_h.")
    return _psi_h_np(zoL)


def friction_velocity(u, z, L, z0m, d0=0.0):
    """Calculate friction velocity u*."""
    term1 = np.log(np.maximum(1e-5, (z - d0) / z0m))
    term2 = psi_m((z - d0) / L)
    term3 = psi_m(z0m / L)
    ustar = (u * k) / (term1 - term2 + term3 + 1e-10)
    if isinstance(ustar, xr.DataArray):
        return _update_history(ustar, "Calculated friction_velocity.")
    return ustar


def obukhov_length(ustar, temperature, sensible_heat_flux, rho, latent_heat_flux=None):
    """Calculate Obukhov length L."""

    def _ol_np(us, temp, sh, r, lh=None):
        if lh is not None:
            lv = 2.501e6 - 2361.0 * (temp - 273.15)
            h_v = sh + 0.61 * temp * c_pd * (lh / lv)
        else:
            h_v = sh
        num = -r * c_pd * temp * (us**3)
        den = k * g * h_v

        res = np.full_like(den, np.inf)
        mask = np.abs(den) > 1e-10
        res[mask] = num[mask] / den[mask]
        return res

    if any(
        isinstance(a, xr.DataArray)
        for a in [ustar, temperature, sensible_heat_flux, rho, latent_heat_flux]
    ):
        args = [ustar, temperature, sensible_heat_flux, rho]
        if latent_heat_flux is not None:
            args.append(latent_heat_flux)
        res = xr.apply_ufunc(_ol_np, *args, dask="parallelized", output_dtypes=[float])
        return _update_history(res, "Calculated obukhov_length.")

    return _ol_np(
        np.asarray(ustar),
        np.asarray(temperature),
        np.asarray(sensible_heat_flux),
        np.asarray(rho),
        np.asarray(latent_heat_flux) if latent_heat_flux is not None else None,
    )


def monin_obukhov_stability(z, L, d0=0.0):
    """Calculate z/L."""
    return (z - d0) / L


def richardson_bulk(pt_top, pt_bot, u_top, u_bot, z_top, z_bot):
    """Calculate bulk Richardson number."""
    d_theta = pt_top - pt_bot
    d_u = u_top - u_bot
    d_z = z_top - z_bot
    theta_avg = (pt_top + pt_bot) / 2.0
    num = g * d_theta * d_z
    den = theta_avg * (d_u**2 + 1e-6)
    return num / den


def richardson_gradient(potential_temp, u_wind, v_wind, z_dim="z"):
    """Calculate gradient Richardson number."""
    d_theta_dz = potential_temp.differentiate(z_dim)
    du_dz = u_wind.differentiate(z_dim)
    dv_dz = v_wind.differentiate(z_dim)
    shear_sq = du_dz**2 + dv_dz**2
    rig = (g / potential_temp) * d_theta_dz / (shear_sq + 1e-6)
    return rig


def bowen_ratio(h, le):
    """Calculate Bowen ratio B = H / LE."""
    if isinstance(le, xr.DataArray):
        return xr.where(le != 0, h / le, np.inf)
    return np.where(le != 0, h / le, np.inf)


def aerodynamic_resistance(u, z, z0m, L, d0=0.0):
    """Calculate ra."""
    ustar = friction_velocity(u, z, L, z0m, d0)
    return u / (ustar**2 + 1e-10)


def sensible_heat_flux(t_s, t_a, ra, rho=1.225):
    """Calculate sensible heat flux H."""
    return rho * c_pd * (t_s - t_a) / ra


def latent_heat_flux(vp_s, vp_a, ra, p, rho=1.225):
    """Calculate latent heat flux LE."""
    lv = 2.501e6
    eps = 0.622
    return (rho * eps * lv / p) * (vp_s - vp_a) / ra


def surface_energy_balance(net_rad, soil_flux, h_flux, le_flux):
    """Calculate energy balance residual."""
    return net_rad - soil_flux - h_flux - le_flux


def turbulence_intensity(u_std, u_mean):
    """Calculate turbulence intensity."""
    if isinstance(u_mean, xr.DataArray):
        return xr.where(u_mean != 0, u_std / u_mean, 0.0)
    return np.where(u_mean != 0, u_std / u_mean, 0.0)


def roughness_length_from_profile(z, u, ustar, L):
    """Estimate z0m from profile."""
    val = np.log(np.maximum(1e-5, z)) - (k * u / (ustar + 1e-10)) - psi_m(z / L)
    return np.exp(val)
