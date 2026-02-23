"""
Statistical operations for atmospheric data analysis and micrometeorology calculations.
Provides backward compatibility for existing tests while following Aero Protocol.
"""

import numpy as np
import xarray as xr
from typing import Any, Union, Optional, Tuple

# Import from micrometeo
from ..micrometeo import micrometeo_calculations as mm


def _update_history(obj: Any, msg: str) -> Any:
    """
    Update history attribute of an xarray DataArray.

    Parameters
    ----------
    obj : Any
        The data object to update.
    msg : str
        The history message to add.

    Returns
    -------
    Any
        The updated object.
    """
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
    return obj


def bulk_richardson_number(
    u_wind: Union[np.ndarray, xr.DataArray],
    v_wind: Union[np.ndarray, xr.DataArray],
    potential_temperature: Union[np.ndarray, xr.DataArray],
    height: Union[np.ndarray, xr.DataArray],
    method: str = "standard",
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate bulk Richardson number.

    Parameters
    ----------
    u_wind : array-like
        U-component of wind (m/s).
    v_wind : array-like
        V-component of wind (m/s).
    potential_temperature : array-like
        Potential temperature (K).
    height : array-like
        Height levels (m).
    method : str, optional
        Calculation method, default 'standard'.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        Bulk Richardson number.
    """
    if isinstance(u_wind, xr.DataArray):
        # Implementation using xarray for laziness
        # Assuming height is a dimension or coordinate
        pt = potential_temperature
        z = height
        pt_top = pt.isel(lev=-1) if "lev" in pt.dims else pt[-1]
        pt_bot = pt.isel(lev=0) if "lev" in pt.dims else pt[0]
        z_top = z.isel(lev=-1) if "lev" in z.dims else z[-1]
        z_bot = z.isel(lev=0) if "lev" in z.dims else z[0]

        spd = np.sqrt(u_wind**2 + v_wind**2)
        spd_top = spd.isel(lev=-1) if "lev" in spd.dims else spd[-1]

        return mm.richardson_bulk(pt_top, pt_bot, spd_top, 0.0, z_top, z_bot)
    else:
        # NumPy/Dask
        pt = np.asanyarray(potential_temperature)
        z = np.asanyarray(height)
        if pt.ndim > 0 and len(pt) >= 2:
            pt_top, pt_bot = pt[-1], pt[0]
            z_top, z_bot = z[-1], z[0]
            spd = np.sqrt(np.asanyarray(u_wind) ** 2 + np.asanyarray(v_wind) ** 2)
            if np.asanyarray(spd).ndim > 0:
                spd_top = spd[-1]
            else:
                spd_top = spd
            return mm.richardson_bulk(pt_top, pt_bot, spd_top, 0.0, z_top, z_bot)
        return 0.0


def friction_velocity(
    wind_speed: Union[float, np.ndarray, xr.DataArray],
    surface_roughness: float,
    stability_param: Union[float, np.ndarray, xr.DataArray] = 0.0,
    height: float = 10.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate friction velocity.

    Parameters
    ----------
    wind_speed : array-like
        Wind speed (m/s).
    surface_roughness : float
        Surface roughness length z0 (m).
    stability_param : array-like, optional
        Stability parameter z/L, default 0.
    height : float, optional
        Measurement height (m), default 10.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        Friction velocity u* (m/s).
    """
    if isinstance(stability_param, xr.DataArray):
        L = xr.where(stability_param != 0, height / stability_param, 1e10)
    else:
        z_L = np.asanyarray(stability_param)
        L = np.where(z_L != 0, height / z_L, 1e10)
    return mm.friction_velocity(wind_speed, height, L, surface_roughness)


def obukhov_length(
    ustar: Union[float, np.ndarray, xr.DataArray],
    sensible_heat_flux: Union[float, np.ndarray, xr.DataArray],
    potential_temperature: Union[float, np.ndarray, xr.DataArray],
    rho: float = 1.225,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate Obukhov length.

    Parameters
    ----------
    ustar : array-like
        Friction velocity (m/s).
    sensible_heat_flux : array-like
        Sensible heat flux (W/m^2).
    potential_temperature : array-like
        Potential temperature (K).
    rho : float, optional
        Air density (kg/m^3), default 1.225.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        Obukhov length L (m).
    """
    return mm.obukhov_length(ustar, potential_temperature, sensible_heat_flux, rho)


def monin_obukhov_length(
    ustar: Union[float, np.ndarray, xr.DataArray],
    sensible_heat_flux: Union[float, np.ndarray, xr.DataArray],
    potential_temperature: Union[float, np.ndarray, xr.DataArray],
    rho: float = 1.225,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Alias for obukhov_length."""
    return obukhov_length(ustar, sensible_heat_flux, potential_temperature, rho)


def sensible_heat_flux(
    t_air: Union[float, np.ndarray, xr.DataArray],
    t_surf: Union[float, np.ndarray, xr.DataArray],
    wind_speed: Union[float, np.ndarray, xr.DataArray],
    stability_param: Union[float, np.ndarray, xr.DataArray] = 0.0,
    height: float = 10.0,
    z0m: float = 0.01,
    rho: float = 1.225,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate sensible heat flux.

    Parameters
    ----------
    t_air : array-like
        Air temperature at height z (K).
    t_surf : array-like
        Surface temperature (K).
    wind_speed : array-like
        Wind speed at height z (m/s).
    stability_param : array-like, optional
        Stability parameter z/L.
    height : float, optional
        Height z (m).
    z0m : float, optional
        Roughness length (m).
    rho : float, optional
        Air density (kg/m^3).

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        Sensible heat flux H (W/m^2).
    """
    ustar = friction_velocity(wind_speed, z0m, stability_param, height)
    ra = wind_speed / (ustar**2 + 1e-10)
    return mm.sensible_heat_flux(t_surf, t_air, ra, rho=rho)


def latent_heat_flux(
    vp_air: Union[float, np.ndarray, xr.DataArray],
    vp_surf: Union[float, np.ndarray, xr.DataArray],
    aerodynamic_res: Union[float, np.ndarray, xr.DataArray],
    pressure: float = 101325.0,
    rho: float = 1.225,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate latent heat flux.

    Parameters
    ----------
    vp_air : array-like
        Vapor pressure in air (Pa).
    vp_surf : array-like
        Vapor pressure at surface (Pa).
    aerodynamic_res : array-like
        Aerodynamic resistance (s/m).
    pressure : float, optional
        Atmospheric pressure (Pa).
    rho : float, optional
        Air density (kg/m^3).

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        Latent heat flux LE (W/m^2).
    """
    return mm.latent_heat_flux(vp_surf, vp_air, aerodynamic_res, pressure, rho=rho)


def momentum_flux(
    u_prime: Union[np.ndarray, xr.DataArray],
    w_prime: Union[np.ndarray, xr.DataArray],
    air_density: float = 1.225,
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate momentum flux (Reynolds stress).

    Parameters
    ----------
    u_prime : array-like
        Fluctuation in horizontal wind.
    w_prime : array-like
        Fluctuation in vertical wind.
    air_density : float, optional
        Air density (kg/m^3).

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        Momentum flux (Pa).
    """
    if isinstance(u_prime, xr.DataArray):
        uw = (u_prime * w_prime).mean()
        return -air_density * uw
    else:
        uw = np.mean(np.asanyarray(u_prime) * np.asanyarray(w_prime))
        return -air_density * uw


def turbulence_kinetic_energy(
    u_prime: Union[np.ndarray, xr.DataArray],
    v_prime: Union[np.ndarray, xr.DataArray],
    w_prime: Union[np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate Turbulence Kinetic Energy (TKE).

    Parameters
    ----------
    u_prime, v_prime, w_prime : array-like
        Fluctuations in wind components.

    Returns
    -------
    Union[float, np.ndarray, xr.DataArray]
        TKE (m^2/s^2).
    """
    if isinstance(u_prime, xr.DataArray):
        up2 = (u_prime**2).mean()
        vp2 = (v_prime**2).mean()
        wp2 = (w_prime**2).mean()
    else:
        up2 = np.mean(np.asanyarray(u_prime) ** 2)
        vp2 = np.mean(np.asanyarray(v_prime) ** 2)
        wp2 = np.mean(np.asanyarray(w_prime) ** 2)
    return 0.5 * (up2 + vp2 + wp2)


def standard_deviation(
    x: Union[np.ndarray, xr.DataArray], dim: Optional[str] = None
) -> Any:
    """Calculate standard deviation."""
    if isinstance(x, xr.DataArray):
        return x.std(dim=dim, ddof=1)
    if np.size(x) <= 1:
        return np.nan if np.size(x) == 0 else 0.0
    return np.std(np.asanyarray(x), ddof=1)


def correlation_coefficient(x: Any, y: Any, dim: Optional[str] = None) -> Any:
    """Calculate correlation coefficient."""
    if isinstance(x, xr.DataArray):
        return xr.corr(x, y, dim=dim)
    if np.size(x) <= 1:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def covariance(x: Any, y: Any) -> Any:
    """Calculate covariance."""
    if isinstance(x, xr.DataArray):
        return xr.cov(x, y)
    if np.size(x) <= 1:
        return np.nan
    return np.cov(x, y, ddof=1)[0, 1]


def stability_parameter(
    height: float, obukhov_l: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate z/L."""
    return height / (obukhov_l + 1e-10)


def obukhov_stability_parameter(
    height: float, obukhov_l: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Alias for stability_parameter."""
    return stability_parameter(height, obukhov_l)


def psi_momentum(
    zol: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate psi_m."""
    return mm.psi_m(zol)


def psi_heat(
    zol: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate psi_h."""
    return mm.psi_h(zol)


def aerodynamic_resistance(
    u: Union[float, np.ndarray, xr.DataArray],
    z: float,
    z0m: float,
    L: Union[float, np.ndarray, xr.DataArray],
    d0: float = 0.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate aerodynamic resistance."""
    return mm.aerodynamic_resistance(u, z, z0m, L, d0)


def surface_energy_balance(
    net_rad: Union[float, np.ndarray, xr.DataArray],
    soil_flux: Union[float, np.ndarray, xr.DataArray],
    h_flux: Union[float, np.ndarray, xr.DataArray],
    le_flux: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate surface energy balance residual."""
    return mm.surface_energy_balance(net_rad, soil_flux, h_flux, le_flux)


def friction_velocity_from_wind(
    wind_speed: Union[float, np.ndarray, xr.DataArray],
    height: float,
    roughness_length: float,
    stability_parameter_val: Union[float, np.ndarray, xr.DataArray] = 0.0,
    displacement_height: float = 0.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate friction velocity from wind profile."""
    if isinstance(stability_parameter_val, xr.DataArray):
        L = xr.where(
            stability_parameter_val != 0, height / stability_parameter_val, 1e10
        )
    else:
        val = np.asanyarray(stability_parameter_val)
        L = np.where(val != 0, height / val, 1e10)
    return mm.friction_velocity(
        wind_speed, height, L, roughness_length, displacement_height
    )


def atmospheric_boundary_layer_height(
    potential_temperature: xr.DataArray,
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    z_dim: str = "z",
    critical_richardson: float = 0.25,
) -> xr.DataArray:
    """Calculate ABL height using bulk Richardson method."""
    z = potential_temperature[z_dim]
    theta_s = potential_temperature.isel({z_dim: 0})
    delta_theta = potential_temperature - theta_s
    delta_z = z - z.isel({z_dim: 0})
    delta_u2 = (u_wind - u_wind.isel({z_dim: 0})) ** 2 + (
        v_wind - v_wind.isel({z_dim: 0})
    ) ** 2
    rib = (9.80665 / theta_s) * delta_theta * delta_z / (delta_u2 + 1e-6)
    return z.where(rib > critical_richardson).min(dim=z_dim)


def turbulence_intensity(
    u_std: Union[float, np.ndarray, xr.DataArray],
    u_mean: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate turbulence intensity."""
    return mm.turbulence_intensity(u_std, u_mean)


def xarray_bulk_richardson_number(
    u_wind: xr.DataArray,
    v_wind: xr.DataArray,
    potential_temperature: xr.DataArray,
    height: xr.DataArray,
    method: str = "standard",
) -> xr.DataArray:
    """Xarray wrapper for bulk Richardson number."""
    return bulk_richardson_number(u_wind, v_wind, potential_temperature, height, method)  # type: ignore


def xarray_monin_obukhov_length(
    friction_velocity: xr.DataArray,
    temperature: xr.DataArray,
    air_density: xr.DataArray,
    specific_heat: Any,
    sensible_heat_flux: xr.DataArray,
    latent_heat_flux: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Xarray wrapper for Obukhov length."""
    return mm.obukhov_length(
        friction_velocity,
        temperature,
        sensible_heat_flux,
        air_density,
        latent_heat_flux,
    )  # type: ignore


def xarray_surface_energy_balance(
    net_radiation: xr.DataArray,
    soil_heat_flux: xr.DataArray,
    sensible_heat_flux: xr.DataArray,
    latent_heat_flux: xr.DataArray,
) -> xr.DataArray:
    """Xarray wrapper for surface energy balance."""
    return mm.surface_energy_balance(
        net_radiation, soil_heat_flux, sensible_heat_flux, latent_heat_flux
    )  # type: ignore


def xarray_turbulent_fluxes_from_similarity(
    wind_speed: xr.DataArray,
    air_temperature: xr.DataArray,
    surface_temperature: xr.DataArray,
    vapor_pressure_air: xr.DataArray,
    vapor_pressure_surface: xr.DataArray,
    height: float,
    roughness_length: float,
    stability_parameter: xr.DataArray,
    displacement_height: float = 0.0,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Xarray wrapper for turbulent flux calculation."""
    ra = mm.aerodynamic_resistance(
        wind_speed,
        height,
        roughness_length,
        height / stability_parameter,
        displacement_height,
    )
    h = mm.sensible_heat_flux(surface_temperature, air_temperature, ra)
    le = mm.latent_heat_flux(vapor_pressure_surface, vapor_pressure_air, ra, 101325.0)
    return h, le  # type: ignore
