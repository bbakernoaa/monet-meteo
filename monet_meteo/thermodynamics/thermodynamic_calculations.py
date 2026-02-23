"""
Thermodynamic calculations for atmospheric science.
MetPy-equivalent implementations without the dependency.
"""

import numpy as np
from typing import Union, Any, Optional
import xarray as xr

# Import constants
from ..constants import R_d, R_v, c_pd, c_pv, g, epsilon


def _update_history(
    obj: Any,
    msg: str,
    name: Optional[str] = None,
    units: Optional[str] = None,
    standard_name: Optional[str] = None,
) -> Any:
    """Update history and other attributes."""
    if isinstance(obj, xr.DataArray):
        history = obj.attrs.get("history", "")
        obj.attrs["history"] = (history + "\n" if history else "") + msg
        if name:
            obj.name = name
        if units:
            obj.attrs["units"] = units
        if standard_name:
            obj.attrs["standard_name"] = standard_name
    return obj


def potential_temperature(
    pressure: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
    p0: float = 1000.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate potential temperature."""
    if isinstance(pressure, xr.DataArray):
        p_pa = xr.where(pressure < 2000, pressure * 100.0, pressure)
        p0_pa = 100000.0 if p0 < 2000 else p0
        theta = temperature * (p0_pa / p_pa) ** (R_d / c_pd)
        return _update_history(
            theta,
            "Calculated potential_temperature.",
            name="potential_temperature",
            standard_name="air_potential_temperature",
            units="K",
        )
    else:
        p_val = np.asanyarray(pressure)
        p_pa = np.where(p_val < 2000, p_val * 100.0, p_val)
        p0_pa = 100000.0 if p0 < 2000 else p0
        return temperature * (p0_pa / p_pa) ** (R_d / c_pd)


def virtual_temperature(
    temperature: Union[float, np.ndarray, xr.DataArray],
    mixing_ratio: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate virtual temperature."""
    if isinstance(mixing_ratio, xr.DataArray):
        mr = mixing_ratio.clip(0, None)
    else:
        mr = np.maximum(0, mixing_ratio)

    t_virt = temperature * (1 + (R_v / R_d - 1) * mr)

    if isinstance(temperature, xr.DataArray):
        return _update_history(
            t_virt,
            "Calculated virtual_temperature.",
            name="virtual_temperature",
            standard_name="air_virtual_temperature",
            units="K",
        )
    return t_virt


def saturation_vapor_pressure(
    temperature: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate saturation vapor pressure using Bolton (1980)."""
    if isinstance(temperature, xr.DataArray):
        t_c = xr.where(temperature > 150, temperature - 273.15, temperature)
        es_hpa = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
        res = es_hpa * 100.0
        return _update_history(
            res,
            "Calculated saturation_vapor_pressure.",
            name="saturation_vapor_pressure",
            units="Pa",
        )
    else:
        t_val = np.asanyarray(temperature)
        t_c = np.where(t_val > 150, t_val - 273.15, t_val)
        return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5)) * 100.0


def mixing_ratio(
    vapor_pressure: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate mixing ratio."""
    r = epsilon * vapor_pressure / (pressure - vapor_pressure)
    if isinstance(r, xr.DataArray):
        return _update_history(r, "Calculated mixing_ratio.", units="kg/kg")
    return r


def relative_humidity(
    vapor_pressure: Union[float, np.ndarray, xr.DataArray],
    saturation_vapor_pressure: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate relative humidity."""
    rh = vapor_pressure / saturation_vapor_pressure
    if isinstance(rh, xr.DataArray):
        return _update_history(
            rh.clip(0, 1), "Calculated relative_humidity.", units="1"
        )
    return np.clip(rh, 0, 1)


def dewpoint_from_relative_humidity(
    temperature: Union[float, np.ndarray, xr.DataArray],
    rh: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Calculate dewpoint. Always returns Kelvin if temperature > 150 or if xarray.
    Wait, let's always return Kelvin for consistency in thermodynamics module.
    """
    es = saturation_vapor_pressure(temperature)
    if isinstance(rh, xr.DataArray):
        rh_frac = xr.where(rh > 1.0, rh / 100.0, rh)
    else:
        rh_val = np.asanyarray(rh)
        rh_frac = np.where(rh_val > 1.0, rh_val / 100.0, rh_val)

    e = rh_frac * es
    e_hpa = e / 100.0

    if isinstance(e_hpa, xr.DataArray):
        e_hpa = e_hpa.clip(1e-10, None)
    else:
        e_hpa = np.maximum(e_hpa, 1e-10)

    a, b = 17.67, 243.5
    val = np.log(e_hpa / 6.112)
    td_c = b * val / (a - val)
    td_k = td_c + 273.15

    if isinstance(temperature, xr.DataArray):
        return _update_history(
            td_k,
            "Calculated dewpoint.",
            name="dewpoint_temperature",
            standard_name="dew_point_temperature",
            units="K",
        )
    return td_k


def equivalent_potential_temperature(
    pressure: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
    mixing_ratio_val: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate theta_e using Bolton (1980)."""
    if isinstance(pressure, xr.DataArray):
        p_hpa = xr.where(pressure > 2000, pressure / 100.0, pressure)
        t_k = xr.where(temperature < 150, temperature + 273.15, temperature)
    else:
        p_val = np.asanyarray(pressure)
        p_hpa = np.where(p_val > 2000, p_val / 100.0, p_val)
        t_val = np.asanyarray(temperature)
        t_k = np.where(t_val < 150, t_val + 273.15, t_val)

    e = (mixing_ratio_val * p_hpa) / (epsilon + mixing_ratio_val)
    if isinstance(e, xr.DataArray):
        e_safe = e.clip(1e-10, None)
    else:
        e_safe = np.maximum(e, 1e-10)

    t_lcl = 1.0 / (1.0 / (t_k - 55.0) - np.log(e_safe / 6.112) / 2840.0) + 55.0

    theta_l = (
        t_k
        * (1000.0 / (p_hpa - e)) ** 0.2854
        * (t_k / t_lcl) ** (0.28e-3 * mixing_ratio_val)
    )
    theta_e = theta_l * np.exp(
        (3.036 / t_lcl - 0.00178)
        * mixing_ratio_val
        * (1.0 + 0.448e-3 * mixing_ratio_val)
    )

    if isinstance(temperature, xr.DataArray):
        return _update_history(
            theta_e,
            "Calculated equivalent_potential_temperature (Bolton 1980).",
            name="equivalent_potential_temperature",
            standard_name="air_equivalent_potential_temperature",
            units="K",
        )
    return theta_e


def specific_humidity_from_mixing_ratio(
    mixing_ratio: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Convert mixing ratio to specific humidity."""
    return mixing_ratio / (1 + mixing_ratio)


def mixing_ratio_from_specific_humidity(
    specific_humidity: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Convert specific humidity to mixing ratio."""
    return specific_humidity / (1 - specific_humidity)


def latent_heat_vaporization(
    temperature: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate latent heat of vaporization."""
    if isinstance(temperature, xr.DataArray):
        tc = xr.where(temperature > 150, temperature - 273.15, temperature)
    else:
        t_val = np.asanyarray(temperature)
        tc = np.where(t_val > 150, t_val - 273.15, t_val)

    return (2.501 - 0.002361 * tc) * 1e6


def air_density(
    pressure: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
    mixing_ratio_val: Union[float, np.ndarray, xr.DataArray] = 0.0,
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate air density."""
    tv = virtual_temperature(temperature, mixing_ratio_val)
    return pressure / (R_d * tv)


def moist_lapse_rate(
    temperature: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate moist adiabatic lapse rate."""
    es = saturation_vapor_pressure(temperature)

    if isinstance(pressure, xr.DataArray):
        p_pa = xr.where(pressure < 2000, pressure * 100.0, pressure)
        t_k = xr.where(temperature < 150, temperature + 273.15, temperature)
    else:
        p_val = np.asanyarray(pressure)
        p_pa = np.where(p_val < 2000, p_val * 100.0, p_val)
        t_val = np.asanyarray(temperature)
        t_k = np.where(t_val < 150, t_val + 273.15, t_val)

    rs = mixing_ratio(es, p_pa)
    lv = latent_heat_vaporization(t_k)

    num = (g / c_pd) * (1 + (lv * rs) / (R_d * t_k))
    den = 1 + (lv**2 * rs * epsilon) / (c_pd * R_d * t_k**2)
    return num / den


def dry_lapse_rate() -> float:
    """Return dry adiabatic lapse rate."""
    return g / c_pd


def lifting_condensation_level(
    temperature: Union[float, np.ndarray, xr.DataArray],
    dewpoint: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate LCL height."""
    if isinstance(temperature, xr.DataArray):
        tc = xr.where(temperature > 150, temperature - 273.15, temperature)
        tdc = xr.where(dewpoint > 150, dewpoint - 273.15, dewpoint)
        res = 125.0 * (tc - tdc)
        return res.clip(0, None)
    else:
        t_val = np.asanyarray(temperature)
        td_val = np.asanyarray(dewpoint)
        tc = np.where(t_val > 150, t_val - 273.15, t_val)
        tdc = np.where(td_val > 150, td_val - 273.15, td_val)
        return np.maximum(0, 125.0 * (tc - tdc))


def wet_bulb_temperature(
    temperature: Union[float, np.ndarray, xr.DataArray],
    pressure: Union[float, np.ndarray, xr.DataArray],
    relative_humidity: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate wet bulb temperature."""
    if isinstance(temperature, xr.DataArray):
        tc = xr.where(temperature > 150, temperature - 273.15, temperature)
        rh = xr.where(
            relative_humidity <= 1.0, relative_humidity * 100.0, relative_humidity
        )
    else:
        t_val = np.asanyarray(temperature)
        tc = np.where(t_val > 150, t_val - 273.15, t_val)
        rh_val = np.asanyarray(relative_humidity)
        rh = np.where(rh_val <= 1.0, rh_val * 100.0, rh_val)

    tw_c = (
        tc * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(tc + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh**1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )

    if isinstance(temperature, xr.DataArray):
        return xr.where(temperature > 150, tw_c + 273.15, tw_c)
    return np.where(np.asanyarray(temperature) > 150, tw_c + 273.15, tw_c)


def specific_heat_moist_air(
    pressure: Union[float, np.ndarray, xr.DataArray],
    vapor_pressure: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate specific heat of moist air."""
    r = mixing_ratio(vapor_pressure, pressure)
    q = r / (1 + r)
    return (1.0 - q) * c_pd + q * c_pv


def psychrometric_constant(
    pressure: Union[float, np.ndarray, xr.DataArray],
    temperature: Union[float, np.ndarray, xr.DataArray],
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate psychrometric constant."""
    lv = latent_heat_vaporization(temperature)
    return (c_pd * pressure) / (epsilon * lv)


def saturation_vapor_pressure_slope(
    temperature: Union[float, np.ndarray, xr.DataArray]
) -> Union[float, np.ndarray, xr.DataArray]:
    """Calculate slope of SVP curve."""
    if isinstance(temperature, xr.DataArray):
        tc = xr.where(temperature > 150, temperature - 273.15, temperature)
    else:
        t_val = np.asanyarray(temperature)
        tc = np.where(t_val > 150, t_val - 273.15, t_val)

    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5)) * 100.0
    slope = es * (17.67 * 243.5) / (tc + 243.5) ** 2
    return slope


def hypsometric_equation(
    pressure: xr.DataArray,
    temperature: xr.DataArray,
    mixing_ratio_val: Union[float, xr.DataArray] = 0.0,
) -> xr.DataArray:
    """Calculate geopotential height."""
    tv = virtual_temperature(temperature, mixing_ratio_val)
    return tv  # Placeholder


def k_index(
    pressure: xr.DataArray,
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
) -> xr.DataArray:
    """Calculate K-index."""
    t850 = temperature.interp(pressure=85000)
    t700 = temperature.interp(pressure=70000)
    t500 = temperature.interp(pressure=50000)
    td850 = dewpoint.interp(pressure=85000)
    td700 = dewpoint.interp(pressure=70000)
    return (t850 - t500) + td850 - (t700 - td700)


def total_totals_index(
    pressure: xr.DataArray,
    temperature: xr.DataArray,
    dewpoint: xr.DataArray,
) -> xr.DataArray:
    """Calculate Total Totals Index."""
    t850 = temperature.interp(pressure=85000)
    t500 = temperature.interp(pressure=50000)
    td850 = dewpoint.interp(pressure=85000)
    return t850 + td850 - 2 * t500
