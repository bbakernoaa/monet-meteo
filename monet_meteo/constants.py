"""
Constants for atmospheric calculations.

This module defines physical constants used in meteorological calculations.
"""

from typing import Final

# Physical constants
# Gas constant for dry air (J kg-1 K-1)
R_d: Final[float] = 287.04
# Gas constant for water vapor (J kg-1 K-1)
R_v: Final[float] = 461.5
# Specific heat of dry air at constant pressure (J kg-1 K-1)
c_pd: Final[float] = 1004.0
# Specific heat of water vapor at constant pressure (J kg-1 K-1)
c_pv: Final[float] = 1869.0
# Specific heat of liquid water (J kg-1 K-1)
c_pw: Final[float] = 4190.0
# Ratio of molecular weights of water to dry air
epsilon: Final[float] = R_d / R_v  # ≈ 0.622
# Acceleration due to gravity (m s-2)
g: Final[float] = 9.80665
# Von Karman constant
k: Final[float] = 0.4
# Reference pressure (Pa)
p0: Final[float] = 100000.0  # 1000 hPa
# Stefan-Boltzmann constant (W m-2 K-4)
sigma: Final[float] = 5.67e-8
# Latent heat of vaporization at 0°C (J kg-1)
L_v0: Final[float] = 2.501e6
# Latent heat of fusion at 0°C (J kg-1)
L_f0: Final[float] = 3.337e5
# Latent heat of sublimation at 0°C (J kg-1)
L_s0: Final[float] = L_v0 + L_f0
# Earth's radius (m)
R_earth: Final[float] = 6.371e6
# Earth's rotation rate (s-1)
Omega: Final[float] = 7.292e-5