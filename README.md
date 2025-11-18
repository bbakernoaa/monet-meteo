# Monet Meteo

A comprehensive meteorological library for atmospheric sciences that provides tools for atmospheric calculations including thermodynamic variables, derived parameters, and dynamic calculations.

## Features

- **Thermodynamic Calculations**: Potential temperature, equivalent potential temperature, virtual temperature, saturation vapor pressure, mixing ratio, lapse rates, and more.
- **Derived Parameters**: Heat index, wind chill, lifting condensation level, wet bulb temperature, dew point temperature, and other derived meteorological parameters.
- **Dynamic Calculations**: Vorticity, divergence, geostrophic wind, gradient wind, absolute vorticity, potential vorticity, and other dynamic parameters.
- **Data Models**: Structured data models for atmospheric profiles and related meteorological data.
- **Type Hints**: Full type hinting for better code quality and IDE support.
- **xarray Support**: Compatible with xarray DataArrays for multi-dimensional meteorological data.

## Installation

```bash
pip install monet-meteo
```

## Usage

### Basic Example

```python
import numpy as np
from monet_meteo.thermodynamics import potential_temperature
from monet_meteo.derived import heat_index
from monet_meteo.dynamics import geostrophic_wind

# Calculate potential temperature
pressure = 85000.0  # Pa
temperature = 293.15 # K
theta = potential_temperature(pressure, temperature)
print(f"Potential temperature: {theta} K")

# Calculate heat index
temp_f = 90.0  # °F
rel_humidity = 70.0  # %
hi = heat_index(temp_f, rel_humidity)
print(f"Heat index: {hi} °F")

# Calculate geostrophic wind from height field
height = np.array([58800, 58600, 58400])  # Geopotential height in m²/s²
dx = dy = 100000  # 100 km grid spacing
latitude = np.radians(45.0)  # Latitude in radians
u_g, v_g = geostrophic_wind(height, dx, dy, latitude)
print(f"Geostrophic wind: u={u_g} m/s, v={v_g} m/s")
```

### Using Data Models

```python
from monet_meteo.models import AtmosphericProfile

# Create an atmospheric profile
profile = AtmosphericProfile(
    pressure=np.array([10000, 95000, 90000, 85000]),  # Pa
    temperature=np.array([293.15, 288.15, 283.15, 278.15])  # K
)

# Calculate thermodynamic properties
profile.calculate_thermodynamic_properties()
print(f"Potential temperatures: {profile.potential_temperature} K")
```

## Modules

### Thermodynamics
- `potential_temperature()`: Calculate potential temperature using the Poisson equation
- `equivalent_potential_temperature()`: Calculate equivalent potential temperature
- `virtual_temperature()`: Calculate virtual temperature
- `saturation_vapor_pressure()`: Calculate saturation vapor pressure using Clausius-Clapeyron equation
- `mixing_ratio()`: Calculate mixing ratio
- `relative_humidity()`: Calculate relative humidity
- `dewpoint_from_relative_humidity()`: Calculate dewpoint from temperature and relative humidity
- `wet_bulb_temperature()`: Calculate wet bulb temperature
- `dry_lapse_rate()`: Calculate dry adiabatic lapse rate
- `moist_lapse_rate()`: Calculate moist adiabatic lapse rate
- `lifting_condensation_level()`: Calculate lifting condensation level

### Derived Parameters
- `heat_index()`: Calculate heat index using Rothfusz regression
- `wind_chill()`: Calculate wind chill temperature
- `dewpoint_temperature()`: Calculate dewpoint temperature from temperature and relative humidity
- `actual_vapor_pressure()`: Calculate actual vapor pressure from dewpoint
- `saturation_vapor_pressure()`: Calculate saturation vapor pressure
- `lifting_condensation_level()`: Calculate lifting condensation level
- `wet_bulb_temperature()`: Calculate wet bulb temperature

### Dynamics
- `coriolis_parameter()`: Calculate Coriolis parameter
- `relative_vorticity()`: Calculate relative vorticity
- `absolute_vorticity()`: Calculate absolute vorticity
- `divergence()`: Calculate horizontal divergence
- `geostrophic_wind()`: Calculate geostrophic wind from height field
- `gradient_wind()`: Calculate gradient wind speed
- `potential_vorticity()`: Calculate Ertel's potential vorticity
- `vertical_velocity_pressure()`: Convert omega to geometric vertical velocity
- `omega_to_w()`: Convert omega (pressure vertical velocity) to w (geometric vertical velocity)

## Data Models

### AtmosphericProfile
A complete atmospheric profile with pressure, temperature, and optional thermodynamic properties.

### WindProfile
A wind profile with height, u-wind, and v-wind components.

### ThermodynamicProfile
A profile with thermodynamic properties like pressure, temperature, dewpoint, mixing ratio, and relative humidity.

### DerivedParameters
A collection of derived meteorological parameters.

## Constants

The library includes commonly used atmospheric constants:
- `R_d`: Gas constant for dry air (287.04 J kg⁻¹ K⁻¹)
- `R_v`: Gas constant for water vapor (461.5 J kg⁻¹ K⁻¹)
- `c_pd`: Specific heat of dry air at constant pressure (1004.0 J kg⁻¹ K⁻¹)
- `g`: Acceleration due to gravity (9.8065 m s⁻²)
- `Omega`: Earth's rotation rate (7.292e-5 s⁻¹)
- And more...

## Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This package builds upon meteorological equations and algorithms from various sources including the American Meteorological Society, NOAA, and standard atmospheric science textbooks.
