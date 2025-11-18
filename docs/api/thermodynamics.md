# Thermodynamics Module

The thermodynamics module provides comprehensive functions for calculating atmospheric thermodynamic properties, including potential temperature, virtual temperature, mixing ratios, and other fundamental atmospheric variables.

## Functions

### `potential_temperature(pressure, temperature, p0=1000.0)`
Calculate potential temperature using the Poisson equation.

```python
mm.potential_temperature(pressure, temperature, p0=1000.0)
```

**Parameters:**
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Total atmospheric pressure in hPa or mb
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `p0` (float, optional): Reference pressure in hPa, default is 1000.0

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Potential temperature in Kelvin

**Example:**
```python
pressure = 850  # hPa
temperature = 288.15  # K
theta = mm.potential_temperature(pressure, temperature)
print(f"Potential temperature: {theta:.2f} K")
```

### `virtual_temperature(temperature, mixing_ratio)`
Calculate virtual temperature accounting for water vapor effects.

```python
mm.virtual_temperature(temperature, mixing_ratio)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `mixing_ratio` (float, numpy.ndarray, or xarray.DataArray): Mixing ratio in kg/kg

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Virtual temperature in Kelvin

**Example:**
```python
temperature = 288.15  # K
mixing_ratio = 0.008  # kg/kg
virtual_temp = mm.virtual_temperature(temperature, mixing_ratio)
```

### `saturation_vapor_pressure(temperature)`
Calculate saturation vapor pressure using the Clausius-Clapeyron equation.

```python
mm.saturation_vapor_pressure(temperature)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Saturation vapor pressure in Pascal

**Example:**
```python
temperature = 288.15  # K
es = mm.saturation_vapor_pressure(temperature)
print(f"Saturation vapor pressure: {es:.2f} Pa")
```

### `mixing_ratio(vapor_pressure, pressure)`
Calculate the mixing ratio from vapor pressure and total pressure.

```python
mm.mixing_ratio(vapor_pressure, pressure)
```

**Parameters:**
- `vapor_pressure` (float, numpy.ndarray, or xarray.DataArray): Vapor pressure in Pascal
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Total pressure in Pascal

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Mixing ratio in kg/kg

**Example:**
```python
vapor_pressure = 1500  # Pa
pressure = 85000  # Pa
mixing_ratio = mm.mixing_ratio(vapor_pressure, pressure)
```

### `relative_humidity(vapor_pressure, saturation_vapor_pressure)`
Calculate relative humidity from vapor pressure and saturation vapor pressure.

```python
mm.relative_humidity(vapor_pressure, saturation_vapor_pressure)
```

**Parameters:**
- `vapor_pressure` (float, numpy.ndarray, or xarray.DataArray): Actual vapor pressure in Pascal
- `saturation_vapor_pressure` (float, numpy.ndarray, or xarray.DataArray): Saturation vapor pressure in Pascal

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Relative humidity as dimensionless value (0-1)

**Example:**
```python
vapor_pressure = 1500  # Pa
sat_vapor_pressure = 2000  # Pa
rh = mm.relative_humidity(vapor_pressure, sat_vapor_pressure)
print(f"Relative humidity: {rh:.1%}")
```

### `dewpoint_from_relative_humidity(temperature, relative_humidity)`
Calculate dewpoint temperature from temperature and relative humidity.

```python
mm.dewpoint_from_relative_humidity(temperature, relative_humidity)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `relative_humidity` (float, numpy.ndarray, or xarray.DataArray): Relative humidity as dimensionless value (0-1)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Dewpoint temperature in Kelvin

**Example:**
```python
temperature = 293.15  # K
relative_humidity = 0.7  # 70%
dewpoint = mm.dewpoint_from_relative_humidity(temperature, relative_humidity)
```

### `equivalent_potential_temperature(pressure, temperature, mixing_ratio_val)`
Calculate equivalent potential temperature.

```python
mm.equivalent_potential_temperature(pressure, temperature, mixing_ratio_val)
```

**Parameters:**
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Total atmospheric pressure in Pascal
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `mixing_ratio_val` (float, numpy.ndarray, or xarray.DataArray): Mixing ratio in kg/kg

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Equivalent potential temperature in Kelvin

**Example:**
```python
pressure = 85000  # Pa
temperature = 288.15  # K
mixing_ratio = 0.008  # kg/kg
theta_e = mm.equivalent_potential_temperature(pressure, temperature, mixing_ratio)
```

### `wet_bulb_temperature(temperature, pressure, relative_humidity)`
Calculate wet bulb temperature using Stull (2011) approximation.

```python
mm.wet_bulb_temperature(temperature, pressure, relative_humidity)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Total atmospheric pressure in Pascal
- `relative_humidity` (float, numpy.ndarray, or xarray.DataArray): Relative humidity as dimensionless value (0-1)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Wet bulb temperature in Kelvin

**Example:**
```python
temperature = 298.15  # K
pressure = 101325  # Pa
relative_humidity = 0.6  # 60%
wetbulb = mm.wet_bulb_temperature(temperature, pressure, relative_humidity)
```

### `moist_lapse_rate(temperature, vapor_pressure, pressure)`
Calculate moist-adiabatic lapse rate.

```python
mm.moist_lapse_rate(temperature, vapor_pressure, pressure)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `vapor_pressure` (float, numpy.ndarray, or xarray.DataArray): Vapor pressure in Pascal
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Total pressure in Pascal

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Moist-adiabatic lapse rate in K/m

**Example:**
```python
temperature = 288.15  # K
vapor_pressure = 1500  # Pa
pressure = 85000  # Pa
moist_gamma = mm.moist_lapse_rate(temperature, vapor_pressure, pressure)
```

### `dry_lapse_rate()`
Calculate dry-adiabatic lapse rate (constant).

```python
mm.dry_lapse_rate()
```

**Parameters:**
- None

**Returns:**
- float: Dry-adiabatic lapse rate in K/m

**Example:**
```python
dry_gamma = mm.dry_lapse_rate()
print(f"Dry lapse rate: {dry_gamma:.4f} K/m")
```

### `lifting_condensation_level(temperature, dewpoint)`
Calculate lifting condensation level height.

```python
mm.lifting_condensation_level(temperature, dewpoint)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Surface air temperature in Kelvin
- `dewpoint` (float, numpy.ndarray, or xarray.DataArray): Surface dewpoint temperature in Kelvin

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Lifting condensation level height in meters

**Example:**
```python
temperature = 298.15  # K
dewpoint = 288.15  # K
lcl = mm.lifting_condensation_level(temperature, dewpoint)
print(f"LCL height: {lcl:.1f} m")
```

## Data Models

### `AtmosphericProfile`
A data class representing a complete atmospheric profile.

```python
mm.AtmosphericProfile(pressure, temperature, ...)
```

**Attributes:**
- `pressure`: Pressure levels (Pa)
- `temperature`: Temperature profile (K)
- `potential_temperature`: Potential temperature profile (K), optional
- `equivalent_potential_temperature`: Equivalent potential temperature profile (K), optional
- `virtual_temperature`: Virtual temperature profile (K), optional
- `u_wind`: Eastward wind component (m/s), optional
- `v_wind`: Northward wind component (m/s), optional
- `mixing_ratio`: Water vapor mixing ratio (kg/kg), optional
- `relative_humidity`: Relative humidity (0-1), optional
- `height`: Height levels (m), optional

**Methods:**
- `calculate_thermodynamic_properties()`: Calculate derived thermodynamic properties

**Example:**
```python
profile = mm.AtmosphericProfile(
    pressure=np.array([1000, 850, 700, 500]),
    temperature=np.array([298.15, 285.15, 273.15, 250.15])
)
profile.calculate_thermodynamic_properties()
print(f"Potential temperature: {profile.potential_temperature}")
```

### `ThermodynamicProfile`
A data class representing thermodynamic properties of the atmosphere.

```python
mm.ThermodynamicProfile(pressure, temperature, ...)
```

**Attributes:**
- `pressure`: Pressure levels (Pa)
- `temperature`: Temperature profile (K)
- `dewpoint`: Dewpoint temperature profile (K), optional
- `mixing_ratio`: Water vapor mixing ratio (kg/kg), optional
- `relative_humidity`: Relative humidity (0-1), optional

## Constants

The thermodynamics module uses physical constants defined in [`monet_meteo.constants`](../constants.md):

- `R_d`: Gas constant for dry air (287.04 J kg⁻¹ K⁻¹)
- `R_v`: Gas constant for water vapor (461.5 J kg⁻¹ K⁻¹)
- `c_pd`: Specific heat of dry air at constant pressure (1004.0 J kg⁻¹ K⁻¹)
- `c_pv`: Specific heat of water vapor at constant pressure (1869.0 J kg⁻¹ K⁻¹)
- `g`: Acceleration due to gravity (9.80665 m s⁻²)
- `epsilon`: Ratio of molecular weights of water to dry air (0.622)
- `p0`: Reference pressure (100000.0 Pa)
- `L_v0`: Latent heat of vaporization at 0°C (2.501×10⁶ J kg⁻¹)

## Usage Patterns

### Basic Thermodynamic Calculations
```python
import monet_meteo as mm
import numpy as np

# Define atmospheric conditions
pressure = 850.0  # hPa
temperature = 288.15  # K
dewpoint = 283.15  # K

# Calculate derived properties
theta = mm.potential_temperature(pressure, temperature)
virtual_temp = mm.virtual_temperature(temperature, 0.008)
mixing_ratio = mm.mixing_ratio(
    mm.saturation_vapor_pressure(dewpoint), 
    pressure * 100
)
```

### Atmospheric Profile Analysis
```python
# Create atmospheric profile
pressure_levels = np.array([1000, 850, 700, 500, 300])  # hPa
temperatures = np.array([298.15, 285.15, 273.15, 250.15, 230.15])  # K

profile = mm.AtmosphericProfile(
    pressure=pressure_levels,
    temperature=temperatures
)

# Calculate thermodynamic properties
profile.calculate_thermodynamic_properties()

# Access results
print(f"Potential temperatures: {profile.potential_temperature}")
print(f"Virtual temperatures: {profile.virtual_temperature}")
```

### Batch Processing with xarray
```python
import xarray as xr

# Create xarray dataset
ds = xr.Dataset({
    'temperature': (['time', 'level'], np.random.rand(24, 5) * 30 + 250),
    'pressure': (['level'], [1000, 850, 700, 500, 300]),
})

# Calculate potential temperature
ds['theta'] = mm.potential_temperature(ds.pressure, ds.temperature)

# Calculate mixing ratio
ds['vapor_pressure'] = ds.rh * mm.saturation_vapor_pressure(ds.temperature) / 100
ds['mixing_ratio'] = mm.mixing_ratio(ds.vapor_pressure, ds.pressure * 100)
```

## Error Handling

The functions include input validation to ensure physical合理性:

- Pressure values must be positive
- Temperature values must be above absolute zero
- Mixing ratios must be non-negative
- Relative humidity values must be between 0 and 1

### Common Errors
```python
# Error: Negative pressure
try:
    mm.potential_temperature(-100, 288.15)
except ValueError as e:
    print(f"Error: {e}")

# Error: Temperature below absolute zero
try:
    mm.saturation_vapor_pressure(-1)
except ValueError as e:
    print(f"Error: {e}")

# Error: Relative humidity > 1
try:
    mm.dewpoint_from_relative_humidity(288.15, 1.5)
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Vectorization
All functions support numpy arrays for efficient vectorized operations:

```python
import numpy as np

# Vectorized calculation for multiple profiles
pressures = np.array([1000, 850, 700, 500])
temperatures = np.array([298, 290, 280, 270])
thetas = mm.potential_temperature(pressures, temperatures)
```

### Memory Efficiency
For large datasets, consider processing in chunks:

```python
def process_large_dataset(data_chunk):
    """Process a chunk of atmospheric data"""
    result = {}
    result['theta'] = mm.potential_temperature(
        data_chunk['pressure'], 
        data_chunk['temperature']
    )
    result['mixing_ratio'] = mm.mixing_ratio(
        data_chunk['vapor_pressure'],
        data_chunk['pressure']
    )
    return result
```

## References

- Bolton, D. (1980). The Computation of Potential Temperature. Monthly Weather Review, 108(7), 1046-1053.
- Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity and Air Temperature. Journal of Applied Meteorology and Climatology, 50(11), 2267-2269.
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/

## See Also

- [Unit Conversions](../units.md) - Convert between different meteorological units
- [Statistical Analysis](../statistical.md) - Statistical and micrometeorological functions
- [Dynamic Calculations](../dynamics.md) - Dynamic meteorology functions
- [Data Models](../models.md) - Structured data models for atmospheric data