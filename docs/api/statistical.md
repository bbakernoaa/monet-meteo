# Statistical Analysis Module

The statistical module provides functions for atmospheric boundary layer analysis, micrometeorological calculations, and statistical methods commonly used in meteorology and climate research.

## Functions

### `bulk_richardson_number(u_wind, v_wind, potential_temperature, height)`
Calculate bulk Richardson number for atmospheric stability assessment.

```python
mm.bulk_richardson_number(u_wind, v_wind, potential_temperature, height)
```

**Parameters:**
- `u_wind` (float, numpy.ndarray, or xarray.DataArray): Eastward wind component (m/s)
- `v_wind` (float, numpy.ndarray, or xarray.DataArray): Northward wind component (m/s)
- `potential_temperature` (float, numpy.ndarray, or xarray.DataArray): Potential temperature (K)
- `height` (float, numpy.ndarray, or xarray.DataArray): Height above surface (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Bulk Richardson number (dimensionless)

**Example:**
```python
u = 5.0  # m/s
v = 2.0  # m/s
theta = 300.0  # K
height = 100.0  # m
ri = mm.bulk_richardson_number(u, v, theta, height)
print(f"Bulk Richardson number: {ri:.3f}")
```

### `monin_obukhov_length(friction_velocity, temperature, air_density, specific_heat, sensible_heat_flux, latent_heat_flux=None)`
Calculate Monin-Obukhov length for atmospheric stability.

```python
mm.monin_obukhov_length(friction_velocity, temperature, air_density, specific_heat, sensible_heat_flux, latent_heat_flux=None)
```

**Parameters:**
- `friction_velocity` (float, numpy.ndarray, or xarray.DataArray): Friction velocity (m/s)
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature (K)
- `air_density` (float, numpy.ndarray, or xarray.DataArray): Air density (kg/m³)
- `specific_heat` (float, numpy.ndarray, or xarray.DataArray): Specific heat capacity (J/kg/K)
- `sensible_heat_flux` (float, numpy.ndarray, or xarray.DataArray): Sensible heat flux (W/m²)
- `latent_heat_flux` (float, numpy.ndarray, or xarray.DataArray, optional): Latent heat flux (W/m²)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Monin-Obukhov length (m)

**Example:**
```python
ustar = 0.3  # m/s
temp = 293.15  # K
rho = 1.2  # kg/m³
cp = 1004  # J/kg/K
H = 150  # W/m²
L = mm.monin_obukhov_length(ustar, temp, rho, cp, H)
print(f"Monin-Obukhov length: {L:.1f} m")
```

### `stability_parameter(z, L)`
Calculate stability parameter z/L.

```python
mm.stability_parameter(z, L)
```

**Parameters:**
- `z` (float, numpy.ndarray, or xarray.DataArray): Height above surface (m)
- `L` (float, numpy.ndarray, or xarray.DataArray): Monin-Obukhov length (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Stability parameter z/L (dimensionless)

**Example:**
```python
height = 10.0  # m
L = 100.0  # m
z_over_L = mm.stability_parameter(height, L)
print(f"Stability parameter: {z_over_L:.3f}")
```

### `psi_momentum(stability_parameter)`
Calculate stability correction function for momentum (Ψ_m) in Monin-Obukhov similarity theory.

```python
mm.psi_momentum(stability_parameter)
```

**Parameters:**
- `stability_parameter` (float, numpy.ndarray, or xarray.DataArray): Stability parameter z/L

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Stability correction for momentum Ψ_m

**Example:**
```python
z_over_L = 0.1  # Stable conditions
psi_m = mm.psi_momentum(z_over_L)
print(f"Momentum stability correction: {psi_m:.3f}")
```

### `psi_heat(stability_parameter)`
Calculate stability correction function for heat (Ψ_h) in Monin-Obukhov similarity theory.

```python
mm.psi_heat(stability_parameter)
```

**Parameters:**
- `stability_parameter` (float, numpy.ndarray, or xarray.DataArray): Stability parameter z/L

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Stability correction for heat Ψ_h

**Example:**
```python
z_over_L = -0.1  # Unstable conditions
psi_h = mm.psi_heat(z_over_L)
print(f"Heat stability correction: {psi_h:.3f}")
```

### `aerodynamic_resistance(ustar, z0, z, L=None)`
Calculate aerodynamic resistance.

```python
mm.aerodynamic_resistance(ustar, z0, z, L=None)
```

**Parameters:**
- `ustar` (float, numpy.ndarray, or xarray.DataArray): Friction velocity (m/s)
- `z0` (float, numpy.ndarray, or xarray.DataArray): Roughness length (m)
- `z` (float, numpy.ndarray, or xarray.DataArray): Measurement height (m)
- `L` (float, numpy.ndarray, or xarray.DataArray, optional): Monin-Obukhov length (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Aerodynamic resistance (s/m)

**Example:**
```python
ustar = 0.3  # m/s
z0 = 0.01  # m (short grass)
z = 2.0  # m
L = 100.0  # m
ra = mm.aerodynamic_resistance(ustar, z0, z, L)
print(f"Aerodynamic resistance: {ra:.3f} s/m")
```

### `surface_energy_balance(net_radiation, soil_heat_flux, sensible_heat_flux, latent_heat_flux)`
Calculate surface energy balance residual.

```python
mm.surface_energy_balance(net_radiation, soil_heat_flux, sensible_heat_flux, latent_heat_flux)
```

**Parameters:**
- `net_radiation` (float, numpy.ndarray, or xarray.DataArray): Net radiation (W/m²)
- `soil_heat_flux` (float, numpy.ndarray, or xarray.DataArray): Soil heat flux (W/m²)
- `sensible_heat_flux` (float, numpy.ndarray, or xarray.DataArray): Sensible heat flux (W/m²)
- `latent_heat_flux` (float, numpy.ndarray, or xarray.DataArray): Latent heat flux (W/m²)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Energy balance residual (W/m²)

**Example:**
```python
Rn = 500  # W/m²
G = 50  # W/m²
H = 150  # W/m²
LE = 280  # W/m²
residual = mm.surface_energy_balance(Rn, G, H, LE)
print(f"Energy balance residual: {residual:.1f} W/m²")
```

### `sensible_heat_flux(air_temp, surface_temp, aerodynamic_resistance)`
Calculate sensible heat flux.

```python
mm.sensible_heat_flux(air_temp, surface_temp, aerodynamic_resistance)
```

**Parameters:**
- `air_temp` (float, numpy.ndarray, or xarray.DataArray): Air temperature (K)
- `surface_temp` (float, numpy.ndarray, or xarray.DataArray): Surface temperature (K)
- `aerodynamic_resistance` (float, numpy.ndarray, or xarray.DataArray): Aerodynamic resistance (s/m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Sensible heat flux (W/m²)

**Example:**
```python
T_air = 293.15  # K
T_surface = 298.15  # K
ra = 50  # s/m
H = mm.sensible_heat_flux(T_air, T_surface, ra)
print(f"Sensible heat flux: {H:.1f} W/m²")
```

### `latent_heat_flux(vapor_pressure, saturation_vapor_pressure, aerodynamic_resistance, temperature)`
Calculate latent heat flux.

```python
mm.latent_heat_flux(vapor_pressure, saturation_vapor_pressure, aerodynamic_resistance, temperature)
```

**Parameters:**
- `vapor_pressure` (float, numpy.ndarray, or xarray.DataArray): Vapor pressure (Pa)
- `saturation_vapor_pressure` (float, numpy.ndarray, or xarray.DataArray): Saturation vapor pressure (Pa)
- `aerodynamic_resistance` (float, numpy.ndarray, or xarray.DataArray): Aerodynamic resistance (s/m)
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature (K)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Latent heat flux (W/m²)

**Example:**
```python
e = 1500  # Pa
es = 2000  # Pa
ra = 50  # s/m
T = 293.15  # K
LE = mm.latent_heat_flux(e, es, ra, T)
print(f"Latent heat flux: {LE:.1f} W/m²")
```

### `friction_velocity_from_wind(wind_speed, height, z0, L=None)`
Calculate friction velocity from wind speed.

```python
mm.friction_velocity_from_wind(wind_speed, height, z0, L=None)
```

**Parameters:**
- `wind_speed` (float, numpy.ndarray, or xarray.DataArray): Wind speed (m/s)
- `height` (float, numpy.ndarray, or xarray.DataArray): Measurement height (m)
- `z0` (float, numpy.ndarray, or xarray.DataArray): Roughness length (m)
- `L` (float, numpy.ndarray, or xarray.DataArray, optional): Monin-Obukhov length (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Friction velocity (m/s)

**Example:**
```python
wind_speed = 5.0  # m/s
height = 10.0  # m
z0 = 0.1  # m (rough terrain)
ustar = mm.friction_velocity_from_wind(wind_speed, height, z0)
print(f"Friction velocity: {ustar:.3f} m/s")
```

### `atmospheric_boundary_layer_height(stability_parameter, friction_velocity, surface_heat_flux)`
Calculate atmospheric boundary layer height.

```python
mm.atmospheric_boundary_layer_height(stability_parameter, friction_velocity, surface_heat_flux)
```

**Parameters:**
- `stability_parameter` (float, numpy.ndarray, or xarray.DataArray): Stability parameter
- `friction_velocity` (float, numpy.ndarray, or xarray.DataArray): Friction velocity (m/s)
- `surface_heat_flux` (float, numpy.ndarray, or xarray.DataArray): Surface heat flux (W/m²)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Atmospheric boundary layer height (m)

**Example:**
```python
z_over_L = 0.1  # Stable
ustar = 0.3  # m/s
H = 150  # W/m²
h = mm.atmospheric_boundary_layer_height(z_over_L, ustar, H)
print(f"Boundary layer height: {h:.1f} m")
```

### `turbulence_intensity(u_component, v_component, w_component=None)`
Calculate turbulence intensity.

```python
mm.turbulence_intensity(u_component, v_component, w_component=None)
```

**Parameters:**
- `u_component` (float, numpy.ndarray, or xarray.DataArray): U-component of wind (m/s)
- `v_component` (float, numpy.ndarray, or xarray.DataArray): V-component of wind (m/s)
- `w_component` (float, numpy.ndarray, or xarray.DataArray, optional): W-component of wind (m/s)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Turbulence intensity

**Example:**
```python
u = np.array([5.0, 5.2, 4.8, 5.1, 4.9])  # m/s
v = np.array([2.0, 2.1, 1.9, 2.0, 2.1])  # m/s
TI = mm.turbulence_intensity(u, v)
print(f"Turbulence intensity: {TI:.3f}")
```

### `obukhov_stability_parameter(temperature_difference, wind_speed, surface_roughness, height)`
Obukhov stability parameter calculation.

```python
mm.obukhov_stability_parameter(temperature_difference, wind_speed, surface_roughness, height)
```

**Parameters:**
- `temperature_difference` (float, numpy.ndarray, or xarray.DataArray): Temperature difference (K)
- `wind_speed` (float, numpy.ndarray, or xarray.DataArray): Wind speed (m/s)
- `surface_roughness` (float, numpy.ndarray, or xarray.DataArray): Surface roughness (m)
- `height` (float, numpy.ndarray, or xarray.DataArray): Measurement height (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Obukhov stability parameter

**Example:**
```python
dT = 2.0  # K
wind = 5.0  # m/s
z0 = 0.1  # m
z = 10.0  # m
L_inv = mm.obukhov_stability_parameter(dT, wind, z0, z)
print(f"Obukhov stability parameter: {L_inv:.4f} m⁻¹")
```

## Xarray Wrapper Functions

### `xarray_bulk_richardson_number(u_wind, v_wind, potential_temperature, height)`
Bulk Richardson number for xarray DataArrays.

```python
mm.xarray_bulk_richardson_number(u_wind, v_wind, potential_temperature, height)
```

### `xarray_monin_obukhov_length(...)`
Monin-Obukhov length for xarray DataArrays.

### `xarray_surface_energy_balance(...)`
Surface energy balance for xarray DataArrays.

### `xarray_turbulent_fluxes_from_similarity(...)`
Turbulent fluxes from similarity theory for xarray DataArrays.

## Usage Patterns

### Basic Stability Analysis
```python
import monet_meteo as mm
import numpy as np

# Define atmospheric conditions
u_wind = 5.0  # m/s
v_wind = 2.0  # m/s
potential_temp = 300.0  # K
height = 100.0  # m

# Calculate bulk Richardson number
ri = mm.bulk_richardson_number(u_wind, v_wind, potential_temp, height)

# Classify stability
if ri < 0.0:
    stability = "Unstable"
elif ri < 0.25:
    stability = "Neutral"
elif ri < 1.0:
    stability = "Stable"
else:
    stability = "Very Stable"

print(f"Richardson number: {ri:.3f} ({stability})")
```

### Monin-Obukhov Similarity Theory
```python
# Define surface layer parameters
ustar = 0.3  # m/s
temperature = 293.15  # K
air_density = 1.2  # kg/m³
specific_heat = 1004  # J/kg/K
sensible_heat_flux = 150  # W/m²

# Calculate Monin-Obukhov length
L = mm.monin_obukhov_length(ustar, temperature, air_density, specific_heat, sensible_heat_flux)

# Calculate stability parameter
z = 10.0  # m
z_over_L = mm.stability_parameter(z, L)

# Calculate stability corrections
psi_m = mm.psi_momentum(z_over_L)
psi_h = mm.psi_heat(z_over_L)

print(f"Monin-Obukhov length: {L:.1f} m")
print(f"Stability parameter: {z_over_L:.3f}")
print(f"Momentum correction: {psi_m:.3f}")
print(f"Heat correction: {psi_h:.3f}")
```

### Surface Energy Balance
```python
# Define energy balance components
net_radiation = 500  # W/m²
soil_heat_flux = 50  # W/m²
sensible_heat_flux = 150  # W/m²
latent_heat_flux = 280  # W/m²

# Calculate energy balance residual
residual = mm.surface_energy_balance(net_radiation, soil_heat_flux, sensible_heat_flux, latent_heat_flux)

print(f"Energy balance closure: {residual:.1f} W/m²")
print(f"Balance quality: {'Good' if abs(residual) < 20 else 'Poor'}")
```

### Flux Calculations
```python
# Define atmospheric parameters
air_temp = 293.15  # K
surface_temp = 298.15  # K
vapor_pressure = 1500  # Pa
saturation_vapor_pressure = 2000  # Pa
aerodynamic_resistance = 50  # s/m

# Calculate heat fluxes
sensible_flux = mm.sensible_heat_flux(air_temp, surface_temp, aerodynamic_resistance)
latent_flux = mm.latent_heat_flux(vapor_pressure, saturation_vapor_pressure, aerodynamic_resistance, air_temp)

print(f"Sensible heat flux: {sensible_flux:.1f} W/m²")
print(f"Latent heat flux: {latent_flux:.1f} W/m²")

# Calculate Bowen ratio
bowen_ratio = sensible_flux / latent_flux if latent_flux != 0 else float('inf')
print(f"Bowen ratio: {bowen_ratio:.3f}")
```

## Advanced Applications

### Micrometeorological Field Campaign Analysis
```python
def analyze_micrometeorological_data(u, v, w, temperature, height, z0):
    """
    Comprehensive micrometeorological analysis
    """
    # Calculate basic statistics
    wind_speed = np.sqrt(u**2 + v**2)
    
    # Calculate stability parameters
    ri = mm.bulk_richardson_number(u, v, temperature, height)
    ustar = mm.friction_velocity_from_wind(wind_speed, height, z0)
    
    # Calculate turbulence statistics
    TI = mm.turbulence_intensity(u, v, w)
    
    # Classify atmospheric conditions
    stability_class = classify_atmospheric_stability(ri)
    
    return {
        'wind_speed': wind_speed,
        'friction_velocity': ustar,
        'richardson_number': ri,
        'turbulence_intensity': TI,
        'stability_class': stability_class
    }

def classify_atmospheric_stability(ri):
    """Classify atmospheric stability based on Richardson number"""
    if ri < -0.5:
        return "Very Unstable"
    elif ri < 0.0:
        return "Unstable"
    elif ri < 0.25:
        return "Neutral"
    elif ri < 1.0:
        return "Stable"
    else:
        return "Very Stable"

# Example usage
# micromet_data = analyze_micrometeorological_data(u_wind, v_wind, w_wind, temp, height, z0)
```

### Eddy Covariance Data Processing
```python
def process_eddy_covariance_data(u, v, w, T, CO2, metadata):
    """
    Process eddy covariance data to calculate turbulent fluxes
    """
    # Calculate fluxes
    momentum_flux = np.mean(u * w)
    sensible_heat_flux = np.mean(w * T)
    CO2_flux = np.mean(w * CO2)
    
    # Calculate friction velocity
    ustar = np.sqrt(abs(momentum_flux))
    
    # Calculate Monin-Obukhov length
    L = mm.monin_obukhov_length(
        ustar, np.mean(T), metadata['air_density'], 
        metadata['specific_heat'], sensible_heat_flux
    )
    
    # Calculate stability corrections
    z_over_L = metadata['height'] / L
    psi_m = mm.psi_momentum(z_over_L)
    psi_h = mm.psi_heat(z_over_L)
    
    return {
        'momentum_flux': momentum_flux,
        'sensible_heat_flux': sensible_heat_flux,
        'CO2_flux': CO2_flux,
        'friction_velocity': ustar,
        'monin_obukhov_length': L,
        'stability_corrections': {'psi_m': psi_m, 'psi_h': psi_h}
    }

# Example usage
# flux_data = process_eddy_covariance_data(u, v, w, temperature, co2, metadata)
```

### Boundary Layer Height Detection
```python
def detect_boundary_layer_height(wind_speed, potential_temperature, time_height_matrix):
    """
    Detect atmospheric boundary layer height from wind and temperature profiles
    """
    # Calculate bulk Richardson number at different heights
    ri_profiles = []
    for i in range(len(time_height_matrix['height'])):
        u_profile = time_height_matrix['u_wind'][:, i]
        v_profile = time_height_matrix['v_wind'][:, i]
        theta_profile = time_height_matrix['potential_temperature'][:, i]
        height_profile = time_height_matrix['height'][i]
        
        ri = mm.bulk_richardson_number(u_profile, v_profile, theta_profile, height_profile)
        ri_profiles.append(ri)
    
    # Find height where Ri > critical value (typically 0.25)
    critical_ri = 0.25
    bl_heights = []
    for ri_profile in ri_profiles:
        bl_idx = np.where(ri_profile > critical_ri)[0]
        if len(bl_idx) > 0:
            bl_height = time_height_matrix['height'][bl_idx[0]]
        else:
            bl_height = np.nan
        bl_heights.append(bl_height)
    
    return np.array(bl_heights)

# Example usage
# bl_heights = detect_boundary_layer_height(wind_data, temp_data, profile_data)
```

## Error Handling

The functions include input validation for meteorological合理性:

- Wind speeds should be non-negative
- Heights should be positive
- Temperatures should be above absolute zero
- Flux values should be physically reasonable
- Richardson numbers should be finite

### Common Errors
```python
# Error: Negative wind speed
try:
    mm.bulk_richardson_number(-5, 2, 300, 100)
except ValueError as e:
    print(f"Error: {e}")

# Error: Zero height
try:
    mm.aerodynamic_resistance(0.3, 0.01, 0)
except ValueError as e:
    print(f"Error: {e}")

# Error: Negative temperature
try:
    mm.monin_obukhov_length(0.3, -274, 1.2, 1004, 150)
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Vectorization
All functions support numpy arrays for efficient vectorized operations:

```python
# Vectorized calculation for multiple time steps
u_array = np.array([5.0, 5.2, 4.8, 5.1, 4.9])
v_array = np.array([2.0, 2.1, 1.9, 2.0, 2.1])
ri_array = mm.bulk_richardson_number(u_array, v_array, 300.0, 100.0)
```

### Memory Efficiency
For large datasets, use chunked processing:

```python
def process_large_flux_dataset(data_chunk):
    """Process a chunk of flux data"""
    result = {}
    result['sensible_heat_flux'] = mm.sensible_heat_flux(
        data_chunk['air_temp'],
        data_chunk['surface_temp'],
        data_chunk['aerodynamic_resistance']
    )
    result['latent_heat_flux'] = mm.latent_heat_flux(
        data_chunk['vapor_pressure'],
        data_chunk['saturation_vapor_pressure'],
        data_chunk['aerodynamic_resistance'],
        data_chunk['air_temp']
    )
    return result
```

## References

- Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer Academic Publishers.
- Garratt, J.R. (1994). The Atmospheric Boundary Layer. Cambridge University Press.
- Foken, T. (2008). Micrometeorology. Springer.
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/

## See Also

- [Thermodynamics Module](../thermodynamics.md) - Thermodynamic variable calculations
- [Dynamic Calculations](../dynamics.md) - Dynamic meteorology functions
- [Unit Conversions](../units.md) - Meteorological unit conversion utilities
- [Data Models](../models.md) - Structured data models for atmospheric data
- [Xarray Integration](../io.md) - Integration with xarray for gridded data