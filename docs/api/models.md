# Data Models Module

The models module provides structured data classes for representing atmospheric profiles, wind profiles, and derived meteorological parameters. These classes ensure data consistency and provide convenient methods for atmospheric data analysis.

## Data Classes

### `AtmosphericProfile`
A comprehensive data class representing a complete atmospheric profile.

```python
mm.AtmosphericProfile(pressure, temperature, ...)
```

**Attributes:**
- `pressure` (Union[np.ndarray, xr.DataArray]): Pressure levels (Pa)
- `temperature` (Union[np.ndarray, xr.DataArray]): Temperature profile (K)
- `potential_temperature` (Optional[Union[np.ndarray, xr.DataArray]]): Potential temperature profile (K)
- `equivalent_potential_temperature` (Optional[Union[np.ndarray, xr.DataArray]]): Equivalent potential temperature profile (K)
- `virtual_temperature` (Optional[Union[np.ndarray, xr.DataArray]]): Virtual temperature profile (K)
- `u_wind` (Optional[Union[np.ndarray, xr.DataArray]]): Eastward wind component (m/s)
- `v_wind` (Optional[Union[np.ndarray, xr.DataArray]]): Northward wind component (m/s)
- `mixing_ratio` (Optional[Union[np.ndarray, xr.DataArray]]): Water vapor mixing ratio (kg/kg)
- `relative_humidity` (Optional[Union[np.ndarray, xr.DataArray]]): Relative humidity (0-1)
- `height` (Optional[Union[np.ndarray, xr.DataArray]]): Height levels (m)

**Methods:**
- `calculate_thermodynamic_properties()`: Calculate derived thermodynamic properties if not already provided

**Example:**
```python
import monet_meteo as mm
import numpy as np

# Create atmospheric profile
profile = mm.AtmosphericProfile(
    pressure=np.array([1000, 850, 700, 500, 300]),  # hPa
    temperature=np.array([298.15, 285.15, 273.15, 250.15, 230.15]),  # K
    mixing_ratio=np.array([0.015, 0.008, 0.003, 0.001, 0.0001])  # kg/kg
)

# Calculate derived properties
profile.calculate_thermodynamic_properties()

# Access results
print(f"Pressure levels: {profile.pressure}")
print(f"Potential temperature: {profile.potential_temperature}")
print(f"Equivalent potential temperature: {profile.equivalent_potential_temperature}")
```

**Data Validation:**
The class automatically validates that pressure and temperature arrays have the same shape:

```python
try:
    # This will raise ValueError due to shape mismatch
    invalid_profile = mm.AtmosphericProfile(
        pressure=np.array([1000, 850, 700]),
        temperature=np.array([298.15, 285.15])
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### `WindProfile`
A data class representing a wind profile with height information.

```python
mm.WindProfile(height, u_wind, v_wind)
```

**Attributes:**
- `height` (Union[np.ndarray, xr.DataArray]): Height levels (m)
- `u_wind` (Union[np.ndarray, xr.DataArray]): Eastward wind component (m/s)
- `v_wind` (Union[np.ndarray, xr.DataArray]): Northward wind component (m/s)

**Methods:**
- `wind_speed()`: Calculate wind speed from u and v components
- `wind_direction()`: Calculate wind direction from u and v components (in degrees)

**Example:**
```python
# Create wind profile
wind_profile = mm.WindProfile(
    height=np.array([10, 50, 100, 200]),  # m
    u_wind=np.array([5.0, 8.0, 10.0, 12.0]),  # m/s
    v_wind=np.array([2.0, 3.0, 4.0, 5.0])   # m/s
)

# Calculate derived properties
wind_speed = wind_profile.wind_speed()
wind_direction = wind_profile.wind_direction()

print(f"Wind speed: {wind_speed} m/s")
print(f"Wind direction: {wind_direction} degrees")
```

**Data Validation:**
The class validates that all arrays have the same shape:

```python
try:
    # This will raise ValueError due to shape mismatch
    invalid_wind = mm.WindProfile(
        height=np.array([10, 50, 100]),
        u_wind=np.array([5.0, 8.0]),
        v_wind=np.array([2.0, 3.0])
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### `ThermodynamicProfile`
A data class representing thermodynamic properties of the atmosphere.

```python
mm.ThermodynamicProfile(pressure, temperature, ...)
```

**Attributes:**
- `pressure` (Union[np.ndarray, xr.DataArray]): Pressure levels (Pa)
- `temperature` (Union[np.ndarray, xr.DataArray]): Temperature profile (K)
- `dewpoint` (Optional[Union[np.ndarray, xr.DataArray]]): Dewpoint temperature profile (K)
- `mixing_ratio` (Optional[Union[np.ndarray, xr.DataArray]]): Water vapor mixing ratio (kg/kg)
- `relative_humidity` (Optional[Union[np.ndarray, xr.DataArray]]): Relative humidity (0-1)

**Example:**
```python
# Create thermodynamic profile
thermo_profile = mm.ThermodynamicProfile(
    pressure=np.array([1000, 850, 700]) * 100,  # Convert to Pa
    temperature=np.array([298.15, 285.15, 273.15]),  # K
    relative_humidity=np.array([0.7, 0.8, 0.9])  # 70%, 80%, 90%
)

# Access data
print(f"Pressure: {thermo_profile.pressure} Pa")
print(f"Temperature: {thermo_profile.temperature} K")
print(f"Relative humidity: {thermo_profile.relative_humidity}")
```

**Data Validation:**
The class validates pressure and temperature arrays have the same shape and ensures at least one moisture variable is provided.

### `DerivedParameters`
A data class for representing derived meteorological parameters.

```python
mm.DerivedParameters(...)
```

**Attributes:**
- `heat_index` (Optional[Union[np.ndarray, xr.DataArray]]): Heat index values (°F)
- `wind_chill` (Optional[Union[np.ndarray, xr.DataArray]]): Wind chill temperatures (°F)
- `lcl_height` (Optional[Union[np.ndarray, xr.DataArray]]): Lifting condensation level height (m)
- `wet_bulb_temp` (Optional[Union[np.ndarray, xr.DataArray]]): Wet bulb temperature (K)

**Example:**
```python
# Create derived parameters
derived = mm.DerivedParameters(
    heat_index=np.array([95, 100, 105]),  # °F
    wind_chill=np.array([20, 15, 10]),    # °F
    lcl_height=np.array([500, 600, 700]), # m
    wet_bulb_temp=np.array([295, 290, 285]) # K
)

# Access data
print(f"Heat index: {derived.heat_index} °F")
print(f"Wind chill: {derived.wind_chill} °F")
print(f"LCL height: {derived.lcl_height} m")
```

## Usage Patterns

### Basic Profile Creation and Analysis
```python
import monet_meteo as mm
import numpy as np

def create_standard_atmosphere_profile():
    """Create a standard atmosphere profile"""
    # Define pressure levels
    pressure_levels = np.logspace(2, 3.5, 20)  # 100 to 3162 hPa
    
    # Standard atmosphere temperature profile
    temperature_levels = 288.15 - 0.0065 * (pressure_levels - 101325) / 9.80665 * 287.04
    
    # Create profile
    profile = mm.AtmosphericProfile(
        pressure=pressure_levels * 100,  # Convert to Pa
        temperature=temperature_levels
    )
    
    # Calculate derived properties
    profile.calculate_thermodynamic_properties()
    
    return profile

# Example usage
std_atmosphere = create_standard_atmosphere_profile()
print(f"Profile shape: {std_atmosphere.pressure.shape}")
print(f"Potential temperature range: {std_atmosphere.potential_temperature.min():.1f} - {std_atmosphere.potential_temperature.max():.1f} K")
```

### Wind Profile Analysis
```python
def analyze_wind_shear(wind_profile):
    """
    Analyze wind shear from wind profile
    """
    # Calculate wind speed and direction
    wind_speed = wind_profile.wind_speed()
    wind_direction = wind_profile.wind_direction()
    
    # Calculate wind shear (speed difference between levels)
    wind_shear = np.diff(wind_speed)
    
    # Calculate directional shear (direction difference between levels)
    directional_shear = np.diff(wind_direction)
    
    # Normalize directional shear to [-180, 180]
    directional_shear = np.where(directional_shear > 180, directional_shear - 360, directional_shear)
    directional_shear = np.where(directional_shear < -180, directional_shear + 360, directional_shear)
    
    return {
        'wind_speed': wind_speed,
        'wind_direction': wind_direction,
        'wind_shear': wind_shear,
        'directional_shear': directional_shear,
        'height_diff': np.diff(wind_profile.height)
    }

# Example usage
wind_profile = mm.WindProfile(
    height=np.array([10, 50, 100, 200, 500]),
    u_wind=np.array([5, 8, 10, 12, 15]),
    v_wind=np.array([2, 3, 4, 5, 6])
)

shear_analysis = analyze_wind_shear(wind_profile)
print(f"Maximum wind shear: {np.max(shear_analysis['wind_shear']):.2f} m/s per 100m")
```

### Thermodynamic Profile Analysis
```python
def analyze_atmospheric_stability(thermo_profile):
    """
    Analyze atmospheric stability from thermodynamic profile
    """
    # Calculate lapse rate
    pressure_diff = np.diff(thermo_profile.pressure)
    temp_diff = np.diff(thermo_profile.temperature)
    lapse_rate = -temp_diff / pressure_diff * 9.80665 / 287.04
    
    # Classify stability
    dry_adiabatic_lapse_rate = 9.8  # K/km
    moist_adiabatic_lapse_rate = 6.5  # K/km
    
    stability = []
    for lr in lapse_rate:
        if lr > dry_adiabatic_lapse_rate:
            stability.append("Absolutely unstable")
        elif lr > moist_adiabatic_lapse_rate:
            stability.append("Conditionally unstable")
        else:
            stability.append("Stable")
    
    return {
        'lapse_rate': lapse_rate,
        'stability_classification': stability,
        'temperature_gradient': temp_diff / pressure_diff * 1000  # K/km
    }

# Example usage
thermo_profile = mm.ThermodynamicProfile(
    pressure=np.array([1000, 850, 700, 500]) * 100,
    temperature=np.array([298.15, 285.15, 273.15, 250.15])
)

stability_analysis = analyze_atmospheric_stability(thermo_profile)
print(f"Stability classifications: {stability_analysis['stability_classification']}")
```

### Profile Interpolation and Comparison
```python
def compare_atmospheric_profiles(profile1, profile2):
    """
    Compare two atmospheric profiles on common pressure levels
    """
    # Find common pressure levels
    common_pressure = np.intersect1d(profile1.pressure, profile2.pressure)
    
    # Interpolate profile1 to common levels
    from scipy import interpolate
    interp_temp1 = interpolate.interp1d(profile1.pressure, profile1.temperature, 
                                       bounds_error=False, fill_value='extrapolate')
    interp_temp2 = interpolate.interp1d(profile2.pressure, profile2.temperature, 
                                       bounds_error=False, fill_value='extrapolate')
    
    temp1_interp = interp_temp1(common_pressure)
    temp2_interp = interp_temp2(common_pressure)
    
    # Calculate differences
    temp_difference = temp1_interp - temp2_interp
    
    return {
        'common_pressure': common_pressure,
        'profile1_temperature': temp1_interp,
        'profile2_temperature': temp2_interp,
        'temperature_difference': temp_difference
    }

# Example usage
profile1 = mm.AtmosphericProfile(
    pressure=np.array([1000, 850, 700, 500]) * 100,
    temperature=np.array([298.15, 285.15, 273.15, 250.15])
)

profile2 = mm.AtmosphericProfile(
    pressure=np.array([1000, 800, 600, 400]) * 100,
    temperature=np.array([300.15, 280.15, 260.15, 240.15])
)

comparison = compare_atmospheric_profiles(profile1, profile2)
print(f"Maximum temperature difference: {np.max(np.abs(comparison['temperature_difference'])):.2f} K")
```

## Advanced Applications

### Radiosonde Data Processing
```python
def process_radiosonde_data(pressure, temperature, dewpoint, u_wind, v_wind, height=None):
    """
    Process raw radiosonde data into structured profiles
    """
    # Convert to appropriate units if needed
    pressure_pa = pressure * 100 if pressure.max() < 10000 else pressure
    
    # Create comprehensive atmospheric profile
    profile = mm.AtmosphericProfile(
        pressure=pressure_pa,
        temperature=temperature,
        u_wind=u_wind,
        v_wind=v_wind,
        height=height if height is not None else pressure_pa / 9.80665 * 287.04  # Approximate height
    )
    
    # Calculate derived properties
    profile.calculate_thermodynamic_properties()
    
    # Create wind profile
    wind_profile = mm.WindProfile(
        height=profile.height,
        u_wind=u_wind,
        v_wind=v_wind
    )
    
    # Create thermodynamic profile
    thermo_profile = mm.TermodynamicProfile(
        pressure=pressure_pa,
        temperature=temperature,
        dewpoint=dewpoint
    )
    
    return {
        'atmospheric_profile': profile,
        'wind_profile': wind_profile,
        'thermodynamic_profile': thermo_profile,
        'derived_parameters': calculate_derived_parameters(thermo_profile)
    }

def calculate_derived_parameters(thermo_profile):
    """Calculate derived parameters from thermodynamic profile"""
    # Calculate lifting condensation level
    surface_temp = thermo_profile.temperature[0]
    surface_dewpoint = thermo_profile.dewpoint[0]
    lcl_height = mm.lifting_condensation_level(surface_temp, surface_dewpoint)
    
    # Calculate heat index (simplified)
    temp_c = mm.convert_temperature(surface_temp, 'K', 'C')
    rh = 0.7  # Assume 70% relative humidity
    heat_index = mm.heat_index(surface_temp, rh)
    heat_index_c = mm.convert_temperature(heat_index, 'K', 'C')
    
    return mm.DerivedParameters(
        lcl_height=np.array([lcl_height]),
        heat_index=np.array([heat_index_c])
    )

# Example usage
# radiosonde_data = process_radiosonde_data(pressure_hpa, temperature_k, dewpoint_k, u_wind, v_wind)
```

### Model Data Validation
```python
def validate_model_output(model_data, obs_data, tolerance=0.1):
    """
    Validate model output against observations using profile models
    """
    # Create profiles from model data
    model_profile = mm.AtmosphericProfile(
        pressure=model_data['pressure'],
        temperature=model_data['temperature'],
        u_wind=model_data.get('u_wind'),
        v_wind=model_data.get('v_wind')
    )
    
    # Create profiles from observation data
    obs_profile = mm.AtmosphericProfile(
        pressure=obs_data['pressure'],
        temperature=obs_data['temperature'],
        u_wind=obs_data.get('u_wind'),
        v_wind=obs_data.get('v_wind')
    )
    
    # Interpolate to common levels for comparison
    comparison = compare_atmospheric_profiles(model_profile, obs_profile)
    
    # Calculate statistics
    temp_rmse = np.sqrt(np.mean(comparison['temperature_difference']**2))
    temp_bias = np.mean(comparison['temperature_difference'])
    
    validation_results = {
        'temperature_rmse': temp_rmse,
        'temperature_bias': temp_bias,
        'is_within_tolerance': temp_rmse <= tolerance,
        'comparison_data': comparison
    }
    
    return validation_results

# Example usage
# validation = validate_model_output(model_output, observations, tolerance=0.5)
```

### Climate Data Analysis
```python
def analyze_climate_profile_climatology(profile_data, time_dim='time'):
    """
    Analyze climatology of atmospheric profiles
    """
    # Calculate seasonal means
    seasonal_profiles = {}
    
    for season in ['DJF', 'MAM', 'JJA', 'SON']:
        season_mask = profile_data['season'] == season
        if np.any(season_mask):
            seasonal_profile = mm.AtmosphericProfile(
                pressure=np.mean(profile_data['pressure'][season_mask], axis=0),
                temperature=np.mean(profile_data['temperature'][season_mask], axis=0),
                u_wind=np.mean(profile_data['u_wind'][season_mask], axis=0),
                v_wind=np.mean(profile_data['v_wind'][season_mask], axis=0)
            )
            seasonal_profiles[season] = seasonal_profile
    
    # Calculate long-term mean profile
    long_term_profile = mm.AtmosphericProfile(
        pressure=np.mean(profile_data['pressure'], axis=0),
        temperature=np.mean(profile_data['temperature'], axis=0),
        u_wind=np.mean(profile_data['u_wind'], axis=0),
        v_wind=np.mean(profile_data['v_wind'], axis=0)
    )
    
    return {
        'long_term_mean': long_term_profile,
        'seasonal_means': seasonal_profiles,
        'climatological_statistics': calculate_climatological_statistics(profile_data)
    }

def calculate_climatological_statistics(profile_data):
    """Calculate climatological statistics"""
    stats = {}
    
    for var in ['temperature', 'u_wind', 'v_wind']:
        stats[f'{var}_mean'] = np.mean(profile_data[var], axis=0)
        stats[f'{var}_std'] = np.std(profile_data[var], axis=0)
        stats[f'{var}_max'] = np.max(profile_data[var], axis=0)
        stats[f'{var}_min'] = np.min(profile_data[var], axis=0)
    
    return stats

# Example usage
# climatology = analyze_climate_profile_climatology(climate_data)
```

## Error Handling

The data models include comprehensive validation:

### Common Errors
```python
# Error: Shape mismatch
try:
    profile = mm.AtmosphericProfile(
        pressure=np.array([1000, 850]),
        temperature=np.array([298.15])
    )
except ValueError as e:
    print(f"Shape mismatch error: {e}")

# Error: Inconsistent array types
try:
    profile = mm.AtmosphericProfile(
        pressure=np.array([1000, 850]),
        temperature=[298.15, 285.15]  # List instead of array
    )
except ValueError as e:
    print(f"Array type error: {e}")

# Error: Missing required parameters for thermodynamic profile
try:
    thermo_profile = mm.TermodynamicProfile(
        pressure=np.array([1000, 850]) * 100,
        temperature=np.array([298.15, 285.15])
    )
    # This is valid but may not have complete thermodynamic information
    pass
except ValueError as e:
    print(f"Validation error: {e}")
```

## Performance Considerations

### Memory Efficiency
For large datasets, consider processing profiles individually:

```python
def process_large_profile_dataset(profile_data, chunk_size=1000):
    """Process large dataset of profiles in chunks"""
    processed_profiles = []
    
    for i in range(0, len(profile_data), chunk_size):
        chunk = profile_data[i:i+chunk_size]
        
        # Process chunk
        chunk_profiles = []
        for j in range(len(chunk)):
            profile = mm.AtmosphericProfile(
                pressure=chunk['pressure'][j],
                temperature=chunk['temperature'][j],
                u_wind=chunk['u_wind'][j],
                v_wind=chunk['v_wind'][j]
            )
            profile.calculate_thermodynamic_properties()
            chunk_profiles.append(profile)
        
        processed_profiles.extend(chunk_profiles)
    
    return processed_profiles
```

### Vectorized Operations
Leverage numpy operations for efficient calculations:

```python
def vectorized_profile_analysis(profiles):
    """Perform vectorized analysis on multiple profiles"""
    # Stack profiles for vectorized operations
    all_temps = np.stack([p.temperature for p in profiles])
    all_pressures = np.stack([p.pressure for p in profiles])
    
    # Vectorized calculation of potential temperature
    potential_temps = []
    for i in range(len(profiles)):
        theta = mm.potential_temperature(all_pressures[i], all_temps[i])
        potential_temps.append(theta)
    
    return potential_temps
```

## Data Persistence

### Saving and Loading Profiles
```python
import pickle
import json

def save_profile_to_file(profile, filename):
    """Save atmospheric profile to file"""
    with open(filename, 'wb') as f:
        pickle.dump(profile, f)

def load_profile_from_file(filename):
    """Load atmospheric profile from file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def export_profile_to_json(profile, filename):
    """Export profile data to JSON format"""
    data = {
        'pressure': profile.pressure.tolist(),
        'temperature': profile.temperature.tolist(),
        'u_wind': profile.u_wind.tolist() if profile.u_wind is not None else None,
        'v_wind': profile.v_wind.tolist() if profile.v_wind is not None else None,
        'mixing_ratio': profile.mixing_ratio.tolist() if profile.mixing_ratio is not None else None
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# Example usage
# save_profile_to_file(atmospheric_profile, 'profile.pkl')
# loaded_profile = load_profile_from_file('profile.pkl')
# export_profile_to_json(atmospheric_profile, 'profile.json')
```

## References

- World Meteorological Organization (WMO). (2018). Guide to Meteorological Instruments and Methods of Observation.
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/
- U.S. Standard Atmosphere. (1976). U.S. Government Printing Office.

## See Also

- [Thermodynamics Module](thermodynamics.md) - Thermodynamic variable calculations
- [Dynamic Calculations](dynamics.md) - Dynamic meteorology functions
- [Statistical Analysis](statistical.md) - Statistical and micrometeorological functions
- [Unit Conversions](units.md) - Meteorological unit conversion utilities
- [Coordinates](coordinates.md) - Coordinate transformation utilities