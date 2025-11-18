# Basic Usage Examples

This page provides practical examples of Monet-Meteo's core functionality. Each example demonstrates common meteorological calculations and data processing tasks.

## 🌡️ Thermodynamic Calculations

### Example 1: Basic Atmospheric Properties

```python
import monet_meteo as mm
import numpy as np

# Define atmospheric conditions
pressure = 850.0  # hPa
temperature = 288.15  # K (15°C)
dewpoint = 283.15  # K (10°C)

# Calculate thermodynamic properties
theta = mm.potential_temperature(pressure, temperature)
virtual_temp = mm.virtual_temperature(temperature, 0.008)  # mixing ratio
sat_vapor_pressure = mm.saturation_vapor_pressure(temperature)
mixing_ratio = mm.mixing_ratio(sat_vapor_pressure, pressure * 100)
relative_humidity = mm.relative_humidity(sat_vapor_pressure * 0.7, sat_vapor_pressure)

print(f"Potential Temperature: {theta:.2f} K")
print(f"Virtual Temperature: {virtual_temp:.2f} K")
print(f"Saturation Vapor Pressure: {sat_vapor_pressure:.2f} Pa")
print(f"Mixing Ratio: {mixing_ratio:.6f} kg/kg")
print(f"Relative Humidity: {relative_humidity:.2%}")
```

### Example 2: LCL and Wet Bulb Temperature

```python
import numpy as np

# Surface conditions
surface_temp = 298.15  # K (25°C)
surface_dewpoint = 288.15  # K (15°C)
pressure = 1013.25  # hPa

# Calculate lifting condensation level
lcl_height = mm.lifting_condensation_level(surface_temp, surface_dewpoint)
print(f"Lifting Condensation Level: {lcl_height:.1f} m")

# Calculate wet bulb temperature
wetbulb_temp = mm.wet_bulb_temperature(surface_temp, pressure * 100, relative_humidity)
print(f"Wet Bulb Temperature: {wetbulb_temp:.2f} K ({mm.convert_temperature(wetbulb_temp, 'K', 'C'):.2f}°C)")

# Calculate equivalent potential temperature
theta_e = mm.equivalent_potential_temperature(pressure * 100, surface_temp, mixing_ratio)
print(f"Equivalent Potential Temperature: {theta_e:.2f} K")
```

## 🔄 Unit Conversions

### Example 3: Comprehensive Unit Conversion

```python
# Temperature conversions
temperatures = {
    'kelvin': 288.15,
    'celsius': 15.0,
    'fahrenheit': 59.0
}

print("Temperature Conversions:")
for temp_unit, temp_value in temperatures.items():
    converted = {
        'K': mm.convert_temperature(temp_value, temp_unit, 'K'),
        'C': mm.convert_temperature(temp_value, temp_unit, 'C'),
        'F': mm.convert_temperature(temp_value, temp_unit, 'F')
    }
    print(f"{temp_value:.2f} {temp_unit.upper()} = "
          f"{converted['K']:.2f} K, "
          f"{converted['C']:.2f}°C, "
          f"{converted['F']:.2f}°F")

# Pressure conversions
pressure_hpa = 1013.25
print(f"\nPressure Conversions:")
print(f"{pressure_hpa:.2f} hPa = "
      f"{mm.convert_pressure(pressure_hpa, 'hPa', 'Pa'):.2f} Pa = "
      f"{mm.convert_pressure(pressure_hpa, 'hPa', 'mmHg'):.2f} mmHg = "
      f"{mm.convert_pressure(pressure_hpa, 'hPa', 'atm'):.4f} atm")

# Wind speed conversions
wind_speeds = {
    'm/s': 10.0,
    'km/h': 36.0,
    'knots': 19.44,
    'mph': 22.37
}

print("\nWind Speed Conversions:")
for unit, value in wind_speeds.items():
    converted = mm.convert_wind_speed(value, unit, 'm/s')
    print(f"{value:.2f} {unit} = {converted:.2f} m/s")
```

## 🌊 Atmospheric Profiles

### Example 4: Creating and Analyzing Atmospheric Profiles

```python
import numpy as np

# Create atmospheric sounding data
pressure_levels = np.array([1000, 850, 700, 500, 300, 200])  # hPa
temperatures = np.array([298.15, 285.15, 273.15, 250.15, 230.15, 220.15])  # K
dewpoints = np.array([295.15, 280.15, 268.15, 240.15, 210.15, 200.15])  # K

# Create atmospheric profile
profile = mm.AtmosphericProfile(
    pressure=pressure_levels,
    temperature=temperatures
)

# Calculate additional properties
profile.calculate_thermodynamic_properties()

# Create wind profile
heights = np.array([0, 100, 500, 1000, 2000])  # m
u_wind = np.array([0, 2, 5, 8, 12])  # m/s
v_wind = np.array([0, 1, 3, 6, 9])  # m/s

wind_profile = mm.WindProfile(
    height=heights,
    u_wind=u_wind,
    v_wind=v_wind
)

# Analyze profiles
print("Atmospheric Profile Analysis:")
print(f"Pressure levels: {profile.pressure} hPa")
print(f"Potential temperatures: {profile.potential_temperature} K")
print(f"Wind speeds: {wind_profile.wind_speed()} m/s")
print(f"Wind directions: {wind_profile.wind_direction()}°")
```

## 🌐 Geographic Calculations

### Example 5: Distance and Bearing Calculations

```python
# Major city coordinates
cities = {
    'New York': (40.7128, -74.0060),
    'Los Angeles': (34.0522, -118.2437),
    'Chicago': (41.8781, -87.6298),
    'Houston': (29.7604, -95.3698),
    'Miami': (25.7617, -80.1918)
}

print("Geographic Distance Calculations:")
city_names = list(cities.keys())
for i, city1 in enumerate(city_names):
    lat1, lon1 = cities[city1]
    for city2 in city_names[i+1:]:
        lat2, lon2 = cities[city2]
        
        # Calculate distance
        distance = mm.calculate_distance(lat1, lon1, lat2, lon2)
        bearing = mm.bearing(lat1, lon1, lat2, lon2)
        
        print(f"{city1} to {city2}: "
              f"{distance/1000:.0f} km, "
              f"Bearing: {bearing:.0f}°")

# Convert coordinates between systems
lat, lon = 40.7128, -74.0060  # New York
x, y, z = mm.latlon_to_cartesian(lat, lon)
print(f"\nCartesian coordinates: x={x:.2f}, y={y:.2f}, z={z:.2f}")
```

## 📊 Statistical Analysis

### Example 6: Atmospheric Stability Analysis

```python
import numpy as np

# Surface layer data
heights = np.array([2, 10, 50, 100])  # m
wind_speeds = np.array([1.5, 3.0, 6.0, 8.5])  # m/s
temperatures = np.array([294.15, 293.15, 292.15, 291.15])  # K
u_wind = np.array([0.5, 1.0, 2.0, 3.0])  # m/s
v_wind = np.array([1.0, 2.0, 4.0, 6.0])  # m/s

# Calculate stability parameters
potential_temp = mm.potential_temperature(1013.25, temperatures)
ri = mm.bulk_richardson_number(u_wind, v_wind, potential_temp, heights)

print("Atmospheric Stability Analysis:")
for i, (h, ri_val) in enumerate(zip(heights, ri)):
    stability = "Unstable" if ri_val < 0.0 else "Neutral" if ri_val < 0.25 else "Stable"
    print(f"Height {h}m: Ri = {ri_val:.3f} ({stability})")

# Monin-Obukhov calculation
friction_velocity = mm.friction_velocity_from_wind(wind_speeds[-1], heights[-1], 0.1)
L = mm.monin_obukhov_length(
    friction_velocity, temperatures[-1], 1.225, 1004, 150, 50
)

print(f"\nFriction velocity: {friction_velocity:.3f} m/s")
print(f"Monin-Obukhov length: {L:.1f} m")
stability = "Stable" if L > 0 else "Unstable" if L != float('inf') else "Neutral"
print(f"Atmospheric stability: {stability}")
```

## 🔄 Data Interpolation

### Example 7: Vertical Interpolation of Atmospheric Profiles

```python
import numpy as np

# Original sounding data
original_pressure = np.array([1000, 850, 700, 500, 300, 200])  # hPa
original_temperature = np.array([298.15, 285.15, 273.15, 250.15, 230.15, 220.15])  # K
original_humidity = np.array([0.8, 0.7, 0.6, 0.4, 0.2, 0.1])  # relative humidity

# Target pressure levels for interpolation
target_pressure = np.linspace(1000, 200, 20)  # hPa

# Interpolate temperature
interpolated_temp = mm.interpolate_vertical(
    original_temperature, original_pressure, target_pressure, method='log'
)

# Interpolate humidity
interpolated_humidity = mm.interpolate_vertical(
    original_humidity, original_pressure, target_pressure, method='linear'
)

# Create new profile with interpolated data
interpolated_profile = mm.AtmosphericProfile(
    pressure=target_pressure,
    temperature=interpolated_temp,
    relative_humidity=interpolated_humidity
)

print("Vertical Interpolation Results:")
print(f"Original levels: {len(original_pressure)} points")
print(f"Interpolated levels: {len(target_pressure)} points")
print(f"Temperature range: {interpolated_temp.min():.1f} - {interpolated_temp.max():.1f} K")
print(f"Humidity range: {interpolated_humidity.min():.1%} - {interpolated_humidity.max():.1%}")
```

## 🌟 Combined Example: Weather Analysis Pipeline

```python
import monet_meteo as mm
import numpy as np

def analyze_weather_conditions(pressure, temperature, dewpoint, wind_speed, wind_dir):
    """
    Complete weather analysis pipeline
    """
    # Convert units if needed
    pressure_pa = mm.convert_pressure(pressure, 'hPa', 'Pa')
    
    # Calculate thermodynamic properties
    theta = mm.potential_temperature(pressure, temperature)
    mixing_ratio = mm.mixing_ratio(
        mm.saturation_vapor_pressure(dewpoint), pressure_pa
    )
    
    # Calculate comfort indices
    heat_index = mm.heat_index(temperature, relative_humidity=0.6)
    wind_chill = mm.wind_chill(temperature, wind_speed)
    
    # Stability analysis
    u_wind, v_wind = mm.wind_components(wind_speed, wind_dir)
    ri = mm.bulk_richardson_number(u_wind, v_wind, theta, 10.0)
    
    # Weather classification
    if ri < -0.5:
        stability = "Very Unstable"
    elif ri < 0.0:
        stability = "Unstable"
    elif ri < 0.25:
        stability = "Neutral"
    elif ri < 1.0:
        stability = "Stable"
    else:
        stability = "Very Stable"
    
    return {
        'potential_temperature': theta,
        'mixing_ratio': mixing_ratio,
        'heat_index': heat_index,
        'wind_chill': wind_chill,
        'richardson_number': ri,
        'stability_class': stability
    }

# Example weather station data
weather_data = [
    {'pressure': 1013.25, 'temperature': 298.15, 'dewpoint': 290.15, 
     'wind_speed': 5.0, 'wind_dir': 270},
    {'pressure': 1010.0, 'temperature': 295.15, 'dewpoint': 288.15, 
     'wind_speed': 8.0, 'wind_dir': 315},
    {'pressure': 1005.0, 'temperature': 292.15, 'dewpoint': 285.15, 
     'wind_speed': 3.0, 'wind_dir': 180}
]

print("Weather Analysis Results:")
for i, data in enumerate(weather_data):
    results = analyze_weather_conditions(**data)
    print(f"\nStation {i+1}:")
    print(f"  Conditions: {data['temperature']-273.15:.1f}°C, "
          f"{data['wind_speed']:.1f} m/s wind")
    print(f"  Stability: {results['stability_class']}")
    print(f"  Potential Temperature: {results['potential_temperature']:.1f} K")
    print(f"  Mixing Ratio: {results['mixing_ratio']:.4f} kg/kg")
```

## 📝 Tips for Best Usage

1. **Input Validation**: Always check input units and ranges
2. **Array Operations**: Use numpy arrays for batch calculations
3. **Unit Consistency**: Maintain consistent units throughout calculations
4. **Error Handling**: Implement proper error handling for edge cases
5. **Performance**: Use vectorized operations for large datasets

## 🔄 Next Steps

- Explore more advanced examples in the [Tutorials](../../tutorials/) section
- Check the [API Reference](../../api/) for detailed function documentation
- Learn about [Xarray Integration](../../api/io.md) for gridded data processing
- Discover [Real-World Applications](../../tutorials/real-world-examples.md)