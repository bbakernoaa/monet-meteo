# Quick Start Guide

Welcome to Monet-Meteo! This guide will help you get started with the library quickly.

## 📦 Installation

### Prerequisites
- Python 3.8+
- NumPy
- xarray (optional, for advanced features)

### Basic Installation
```bash
pip install monet-meteo
```

### Development Installation
```bash
git clone https://github.com/noaa-arl/monet-meteo.git
cd monet-meteo
pip install -e .
```

### Optional Dependencies
For full functionality, install optional dependencies:
```bash
pip install xarray dask pint matplotlib cartopy
```

## 🚀 First Steps

### Basic Import
```python
import monet_meteo as mm
```

### Check Version
```python
print(mm.__version__)
```

## 🌡️ Basic Thermodynamic Calculations

### Potential Temperature
Calculate potential temperature using the Poisson equation:

```python
# Input data
pressure = 850  # hPa
temperature = 288.15  # K

# Calculate potential temperature
theta = mm.potential_temperature(pressure, temperature)
print(f"Potential temperature: {theta:.2f} K")
```

### Mixing Ratio and Relative Humidity
```python
# Input data
vapor_pressure = 1500  # Pa
total_pressure = 85000  # Pa

# Calculate mixing ratio
mixing_ratio = mm.mixing_ratio(vapor_pressure, total_pressure)
print(f"Mixing ratio: {mixing_ratio:.6f} kg/kg")

# Calculate saturation vapor pressure
sat_vapor_pressure = mm.saturation_vapor_pressure(temperature)
print(f"Saturation vapor pressure: {sat_vapor_pressure:.2f} Pa")
```

### Virtual Temperature
```python
# Calculate virtual temperature
virtual_temp = mm.virtual_temperature(temperature, mixing_ratio)
print(f"Virtual temperature: {virtual_temp:.2f} K")
```

## 🔄 Unit Conversions

### Temperature Conversion
```python
# Convert between temperature units
temp_celsius = mm.convert_temperature(288.15, 'K', 'C')
temp_fahrenheit = mm.convert_temperature(288.15, 'K', 'F')
temp_kelvin = mm.convert_temperature(15.0, 'C', 'K')

print(f"288.15 K = {temp_celsius:.2f}°C")
print(f"288.15 K = {temp_fahrenheit:.2f}°F")
print(f"15.0°C = {temp_kelvin:.2f} K")
```

### Pressure Conversion
```python
# Convert between pressure units
pressure_pa = mm.convert_pressure(1013.25, 'hPa', 'Pa')
pressure_atm = mm.convert_pressure(1013.25, 'hPa', 'atm')
pressure_mmhg = mm.convert_pressure(1013.25, 'hPa', 'mmHg')

print(f"1013.25 hPa = {pressure_pa:.2f} Pa")
print(f"1013.25 hPa = {pressure_atm:.4f} atm")
print(f"1013.25 hPa = {pressure_mmhg:.2f} mmHg")
```

### Wind Speed Conversion
```python
# Convert wind speed units
wind_mps = mm.convert_wind_speed(10, 'km/h', 'm/s')
wind_knots = mm.convert_wind_speed(10, 'm/s', 'knots')
wind_mph = mm.convert_wind_speed(10, 'm/s', 'mph')

print(f"10 km/h = {wind_mps:.2f} m/s")
print(f"10 m/s = {wind_knots:.2f} knots")
print(f"10 m/s = {wind_mph:.2f} mph")
```

## 🌊 Atmospheric Profiles

### Using Data Models
```python
import numpy as np

# Create atmospheric profile data
pressure_levels = np.array([1000, 850, 700, 500, 300])  # hPa
temperatures = np.array([288.15, 275.15, 260.15, 230.15, 210.15])  # K

# Create atmospheric profile
profile = mm.AtmosphericProfile(
    pressure=pressure_levels,
    temperature=temperatures
)

# Calculate thermodynamic properties
profile.calculate_thermodynamic_properties()
print(f"Potential temperature: {profile.potential_temperature}")
```

### Wind Profile
```python
# Create wind profile
heights = np.array([10, 50, 100, 200])  # m
u_wind = np.array([2.5, 5.0, 7.5, 10.0])  # m/s
v_wind = np.array([1.0, 2.0, 3.0, 4.0])  # m/s

wind_profile = mm.WindProfile(
    height=heights,
    u_wind=u_wind,
    v_wind=v_wind
)

# Calculate wind properties
wind_speed = wind_profile.wind_speed()
wind_direction = wind_profile.wind_direction()

print(f"Wind speed: {wind_speed}")
print(f"Wind direction: {wind_direction}")
```

## 🌐 Coordinate Systems

### Distance Calculation
```python
# Calculate distance between two points
lat1, lon1 = 40.7128, -74.0060  # New York
lat2, lon2 = 34.0522, -118.2437  # Los Angeles

distance = mm.calculate_distance(lat1, lon1, lat2, lon2)
print(f"Distance: {distance/1000:.2f} km")
```

### Bearing Calculation
```python
# Calculate bearing from point A to point B
bearing = mm.bearing(lat1, lon1, lat2, lon2)
print(f"Bearing: {bearing:.2f}°")
```

## 📊 Statistical Analysis

### Monin-Obukhov Length
```python
# Calculate Monin-Obukhov length for atmospheric stability
friction_velocity = 0.4  # m/s
temperature = 293.15  # K
air_density = 1.225  # kg/m³
specific_heat = 1004  # J/kg/K
sensible_heat_flux = 150  # W/m²

L = mm.monin_obukhov_length(
    friction_velocity, temperature, air_density, 
    specific_heat, sensible_heat_flux
)
print(f"Monin-Obukhov length: {L:.2f} m")
```

### Bulk Richardson Number
```python
# Calculate bulk Richardson number
u_wind = 5.0  # m/s
v_wind = 2.0  # m/s
potential_temp = 300.0  # K
height = 100.0  # m

ri = mm.bulk_richardson_number(u_wind, v_wind, potential_temp, height)
print(f"Bulk Richardson number: {ri:.4f}")
```

## 🔄 Data Interpolation

### Vertical Interpolation
```python
# Create sample data
old_pressure = np.array([1000, 850, 700, 500])  # hPa
old_temperature = np.array([288.15, 275.15, 265.15, 240.15])  # K

# Interpolate to new pressure levels
new_pressure = np.array([900, 800, 600, 400])  # hPa
interpolated_temp = mm.interpolate_vertical(
    old_temperature, old_pressure, new_pressure,
    method='log'
)

print(f"Interpolated temperatures: {interpolated_temp}")
```

## 📈 Advanced Usage with Xarray

If you have xarray installed, you can work with gridded climate data:

```python
import xarray as xr

# Create sample xarray dataset
data = xr.Dataset({
    'temperature': (['time', 'level', 'lat', 'lon'], np.random.rand(24, 5, 10, 10) * 30 + 250),
    'pressure': (['level'], [1000, 850, 700, 500, 300]),
    'lat': (['lat'], np.linspace(-45, 45, 10)),
    'lon': (['lon'], np.linspace(-180, 180, 10)),
    'time': (['time'], pd.date_range('2023-01-01', periods=24))
})

# Calculate potential temperature
data['theta'] = mm.potential_temperature(data.pressure, data.temperature)

# Convert units
data['temp_celsius'] = mm.convert_temperature(data.temperature, 'K', 'C')
```

## 📋 Next Steps

- Browse the [API Reference](api/) for detailed function documentation
- Check out the [Tutorials](tutorials/) for more complex examples
- Explore the [Real-World Examples](tutorials/real-world-examples.md) for practical applications

## 🆘 Need Help?

- Check the [Troubleshooting Guide](advanced/troubleshooting.md)
- Review the [Best Practices](advanced/best-practices.md)
- Visit our [GitHub Issues](https://github.com/noaa-arl/monet-meteo/issues)