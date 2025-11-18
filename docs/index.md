# Monet-Meteo Documentation

Welcome to the monet-meteo library documentation. This comprehensive toolkit provides tools for atmospheric calculations, including thermodynamic variables, derived parameters, dynamic calculations, unit conversions, interpolations, and coordinate transformations.

## 🌟 Overview

The monet-meteo library is designed for atmospheric scientists, meteorologists, and climate researchers who need reliable, efficient tools for atmospheric calculations. The library follows established meteorological conventions and provides both high-level convenience functions and low-level computational routines.

### Key Features

- **Comprehensive Thermodynamics**: Calculate potential temperature, equivalent potential temperature, mixing ratios, and more
- **Dynamic Meteorology**: Vorticity, divergence, geostrophic wind, and atmospheric dynamics
- **Statistical Analysis**: Monin-Obukhov similarity theory, surface energy balance, turbulent fluxes
- **Unit Conversions**: Complete meteorological unit conversion utilities
- **Coordinate Transformations**: Geographic projections, coordinate system conversions
- **Data Interpolation**: Advanced interpolation methods for atmospheric data
- **I/O Operations**: NetCDF handling with xarray integration and CF compliance
- **Data Models**: Structured classes for atmospheric profiles and data organization

## 📚 Documentation Structure

### Getting Started

- **[Quick Start Guide](quickstart.md)** - Installation and basic usage
- **[User Guide](userguide.md)** - Comprehensive usage instructions and workflows
- **[Installation](installation.md)** - Detailed installation instructions

### API Reference

- **[API Overview](api/index.md)** - Complete API reference and module index
- **[Thermodynamics](api/thermodynamics.md)** - Thermodynamic calculations
- **[Derived Parameters](api/derived.md)** - Comfort indices and derived variables
- **[Dynamic Calculations](api/dynamics.md)** - Atmospheric dynamics
- **[Statistical Analysis](api/statistical.md)** - Micrometeorology and statistics
- **[Unit Conversions](api/units.md)** - Meteorological unit conversions
- **[Data Models](api/models.md)** - Structured data classes
- **[Coordinates](api/coordinates.md)** - Geographic calculations
- **[Interpolation](api/interpolation.md)** - Data interpolation methods
- **[I/O Operations](api/io.md)** - File operations and xarray integration

### Examples and Tutorials

- **[Basic Examples](examples/basic.md)** - Simple usage examples
- **[Climate Data Processing](examples/climate.md)** - Climate dataset processing
- **[Model Output Analysis](examples/models.md)** - Model output processing
- **[Real-time Processing](examples/realtime.md)** - Real-time data processing
- **[Visualization Examples](examples/visualization.md)** - Plotting and visualization

### Advanced Topics

- **[Performance Optimization](advanced/performance.md)** - Performance tuning and optimization
- **[Best Practices](advanced/bestpractices.md)** - Development best practices
- **[Integration Guide](advanced/integration.md)** - Integration with other tools
- **[Contributing](advanced/contributing.md)** - Contribution guidelines

### Reference

- **[Physical Constants](reference/constants.md)** - Atmospheric constants and parameters
- **[Glossary](reference/glossary.md)** - Meteorological terminology
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions
- **[Changelog](reference/changelog.md)** - Version history and changes

## 🚀 Quick Start

### Installation

```bash
pip install monet-meteo
```

### Basic Usage

```python
import monet_meteo as mm
import numpy as np

# Calculate potential temperature
pressure = np.array([100000, 85000, 70000])  # Pa
temperature = np.array([298.15, 285.15, 273.15])  # K
potential_temp = mm.potential_temperature(pressure, temperature)

# Convert units
pressure_pa = mm.convert_pressure(1013.25, 'hPa', 'Pa')
temperature_c = mm.convert_temperature(298.15, 'K', 'C')

# Geographic calculations
distance = mm.calculate_distance(40.7, -74.0, 51.5, -0.1)  # NYC to London
```

### Working with NetCDF Data

```python
import xarray as xr
import monet_meteo as mm

# Load data
dataset = mm.load_netcdf_dataset('weather_data.nc')

# Convert units
dataset['pressure'] = mm.xr_convert_pressure(dataset['pressure'], 'hPa', 'Pa')

# Interpolate to pressure levels
pressure_levels = xr.DataArray([1000, 850, 700], dims=['pressure'])
interpolated = mm.xr_interpolate_vertical(
    dataset['temperature'], 'level', pressure_levels
)
```

## 📊 Core Modules

### Thermodynamics
Calculate fundamental atmospheric properties:

```python
# Basic thermodynamic properties
theta = mm.potential_temperature(pressure, temperature)
mixing_ratio = mm.mixing_ratio(vapor_pressure, total_pressure)
rh = mm.relative_humidity(temperature, dewpoint)
```

### Dynamic Calculations
Perform atmospheric dynamics calculations:

```python
# Vorticity and divergence
vorticity = mm.relative_vorticity(u_wind, v_wind, dx, dy)
divergence = mm.divergence(u_wind, v_wind, dx, dy)

# Wind calculations
geostrophic_u, geostrophic_v = mm.geostrophic_wind(pressure_gradient, latitude)
```

### Statistical Analysis
Micrometeorological and statistical functions:

```python
# Monin-Obukhov similarity theory
monin_obukhov_length = mm.monin_obukhov_length(
    friction_velocity, height, sensible_heat_flux, latent_heat_flux
)

# Surface energy balance
energy_balance = mm.surface_energy_balance(
    incoming_shortwave, incoming_longwave, outgoing_longwave,
    sensible_heat, latent_heat, ground_heat
)
```

### Unit Conversions
Comprehensive meteorological unit conversions:

```python
# Pressure conversions
pressure_pa = mm.convert_pressure(pressure_hpa, 'hPa', 'Pa')

# Temperature conversions
temperature_c = mm.convert_temperature(temperature_k, 'K', 'C')

# Wind speed conversions
wind_knots = mm.convert_wind_speed(wind_mps, 'm/s', 'knots')
```

## 🔧 Advanced Features

### Data Models
Use structured data classes:

```python
# Create atmospheric profile
profile = mm.AtmosphericProfile(
    pressure=pressure_levels,
    temperature=temperature_profile,
    mixing_ratio=mixing_ratio_profile
)

# Calculate derived properties
profile.calculate_thermodynamic_properties()
```

### Coordinate Transformations
Handle geographic calculations:

```python
# Geographic to Cartesian conversion
x, y, z = mm.latlon_to_cartesian(latitude, longitude, elevation)

# Distance and bearing calculations
distance = mm.calculate_distance(lat1, lon1, lat2, lon2)
bearing = mm.bearing(lat1, lon1, lat2, lon2)
```

### Advanced Interpolation
Interpolate atmospheric data:

```python
# Vertical interpolation
interpolated_temp = mm.interpolate_vertical(
    temperature, old_pressure_levels, new_pressure_levels, method='log'
)

# Horizontal interpolation
interpolated_field = mm.interpolate_horizontal(
    data, old_lons, old_lats, new_lons, new_lats, method='cubic'
)
```

## 🎯 Use Cases

### Climate Research
```python
def analyze_climate_trends(climate_data):
    # Calculate climatological statistics
    seasonal_means = climate_data.groupby('season').mean()
    
    # Analyze temperature trends with thermodynamics
    potential_temps = mm.potential_temperature(
        climate_data['pressure'], climate_data['temperature']
    )
    
    return seasonal_means, potential_temps
```

### Weather Forecasting
```python
def process_model_output(model_data):
    # Interpolate to pressure levels
    standard_levels = [1000, 850, 700, 500, 300]  # hPa
    
    for variable in ['temperature', 'u_wind', 'v_wind', 'humidity']:
        if variable in model_data:
            if variable in ['u_wind', 'v_wind']:
                u_interp, v_interp = mm.xr_interpolate_wind_components(
                    model_data['u_wind'], model_data['v_wind'],
                    model_data['pressure'], standard_levels
                )
                model_data[f'{variable}_interp'] = u_interp
                model_data[f'{variable}_interp'] = v_interp
            else:
                model_data[f'{variable}_interp'] = mm.xr_interpolate_vertical(
                    model_data[variable], 'pressure', standard_levels
                )
    
    return model_data
```

### Environmental Monitoring
```python
def calculate_surface_fluxes(met_data):
    # Calculate Monin-Obukhov length
    monin_obukhov_length = mm.monin_obukhov_length(
        met_data['friction_velocity'],
        met_data['measurement_height'],
        met_data['sensible_heat_flux'],
        met_data['latent_heat_flux']
    )
    
    # Calculate surface energy balance
    energy_balance = mm.surface_energy_balance(
        met_data['incoming_shortwave'],
        met_data['incoming_longwave'],
        met_data['outgoing_longwave'],
        met_data['sensible_heat_flux'],
        met_data['latent_heat_flux'],
        met_data['ground_heat_flux']
    )
    
    return {
        'monin_obukhov_length': monin_obukhov_length,
        'energy_balance': energy_balance
    }
```

## 📈 Performance Features

### Vectorized Operations
All functions support numpy arrays for efficient vectorized calculations:

```python
# Vectorized calculation for multiple profiles
pressure_profiles = np.random.rand(100, 10)  # 100 profiles, 10 levels
temperature_profiles = np.random.rand(100, 10) + 273.15

potential_temps = mm.potential_temperature(pressure_profiles, temperature_profiles)
```

### Memory Efficiency
Process large datasets efficiently:

```python
# Process large datasets in chunks
def process_large_climate_data(filepath, chunk_size=1000):
    dataset = mm.load_netcdf_dataset(filepath)
    
    for i in range(0, len(dataset.time), chunk_size):
        chunk = dataset.isel(time=slice(i, i + chunk_size))
        process_chunk(chunk)
```

### Dask Integration
Leverage dask for parallel processing of large datasets:

```python
import dask.array as da

# Create dask array for parallel processing
dask_array = da.from_array(large_data, chunks=(1000, 1000))

# Parallel interpolation
result = mm.xr_interpolate_with_dask(
    dask_array, old_coords, new_coords, method='linear'
)
```

## 🔗 Integration with Other Tools

### Matplotlib Visualization
```python
import matplotlib.pyplot as plt
import monet_meteo as mm

# Plot atmospheric profile
altitude = mm.pressure_to_altitude(pressure_levels)
plt.plot(temperature - 273.15, altitude / 1000)
plt.xlabel('Temperature (°C)')
plt.ylabel('Altitude (km)')
```

### Cartopy Mapping
```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Plot weather data on map
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.contourf(lon, lat, temperature, transform=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
```

### Pandas DataFrames
```python
import pandas as pd
import monet_meteo as mm

# Calculate derived parameters in pandas DataFrame
weather_df['heat_index'] = weather_df.apply(
    lambda row: mm.heat_index(row['temperature_k'], row['relative_humidity']),
    axis=1
)
```

## 🤝 Contributing

We welcome contributions to the monet-meteo library! Please see our [contributing guide](advanced/contributing.md) for:

- Development setup instructions
- Code style guidelines
- Testing procedures
- Pull request process

### Community Support

- **Issues**: Report bugs and request features on [GitHub Issues]
- **Discussions**: Ask questions and share ideas in [GitHub Discussions]
- **Documentation**: Help improve documentation with pull requests

## 📄 License

The monet-meteo library is released under the MIT License. See the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

This library is developed by the NOAA Air Resources Laboratory and builds upon the work of many atmospheric scientists and software developers. We gratefully acknowledge:

- The broader atmospheric science community for foundational algorithms
- Contributors who have helped improve the library
- Users who provide feedback and suggestions

---

**Next Steps:**

1. [Read the Quick Start Guide](quickstart.md) to get up and running quickly
2. Explore the [API Documentation](api/index.md) for detailed function references
3. Check out the [Examples](examples/basic.md) for practical usage scenarios
4. Join our [community](https://github.com/noaa-arlab/monet-meteo/discussions) to get help and share ideas

For the latest updates and announcements, follow us on [GitHub](https://github.com/noaa-arlab/monet-meteo).