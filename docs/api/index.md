# API Reference

Welcome to the monet-meteo library API documentation. This comprehensive API provides tools for atmospheric calculations, including thermodynamic variables, derived parameters, dynamic calculations, unit conversions, interpolations, and coordinate transformations.

## Overview

The monet-meteo library is organized into several modules, each providing specialized functionality for atmospheric science applications:

- **[Thermodynamics](thermodynamics.md)** - Thermodynamic calculations and atmospheric properties
- **[Derived Parameters](derived.md)** - Derived meteorological parameters and comfort indices  
- **[Dynamic Calculations](dynamics.md)** - Dynamic meteorology and atmospheric dynamics
- **[Statistical Analysis](statistical.md)** - Statistical and micrometeorological functions
- **[Unit Conversions](units.md)** - Meteorological unit conversion utilities
- **[Data Models](models.md)** - Structured data models for atmospheric data
- **[Coordinates](coordinates.md)** - Coordinate transformation and geographic calculations
- **[Interpolation](interpolation.md)** - Data interpolation methods for atmospheric data
- **[I/O Operations](io.md)** - Input/output operations and xarray integration

## Quick Access

### Common Tasks

#### Basic Thermodynamic Calculations
```python
import monet_meteo as mm

# Calculate potential temperature
theta = mm.potential_temperature(pressure, temperature)

# Calculate relative humidity
rh = mm.relative_humidity(temperature, dewpoint)

# Calculate mixing ratio
mr = mm.mixing_ratio(vapor_pressure, total_pressure)
```

#### Unit Conversions
```python
# Convert pressure units
pressure_pa = mm.convert_pressure(pressure_hpa, 'hPa', 'Pa')

# Convert temperature units
temp_c = mm.convert_temperature(temp_k, 'K', 'C')

# Convert wind speed
wind_knots = mm.convert_wind_speed(wind_mps, 'm/s', 'knots')
```

#### Coordinate Transformations
```python
# Calculate distance between points
distance = mm.calculate_distance(lat1, lon1, lat2, lon2)

# Convert between coordinate systems
x, y, z = mm.latlon_to_cartesian(lat, lon, elevation)

# Rotate wind components
u_rot, v_rot = mm.rotate_wind_components(u, v, lat, lon)
```

#### Data Interpolation
```python
# Vertical interpolation
interpolated_temp = mm.interpolate_vertical(
    temperature, old_pressure_levels, new_pressure_levels
)

# Horizontal interpolation
interpolated_field = mm.interpolate_horizontal(
    data, old_lons, old_lats, new_lons, new_lats
)
```

### Working with xarray

#### Loading and Processing NetCDF Data
```python
import xarray as xr
import monet_meteo as mm

# Load dataset
dataset = mm.load_netcdf_dataset('weather_data.nc')

# Convert units
dataset['pressure'] = mm.xr_convert_pressure(dataset['pressure'], 'hPa', 'Pa')

# Interpolate to pressure levels
standard_levels = xr.DataArray([1000, 850, 700, 500], dims=['pressure'])
interpolated = mm.xr_interpolate_vertical(
    dataset['temperature'], 'level', standard_levels
)
```

## Module Index

### Core Modules

#### Thermodynamics Module
[thermodynamics.md](thermodynamics.md)

Calculates fundamental thermodynamic properties including:
- Potential temperature and equivalent potential temperature
- Virtual temperature and mixing ratios
- Saturation vapor pressure and relative humidity
- Dewpoint temperature and wet bulb temperature
- Lapse rates (dry and moist)

**Key Functions:**
- [`potential_temperature()`](thermodynamics.md#potential_temperaturepressure-temperature)
- [`equivalent_potential_temperature()`](thermodynamics.md#equivalent_potential_temperaturepressure-temperature-mixing_ratio)
- [`relative_humidity()`](thermodynamics.md#relative_humiditytemperature-dewpoint)
- [`mixing_ratio()`](thermodynamics.md#mixing_ratiovapor_pressure-total_pressure)

#### Derived Parameters Module
[derived.md](derived.md)

Calculates derived meteorological parameters including:
- Heat index and wind chill
- Lifting condensation level
- Wet bulb temperature and dewpoint
- Vapor pressure calculations

**Key Functions:**
- [`heat_index()`](derived.md#heat_indextemperature-relative_humidity)
- [`wind_chill()`](derived.md#wind_chilltemperature-wind_speed)
- [`lifting_condensation_level()`](derived.md#lifting_condensation_leveltemperature-dewpoint)

#### Dynamic Calculations Module
[dynamics.md](dynamics.md)

Performs dynamic meteorology calculations including:
- Vorticity and divergence calculations
- Geostrophic and gradient wind components
- Potential vorticity calculations
- Vertical velocity conversions

**Key Functions:**
- [`absolute_vorticity()`](dynamics.md#absolute_vorticityu-v-pressure-latitude)
- [`relative_vorticity()`](dynamics.md#relative_vorticityu-v-dx-dy)
- [`geostrophic_wind()`](dynamics.md#geostrophic_windpressure-gradient_latitude)

#### Statistical Analysis Module
[statistical.md](statistical.md)

Provides statistical and micrometeorological functions including:
- Monin-Obukhov similarity theory calculations
- Bulk Richardson numbers
- Surface energy balance components
- Turbulent flux calculations

**Key Functions:**
- [`monin_obukhov_length()`](statistical.md#monin_obukhov_lengthfriction_velocity-height-sensible_heat-flux-latent_heat_flux)
- [`bulk_richardson_number()`](statistical.md#bulk_richardson_numberpotential_temperature_difference-wind_speed_difference-height_difference)
- [`surface_energy_balance()`](statistical.md#surface_energy_balanceincoming_shortwave-incoming_longwave-outgoing_longwave-sensible_heat-latent_heat-ground_heat)

### Utility Modules

#### Unit Conversions Module
[units.md](units.md)

Comprehensive unit conversion utilities for:
- Pressure (Pa, hPa, mb, mmHg, inHg, atm)
- Temperature (K, C, F)
- Distance (m, km, ft, mi, nm)
- Wind speed (m/s, knots, km/h, mph)
- Moisture (kg/kg, g/kg, ppm, ppt)
- Concentration (ppm, ppb, ppt, ug/m3)

**Key Functions:**
- [`convert_pressure()`](units.md#convert_pressurevalue-from_unit-to_unit)
- [`convert_temperature()`](units.md#convert_temperaturevalue-from_unit-to_unit)
- [`convert_wind_speed()`](units.md#convert_wind_speedvalue-from_unit-to_unit)

#### Data Models Module
[models.md](models.md)

Structured data classes for:
- `AtmosphericProfile` - Complete atmospheric profiles
- `WindProfile` - Wind profiles with height
- `ThermodynamicProfile` - Thermodynamic properties
- `DerivedParameters` - Derived meteorological parameters

**Key Classes:**
- [`AtmosphericProfile`](models.md#atmosphericprofile)
- [`WindProfile`](models.md#windprofile)
- [`ThermodynamicProfile`](models.md#thermodynamicprofile)

#### Coordinates Module
[coordinates.md](coordinates.md)

Coordinate transformation and geographic calculations:
- Geographic to Cartesian conversions
- Distance and bearing calculations
- Wind component rotation
- Vertical coordinate conversions
- Grid spacing calculations

**Key Functions:**
- [`latlon_to_cartesian()`](coordinates.md#latlon_to_cartesianlat-lon-elevation-00)
- [`calculate_distance()`](coordinates.md#calculate_distancelat1-lon1-lat2-lon2-method-haversine)
- [`rotate_wind_components()`](coordinates.md#rotate_wind_componentsu_wind-v_wind-lat-lon-rotation_method-grid)

#### Interpolation Module
[interpolation.md](interpolation.md)

Data interpolation methods for atmospheric data:
- Vertical interpolation (pressure/altitude)
- Horizontal 2D interpolation
- 3D data interpolation
- Dask-based parallel interpolation

**Key Functions:**
- [`interpolate_vertical()`](interpolation.md#interpolate_verticaldata-old_levels-new_levels-method-linear-axis-0)
- [`interpolate_temperature_pressure()`](interpolation.md#interpolate_temperature_temperature-pressure-new_pressure-method-linear)
- [`interpolate_wind_components()`](interpolation.md#interpolate_wind_componentsu_wind-v_wind-old_levels-new_levels-method-linear-axis-0)

#### I/O Operations Module
[io.md](io.md)

Input/output operations and xarray integration:
- NetCDF file handling
- Xarray DataArray operations
- Coordinate system validation
- CF-compliant dataset creation

**Key Functions:**
- [`xr_convert_pressure()`](io.md#xr_convert_pressuredataarray-from_unit-to_unit)
- [`xr_interpolate_vertical()`](io.md#xr_interpolate_verticaldataarray-old_coord-new_coord-method-linear)
- [`load_netcdf_dataset()`](io.md#load_netcdf_datasetfilepath-decode_times-true-chunks-none)

## Advanced Usage

### Working with Multiple Modules

#### Complete Atmospheric Profile Analysis
```python
import monet_meteo as mm
import numpy as np

# Create atmospheric profile using multiple modules
pressure_levels = np.array([1000, 850, 700, 500]) * 100
temperature_levels = np.array([298.15, 285.15, 273.15, 250.15])
vapor_pressure = np.array([1500, 1200, 800, 400])  # Pa

# Calculate thermodynamic properties
potential_temp = mm.potential_temperature(pressure_levels, temperature_levels)
mixing_ratio = mm.mixing_ratio(vapor_pressure, pressure_levels)
relative_humidity = mm.relative_humidity(temperature_levels, mm.dewpoint_from_vapor_pressure(vapor_pressure))

# Create structured profile
profile = mm.AtmosphericProfile(
    pressure=pressure_levels,
    temperature=temperature_levels,
    potential_temperature=potential_temp,
    mixing_ratio=mixing_ratio,
    relative_humidity=relative_humidity
)

# Calculate derived parameters
derived = mm.DerivedParameters(
    lcl_height=mm.lifting_condensation_level(temperature_levels[0], mm.dewpoint_from_vapor_pressure(vapor_pressure[0])),
    heat_index=mm.heat_index(temperature_levels[0], relative_humidity[0])
)

# Convert to different coordinate system
altitude = mm.pressure_to_altitude(pressure_levels)
```

### Climate Data Processing Pipeline
```python
def process_climate_data(input_file, output_file, target_pressure_levels):
    """
    Complete climate data processing pipeline
    """
    import xarray as xr
    
    # Step 1: Load and validate data
    dataset = mm.load_netcdf_dataset(input_file)
    mm.validate_coordinate_system(dataset)
    
    # Step 2: Convert units
    if 'pressure' in dataset:
        dataset['pressure'] = mm.xr_convert_pressure(dataset['pressure'], 'hPa', 'Pa')
    
    # Step 3: Vertical interpolation
    processed_vars = {}
    for var_name in dataset.data_vars:
        if 'pressure' in dataset[var_name].dims and var_name != 'pressure':
            if var_name in ['u_wind', 'v_wind']:
                u_interp, v_interp = mm.xr_interpolate_wind_components(
                    dataset['u_wind'], dataset['v_wind'], 
                    dataset['pressure'], target_pressure_levels, method='log'
                )
                processed_vars['u_wind'] = u_interp
                processed_vars['v_wind'] = v_interp
            else:
                interp_data = mm.xr_interpolate_vertical(
                    dataset[var_name], 'pressure', target_pressure_levels, method='log'
                )
                processed_vars[var_name] = interp_data
    
    # Step 4: Create final dataset
    final_dataset = xr.Dataset(processed_vars, coords=target_pressure_levels)
    
    # Step 5: Add metadata
    final_dataset.attrs.update({
        'title': 'Processed Climate Data',
        'processing_history': 'Vertical interpolation using monet-meteo',
        'pressure_levels': target_pressure_levels.values
    })
    
    # Step 6: Save results
    mm.save_netcdf_dataset(final_dataset, output_file)
    
    return final_dataset
```

### Ensemble Model Analysis
```python
def analyze_ensemble_models(model_files, analysis_pressure_levels):
    """
    Analyze ensemble of model outputs
    """
    import xarray as xr
    import glob
    
    # Process each model
    ensemble_data = []
    for model_file in model_files:
        # Load and process model data
        dataset = mm.load_netcdf_dataset(model_file)
        
        # Convert to pressure levels
        processed_dataset = process_climate_data(
            model_file, 'temp.nc', analysis_pressure_levels
        )
        
        ensemble_data.append(processed_dataset)
    
    # Calculate ensemble statistics
    stacked = xr.concat(ensemble_data, dim='model')
    ensemble_mean = stacked.mean(dim='model')
    ensemble_std = stacked.std(dim='model')
    
    # Calculate derived quantities
    wind_speed = np.sqrt(ensemble_mean['u_wind']**2 + ensemble_mean['v_wind']**2)
    
    return {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'wind_speed': wind_speed
    }
```

## Performance Guidelines

### Best Practices

1. **Use Vectorized Operations**: All functions support numpy arrays for efficient vectorized operations
2. **Chunk Large Datasets**: Use appropriate chunking when working with large datasets
3. **Leverage xarray**: Use xarray for multi-dimensional data with proper metadata
4. **Validate Input Data**: Use validation functions to ensure data quality
5. **Memory Management**: Monitor memory usage and process in chunks when necessary

### Common Pitfalls

1. **Unit Consistency**: Always ensure consistent units across calculations
2. **Coordinate Systems**: Validate coordinate systems before interpolation
3. **Array Shapes**: Ensure input arrays have compatible shapes for operations
4. **Missing Values**: Handle missing values appropriately in interpolation
5. **Physical Constraints**: Check for physically impossible values (negative temperatures, etc.)

## Error Handling

All functions include comprehensive error handling with meaningful error messages. Common errors include:

- **ValueError**: Invalid input values or parameters
- **TypeError**: Incorrect data types
- **FileNotFoundError**: Missing input files
- **AttributeError**: Missing required attributes or dimensions

## Contributing

The monet-meteo library welcomes contributions. Please refer to the main project documentation for contribution guidelines and development setup.

## References

- World Meteorological Organization (WMO) conventions
- CF (Climate and Forecast) conventions
- International System of Units (SI)
- American Meteorological Society (AMS) glossary

## See Also

- [Quick Start Guide](../quickstart.md) - Getting started with monet-meteo
- [User Guide](../userguide.md) - Detailed usage instructions
- [Examples](../examples.md) - Working examples and tutorials
- [Troubleshooting](../troubleshooting.md) - Common issues and solutions