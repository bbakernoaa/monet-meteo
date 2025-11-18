# User Guide

The monet-meteo library provides comprehensive tools for atmospheric science calculations. This user guide will help you understand how to effectively use the library for various meteorological applications.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Concepts](#basic-concepts)
3. [Core Modules Overview](#core-modules-overview)
4. [Common Workflows](#common-workflows)
5. [Data Processing Pipelines](#data-processing-pipelines)
6. [Best Practices](#best-practices)
7. [Performance Optimization](#performance-optimization)
8. [Integration with Other Tools](#integration-with-other-tools)

## Installation and Setup

### Prerequisites

The monet-meteo library requires the following Python packages:

```bash
pip install numpy xarray scipy pandas matplotlib
```

### Installation

```bash
pip install monet-meteo
```

### Verification

To verify the installation:

```python
import monet_meteo as mm
print(f"monet-meteo version: {mm.__version__}")
print(f"Available modules: {dir(mm)}")
```

## Basic Concepts

### Physical Constants

The library uses standardized atmospheric constants defined in [`monet_meteo.constants`](api/constants.md):

```python
from monet_meteo import constants

print(f"Gas constant for dry air: {constants.R_d} J kg⁻¹ K⁻¹")
print(f"Acceleration due to gravity: {constants.g} m s⁻²")
print(f"Von Karman constant: {constants.k}")
```

### Units and Conversions

Consistent units are crucial for atmospheric calculations. The library provides comprehensive unit conversion utilities:

```python
import monet_meteo as mm

# Pressure conversion
pressure_pa = mm.convert_pressure(1013.25, 'hPa', 'Pa')
print(f"1013.25 hPa = {pressure_pa} Pa")

# Temperature conversion
temp_c = mm.convert_temperature(298.15, 'K', 'C')
print(f"298.15 K = {temp_c}°C")

# Wind speed conversion
wind_knots = mm.convert_wind_speed(10, 'm/s', 'knots')
print(f"10 m/s = {wind_knots} knots")
```

### Data Structures

The library uses several key data structures:

#### Arrays and DataArrays
- **numpy arrays**: For basic numerical operations
- **xarray DataArrays**: For labeled, multi-dimensional data with metadata

#### Profile Models
- **AtmosphericProfile**: Complete atmospheric profiles with pressure, temperature, and derived properties
- **WindProfile**: Wind profiles with height information
- **ThermodynamicProfile**: Thermodynamic properties of the atmosphere

## Core Modules Overview

### Thermodynamics Module

The thermodynamics module provides fundamental atmospheric property calculations:

```python
import monet_meteo as mm
import numpy as np

# Sample atmospheric data
pressure = np.array([100000, 85000, 70000, 50000])  # Pa
temperature = np.array([298.15, 285.15, 273.15, 250.15])  # K
vapor_pressure = np.array([1500, 1200, 800, 400])  # Pa

# Calculate thermodynamic properties
potential_temp = mm.potential_temperature(pressure, temperature)
equivalent_pot_temp = mm.equivalent_potential_temperature(pressure, temperature, vapor_pressure)
mixing_ratio = mm.mixing_ratio(vapor_pressure, pressure)
relative_humidity = mm.relative_humidity(temperature, 280.15)  # Assuming dewpoint

print(f"Potential temperature: {potential_temp} K")
print(f"Mixing ratio: {mixing_ratio} kg/kg")
print(f"Relative humidity: {relative_humidity * 100:.1f}%")
```

### Derived Parameters Module

Calculate comfort indices and derived meteorological parameters:

```python
# Calculate heat index
heat_index = mm.heat_index(305.15, 0.6)  # Temperature in K, RH as fraction
print(f"Heat index: {heat_index:.1f}°F")

# Calculate wind chill
wind_chill = mm.wind_chill(273.15, 10)  # Temperature in K, wind speed in m/s
print(f"Wind chill: {wind_chill:.1f}°F")

# Calculate lifting condensation level
lcl_height = mm.lifting_condensation_level(298.15, 285.15)  # Temperature and dewpoint in K
print(f"LCL height: {lcl_height:.0f} m")
```

### Dynamic Calculations Module

Perform atmospheric dynamics calculations:

```python
# Sample wind data
u_wind = np.array([5, 8, 10, 12])  # m/s
v_wind = np.array([2, 3, 4, 5])    # m/s
dx = 50000  # Grid spacing in meters
dy = 50000  # Grid spacing in meters

# Calculate vorticity
vorticity = mm.relative_vorticity(u_wind, v_wind, dx, dy)
print(f"Relative vorticity: {vorticity:.2e} s⁻¹")

# Calculate divergence
divergence = mm.divergence(u_wind, v_wind, dx, dy)
print(f"Divergence: {divergence:.2e} s⁻¹")

# Calculate geostrophic wind
geostrophic_u, geostrophic_v = mm.geostrophic_wind(
    np.gradient(pressure),  # Pressure gradient
    45.0  # Latitude in degrees
)
print(f"Geostrophic wind: u={geostrophic_u:.2f}, v={geostrophic_v:.2f} m/s")
```

### Coordinates Module

Handle coordinate transformations and geographic calculations:

```python
# Geographic calculations
new_york = (40.7128, -74.0060)
london = (51.5074, -0.1278)

# Calculate distance
distance = mm.calculate_distance(*new_york, *london)
print(f"Distance: {distance/1000:.0f} km")

# Calculate bearing
bearing = mm.bearing(*new_york, *london)
print(f"Bearing: {bearing:.1f}°")

# Coordinate system transformations
x, y, z = mm.latlon_to_cartesian(40.0, -74.0, 100)  # 100m elevation
print(f"Cartesian coordinates: x={x:.0f}, y={y:.0f}, z={z:.0f} m")
```

### Interpolation Module

Interpolate atmospheric data between different coordinate systems:

```python
# Vertical interpolation
pressure_levels = np.array([100000, 85000, 70000, 50000])  # Pa
temperature = np.array([298.15, 285.15, 273.15, 250.15])  # K
new_pressure = np.array([1000, 900, 800, 700, 600, 500]) * 100  # Pa

interpolated_temp = mm.interpolate_temperature_pressure(
    temperature, pressure_levels, new_pressure, method='log'
)
print(f"Interpolated temperature: {interpolated_temp} K")

# Horizontal interpolation
old_lons = np.linspace(-180, 180, 73)
old_lats = np.linspace(-90, 90, 37)
new_lons = np.linspace(-180, 180, 145)
new_lats = np.linspace(-90, 90, 73)

# Create synthetic temperature field
X_old, Y_old = np.meshgrid(old_lons, old_lats)
temperature_field = 288.15 - 0.5 * np.abs(Y_old)

# Interpolate to higher resolution
interpolated_field = mm.interpolate_horizontal(
    temperature_field, old_lons, old_lats, new_lons, new_lats, method='cubic'
)
```

## Common Workflows

### Basic Atmospheric Profile Analysis

```python
def analyze_atmospheric_profile(pressure, temperature, vapor_pressure=None):
    """
    Analyze an atmospheric profile and calculate derived properties
    """
    import monet_meteo as mm
    import numpy as np
    
    # Convert to numpy arrays if needed
    pressure = np.asarray(pressure)
    temperature = np.asarray(temperature)
    
    # Calculate basic thermodynamic properties
    potential_temp = mm.potential_temperature(pressure, temperature)
    
    results = {
        'pressure': pressure,
        'temperature': temperature,
        'potential_temperature': potential_temp
    }
    
    # Add moisture calculations if vapor pressure is provided
    if vapor_pressure is not None:
        vapor_pressure = np.asarray(vapor_pressure)
        mixing_ratio = mm.mixing_ratio(vapor_pressure, pressure)
        relative_humidity = mm.relative_humidity(temperature, 
                                               mm.dewpoint_from_vapor_pressure(vapor_pressure))
        
        results.update({
            'vapor_pressure': vapor_pressure,
            'mixing_ratio': mixing_ratio,
            'relative_humidity': relative_humidity
        })
        
        # Calculate equivalent potential temperature
        results['equivalent_potential_temperature'] = mm.equivalent_potential_temperature(
            pressure, temperature, mixing_ratio
        )
    
    # Convert pressure to altitude for easier interpretation
    results['altitude'] = mm.pressure_to_altitude(pressure)
    
    return results

# Example usage
profile_data = analyze_atmospheric_profile(
    pressure=[100000, 85000, 70000, 50000],
    temperature=[298.15, 285.15, 273.15, 250.15],
    vapor_pressure=[1500, 1200, 800, 400]
)
```

### Weather Data Processing

```python
def process_weather_data(weather_dict):
    """
    Process raw weather data into standard format
    """
    import monet_meteo as mm
    
    processed_data = {}
    
    # Convert units
    if 'pressure' in weather_dict:
        processed_data['pressure_pa'] = mm.convert_pressure(
            weather_dict['pressure'], 'hPa', 'Pa'
        )
    
    if 'temperature' in weather_dict:
        processed_data['temperature_k'] = mm.convert_temperature(
            weather_dict['temperature'], 'C', 'K'
        )
        processed_data['temperature_c'] = weather_dict['temperature']
    
    if 'wind_speed' in weather_dict:
        processed_data['wind_speed_ms'] = mm.convert_wind_speed(
            weather_dict['wind_speed'], 'km/h', 'm/s'
        )
        processed_data['wind_speed_knots'] = mm.convert_wind_speed(
            weather_dict['wind_speed'], 'km/h', 'knots'
        )
    
    # Calculate derived quantities
    if 'temperature_k' in processed_data and 'relative_humidity' in weather_dict:
        processed_data['heat_index'] = mm.heat_index(
            processed_data['temperature_k'], weather_dict['relative_humidity']
        )
        processed_data['heat_index_c'] = mm.convert_temperature(
            processed_data['heat_index'], 'K', 'C'
        )
    
    return processed_data

# Example usage
raw_weather = {
    'pressure': 1013.25,  # hPa
    'temperature': 25.0,   # °C
    'wind_speed': 36.0,    # km/h
    'relative_humidity': 0.6
}

processed = process_weather_data(raw_weather)
```

### Climate Data Analysis

```python
def analyze_climate_trends(climate_data, time_axis='time'):
    """
    Analyze climate data trends using monet-meteo utilities
    """
    import monet_meteo as mm
    import numpy as np
    import xarray as xr
    
    # Ensure data is in xarray format for easier manipulation
    if not isinstance(climate_data, xr.Dataset):
        climate_data = xr.Dataset(climate_data)
    
    # Calculate seasonal means
    climate_data_seasonal = climate_data.groupby('season').mean()
    
    # Calculate climatological statistics
    climatological_mean = climate_data.mean(dim='time')
    climatological_std = climate_data.std(dim='time')
    
    # Analyze temperature trends
    if 'temperature' in climate_data:
        # Calculate potential temperature for stability analysis
        pressure_levels = climate_data['pressure'].values if 'pressure' in climate_data else 101325
        potential_temp = mm.potential_temperature(
            pressure_levels, climate_data['temperature']
        )
        
        # Add to results
        climate_data['potential_temperature'] = (climate_data['temperature'].dims, potential_temp)
    
    # Wind analysis
    if 'u_wind' in climate_data and 'v_wind' in climate_data:
        wind_speed = np.sqrt(climate_data['u_wind']**2 + climate_data['v_wind']**2)
        wind_direction = np.degrees(np.arctan2(-climate_data['u_wind'], -climate_data['v_wind'])) % 360
        
        climate_data['wind_speed'] = (climate_data['u_wind'].dims, wind_speed)
        climate_data['wind_direction'] = (climate_data['u_wind'].dims, wind_direction)
    
    return {
        'seasonal_means': climate_data_seasonal,
        'climatological_mean': climatological_mean,
        'climatological_std': climatological_std,
        'processed_data': climate_data
    }

# Example usage
# climate_analysis = analyze_climate_trends(climate_dataset)
```

## Data Processing Pipelines

### NetCDF Data Processing Pipeline

```python
def process_netcdf_pipeline(input_file, output_file, processing_steps):
    """
    Complete NetCDF data processing pipeline
    
    Parameters:
    - input_file: Path to input NetCDF file
    - output_file: Path to output NetCDF file
    - processing_steps: List of processing functions to apply
    """
    import monet_meteo as mm
    import xarray as xr
    
    # Step 1: Load data
    dataset = mm.load_netcdf_dataset(input_file)
    
    # Step 2: Apply processing steps
    for step in processing_steps:
        dataset = step(dataset)
    
    # Step 3: Validate and add metadata
    mm.validate_coordinate_system(dataset)
    
    # Add processing history
    if 'history' not in dataset.attrs:
        dataset.attrs['history'] = 'Processed using monet-meteo'
    else:
        dataset.attrs['history'] += '; Processed using monet-meteo'
    
    # Step 4: Save results
    mm.save_netcdf_dataset(dataset, output_file)
    
    return dataset

def unit_conversion_step(dataset):
    """Apply unit conversions to dataset"""
    # Convert pressure if needed
    if 'pressure' in dataset and dataset['pressure'].attrs.get('units') == 'hPa':
        dataset['pressure'] = mm.xr_convert_pressure(dataset['pressure'], 'hPa', 'Pa')
        dataset['pressure'].attrs['units'] = 'Pa'
    
    # Convert temperature if needed
    if 'temperature' in dataset and dataset['temperature'].attrs.get('units') == 'K':
        dataset['temperature_c'] = mm.xr_convert_temperature(
            dataset['temperature'], 'K', 'C'
        )
        dataset['temperature_c'].attrs = {
            'units': 'C',
            'long_name': 'Air Temperature',
            'standard_name': 'air_temperature'
        }
    
    return dataset

def vertical_interpolation_step(dataset, target_levels):
    """Apply vertical interpolation to dataset"""
    # Define target pressure levels
    if isinstance(target_levels, (list, np.ndarray)):
        target_pressure = xr.DataArray(target_levels, dims=['pressure'])
    else:
        target_pressure = target_levels
    
    # Interpolate all variables with pressure dimension
    for var_name in dataset.data_vars:
        if 'pressure' in dataset[var_name].dims and var_name != 'pressure':
            if var_name in ['u_wind', 'v_wind']:
                # Interpolate wind components
                u_interp, v_interp = mm.xr_interpolate_wind_components(
                    dataset['u_wind'], dataset['v_wind'],
                    dataset['pressure'], target_pressure, method='log'
                )
                
                dataset[f'{var_name}_interp'] = u_interp
                dataset[f'{var_name}_interp'] = v_interp
            else:
                # Interpolate other variables
                interp_data = mm.xr_interpolate_vertical(
                    dataset[var_name], 'pressure', target_pressure, method='log'
                )
                dataset[f'{var_name}_interp'] = interp_data
    
    return dataset

# Example usage
# processing_pipeline = [
#     lambda ds: unit_conversion_step(ds),
#     lambda ds: vertical_interpolation_step(ds, [1000, 850, 700, 500] * 100)
# ]
# processed_data = process_netcdf_pipeline('input.nc', 'output.nc', processing_pipeline)
```

### Real-time Data Processing

```python
def real_time_weather_processing(observation_data, model_background, output_file):
    """
    Process real-time weather observations with model background
    """
    import monet_meteo as mm
    import xarray as xr
    
    # Step 1: Load and prepare observation data
    obs_dataset = mm.load_netcdf_dataset(observation_data)
    
    # Step 2: Process observations to model grid
    processed_obs = {}
    
    for var_name in ['temperature', 'u_wind', 'v_wind', 'humidity']:
        if var_name in obs_dataset:
            # Interpolate observations to model grid
            if var_name in ['u_wind', 'v_wind']:
                # Wind component interpolation
                obs_var = obs_dataset[var_name]
                model_var = model_background[var_name]
                
                # Interpolate using dask for efficiency
                interp_var = mm.xr_interpolate_with_dask(
                    obs_var,
                    {'lon': obs_dataset.lon, 'lat': obs_dataset.lat},
                    {'lon': model_background.lon, 'lat': model_background.lat},
                    method='linear'
                )
                
                processed_obs[var_name] = interp_var
            else:
                # Regular variable interpolation
                obs_var = obs_dataset[var_name]
                interp_var = mm.xr_interpolate_with_dask(
                    obs_var,
                    {'lon': obs_dataset.lon, 'lat': obs_dataset.lat},
                    {'lon': model_background.lon, 'lat': model_background.lat},
                    method='linear'
                )
                processed_obs[var_name] = interp_var
    
    # Step 3: Calculate analysis increments
    analysis_inc = {}
    for var_name in processed_obs:
        analysis_inc[var_name] = processed_obs[var_name] - model_background[var_name]
    
    # Step 4: Create analysis dataset
    analysis_dataset = xr.Dataset(analysis_inc, coords=model_background.coords)
    
    # Step 5: Add metadata
    analysis_dataset.attrs = {
        'title': 'Analysis Increments',
        'source': 'Real-time processing pipeline',
        'processing_time': pd.Timestamp.now().isoformat(),
        'background_source': model_background.attrs.get('source', 'Unknown'),
        'observation_source': obs_dataset.attrs.get('source', 'Unknown')
    }
    
    # Step 6: Save analysis
    mm.save_netcdf_dataset(analysis_dataset, output_file)
    
    return analysis_dataset
```

### Ensemble Processing

```python
def process_ensemble_models(model_files, common_grid):
    """
    Process multiple model outputs to common grid for ensemble analysis
    """
    import monet_meteo as mm
    import xarray as xr
    import glob
    
    ensemble_datasets = []
    
    for model_file in model_files:
        # Load individual model
        model_dataset = mm.load_netcdf_dataset(model_file, chunks={'time': 1})
        
        # Convert to common grid
        model_regridded = {}
        
        for var_name in ['temperature', 'u_wind', 'v_wind', 'humidity']:
            if var_name in model_dataset:
                # Interpolate to common grid
                if var_name in ['u_wind', 'v_wind']:
                    # Wind components
                    u_data = model_dataset['u_wind']
                    v_data = model_dataset['v_wind']
                    
                    u_interp = mm.xr_interpolate_with_dask(
                        u_data,
                        {'lon': model_dataset.lon, 'lat': model_dataset.lat},
                        {'lon': common_grid['lon'], 'lat': common_grid['lat']},
                        method='cubic'
                    )
                    
                    v_interp = mm.xr_interpolate_with_dask(
                        v_data,
                        {'lon': model_dataset.lon, 'lat': model_dataset.lat},
                        {'lon': common_grid['lon'], 'lat': common_grid['lat']},
                        method='cubic'
                    )
                    
                    model_regridded['u_wind'] = u_interp
                    model_regridded['v_wind'] = v_interp
                else:
                    # Regular variables
                    interp_data = mm.xr_interpolate_with_dask(
                        model_dataset[var_name],
                        {'lon': model_dataset.lon, 'lat': model_dataset.lat},
                        {'lon': common_grid['lon'], 'lat': common_grid['lat']},
                        method='cubic'
                    )
                    model_regridded[var_name] = interp_data
        
        # Create regridded dataset
        regridded_dataset = xr.Dataset(model_regridded, coords=common_grid)
        regridded_dataset.attrs['source'] = model_dataset.attrs.get('source', 'Unknown')
        
        ensemble_datasets.append(regridded_dataset)
    
    # Calculate ensemble statistics
    stacked = xr.concat(ensemble_datasets, dim='model')
    ensemble_mean = stacked.mean(dim='model')
    ensemble_std = stacked.std(dim='model')
    
    return {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'members': ensemble_datasets
    }

# Example usage
# common_grid = {'lon': model_lons, 'lat': model_lats}
# ensemble_stats = process_ensemble_models(model_files, common_grid)
```

## Best Practices

### Data Quality

1. **Validate Input Data**: Always check for physically impossible values
   ```python
   # Check for negative temperatures
   if np.any(temperature < 0):
       raise ValueError("Temperature cannot be negative in Kelvin")
   
   # Check for negative pressures
   if np.any(pressure < 0):
       raise ValueError("Pressure cannot be negative")
   ```

2. **Use Appropriate Units**: Convert all data to consistent units before processing
   ```python
   # Standardize units before calculations
   pressure_pa = mm.convert_pressure(pressure_input, input_unit, 'Pa')
   temperature_k = mm.convert_temperature(temperature_input, input_unit, 'K')
   ```

3. **Handle Missing Values**: Properly handle NaN and missing data
   ```python
   # Check for and handle missing values
   if np.isnan(temperature).any():
       print("Warning: Missing temperature values detected")
       # Interpolate or fill missing values
   ```

### Performance Optimization

1. **Use Vectorized Operations**: Leverage numpy's vectorized operations
   ```python
   # Good: Vectorized operation
   potential_temps = mm.potential_temperature(pressure_array, temperature_array)
   
   # Bad: Loop operation
   potential_temps = []
   for p, t in zip(pressure_array, temperature_array):
       potential_temps.append(mm.potential_temperature(p, t))
   ```

2. **Chunk Large Datasets**: Use appropriate chunking for memory efficiency
   ```python
   # Process large datasets in chunks
   chunk_size = 1000
   for i in range(0, len(data), chunk_size):
       chunk = data[i:i+chunk_size]
       process_chunk(chunk)
   ```

3. **Leverage Dask**: Use dask for parallel processing of large datasets
   ```python
   import dask.array as da
   
   # Create dask array for parallel processing
   dask_array = da.from_array(large_array, chunks=(1000, 1000))
   result = da.map_blocks(process_function, dask_array)
   ```

### Code Organization

1. **Modular Design**: Break down complex processing into reusable functions
   ```python
   def load_and_validate_data(filepath):
       """Load and validate meteorological data"""
       data = mm.load_netcdf_dataset(filepath)
       mm.validate_coordinate_system(data)
       return data
   
   def calculate_derived_variables(data):
       """Calculate derived meteorological variables"""
       # Implementation
       pass
   ```

2. **Error Handling**: Implement comprehensive error handling
   ```python
   def safe_calculation(func, *args, **kwargs):
       """Safely execute calculations with error handling"""
       try:
           return func(*args, **kwargs)
       except ValueError as e:
           print(f"Calculation error: {e}")
           return None
       except Exception as e:
           print(f"Unexpected error: {e}")
           return None
   ```

3. **Documentation**: Document functions and data structures
   ```python
   def process_atmospheric_profile(pressure, temperature, moisture_data=None):
       """
       Process atmospheric profile and calculate derived properties.
       
       Parameters:
       - pressure: Pressure levels in Pa
       - temperature: Temperature profile in K
       - moisture_data: Optional moisture information
       
       Returns:
       - Dictionary with processed atmospheric data
       """
       # Implementation
       pass
   ```

## Performance Optimization

### Memory Management

1. **Monitor Memory Usage**: Track memory consumption during processing
   ```python
   import psutil
   import os
   
   def get_memory_usage():
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / 1024**2  # MB
   
   print(f"Memory usage: {get_memory_usage():.1f} MB")
   ```

2. **Process in Chunks**: For very large datasets, process in manageable chunks
   ```python
   def process_large_dataset(dataset, chunk_size=100):
       results = []
       for i in range(0, len(dataset), chunk_size):
           chunk = dataset.isel(time=slice(i, i + chunk_size))
           result = process_chunk(chunk)
           results.append(result)
       return xr.concat(results, dim='time')
   ```

3. **Use Efficient Data Types**: Choose appropriate data types for memory efficiency
   ```python
   # Use float32 instead of float64 when precision allows
   temperature = temperature.astype('float32')
   pressure = pressure.astype('float32')
   ```

### Parallel Processing

1. **Dask Integration**: Leverage dask for parallel processing
   ```python
   import monet_meteo as mm
   import dask.array as da
   
   # Create dask array
   dask_array = da.from_array(large_data, chunks=(1000, 1000))
   
   # Parallel interpolation
   result = mm.xr_interpolate_with_dask(
       dask_array,
       {'x': old_coords, 'y': old_coords},
       {'x': new_coords, 'y': new_coords},
       method='linear'
   )
   ```

2. **Multiprocessing**: Use multiprocessing for CPU-bound tasks
   ```python
   from multiprocessing import Pool
   
   def parallel_processing(data_list):
       with Pool(processes=4) as pool:
           results = pool.map(process_single_profile, data_list)
       return results
   ```

### Algorithm Optimization

1. **Use Optimized Algorithms**: Choose the most efficient algorithms for your use case
   ```python
   # For large datasets, use optimized interpolation
   # Instead of scipy.interp1d, use monet_meteo's optimized functions
   interpolated = mm.interpolate_vertical(data, old_levels, new_levels)
   ```

2. **Cache Results**: Cache intermediate results when possible
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=128)
   def calculate_potential_temperature(pressure, temperature):
       return mm.potential_temperature(pressure, temperature)
   ```

## Integration with Other Tools

### Matplotlib Integration

```python
import matplotlib.pyplot as plt
import monet_meteo as mm

def plot_atmospheric_profile(pressure, temperature, title="Atmospheric Profile"):
    """Plot atmospheric temperature profile"""
    # Convert pressure to altitude for plotting
    altitude = mm.pressure_to_altitude(pressure)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot temperature profile
    ax.plot(temperature - 273.15, altitude / 1000, 'b-', linewidth=2, label='Temperature')
    
    # Plot potential temperature
    potential_temp = mm.potential_temperature(pressure, temperature)
    ax.plot(potential_temp - 273.15, altitude / 1000, 'r--', linewidth=2, label='Potential Temperature')
    
    ax.invert_yaxis()
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Altitude (km)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig, ax

# Example usage
# fig, ax = plot_atmospheric_profile(pressure_levels, temperature_levels)
```

### Pandas Integration

```python
import pandas as pd
import monet_meteo as mm

def pandas_weather_analysis(weather_dataframe):
    """Analyze weather data using pandas and monet-meteo"""
    # Convert units in pandas DataFrame
    weather_dataframe['pressure_pa'] = weather_dataframe['pressure_hpa'].apply(
        lambda x: mm.convert_pressure(x, 'hPa', 'Pa')
    )
    
    weather_dataframe['temperature_k'] = weather_dataframe['temperature_c'].apply(
        lambda x: mm.convert_temperature(x, 'C', 'K')
    )
    
    # Calculate derived parameters
    weather_dataframe['heat_index'] = weather_dataframe.apply(
        lambda row: mm.heat_index(
            mm.convert_temperature(row['temperature_c'], 'C', 'K'),
            row['relative_humidity']
        ),
        axis=1
    )
    
    # Calculate wind speed from components
    if 'u_wind_ms' in weather_dataframe.columns and 'v_wind_ms' in weather_dataframe.columns:
        weather_dataframe['wind_speed_ms'] = np.sqrt(
            weather_dataframe['u_wind_ms']**2 + weather_dataframe['v_wind_ms']**2
        )
    
    return weather_dataframe

# Example usage
# weather_analysis = pandas_weather_analysis(weather_data)
```

### Cartopy Integration

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import monet_meteo as mm

def plot_weather_map(data, projection=ccrs.PlateCarree()):
    """Plot weather data on map using cartopy"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.OCEAN, color='lightblue')
    ax.add_feature(cfeature.LAND, color='lightgray')
    
    # Plot data
    if 'temperature' in data:
        temp_plot = ax.contourf(
            data['lon'], data['lat'], data['temperature'],
            levels=20, cmap='RdYlBu_r', transform=ccrs.PlateCarree()
        )
        plt.colorbar(temp_plot, ax=ax, label='Temperature (K)')
    
    # Add wind vectors
    if 'u_wind' in data and 'v_wind' in data:
        skip = 5  # Skip some vectors for clarity
        ax.quiver(
            data['lon'][::skip], data['lat'][::skip],
            data['u_wind'][::skip, ::skip], data['v_wind'][::skip, ::skip],
            transform=ccrs.PlateCarree(), scale=200, alpha=0.7
        )
    
    ax.set_title('Weather Map')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    return fig, ax

# Example usage
# fig, ax = plot_weather_map(weather_data)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Process data in smaller chunks
   ```python
   # For memory errors
   import psutil
   if psutil.virtual_memory().percent > 80:
       raise MemoryError("Insufficient memory available")
   ```

2. **Unit Conversion Errors**: Validate units before conversion
   ```python
   # Validate units
   valid_pressure_units = ['Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm']
   if from_unit not in valid_pressure_units:
       raise ValueError(f"Invalid pressure unit: {from_unit}")
   ```

3. **Coordinate System Errors**: Validate coordinate systems
   ```python
   # Validate coordinates
   try:
       mm.validate_coordinate_system(data)
   except ValueError as e:
       print(f"Coordinate system error: {e}")
   ```

### Debugging Tips

1. **Enable Verbose Logging**: Add logging for debugging
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Add debug prints in functions
   logger = logging.getLogger(__name__)
   logger.debug(f"Processing data with shape: {data.shape}")
   ```

2. **Check Data Types**: Verify data types match expectations
   ```python
   # Check data types
   if not isinstance(pressure, (np.ndarray, xr.DataArray)):
       raise TypeError("Pressure must be numpy array or xarray DataArray")
   ```

3. **Test with Sample Data**: Validate functions with known test data
   ```python
   # Test with known values
   test_pressure = 101325  # Pa
   test_temperature = 288.15  # K
   expected_potential_temp = 288.15  # K at 1000 hPa
   
   calculated_potential_temp = mm.potential_temperature(test_pressure, test_temperature)
   assert abs(calculated_potential_temp - expected_potential_temp) < 0.01
   ```

### Performance Monitoring

1. **Timing Functions**: Measure execution time
   ```python
   import time
   
   start_time = time.time()
   result = complex_calculation(data)
   end_time = time.time()
   
   print(f"Calculation completed in {end_time - start_time:.2f} seconds")
   ```

2. **Memory Profiling**: Profile memory usage
   ```python
   import tracemalloc
   
   tracemalloc.start()
   result = memory_intensive_operation(data)
   current, peak = tracemalloc.get_traced_memory()
   tracemalloc.stop()
   
   print(f"Memory usage: {current / 1024**2:.1f} MB, Peak: {peak / 1024**2:.1f} MB")
   ```

## Next Steps

After reviewing this user guide, you may want to explore:

1. **API Reference**: Detailed documentation of all functions and modules
2. **Examples**: Working examples and tutorials
3. **Advanced Topics**: Specialized applications and advanced techniques
4. **Contributing**: Guidelines for contributing to the project

For specific questions or issues, please refer to the [troubleshooting guide](troubleshooting.md) or submit an issue to the project repository.