# Interpolation Module

The interpolation module provides comprehensive utilities for interpolating atmospheric data between different coordinate systems, grid points, and vertical levels. These functions are essential for meteorological data processing, model output analysis, and atmospheric profile interpolation.

## Functions

## Vertical Interpolation

### `pressure_to_altitude(pressure, temperature=None, method='linear')`
Convert pressure levels to altitude using barometric formula or interpolation.

```python
mm.pressure_to_altitude(pressure, temperature=None, method='linear')
```

**Parameters:**
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Pressure levels (Pa)
- `temperature` (float, numpy.ndarray, or xarray.DataArray, optional): Temperature profile (K), default None for barometric formula
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Altitude (m)

**Example:**
```python
import monet_meteo as mm
import numpy as np

# Using barometric formula
altitude = mm.pressure_to_altitude(85000)  # Pa
print(f"Altitude: {altitude:.0f} m")

# Using interpolation with temperature profile
pressure_levels = np.array([100000, 85000, 70000, 50000])
temperatures = np.array([298.15, 285.15, 273.15, 250.15])
altitude_profile = mm.pressure_to_altitude(pressure_levels, temperatures)
```

### `altitude_to_pressure(altitude, reference_pressure=101325, reference_temperature=288.15, method='linear')`
Convert altitude to pressure using barometric formula.

```python
mm.altitude_to_pressure(altitude, reference_pressure=101325, reference_temperature=288.15, method='linear')
```

**Parameters:**
- `altitude` (float, numpy.ndarray, or xarray.DataArray): Altitude (m)
- `reference_pressure` (float, optional): Reference pressure at sea level (Pa), default 101325
- `reference_temperature` (float, optional): Reference temperature at sea level (K), default 288.15

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Pressure (Pa)

**Example:**
```python
# Convert altitude to pressure
pressure = mm.altitude_to_pressure(1000)
print(f"Pressure: {pressure:.0f} Pa")
```

### `interpolate_vertical(data, old_levels, new_levels, method='linear', axis=0)`
Interpolate data along vertical coordinate.

```python
mm.interpolate_vertical(data, old_levels, new_levels, method='linear', axis=0)
```

**Parameters:**
- `data` (float, numpy.ndarray, or xarray.DataArray): Data to interpolate
- `old_levels` (float, numpy.ndarray): Original vertical levels (e.g., pressure in Pa)
- `new_levels` (float, numpy.ndarray): Target vertical levels
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'
- `axis` (int, optional): Axis along which to interpolate, default 0

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Interpolated data

**Example:**
```python
# Interpolate temperature from pressure to sigma coordinates
pressure_levels = np.array([100000, 85000, 70000, 50000, 30000])
sigma_levels = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
temperature = np.array([298.15, 285.15, 273.15, 250.15, 230.15])

interpolated_temp = mm.interpolate_vertical(
    temperature, pressure_levels, sigma_levels, method='log'
)
```

### `interpolate_temperature_pressure(temperature, pressure, new_pressure, method='linear')`
Interpolate temperature to new pressure levels.

```python
mm.interpolate_temperature_pressure(temperature, pressure, new_pressure, method='linear')
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Temperature profile (K)
- `pressure` (float, numpy.ndarray): Pressure profile (Pa)
- `new_pressure` (float, numpy.ndarray): New pressure levels (Pa)
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Interpolated temperature (K)

**Example:**
```python
# Interpolate temperature to standard pressure levels
original_pressure = np.array([100000, 85000, 70000, 50000])
original_temp = np.array([298.15, 285.15, 273.15, 250.15])
standard_levels = np.array([1000, 850, 700, 500, 300, 200]) * 100  # hPa to Pa

interpolated_temp = mm.interpolate_temperature_pressure(
    original_temp, original_pressure, standard_levels
)
```

## Wind Interpolation

### `interpolate_wind_components(u_wind, v_wind, old_levels, new_levels, method='linear', axis=0)`
Interpolate wind components along vertical coordinate.

```python
mm.interpolate_wind_components(u_wind, v_wind, old_levels, new_levels, method='linear', axis=0)
```

**Parameters:**
- `u_wind` (float, numpy.ndarray, or xarray.DataArray): Eastward wind component (m/s)
- `v_wind` (float, numpy.ndarray, or xarray.DataArray): Northward wind component (m/s)
- `old_levels` (float, numpy.ndarray): Original vertical levels
- `new_levels` (float, numpy.ndarray): Target vertical levels
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'
- `axis` (int, optional): Axis along which to interpolate, default 0

**Returns:**
- tuple: (u_interpolated, v_interpolated) interpolated wind components

**Example:**
```python
# Interpolate wind components to new levels
u_original = np.array([5.0, 8.0, 10.0, 12.0, 15.0])
v_original = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
pressure_levels = np.array([1000, 850, 700, 500, 300]) * 100
new_levels = np.array([1000, 800, 600, 400, 200]) * 100

u_interp, v_interp = mm.interpolate_wind_components(
    u_original, v_original, pressure_levels, new_levels, method='log'
)
```

### `interpolate_horizontal(data, x_old, y_old, x_new, y_new, method='linear')`
Interpolate 2D data to new horizontal grid.

```python
mm.interpolate_horizontal(data, x_old, y_old, x_new, y_new, method='linear')
```

**Parameters:**
- `data` (float, numpy.ndarray): 2D data array
- `x_old` (float, numpy.ndarray): Original x coordinates
- `y_old` (float, numpy.ndarray): Original y coordinates
- `x_new` (float, numpy.ndarray): New x coordinates
- `y_new` (float, numpy.ndarray): New y coordinates
- `method` (str, optional): Interpolation method ('linear', 'nearest', 'cubic'), default 'linear'

**Returns:**
- numpy.ndarray: Interpolated 2D data

**Example:**
```python
# Interpolate temperature field to new grid
x_old = np.linspace(-180, 180, 361)
y_old = np.linspace(-90, 90, 181)
x_new = np.linspace(-180, 180, 721)  # Higher resolution
y_new = np.linspace(-90, 90, 361)

# Create synthetic temperature field
X_old, Y_old = np.meshgrid(x_old, y_old)
temperature_field = 288.15 - 0.5 * np.abs(Y_old) - 0.01 * np.sin(X_old * np.pi / 180)

# Interpolate to new grid
temperature_new = mm.interpolate_horizontal(
    temperature_field, x_old, y_old, x_new, y_new, method='cubic'
)
```

## 3D Interpolation

### `interpolate_3d(data, x_old, y_old, z_old, x_new, y_new, z_new, method='linear')`
Interpolate 3D data to new grid.

```python
mm.interpolate_3d(data, x_old, y_old, z_old, x_new, y_new, z_new, method='linear')
```

**Parameters:**
- `data` (float, numpy.ndarray): 3D data array
- `x_old` (float, numpy.ndarray): Original x coordinates
- `y_old` (float, numpy.ndarray): Original y coordinates
- `z_old` (float, numpy.ndarray): Original z coordinates
- `x_new` (float, numpy.ndarray): New x coordinates
- `y_new` (float, numpy.ndarray): New y coordinates
- `z_new` (float, numpy.ndarray): New z coordinates
- `method` (str, optional): Interpolation method ('linear', 'nearest'), default 'linear'

**Returns:**
- numpy.ndarray: Interpolated 3D data

**Example:**
```python
# Interpolate 3D atmospheric data
x_old = np.linspace(-180, 180, 72)
y_old = np.linspace(-90, 90, 36)
z_old = np.array([1000, 850, 700, 500, 300, 200]) * 100  # Pressure levels

# Create synthetic 3D temperature field
X_old, Y_old, Z_old = np.meshgrid(x_old, y_old, z_old, indexing='ij')
temperature_3d = 288.15 - 0.5 * np.abs(Y_old) - 0.01 * Z_old / 1000

# New grid
x_new = np.linspace(-180, 180, 144)
y_new = np.linspace(-90, 90, 72)
z_new = np.array([1000, 900, 800, 700, 600, 500, 400, 300, 200]) * 100

# Interpolate
temperature_3d_new = mm.interpolate_3d(
    temperature_3d, x_old, y_old, z_old, x_new, y_new, z_new
)
```

## Advanced Interpolation

### `interpolate_with_dask(data, old_coords, new_coords, method='linear', chunks=None)`
Interpolate data using dask for parallel processing.

```python
mm.interpolate_with_dask(data, old_coords, new_coords, method='linear', chunks=None)
```

**Parameters:**
- `data` (xarray.DataArray): Data with dask arrays
- `old_coords` (dict): Dictionary of original coordinates
- `new_coords` (dict): Dictionary of new coordinates
- `method` (str, optional): Interpolation method, default 'linear'
- `chunks` (tuple, optional): Chunk size for dask operations

**Returns:**
- xarray.DataArray: Interpolated data

**Example:**
```python
import xarray as xr
import monet_meteo as mm

# Create dask-backed dataset
lat = np.linspace(-90, 90, 181)
lon = np.linspace(-180, 180, 361)
pressure = np.array([1000, 850, 700, 500, 300]) * 100

# Create dask array
temp_data = xr.DataArray(
    dask.array.random.random((5, 181, 361), chunks=(1, 90, 180)),
    dims=['pressure', 'lat', 'lon'],
    coords={'pressure': pressure, 'lat': lat, 'lon': lon}
)

# New coordinates
new_pressure = np.array([1000, 900, 800, 700, 600, 500, 400, 300]) * 100
new_lat = np.linspace(-90, 90, 361)
new_lon = np.linspace(-180, 180, 721)

# Interpolate
interpolated = mm.interpolate_with_dask(
    temp_data, 
    {'pressure': pressure, 'lat': lat, 'lon': lon},
    {'pressure': new_pressure, 'lat': new_lat, 'lon': new_lon},
    method='linear'
)
```

### `pressure_level_interpolation(data, pressure, target_pressure, method='linear')`
Specialized function for interpolating to pressure levels.

```python
mm.pressure_level_interpolation(data, pressure, target_pressure, method='linear')
```

**Parameters:**
- `data` (float, numpy.ndarray, or xarray.DataArray): Data to interpolate
- `pressure` (float, numpy.ndarray): Current pressure levels (Pa)
- `target_pressure` (float, numpy.ndarray): Target pressure levels (Pa)
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Interpolated data

**Example:**
```python
# Interpolate model output to pressure levels
model_pressure = np.array([1000, 950, 900, 850, 800, 750, 700, 650, 600, 550, 500]) * 100
model_data = np.random.rand(len(model_pressure), 181, 361)  # Sample data

standard_levels = np.array([1000, 850, 700, 500, 300, 200]) * 100

interpolated_data = mm.pressure_level_interpolation(
    model_data, model_pressure, standard_levels, method='log'
)
```

## Usage Patterns

### Basic Vertical Interpolation
```python
import monet_meteo as mm
import numpy as np

def interpolate_to_standard_levels(pressure_data, temperature_data, u_wind_data, v_wind_data):
    """
    Interpolate all variables to standard pressure levels
    """
    standard_levels = np.array([1000, 850, 700, 500, 300, 200, 100]) * 100
    
    # Interpolate temperature
    temp_interp = mm.interpolate_temperature_pressure(
        temperature_data, pressure_data, standard_levels
    )
    
    # Interpolate wind components
    u_interp, v_interp = mm.interpolate_wind_components(
        u_wind_data, v_wind_data, pressure_data, standard_levels, method='log'
    )
    
    # Calculate wind speed and direction
    wind_speed = np.sqrt(u_interp**2 + v_interp**2)
    wind_direction = np.degrees(np.arctan2(-u_interp, -v_interp)) % 360
    
    return {
        'pressure_levels': standard_levels,
        'temperature': temp_interp,
        'u_wind': u_interp,
        'v_wind': v_interp,
        'wind_speed': wind_speed,
        'wind_direction': wind_direction
    }

# Example usage
# result = interpolate_to_standard_levels(model_pressure, model_temp, model_u, model_v)
```

### Horizontal Grid Interpolation
```python
def regrid_data_to_higher_resolution(old_data, old_lons, old_lats, factor=2):
    """
    Regrid data to higher resolution
    """
    # Create higher resolution grid
    new_lons = np.linspace(old_lons.min(), old_lons.max(), len(old_lons) * factor)
    new_lats = np.linspace(old_lats.min(), old_lats.max(), len(old_lats) * factor)
    
    # Interpolate each time step
    regridded_data = []
    for time_step in old_data:
        regridded = mm.interpolate_horizontal(
            time_step, old_lons, old_lats, new_lons, new_lats, method='cubic'
        )
        regridded_data.append(regridded)
    
    return np.array(regridded_data), new_lons, new_lats

# Example usage
# regridded, new_lons, new_lats = regrid_data_to_higher_resolution(temperature_data, lons, lats)
```

### 3D Atmospheric Data Interpolation
```python
def interpolate_3d_atmospheric_data(data_3d, old_grid, new_grid):
    """
    Interpolate 3D atmospheric data to new grid
    """
    # Extract coordinates
    old_x, old_y, old_z = old_grid['x'], old_grid['y'], old_grid['z']
    new_x, new_y, new_z = new_grid['x'], new_grid['y'], new_grid['z']
    
    # Interpolate each variable
    interpolated_data = {}
    
    for var_name, var_data in data_3d.items():
        if var_data.ndim == 3:
            interp_var = mm.interpolate_3d(
                var_data, old_x, old_y, old_z, new_x, new_y, new_z, method='linear'
            )
            interpolated_data[var_name] = interp_var
        elif var_data.ndim == 2:
            # Handle 2D variables (surface fields)
            interp_var = mm.interpolate_horizontal(
                var_data, old_x, old_y, new_x, new_y, method='cubic'
            )
            interpolated_data[var_name] = interp_var
    
    return interpolated_data

# Example usage
# new_data = interpolate_3d_atmospheric_data(model_data, old_grid_coords, new_grid_coords)
```

### Dask-Based Processing for Large Datasets
```python
def process_large_dataset_with_dask(dask_data, chunk_size=(1, 90, 180)):
    """
    Process large dataset using dask for parallel interpolation
    """
    # Set up dask chunks
    dask_data = dask_data.chunk(chunk_size)
    
    # Define new coordinates
    new_pressure = np.array([1000, 850, 700, 500, 300, 200]) * 100
    new_lat = np.linspace(-90, 90, 361)
    new_lon = np.linspace(-180, 180, 721)
    
    # Interpolate each variable
    results = {}
    
    for var_name, var_array in dask_data.data_vars.items():
        if 'pressure' in var_array.dims:
            # Vertical interpolation needed
            coords = {
                'pressure': var_array.pressure.values,
                'lat': var_array.lat.values,
                'lon': var_array.lon.values
            }
            new_coords = {
                'pressure': new_pressure,
                'lat': new_lat,
                'lon': new_lon
            }
            
            interpolated = mm.interpolate_with_dask(
                var_array, coords, new_coords, method='linear'
            )
            results[var_name] = interpolated
    
    return results

# Example usage
# dask_results = process_large_dataset_with_dask(dask_dataset)
```

## Advanced Applications

### Model Output Post-Processing
```python
def post_process_model_output(model_data, target_grid):
    """
    Post-process model output for analysis
    """
    # Extract model coordinates
    model_pressure = model_data['pressure'].values
    model_lats = model_data['latitude'].values
    model_lons = model_data['longitude'].values
    
    # Interpolate to standard pressure levels
    standard_levels = np.array([1000, 850, 700, 500, 300, 200, 100]) * 100
    
    post_processed = {}
    
    # Process each variable
    for var_name in ['temperature', 'u_wind', 'v_wind', 'relative_humidity']:
        if var_name in model_data:
            # Vertical interpolation
            if var_name in ['u_wind', 'v_wind']:
                u_data = model_data['u_wind'].values
                v_data = model_data['v_wind'].values
                
                u_interp, v_interp = mm.interpolate_wind_components(
                    u_data, v_data, model_pressure, standard_levels, method='log'
                )
                
                post_processed[f'{var_name}_interpolated'] = {
                    'u': u_interp,
                    'v': v_interp,
                    'pressure_levels': standard_levels,
                    'lats': model_lats,
                    'lons': model_lons
                }
            else:
                data = model_data[var_name].values
                interp_data = mm.pressure_level_interpolation(
                    data, model_pressure, standard_levels, method='log'
                )
                
                post_processed[var_name] = {
                    'data': interp_data,
                    'pressure_levels': standard_levels,
                    'lats': model_lats,
                    'lons': model_lons
                }
    
    # Calculate derived quantities
    if 'u_wind_interpolated' in post_processed and 'v_wind_interpolated' in post_processed:
        u = post_processed['u_wind_interpolated']['u']
        v = post_processed['v_wind_interpolated']['v']
        
        post_processed['wind_speed'] = np.sqrt(u**2 + v**2)
        post_processed['wind_direction'] = np.degrees(np.arctan2(-u, -v)) % 360
    
    return post_processed

# Example usage
# processed = post_process_model_output(model_output, analysis_grid)
```

### Multi-Model Ensemble Interpolation
```python
def interpolate_ensemble_models(ensemble_data, common_grid):
    """
    Interpolate multiple model outputs to common grid for ensemble analysis
    """
    interpolated_ensemble = []
    
    for model_name, model_data in ensemble_data.items():
        # Interpolate this model to common grid
        model_interp = {}
        
        for var_name, var_data in model_data.items():
            # Handle different coordinate systems
            if var_name in ['u_wind', 'v_wind']:
                # Wind component interpolation
                u_interp, v_interp = mm.interpolate_wind_components(
                    var_data['u'], var_data['v'], 
                    var_data['pressure'], common_grid['pressure'],
                    method='log'
                )
                model_interp[var_name] = {
                    'u': u_interp,
                    'v': v_interp
                }
            else:
                # Regular variable interpolation
                interp_data = mm.pressure_level_interpolation(
                    var_data['data'], var_data['pressure'], 
                    common_grid['pressure'], method='log'
                )
                model_interp[var_name] = {'data': interp_data}
        
        interpolated_ensemble.append({
            'model_name': model_name,
            'data': model_interp
        })
    
    return interpolated_ensemble

def calculate_ensemble_statistics(interpolated_ensemble):
    """Calculate ensemble mean and spread"""
    ensemble_mean = {}
    ensemble_spread = {}
    
    # Collect all model data
    all_models = {model['model_name']: model['data'] for model in interpolated_ensemble}
    
    # Calculate statistics for each variable
    for var_name in all_models[list(all_models.keys())[0]].keys():
        ensemble_mean[var_name] = []
        ensemble_spread[var_name] = []
        
        for level_idx in range(common_grid['pressure'].shape[0]):
            level_data = []
            for model_data in all_models.values():
                if var_name in ['u_wind', 'v_wind']:
                    level_data.append(model_data[var_name]['u'][level_idx])
                else:
                    level_data.append(model_data[var_name]['data'][level_idx])
            
            level_data = np.array(level_data)
            ensemble_mean[var_name].append(np.mean(level_data))
            ensemble_spread[var_name].append(np.std(level_data))
        
        ensemble_mean[var_name] = np.array(ensemble_mean[var_name])
        ensemble_spread[var_name] = np.array(ensemble_spread[var_name])
    
    return ensemble_mean, ensemble_spread

# Example usage
# interp_ensemble = interpolate_ensemble_models(ensemble_data, common_pressure_levels)
# ensemble_mean, ensemble_spread = calculate_ensemble_statistics(interp_ensemble)
```

### Real-Time Data Interpolation
```python
def real_time_interpolation(observation_data, model_background, analysis_time):
    """
    Perform real-time data interpolation for data assimilation
    """
    # Extract observation locations and values
    obs_lats = observation_data['latitude']
    obs_lons = observation_data['longitude']
    obs_values = observation_data['values']
    obs_types = observation_data['types']  # 'temperature', 'wind_u', 'wind_v', etc.
    
    # Extract model background at observation locations
    model_background_interp = {}
    
    for obs_type in set(obs_types):
        if obs_type == 'temperature':
            # Interpolate model temperature to observation locations
            model_values = mm.interpolate_horizontal(
                model_background['temperature'],
                model_background['longitude'], model_background['latitude'],
                obs_lons, obs_lats, method='linear'
            )
        elif obs_type in ['wind_u', 'wind_v']:
            # Interpolate wind components
            model_u = mm.interpolate_horizontal(
                model_background['u_wind'],
                model_background['longitude'], model_background['latitude'],
                obs_lons, obs_lats, method='linear'
            )
            model_v = mm.interpolate_horizontal(
                model_background['v_wind'],
                model_background['longitude'], model_background['latitude'],
                obs_lons, obs_lats, method='linear'
            )
            
            if obs_type == 'wind_u':
                model_values = model_u
            else:
                model_values = model_v
        
        model_background_interp[obs_type] = model_values
    
    # Calculate innovation (observation - background)
    innovations = {}
    for obs_type, obs_vals in zip(obs_types, obs_values):
        background_vals = model_background_interp[obs_type]
        innovations[obs_type] = obs_vals - background_vals
    
    return {
        'observation_data': observation_data,
        'model_background': model_background_interp,
        'innovations': innovations,
        'analysis_time': analysis_time
    }

# Example usage
# analysis = real_time_interpolation(observations, model_background, current_time)
```

## Error Handling

The interpolation module includes comprehensive error handling:

### Common Errors
```python
# Error: Invalid interpolation method
try:
    mm.interpolate_vertical(data, old_levels, new_levels, method='invalid')
except ValueError as e:
    print(f"Error: {e}")

# Error: Inconsistent array dimensions
try:
    data = np.random.rand(10, 10)
    old_levels = np.array([1, 2, 3])
    new_levels = np.array([1, 2])
    mm.interpolate_vertical(data, old_levels, new_levels)
except ValueError as e:
    print(f"Error: {e}")

# Error: Negative pressure values
try:
    mm.pressure_to_altitude(-1000)
except ValueError as e:
    print(f"Error: {e}")

# Error: Insufficient data points
try:
    data = np.array([1, 2])
    old_levels = np.array([1, 2])
    new_levels = np.array([1, 2, 3, 4, 5])
    mm.interpolate_vertical(data, old_levels, new_levels)
 except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Vectorization
All functions support numpy arrays for efficient vectorized operations:

```python
import numpy as np

# Vectorized interpolation for multiple profiles
pressure_levels = np.array([100000, 85000, 70000, 50000])
temperatures = np.random.rand(10, len(pressure_levels))  # 10 profiles
new_levels = np.array([1000, 850, 700, 500, 300]) * 100

interpolated_temps = mm.interpolate_temperature_pressure(
    temperatures, pressure_levels, new_levels
)
print(f"Interpolated shape: {interpolated_temps.shape}")
```

### Memory Management
For large datasets, use chunked processing:

```python
def process_large_interpolation_dataset(data_chunk, interpolation_config):
    """Process a chunk of data with interpolation"""
    result = {}
    
    for var_name, var_data in data_chunk.items():
        if var_data.ndim == 3:
            # 3D interpolation
            interp_data = mm.interpolate_3d(
                var_data,
                interpolation_config['old_x'],
                interpolation_config['old_y'],
                interpolation_config['old_z'],
                interpolation_config['new_x'],
                interpolation_config['new_y'],
                interpolation_config['new_z'],
                method='linear'
            )
        else:
            # 2D interpolation
            interp_data = mm.interpolate_horizontal(
                var_data,
                interpolation_config['old_x'],
                interpolation_config['old_y'],
                interpolation_config['new_x'],
                interpolation_config['new_y'],
                method='cubic'
            )
        
        result[var_name] = interp_data
    
    return result
```

### Dask Integration
For very large datasets, leverage dask for parallel processing:

```python
import dask.array as da

def create_dask_interpolation_task(data, old_coords, new_coords, method='linear'):
    """Create dask task for interpolation"""
    return da.map_blocks(
        mm.interpolate_with_dask,
        data, old_coords, new_coords, method,
        dtype=data.dtype,
        chunks=(data.chunksize[0], len(new_coords['y']), len(new_coords['x']))
    )

# Example usage with dask arrays
# dask_task = create_dask_interpolation_task(dask_data, old_coords, new_coords)
```

## References

- Press, W.H., Teukolsky, S.A., Vetterling, W.T., & Flannery, B.P. (2007). Numerical Recipes: The Art of Scientific Computing (3rd ed.).
- National Centers for Environmental Prediction (NCEP). (2003). NCEP Grid Documentation.
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/

## See Also

- [Thermodynamics Module](thermodynamics.md) - Thermodynamic variable calculations
- [Dynamic Calculations](dynamics.md) - Dynamic meteorology functions
- [Statistical Analysis](statistical.md) - Statistical and micrometeorological functions
- [Unit Conversions](units.md) - Meteorological unit conversion utilities
- [Data Models](models.md) - Structured data models for atmospheric data
- [Coordinates](coordinates.md) - Coordinate transformation utilities