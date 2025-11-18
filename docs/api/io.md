# I/O Operations Module

The I/O operations module provides utilities for reading, writing, and processing meteorological data, with special emphasis on xarray integration for netCDF and other gridded data formats. This module is designed to facilitate data loading, processing, and output operations for atmospheric science applications.

## Xarray Integration Functions

### `xr_convert_pressure(dataarray, from_unit, to_unit)`
Convert pressure units in an xarray DataArray.

```python
mm.xr_convert_pressure(dataarray, from_unit, to_unit)
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray containing pressure values
- `from_unit` (str): Source unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')
- `to_unit` (str): Target unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')

**Returns:**
- xarray.DataArray: DataArray with converted pressure values

**Example:**
```python
import monet_meteo as mm
import xarray as xr

# Load pressure data
pressure_data = xr.open_dataset('pressure_data.nc')['pressure']

# Convert from hPa to Pa
pressure_pa = mm.xr_convert_pressure(pressure_data, 'hPa', 'Pa')
```

### `xr_convert_temperature(dataarray, from_unit, to_unit)`
Convert temperature units in an xarray DataArray.

```python
mm.xr_convert_temperature(dataarray, from_unit, to_unit)
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray containing temperature values
- `from_unit` (str): Source unit ('K', 'C', 'F')
- `to_unit` (str): Target unit ('K', 'C', 'F')

**Returns:**
- xarray.DataArray: DataArray with converted temperature values

**Example:**
```python
# Load temperature data
temperature_data = xr.open_dataset('temperature_data.nc')['temperature']

# Convert from Kelvin to Celsius
temperature_c = mm.xr_convert_temperature(temperature_data, 'K', 'C')
```

### `xr_pressure_to_altitude(dataarray, method='barometric')`
Convert pressure to altitude in an xarray DataArray.

```python
mm.xr_pressure_to_altitude(dataarray, method='barometric')
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray containing pressure values (Pa)
- `method` (str, optional): Conversion method ('barometric', 'interpolation'), default 'barometric'

**Returns:**
- xarray.DataArray: DataArray with altitude values (m)

**Example:**
```python
# Convert pressure to altitude
altitude = mm.xr_pressure_to_altitude(pressure_pa)
```

### `xr_altitude_to_pressure(dataarray, method='barometric')`
Convert altitude to pressure in an xarray DataArray.

```python
mm.xr_altitude_to_pressure(dataarray, method='barometric')
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray containing altitude values (m)
- `method` (str, optional): Conversion method ('barometric', 'interpolation'), default 'barometric'

**Returns:**
- xarray.DataArray: DataArray with pressure values (Pa)

**Example:**
```python
# Convert altitude to pressure
pressure = mm.xr_altitude_to_pressure(altitude)
```

## Vertical Interpolation

### `xr_interpolate_vertical(dataarray, old_coord, new_coord, method='linear')`
Interpolate data along vertical coordinate in an xarray DataArray.

```python
mm.xr_interpolate_vertical(dataarray, old_coord, new_coord, method='linear')
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray containing data to interpolate
- `old_coord` (str): Name of the vertical coordinate to interpolate from
- `new_coord` (xarray.DataArray): New coordinate values
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'

**Returns:**
- xarray.DataArray: Interpolated DataArray

**Example:**
```python
# Interpolate temperature from sigma to pressure coordinates
sigma_levels = xr.DataArray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dims=['sigma'])
pressure_levels = xr.DataArray([1000, 850, 700, 500, 300, 200], dims=['pressure'])

# Load sigma-coordinate data
sigma_temp = xr.open_dataset('sigma_temp.nc')['temperature']

# Interpolate to pressure coordinates
pressure_temp = mm.xr_interpolate_vertical(sigma_temp, 'sigma', pressure_levels, method='linear')
```

### `xr_interpolate_temperature_pressure(dataarray, pressure_coord, new_pressure, method='linear')`
Interpolate temperature to new pressure levels in an xarray DataArray.

```python
mm.xr_interpolate_temperature_pressure(dataarray, pressure_coord, new_pressure, method='linear')
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray containing temperature values
- `pressure_coord` (xarray.DataArray): Current pressure coordinate
- `new_pressure` (xarray.DataArray): New pressure levels
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'

**Returns:**
- xarray.DataArray: Interpolated temperature DataArray

**Example:**
```python
# Interpolate temperature to standard pressure levels
standard_levels = xr.DataArray([1000, 850, 700, 500, 300], dims=['pressure'])

interpolated_temp = mm.xr_interpolate_temperature_pressure(
    temperature_data, pressure_data, standard_levels, method='log'
)
```

### `xr_interpolate_wind_components(u_dataarray, v_dataarray, pressure_coord, new_pressure, method='linear')`
Interpolate wind components to new pressure levels in xarray DataArrays.

```python
mm.xr_interpolate_wind_components(u_dataarray, v_dataarray, pressure_coord, new_pressure, method='linear')
```

**Parameters:**
- `u_dataarray` (xarray.DataArray): DataArray containing u-wind component
- `v_dataarray` (xarray.DataArray): DataArray containing v-wind component
- `pressure_coord` (xarray.DataArray): Current pressure coordinate
- `new_pressure` (xarray.DataArray): New pressure levels
- `method` (str, optional): Interpolation method ('linear', 'log'), default 'linear'

**Returns:**
- tuple: (u_interpolated, v_interpolated) interpolated wind components

**Example:**
```python
# Interpolate wind components to pressure levels
u_interp, v_interp = mm.xr_interpolate_wind_components(
    u_wind_data, v_wind_data, pressure_data, standard_levels, method='log'
)
```

## Distance and Geographic Operations

### `xr_calculate_distance(lat1, lon1, lat2, lon2, method='haversine')`
Calculate distance between two points in xarray DataArrays.

```python
mm.xr_calculate_distance(lat1, lon1, lat2, lon2, method='haversine')
```

**Parameters:**
- `lat1, lon1` (xarray.DataArray): Latitude and longitude of first point
- `lat2, lon2` (xarray.DataArray): Latitude and longitude of second point
- `method` (str, optional): Distance calculation method ('haversine', 'vincenty'), default 'haversine'

**Returns:**
- xarray.DataArray: Distance values (m)

**Example:**
```python
# Calculate distances between grid points
distances = mm.xr_calculate_distance(
    grid_lats, grid_lons, target_lat, target_lon, method='haversine'
)
```

### `xr_interpolate_with_dask(dataarray, old_coords, new_coords, method='linear', chunks='auto')`
Interpolate data using dask for parallel processing in xarray.

```python
mm.xr_interpolate_with_dask(dataarray, old_coords, new_coords, method='linear', chunks='auto')
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray to interpolate
- `old_coords` (dict): Dictionary of original coordinates
- `new_coords` (dict): Dictionary of new coordinates
- `method` (str, optional): Interpolation method, default 'linear'
- `chunks` (str or tuple, optional): Chunk specification, default 'auto'

**Returns:**
- xarray.DataArray: Interpolated DataArray with dask arrays

**Example:**
```python
# Interpolate large dataset with dask
interpolated = mm.xr_interpolate_with_dask(
    temperature_data, 
    {'lon': old_lons, 'lat': old_lats},
    {'lon': new_lons, 'lat': new_lats},
    method='cubic'
)
```

## Coordinate and Data Management

### `add_coordinate_metadata(dataarray, coord_name, long_name, units, standard_name=None)`
Add metadata to a coordinate in an xarray DataArray.

```python
mm.add_coordinate_metadata(dataarray, coord_name, long_name, units, standard_name=None)
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray to modify
- `coord_name` (str): Name of the coordinate
- `long_name` (str): Long name description
- `units` (str): Units string
- `standard_name` (str, optional): CF standard name

**Returns:**
- xarray.DataArray: Modified DataArray with metadata

**Example:**
```python
# Add metadata to pressure coordinate
pressure_with_metadata = mm.add_coordinate_metadata(
    pressure_data, 'pressure', 'air pressure', 'Pa', 'air_pressure'
)
```

### `validate_coordinate_system(dataarray)`
Validate coordinate system of an xarray DataArray.

```python
mm.validate_coordinate_system(dataarray)
```

**Parameters:**
- `dataarray` (xarray.DataArray): DataArray to validate

**Returns:**
- bool: True if coordinate system is valid

**Raises:**
- ValueError: If coordinate system is invalid

**Example:**
```python
# Validate coordinate system
is_valid = mm.validate_coordinate_temperature(temperature_data)
if not is_valid:
    raise ValueError("Invalid coordinate system")
```

## NetCDF and File I/O Operations

### `load_netcdf_dataset(filepath, decode_times=True, chunks=None)`
Load a netCDF dataset with meteorological data.

```python
mm.load_netcdf_dataset(filepath, decode_times=True, chunks=None)
```

**Parameters:**
- `filepath` (str): Path to netCDF file
- `decode_times` (bool, optional): Decode time coordinates, default True
- `chunks` (dict, optional): Chunking specification for dask

**Returns:**
- xarray.Dataset: Loaded dataset

**Example:**
```python
# Load meteorological dataset
dataset = mm.load_netcdf_dataset('weather_data.nc', chunks={'time': 10, 'lat': 180, 'lon': 360})
```

### `save_netcdf_dataset(dataset, filepath, encoding=None, unlimited_dims=None)`
Save an xarray Dataset to netCDF format.

```python
mm.save_netcdf_dataset(dataset, filepath, encoding=None, unlimited_dims=None)
```

**Parameters:**
- `dataset` (xarray.Dataset): Dataset to save
- `filepath` (str): Output file path
- `encoding` (dict, optional): Encoding specification
- `unlimited_dims` (list, optional): List of unlimited dimensions

**Example:**
```python
# Save processed dataset
mm.save_netcdf_dataset(processed_dataset, 'processed_weather.nc')
```

### `create_standard_meteorological_dataset(data_dict, coords_dict, attrs_dict)`
Create a standard meteorological dataset with proper CF conventions.

```python
mm.create_standard_meteorological_dataset(data_dict, coords_dict, attrs_dict)
```

**Parameters:**
- `data_dict` (dict): Dictionary of data variables
- `coords_dict` (dict): Dictionary of coordinates
- `attrs_dict` (dict): Global attributes

**Returns:**
- xarray.Dataset: Standard meteorological dataset

**Example:**
```python
# Create standard meteorological dataset
data_vars = {
    'temperature': (['time', 'lat', 'lon'], temperature_data),
    'pressure': (['time', 'lat', 'lon'], pressure_data),
    'u_wind': (['time', 'lat', 'lon'], u_wind_data),
    'v_wind': (['time', 'lat', 'lon'], v_wind_data)
}

coords = {
    'time': time_coords,
    'lat': latitude_coords,
    'lon': longitude_coords
}

attrs = {
    'title': 'Atmospheric Model Output',
    'institution': 'NOAA ARL',
    'source': 'Monet-Meteo Library',
    'history': 'Created by monet_meteo.io module'
}

dataset = mm.create_standard_meteorological_dataset(data_vars, coords, attrs)
```

## Usage Patterns

### Basic Data Loading and Processing
```python
import monet_meteo as mm
import xarray as xr
import numpy as np

def process_meteorological_data(filepath):
    """
    Load and process meteorological data from netCDF file
    """
    # Load dataset with appropriate chunking
    dataset = mm.load_netcdf_dataset(
        filepath, 
        chunks={'time': 10, 'lat': 90, 'lon': 180}
    )
    
    # Convert pressure units if needed
    if 'pressure' in dataset and dataset['pressure'].attrs.get('units') == 'hPa':
        dataset['pressure'] = mm.xr_convert_pressure(dataset['pressure'], 'hPa', 'Pa')
        dataset['pressure'].attrs['units'] = 'Pa'
    
    # Convert temperature units if needed
    if 'temperature' in dataset and dataset['temperature'].attrs.get('units') == 'K':
        dataset['temperature_c'] = mm.xr_convert_temperature(
            dataset['temperature'], 'K', 'C'
        )
        dataset['temperature_c'].attrs = {
            'units': 'C', 
            'long_name': 'Air Temperature',
            'standard_name': 'air_temperature'
        }
    
    # Add metadata to coordinates
    dataset = mm.add_coordinate_metadata(
        dataset, 'lat', 'Latitude', 'degrees_north', 'latitude'
    )
    dataset = mm.add_coordinate_metadata(
        dataset, 'lon', 'Longitude', 'degrees_east', 'longitude'
    )
    
    return dataset

# Example usage
# processed_data = process_meteorological_data('weather_data.nc')
```

### Vertical Coordinate Processing
```python
def vertical_coordinate_conversion(dataset, target_pressure_levels):
    """
    Convert all variables to pressure coordinates
    """
    # Validate input coordinate system
    mm.validate_coordinate_system(dataset)
    
    # Convert pressure coordinate if needed
    if 'pressure' not in dataset.dims and 'level' in dataset.dims:
        # Assume level is actually pressure
        dataset = dataset.rename({'level': 'pressure'})
        dataset['pressure'].attrs = {
            'units': 'Pa',
            'long_name': 'Pressure',
            'standard_name': 'air_pressure'
        }
    
    # Interpolate all variables to standard pressure levels
    processed_data = {}
    
    for var_name, var_data in dataset.data_vars.items():
        if 'pressure' in var_data.dims:
            # Skip pressure coordinate itself
            if var_name == 'pressure':
                continue
            
            # Interpolate to pressure levels
            if var_name in ['u_wind', 'v_wind']:
                # Handle wind components
                u_data = dataset['u_wind']
                v_data = dataset['v_wind']
                
                u_interp, v_interp = mm.xr_interpolate_wind_components(
                    u_data, v_data, dataset['pressure'], target_pressure_levels, method='log'
                )
                
                processed_data['u_wind'] = u_interp
                processed_data['v_wind'] = v_interp
            else:
                # Regular variable interpolation
                interp_data = mm.xr_interpolate_vertical(
                    var_data, 'pressure', target_pressure_levels, method='log'
                )
                processed_data[var_name] = interp_data
        else:
            # 2D variables (surface fields) - no vertical interpolation needed
            processed_data[var_name] = var_data
    
    # Create new dataset
    new_coords = {**dataset.coords, 'pressure': target_pressure_levels}
    new_dataset = xr.Dataset(processed_data, coords=new_coords)
    
    return new_dataset

# Example usage
# standard_levels = xr.DataArray([1000, 850, 700, 500, 300, 200], dims=['pressure'])
# pressure_dataset = vertical_coordinate_conversion(dataset, standard_levels)
```

### Large Dataset Processing with Dask
```python
def process_large_climate_dataset(filepath, output_filepath, chunk_size='auto'):
    """
    Process large climate dataset using dask for parallel processing
    """
    # Load dataset with automatic chunking
    dataset = mm.load_netcdf_dataset(
        filepath, 
        chunks=chunk_size
    )
    
    # Define target coordinates
    target_coords = {
        'lon': np.linspace(-180, 180, 721),
        'lat': np.linspace(-90, 90, 361),
        'pressure': np.array([1000, 850, 700, 500, 300, 200]) * 100
    }
    
    # Process each variable
    processed_vars = {}
    
    for var_name, var_data in dataset.data_vars.items():
        if var_name in ['u_wind', 'v_wind']:
            # Interpolate wind components
            u_data = dataset['u_wind']
            v_data = dataset['v_wind']
            
            # Horizontal interpolation
            u_interp = mm.xr_interpolate_with_dask(
                u_data, 
                {'lon': dataset.lon, 'lat': dataset.lat},
                {'lon': target_coords['lon'], 'lat': target_coords['lat']},
                method='cubic'
            )
            
            v_interp = mm.xr_interpolate_with_dask(
                v_data, 
                {'lon': dataset.lon, 'lat': dataset.lat},
                {'lon': target_coords['lon'], 'lat': target_coords['lat']},
                method='cubic'
            )
            
            # Vertical interpolation
            u_final, v_final = mm.xr_interpolate_wind_components(
                u_interp, v_interp, 
                dataset['pressure'], target_coords['pressure'], 
                method='log'
            )
            
            processed_vars['u_wind'] = u_final
            processed_vars['v_wind'] = v_final
            
        elif var_name in dataset.data_vars and var_name != 'pressure':
            # Regular variable processing
            # Horizontal interpolation
            interp_data = mm.xr_interpolate_with_dask(
                var_data, 
                {'lon': dataset.lon, 'lat': dataset.lat},
                {'lon': target_coords['lon'], 'lat': target_coords['lat']},
                method='cubic'
            )
            
            # Vertical interpolation if needed
            if 'pressure' in interp_data.dims:
                interp_data = mm.xr_interpolate_vertical(
                    interp_data, 'pressure', target_coords['pressure'], method='log'
                )
            
            processed_vars[var_name] = interp_data
    
    # Create new dataset
    new_dataset = xr.Dataset(processed_vars, coords=target_coords)
    
    # Add metadata
    new_dataset.attrs = dataset.attrs
    new_dataset.attrs['processing_history'] = 'Regridded and interpolated using monet_meteo'
    
    # Save processed dataset
    mm.save_netcdf_dataset(new_dataset, output_filepath)
    
    return new_dataset

# Example usage
# processed_climate = process_large_climate_dataset('large_climate.nc', 'processed_climate.nc')
```

### Real-Time Data Processing Pipeline
```python
def real_time_processing_pipeline(observation_file, model_background_file, output_file):
    """
    Process real-time observations with model background
    """
    # Load real-time observations
    obs_dataset = mm.load_netcdf_dataset(observation_file)
    
    # Load model background
    model_dataset = mm.load_netcdf_dataset(model_background_file)
    
    # Process observations to model grid
    processed_obs = {}
    
    for var_name in ['temperature', 'u_wind', 'v_wind', 'humidity']:
        if var_name in obs_dataset:
            # Interpolate observations to model grid
            if var_name in ['u_wind', 'v_wind']:
                # Wind component interpolation
                obs_u = obs_dataset[f'{var_name}_component_u'] if var_name == 'wind' else obs_dataset['u_wind']
                obs_v = obs_dataset[f'{var_name}_component_v'] if var_name == 'wind' else obs_dataset['v_wind']
                
                model_u = model_dataset['u_wind']
                model_v = model_dataset['v_wind']
                
                # Interpolate to model grid
                u_interp = mm.xr_interpolate_with_dask(
                    obs_u,
                    {'lon': obs_dataset.lon, 'lat': obs_dataset.lat},
                    {'lon': model_dataset.lon, 'lat': model_dataset.lat},
                    method='linear'
                )
                
                v_interp = mm.xr_interpolate_with_dask(
                    obs_v,
                    {'lon': obs_dataset.lon, 'lat': obs_dataset.lat},
                    {'lon': model_dataset.lon, 'lat': model_dataset.lat},
                    method='linear'
                )
                
                processed_obs['u_wind'] = u_interp
                processed_obs['v_wind'] = v_interp
            else:
                # Regular variable interpolation
                obs_var = obs_dataset[var_name]
                interp_var = mm.xr_interpolate_with_dask(
                    obs_var,
                    {'lon': obs_dataset.lon, 'lat': obs_dataset.lat},
                    {'lon': model_dataset.lon, 'lat': model_dataset.lat},
                    method='linear'
                )
                processed_obs[var_name] = interp_var
    
    # Create analysis increment (observation - background)
    analysis_inc = {}
    
    for var_name in processed_obs:
        background = model_dataset[var_name]
        analysis_inc[var_name] = processed_obs[var_name] - background
    
    # Create analysis dataset
    analysis_dataset = xr.Dataset(analysis_inc, coords=model_dataset.coords)
    
    # Add metadata
    analysis_dataset.attrs = {
        'title': 'Analysis Increments',
        'source': 'Real-time processing pipeline',
        'processing_time': pd.Timestamp.now().isoformat(),
        'background_source': model_dataset.attrs.get('source', 'Unknown'),
        'observation_source': obs_dataset.attrs.get('source', 'Unknown')
    }
    
    # Save analysis
    mm.save_netcdf_dataset(analysis_dataset, output_file)
    
    return analysis_dataset

# Example usage
# analysis = real_time_processing_pipeline('observations.nc', 'model_background.nc', 'analysis.nc')
```

## Advanced Applications

### Multi-Model Ensemble Processing
```python
def process_ensemble_models(model_files, common_grid):
    """
    Process multiple model outputs to common grid for ensemble analysis
    """
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
    
    return ensemble_datasets

def calculate_ensemble_statistics(ensemble_datasets):
    """Calculate ensemble mean and spread"""
    # Stack all datasets
    stacked = xr.concat(ensemble_datasets, dim='model')
    
    # Calculate statistics
    ensemble_mean = stacked.mean(dim='model')
    ensemble_std = stacked.std(dim='model')
    ensemble_min = stacked.min(dim='model')
    ensemble_max = stacked.max(dim='model')
    
    return {
        'mean': ensemble_mean,
        'std': ensemble_std,
        'min': ensemble_min,
        'max': ensemble_max
    }

# Example usage
# common_grid = {'lon': model_lons, 'lat': model_lats}
# ensemble = process_ensemble_models(model_files, common_grid)
# stats = calculate_ensemble_statistics(ensemble)
```

### Climate Data Processing Pipeline
```python
def climate_data_processing_pipeline(input_dir, output_dir, variables_to_process):
    """
    Process climate data for multi-year analysis
    """
    import glob
    import os
    
    # Get all files in input directory
    file_pattern = os.path.join(input_dir, '*.nc')
    input_files = glob.glob(file_pattern)
    
    # Define common grid
    common_grid = {
        'lon': np.linspace(-180, 180, 360),
        'lat': np.linspace(-90, 90, 180),
        'pressure': np.array([1000, 850, 700, 500, 300, 200]) * 100
    }
    
    # Process each file
    processed_datasets = []
    
    for input_file in input_files:
        # Extract year from filename
        filename = os.path.basename(input_file)
        year = filename.split('_')[1]  # Assuming format like 'data_2020.nc'
        
        print(f"Processing {year}...")
        
        # Load and process data
        dataset = mm.load_netcdf_dataset(input_file)
        
        # Process variables
        processed_data = {}
        
        for var_name in variables_to_process:
            if var_name in dataset:
                # Interpolate to common grid
                if var_name in ['u_wind', 'v_wind']:
                    # Wind components
                    u_data = dataset['u_wind']
                    v_data = dataset['v_wind']
                    
                    u_interp = mm.xr_interpolate_with_dask(
                        u_data,
                        {'lon': dataset.lon, 'lat': dataset.lat},
                        {'lon': common_grid['lon'], 'lat': common_grid['lat']},
                        method='linear'
                    )
                    
                    v_interp = mm.xr_interpolate_with_dask(
                        v_data,
                        {'lon': dataset.lon, 'lat': dataset.lat},
                        {'lon': common_grid['lon'], 'lat': common_grid['lat']},
                        method='linear'
                    )
                    
                    processed_data['u_wind'] = u_interp
                    processed_data['v_wind'] = v_interp
                else:
                    # Regular variables
                    interp_data = mm.xr_interpolate_with_dask(
                        dataset[var_name],
                        {'lon': dataset.lon, 'lat': dataset.lat},
                        {'lon': common_grid['lon'], 'lat': common_grid['lat']},
                        method='linear'
                    )
                    processed_data[var_name] = interp_data
        
        # Create processed dataset
        processed_dataset = xr.Dataset(processed_data, coords=common_grid)
        processed_dataset.attrs['year'] = year
        
        processed_datasets.append(processed_dataset)
    
    # Concatenate all years
    climate_dataset = xr.concat(processed_datasets, dim='time')
    
    # Save final dataset
    output_path = os.path.join(output_dir, 'processed_climate_data.nc')
    mm.save_netcdf_dataset(climate_dataset, output_path)
    
    return climate_dataset

# Example usage
# climate_data = climate_data_processing_pipeline(
#     'input_data/', 'output_data/', 
#     ['temperature', 'u_wind', 'v_wind', 'humidity']
# )
```

## Error Handling

The I/O module includes comprehensive error handling:

### Common Errors
```python
# Error: Invalid file path
try:
    dataset = mm.load_netcdf_dataset('nonexistent_file.nc')
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Error: Invalid coordinate system
try:
    invalid_dataset = xr.Dataset({'data': (['x', 'y'], np.random.rand(10, 10))})
    mm.validate_coordinate_system(invalid_dataset)
except ValueError as e:
    print(f"Coordinate system error: {e}")

# Error: Unit conversion error
try:
    pressure_data = xr.DataArray([1000, 2000], dims=['x'])
    mm.xr_convert_pressure(pressure_data, 'hPa', 'invalid_unit')
except ValueError as e:
    print(f"Unit conversion error: {e}")

# Error: Interpolation error
try:
    small_data = xr.DataArray([1, 2], dims=['x'])
    new_coords = xr.DataArray([1, 2, 3, 4, 5], dims=['x_new'])
    mm.xr_interpolate_vertical(small_data, 'x', new_coords)
except ValueError as e:
    print(f"Interpolation error: {e}")
```

## Performance Considerations

### Chunking Strategy
For large datasets, appropriate chunking is crucial:

```python
# Optimal chunking for different data types
chunking_strategies = {
    'time_series': {'time': 100, 'lat': 180, 'lon': 360},
    'spatial': {'time': 1, 'lat': 90, 'lon': 180},
    'high_resolution': {'time': 1, 'lat': 45, 'lon': 90},
    'very_large': {'time': 1, 'lat': 30, 'lon': 60}
}

# Choose strategy based on data size
def get_optimal_chunking(data_shape, data_size_gb):
    """Determine optimal chunking based on data characteristics"""
    if data_size_gb > 10:  # Very large data
        return chunking_strategies['very_large']
    elif data_shape[1] > 1000:  # High resolution
        return chunking_strategies['high_resolution']
    elif data_shape[0] > 1000:  # Long time series
        return chunking_strategies['time_series']
    else:
        return chunking_strategies['spatial']
```

### Memory Management
Monitor and manage memory usage:

```python
def monitor_processing_memory():
    """Monitor memory usage during data processing"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / 1024**2:.1f} MB")
    print(f"Memory percent: {process.memory_percent():.1f}%")
    
    if process.memory_percent() > 80:
        print("Warning: High memory usage detected")
```

## Constants and Standards

The I/O module follows CF (Climate and Forecast) conventions and uses standard meteorological units:

- **Units**: SI units with common meteorological conventions (Pa for pressure, K for temperature)
- **Coordinate Names**: Standard names like 'lat', 'lon', 'pressure', 'time'
- **Metadata**: CF-compliant attributes including 'long_name', 'units', 'standard_name'
- **Time Handling**: CF-compliant time coordinate handling with calendar support

## References

- NetCDF User's Guide: https://docs.unidata.ucar.edu/netcdf-c/current
- xarray Documentation: https://docs.xarray.dev/
- CF Conventions: https://cfconventions.org/
- Xarray Documentation: https://docs.xarray.dev/en/stable/

## See Also

- [Thermodynamics Module](thermodynamics.md) - Thermodynamic variable calculations
- [Dynamic Calculations](dynamics.md) - Dynamic meteorology functions
- [Statistical Analysis](statistical.md) - Statistical and micrometeorological functions
- [Unit Conversions](units.md) - Meteorological unit conversion utilities
- [Data Models](models.md) - Structured data models for atmospheric data
- [Coordinates](coordinates.md) - Coordinate transformation utilities