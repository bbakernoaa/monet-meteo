# Climate Dataset Processing Tutorial

This tutorial demonstrates advanced climate data processing techniques using Monet-Meteo with xarray and dask for handling large-scale climate datasets.

## 🎯 Tutorial Overview

### Learning Objectives
- Process large climate datasets efficiently with dask
- Perform batch atmospheric calculations on gridded data
- Apply Monet-Meteo functions to xarray datasets
- Handle multi-dimensional climate data time series
- Optimize memory usage for large-scale computations

### Prerequisites
- Monet-Meteo installed (`pip install monet-meteo`)
- xarray and dask installed (`pip install xarray dask`)
- Basic understanding of climate data formats (netCDF)
- Familiarity with xarray data structures

## 📦 Setup

### Install Dependencies
```bash
pip install monet-meteo xarray dask netCDF4 matplotlib cartopy
```

### Import Required Libraries
```python
import monet_meteo as mm
import xarray as xr
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
```

## 🌡️ Processing Climate Model Output

### Example 1: ERA5 Reanalysis Data Processing

```python
def process_era5_data(filepath):
    """
    Process ERA5 reanalysis data to calculate atmospheric variables
    
    Parameters
    ----------
    filepath : str
        Path to ERA5 netCDF file
        
    Returns
    -------
    xarray.Dataset
        Processed dataset with additional variables
    """
    # Open dataset with dask for lazy loading
    ds = xr.open_dataset(
        filepath, 
        chunks={'time': 10, 'latitude': 50, 'longitude': 50}  # Chunking for parallel processing
    )
    
    # Convert units if needed
    ds['temperature'] = mm.convert_temperature(ds.t2m, 'K', 'C')
    ds['pressure'] = mm.convert_pressure(ds.sp, 'Pa', 'hPa')
    
    # Calculate geopotential height from pressure (simplified)
    ds['height'] = mm.pressure_to_altitude(ds.pressure)
    
    # Calculate derived parameters
    ds['potential_temperature'] = xr.apply_ufunc(
        mm.potential_temperature,
        ds.pressure, ds.temperature + 273.15,  # Convert back to K
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized'
    )
    
    # Calculate specific humidity from relative humidity
    ds['sat_vapor_pressure'] = xr.apply_ufunc(
        mm.saturation_vapor_pressure,
        ds.temperature + 273.15,
        input_core_dims=[[]],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized'
    )
    
    ds['vapor_pressure'] = ds.rh * ds.sat_vapor_pressure / 100
    ds['mixing_ratio'] = xr.apply_ufunc(
        mm.mixing_ratio,
        ds.vapor_pressure, ds.pressure * 100,  # Convert hPa to Pa
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized'
    )
    
    return ds

# Example usage
# ds = process_era5_data('era5_2023.nc')
```

### Example 2: Climate Model Ensemble Processing

```python
def process_ensemble_data(file_pattern, variables):
    """
    Process multiple climate model ensemble members
    
    Parameters
    ----------
    file_pattern : str
        Pattern for ensemble files (e.g., 'model_*.nc')
    variables : list
        List of variables to process
        
    Returns
    -------
    xarray.Dataset
        Combined ensemble dataset with processing applied
    """
    # Open multiple files with preprocessing
    def preprocess(ds):
        # Standardize variable names
        name_mapping = {
            'tas': 'temperature',
            'ps': 'pressure',
            'hurs': 'relative_humidity',
            'uas': 'u_wind',
            'vas': 'v_wind'
        }
        
        # Rename variables
        for old_name, new_name in name_mapping.items():
            if old_name in ds:
                ds = ds.rename({old_name: new_name})
        
        # Convert units
        if 'temperature' in ds:
            ds['temperature'] = mm.convert_temperature(ds.temperature, 'K', 'C')
        if 'pressure' in ds:
            ds['pressure'] = mm.convert_pressure(ds.pressure, 'Pa', 'hPa')
            
        return ds
    
    # Open ensemble data with chunks
    ensemble_ds = xr.open_mfdataset(
        file_pattern,
        chunks={'time': 24, 'lat': 30, 'lon': 30},
        preprocess=preprocess,
        parallel=True
    )
    
    # Process each ensemble member
    for member in ensemble_ds.member.values:
        member_ds = ensemble_ds.sel(member=member)
        
        # Calculate atmospheric variables for this member
        member_ds['potential_temperature'] = xr.apply_ufunc(
            mm.potential_temperature,
            member_ds.pressure, member_ds.temperature + 273.15,
            input_core_dims=[[], []],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized'
        )
        
        # Calculate wind speed
        member_ds['wind_speed'] = np.sqrt(member_ds.u_wind**2 + member_ds.v_wind**2)
        
        # Calculate bulk Richardson number
        member_ds['richardson_number'] = xr.apply_ufunc(
            mm.bulk_richardson_number,
            member_ds.u_wind, member_ds.v_wind, member_ds.potential_temperature,
            member_ds.height,
            input_core_dims=[[], [], [], []],
            output_core_dims=[[]],
            vectorize=True,
            dask='parallelized'
        )
    
    return ensemble_ds

# Example usage
# ensemble = process_ensemble_data('ensemble_*.nc', ['temperature', 'pressure'])
```

## 📊 Climate Time Series Analysis

### Example 3: Climate Time Series Processing

```python
def analyze_climate_timeseries(filepath):
    """
    Analyze climate time series with seasonal and trend calculations
    
    Parameters
    ----------
    filepath : str
        Path to climate time series data
        
    Returns
    -------
    dict
        Dictionary containing analysis results
    """
    # Load data with appropriate chunking
    ds = xr.open_dataset(
        filepath, 
        chunks={'time': 365, 'lat': 20, 'lon': 20}  # Annual chunks
    )
    
    # Calculate seasonal means
    seasonal_means = ds.groupby('time.season').mean()
    
    # Calculate climatological means
    climatology = ds.groupby('time.month').mean()
    
    # Calculate anomalies
    monthly_anomalies = ds.groupby('time.month') - climatology
    
    # Calculate trends using linear regression
    def calculate_trend(data):
        """Calculate linear trend over time"""
        time_coords = np.arange(len(data.time))
        trend = np.polyfit(time_coords, data.values, 1)[0]
        return trend
    
    trend_da = xr.apply_ufunc(
        calculate_trend,
        ds.temperature,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized'
    )
    
    # Calculate extreme statistics
    extreme_thresholds = {
        'hot_days': ds.temperature.quantile(0.95, dim='time'),
        'cold_days': ds.temperature.quantile(0.05, dim='time'),
        'heat_waves': (ds.temperature > ds.temperature.quantile(0.95, dim='time')).sum(dim='time')
    }
    
    # Process with Monet-Meteo
    ds['heat_index'] = xr.apply_ufunc(
        mm.heat_index,
        ds.temperature + 273.15,  # Convert to K
        ds.relative_humidity / 100,  # Convert to fraction
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized'
    )
    
    # Calculate heat wave frequency
    heat_wave_threshold = ds.heat_index.quantile(0.95, dim='time')
    heat_wave_frequency = (ds.heat_index > heat_wave_threshold).mean(dim='time')
    
    return {
        'seasonal_means': seasonal_means,
        'monthly_anomalies': monthly_anomalies,
        'trends': trend_da,
        'extremes': extreme_thresholds,
        'heat_wave_frequency': heat_wave_frequency
    }

# Example usage
# analysis = analyze_climate_timeseries('gcm_output.nc')
```

### Example 4: Multi-Model Climate Projections

```python
def process_cmip6_data(base_path, models, scenarios):
    """
    Process CMIP6 multi-model climate projection data
    
    Parameters
    ----------
    base_path : str
        Base path for CMIP6 data
    models : list
        List of CMIP6 model names
    scenarios : list
        List of scenarios (e.g., 'ssp245', 'ssp585')
        
    Returns
    -------
    xarray.Dataset
        Multi-model ensemble dataset
    """
    # Create dataset list
    datasets = []
    
    for model in models:
        for scenario in scenarios:
            # Pattern for CMIP6 files
            pattern = f"{base_path}/{model}/{scenario}/*.nc"
            
            # Load data with preprocessing
            ds = xr.open_mfdataset(
                pattern,
                chunks={'time': 365, 'lat': 30, 'lon': 30},
                parallel=True
            )
            
            # Standardize time coordinate
            ds['time'] = xr.decode_cf(ds).time
            
            # Calculate climate change indices
            ds['txx'] = ds.temperature.max(dim='time')  # Maximum temperature
            ds['tmin'] = ds.temperature.min(dim='time')  # Minimum temperature
            ds['gdd'] = (ds.temperature > 10).sum(dim='time')  # Growing degree days
            
            # Calculate ensemble statistics
            ds['model_mean'] = ds.mean(dim='model')
            ds['model_std'] = ds.std(dim='model')
            ds['model_range'] = ds.max(dim='model') - ds.min(dim='model')
            
            datasets.append(ds)
    
    # Combine datasets
    combined = xr.concat(datasets, dim='scenario')
    
    return combined

# Example usage
# cmip6_data = process_cmip6_data(
#     base_path='/data/cmip6',
#     models=['GFDL-ESM4', 'MRI-ESM2-0', 'IPSL-CM6A-LR'],
#     scenarios=['ssp245', 'ssp585']
# )
```

## 🌍 Geographic Processing

### Example 5: Regional Climate Analysis

```python
def analyze_regional_climate(ds, region_name, bounds):
    """
    Analyze climate data for a specific region
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input climate dataset
    region_name : str
        Name of the region
    bounds : tuple
        (min_lat, max_lat, min_lon, max_lon) bounding box
        
    Returns
    -------
    dict
        Regional climate analysis results
    """
    min_lat, max_lat, min_lon, max_lon = bounds
    
    # Select region
    regional_ds = ds.sel(
        lat=slice(min_lat, max_lat),
        lon=slice(min_lon, max_lon)
    )
    
    # Calculate regional means
    regional_mean = regional_ds.mean(dim=['lat', 'lon'])
    regional_max = regional_ds.max(dim=['lat', 'lon'])
    regional_min = regional_ds.min(dim=['lat', 'lon'])
    
    # Calculate climate indices
    tropical_days = (regional_ds.temperature > 25).sum(dim='time')
    frost_days = (regional_ds.temperature < 0).sum(dim='time')
    growing_season_length = (regional_ds.temperature > 5).sum(dim='time')
    
    # Calculate atmospheric stability
    regional_ds['stability'] = xr.apply_ufunc(
        mm.bulk_richardson_number,
        regional_ds.u_wind, regional_ds.v_wind, regional_ds.potential_temperature,
        regional_ds.height,
        input_core_dims=[[], [], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized'
    )
    
    # Calculate precipitation-related variables
    if 'precipitation' in regional_ds:
        # Convert to mm/day
        regional_ds['precipitation'] = regional_ds.precipitation * 86400
        
        # Calculate wet days (precip > 1mm)
        wet_days = (regional_ds.precipitation > 1).sum(dim='time')
        
        # Calculate extreme precipitation
        extreme_precip = regional_ds.precipitation.quantile(0.95, dim='time')
    else:
        wet_days = None
        extreme_precip = None
    
    return {
        'region_name': region_name,
        'bounds': bounds,
        'regional_mean': regional_mean,
        'regional_max': regional_max,
        'regional_min': regional_min,
        'tropical_days': tropical_days,
        'frost_days': frost_days,
        'growing_season_length': growing_season_length,
        'stability': regional_ds.stability,
        'wet_days': wet_days,
        'extreme_precip': extreme_precip
    }

# Example usage
# northeast_us = analyze_regional_climate(
#     ds, 
#     'Northeast US',
#     (35, 45, -80, -65)
# )
```

### Example 6: Climate Change Detection and Attribution

```python
def detect_climate_change(historical_data, future_data, reference_period=1995-2014):
    """
    Detect and quantify climate change signals
    
    Parameters
    ----------
    historical_data : xarray.Dataset
        Historical climate data
    future_data : xarray.Dataset
        Future climate projection data
    reference_period : tuple
        Reference period for baseline
        
    Returns
    -------
    dict
        Climate change detection results
    """
    # Calculate baseline from reference period
    baseline = historical_data.sel(
        time=slice(f'{reference_period[0]}-01-01', f'{reference_period[1]}-12-31')
    ).mean(dim='time')
    
    # Calculate future means
    future_mean = future_data.mean(dim='time')
    
    # Calculate change signals
    absolute_change = future_mean - baseline
    relative_change = (absolute_change / baseline) * 100
    
    # Calculate change significance using bootstrap
    def bootstrap_trend(data, n_bootstrap=100):
        """Calculate trend uncertainty using bootstrap"""
        trends = []
        for _ in range(n_bootstrap):
            sample = data.isel(time=np.random.choice(len(data.time), len(data.time), replace=True))
            trend = np.polyfit(np.arange(len(sample.time)), sample.values, 1)[0]
            trends.append(trend)
        return np.std(trends)
    
    # Calculate trend uncertainties
    trend_uncertainty = xr.apply_ufunc(
        bootstrap_trend,
        historical_data.temperature,
        input_core_dims=[['time']],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized'
    )
    
    # Calculate exceedance probabilities
    threshold = baseline.temperature + 2 * baseline.temperature.std()
    exceedance_prob = (future_data.temperature > threshold).mean(dim='time')
    
    # Calculate extreme event changes
    historical_extremes = historical_data.temperature.quantile(0.99, dim='time')
    future_extremes = future_data.temperature.quantile(0.99, dim='time')
    extreme_change = future_extremes - historical_extremes
    
    return {
        'baseline': baseline,
        'future_mean': future_mean,
        'absolute_change': absolute_change,
        'relative_change': relative_change,
        'trend_uncertainty': trend_uncertainty,
        'exceedance_probability': exceedance_prob,
        'extreme_change': extreme_change,
        'confidence_level': absolute_change / trend_uncertainty  # Signal-to-noise ratio
    }

# Example usage
# climate_change = detect_climate_change(historical_ds, future_ds)
```

## 📈 Performance Optimization

### Example 7: Optimized Large-Scale Processing

```python
def optimized_climate_processing(data_path, output_path):
    """
    Optimized processing of large climate datasets
    
    Parameters
    ----------
    data_path : str
        Input data path
    output_path : str
        Output path for processed data
        
    Returns
    -------
    None
    """
    # Set up dask for optimal performance
    from dask.distributed import Client
    client = Client(n_workers=4, memory_limit='8GB')
    
    # Configure chunking strategy
    chunk_size = {'time': 730, 'lat': 90, 'lon': 90}  # 2-year chunks
    
    # Load data with optimized chunking
    ds = xr.open_dataset(
        data_path,
        chunks=chunk_size,
        engine='zarr'  # Use zarr for better performance
    )
    
    # Process in memory-efficient chunks
    def process_chunk(chunk):
        """Process a single chunk of data"""
        # Convert units
        chunk['temperature'] = mm.convert_temperature(chunk.t2m, 'K', 'C')
        chunk['pressure'] = mm.convert_pressure(chunk.sp, 'Pa', 'hPa')
        
        # Calculate atmospheric variables
        chunk['potential_temperature'] = mm.potential_temperature(
            chunk.pressure, chunk.temperature + 273.15
        )
        
        # Calculate wind speed
        chunk['wind_speed'] = np.sqrt(chunk.uas**2 + chunk.vas**2)
        
        return chunk
    
    # Apply processing in parallel
    processed_ds = xr.map_blocks(
        process_chunk,
        ds,
        template=ds,  # Preserve structure
        chunks=chunk_size
    )
    
    # Save results with compression
    processed_ds.to_zarr(
        output_path,
        encoding={
            'temperature': {'compressor': zarr.Blosc(cname='zstd', clevel=5)},
            'pressure': {'compressor': zarr.Blosc(cname='zstd', clevel=5)},
            'potential_temperature': {'compressor': zarr.Blosc(cname='zstd', clevel=5)}
        }
    )
    
    # Close dask client
    client.close()
    
    print(f"Processing completed. Results saved to {output_path}")

# Example usage
# optimized_climate_processing('large_dataset.nc', 'processed_data.zarr')
```

## 🎨 Visualization and Analysis

### Example 8: Climate Change Visualization

```python
def visualize_climate_change(change_data, variable_name, title):
    """
    Create visualizations for climate change analysis
    
    Parameters
    ----------
    change_data : xarray.DataArray
        Climate change data
    variable_name : str
        Name of the variable being plotted
    title : str
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create map projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot data
    im = change_data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdBu_r',
        cbar_kwargs={'label': f'Change in {variable_name}'}
    )
    
    # Add geographic features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.OCEANS, color='lightblue')
    ax.add_feature(cfeature.LAND, color='lightgray')
    
    # Set title
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add grid
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    return fig

# Example usage
# fig = visualize_climate_change(
#     climate_change['absolute_change'],
#     'Temperature',
#     'Temperature Change (°C) - 2070-2100 vs 1995-2014'
# )
# fig.savefig('temperature_change.png', dpi=300, bbox_inches='tight')
```

## 📋 Best Practices

### Memory Management
1. **Chunk Appropriately**: Choose chunk sizes that fit in memory
2. **Use Lazy Loading**: Load data with dask for out-of-core processing
3. **Compress Data**: Use efficient compression for large datasets
4. **Process in Batches**: Avoid loading entire datasets into memory

### Performance Optimization
1. **Parallel Processing**: Use dask for parallel computation
2. **Vectorized Operations**: Leverage numpy/xarray vectorization
3. **Efficient Data Structures**: Use appropriate data types
4. **Cache Intermediate Results**: Save processed data for reuse

### Data Quality
1. **Validate Inputs**: Check data ranges and units
2. **Handle Missing Data**: Implement proper missing data handling
3. **Document Processing**: Keep track of processing steps
4. **Version Control**: Use version control for processing scripts

## 🚀 Next Steps

- Explore more advanced climate analysis techniques
- Learn about statistical downscaling methods
- Implement ensemble processing workflows
- Develop custom climate indicators
- Create automated climate monitoring systems

## 📚 Additional Resources

- [xarray Documentation](https://xarray.pydata.org/)
- [Dask Documentation](https://dask.org/)
- [CMIP6 Data Guidelines](https://pcmdi.llnl.gov/CMIP6/)
- [Climate Indices Reference](https://www.climdex.org/)