# Troubleshooting and Best Practices Guide

This guide provides solutions to common issues encountered when using the monet-meteo library, along with best practices for development and usage. Whether you're a new user or an experienced developer, this will help you avoid pitfalls and optimize your workflow.

## Table of Contents

1. [Common Issues and Solutions](#common-issues-and-solutions)
2. [Error Reference](#error-reference)
3. [Best Practices](#best-practices)
4. [Performance Optimization](#performance-optimization)
5. [Debugging Techniques](#debugging-techniques)
6. [Data Quality Assurance](#data-quality-assurance)
7. [Community Support](#community-support)

## Common Issues and Solutions

### Installation and Setup Issues

#### Issue: Installation Fails with Dependencies
**Problem**: Installation fails due to missing dependencies or version conflicts.

**Solution**: 
```bash
# Create a clean virtual environment
python -m venv monet-meteo-env
source monet-meteo-env/bin/activate  # On Windows: monet-meteo-env\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install numpy xarray scipy pandas netCDF4 matplotlib

# Install monet-meteo
pip install monet-meteo
```

#### Issue: Missing Optional Dependencies
**Problem**: Import errors for optional dependencies like xarray or dask.

**Solution**: Install optional dependencies:
```bash
# Install with optional dependencies
pip install monet-meteo[xarray,dask]

# Or install individually
pip install dask
pip install cartopy  # For mapping
```

#### Issue: Platform-Specific Issues
**Problem**: Installation fails on specific platforms (Windows, macOS, Linux).

**Solution**:
- **Windows**: Use conda or ensure you have Visual Studio Build Tools installed
- **macOS**: Use conda or install Xcode command line tools
- **Linux**: Install system dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install python3-dev python3-tk libhdf5-dev
  
  # CentOS/RHEL
  sudo yum install python3-devel python3-tkinter hdf5-devel
  ```

### Data Input Issues

#### Issue: Invalid Data Formats
**Problem**: Data arrays have incorrect shapes or types.

**Solution**: Validate input data before processing:
```python
import numpy as np
import monet_meteo as mm

def validate_atmospheric_data(pressure, temperature):
    """Validate atmospheric data before processing"""
    # Check data types
    if not isinstance(pressure, (np.ndarray, list)) or not isinstance(temperature, (np.ndarray, list)):
        raise TypeError("Pressure and temperature must be numpy arrays or lists")
    
    # Convert to numpy arrays
    pressure = np.asarray(pressure)
    temperature = np.asarray(temperature)
    
    # Check shapes
    if pressure.shape != temperature.shape:
        raise ValueError("Pressure and temperature must have the same shape")
    
    # Check for physically impossible values
    if np.any(pressure <= 0):
        raise ValueError("Pressure must be positive")
    
    if np.any(temperature < 0):
        raise ValueError("Temperature in Kelvin cannot be negative")
    
    return pressure, temperature

# Usage
try:
    pressure, temperature = validate_atmospheric_data(pressure_data, temperature_data)
    result = mm.potential_temperature(pressure, temperature)
except ValueError as e:
    print(f"Data validation error: {e}")
```

#### Issue: Missing Coordinate Information
**Problem**: xarray datasets lack required coordinate metadata.

**Solution**: Add coordinate metadata:
```python
import xarray as xr
import monet_meteo as mm

# Fix missing coordinate metadata
def fix_coordinate_metadata(dataset):
    """Add missing coordinate metadata"""
    if 'lat' in dataset.coords and 'lon' in dataset.coords:
        dataset = mm.add_coordinate_metadata(dataset, 'lat', 'Latitude', 'degrees_north', 'latitude')
        dataset = mm.add_coordinate_metadata(dataset, 'lon', 'Longitude', 'degrees_east', 'longitude')
    
    if 'pressure' in dataset.coords:
        dataset = mm.add_coordinate_metadata(dataset, 'pressure', 'Pressure', 'Pa', 'air_pressure')
    
    if 'time' in dataset.coords:
        dataset = mm.add_coordinate_metadata(dataset, 'time', 'Time', 'hours since 1970-01-01', 'time')
    
    return dataset

# Usage
dataset = fix_coordinate_metadata(dataset)
```

#### Issue: Unit Conversion Errors
**Problem**: Unit conversions fail with invalid units or values.

**Solution**: Validate units before conversion:
```python
def safe_unit_conversion(value, from_unit, to_unit, value_type='pressure'):
    """Safely convert units with validation"""
    valid_units = {
        'pressure': ['Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm'],
        'temperature': ['K', 'C', 'F'],
        'wind_speed': ['m/s', 'mps', 'knots', 'kt', 'km/h', 'kmh', 'mph'],
        'distance': ['m', 'km', 'ft', 'mi', 'nm']
    }
    
    # Validate units
    if value_type not in valid_units:
        raise ValueError(f"Unknown value type: {value_type}")
    
    if from_unit not in valid_units[value_type]:
        raise ValueError(f"Invalid {value_type} unit: {from_unit}")
    
    if to_unit not in valid_units[value_type]:
        raise ValueError(f"Invalid {value_type} unit: {to_unit}")
    
    # Validate values
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"{value_type} cannot be negative")
    
    try:
        converted = mm.convert_pressure(value, from_unit, to_unit) if value_type == 'pressure' else \
                   mm.convert_temperature(value, from_unit, to_unit) if value_type == 'temperature' else \
                   mm.convert_wind_speed(value, from_unit, to_unit) if value_type == 'wind_speed' else \
                   mm.convert_distance(value, from_unit, to_unit)
        
        return converted
    except Exception as e:
        print(f"Unit conversion failed: {e}")
        return None

# Usage
try:
    pressure_pa = safe_unit_conversion(1013.25, 'hPa', 'Pa', 'pressure')
except ValueError as e:
    print(f"Unit conversion error: {e}")
```

### Computation Issues

#### Issue: Numerical Instability
**Problem**: Calculations produce NaN or infinite values.

**Solution**: Add numerical stability checks:
```python
import numpy as np
import monet_meteo as mm

def stable_potential_temperature(pressure, temperature):
    """Calculate potential temperature with numerical stability checks"""
    # Add small values to avoid division by zero
    pressure = np.where(pressure <= 0, 1e-10, pressure)
    temperature = np.where(temperature <= 0, 1e-10, temperature)
    
    try:
        result = mm.potential_temperature(pressure, temperature)
        
        # Check for invalid results
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            raise ValueError("Calculation produced NaN or infinite values")
        
        return result
    except Exception as e:
        print(f"Potential temperature calculation failed: {e}")
        return np.full_like(temperature, np.nan)

# Usage
try:
    theta = stable_potential_temperature(pressure_data, temperature_data)
except Exception as e:
    print(f"Calculation error: {e}")
```

#### Issue: Memory Errors
**Problem**: Processing large datasets causes memory errors.

**Solution**: Process data in chunks:
```python
import numpy as np
import monet_meteo as mm

def process_large_dataset(dataset, chunk_size=1000):
    """Process large dataset in chunks to avoid memory errors"""
    results = []
    
    for i in range(0, len(dataset), chunk_size):
        chunk = dataset.isel(time=slice(i, min(i + chunk_size, len(dataset))))
        
        try:
            # Process chunk
            processed_chunk = mm.xr_interpolate_vertical(
                chunk['temperature'], 'pressure', chunk['pressure'], method='log'
            )
            results.append(processed_chunk)
        except Exception as e:
            print(f"Failed to process chunk {i}: {e}")
            continue
    
    # Combine results
    if results:
        return xr.concat(results, dim='time')
    else:
        raise RuntimeError("No chunks were successfully processed")

# Usage
try:
    result = process_large_dataset(large_dataset, chunk_size=500)
except MemoryError:
    print("Memory error: Try reducing chunk_size")
except Exception as e:
    print(f"Processing error: {e}")
```

#### Issue: Slow Performance
**Problem**: Calculations are slow for large datasets.

**Solution**: Optimize performance:
```python
import time
import numpy as np
import monet_meteo as mm

def optimize_performance():
    """Performance optimization techniques"""
    
    # 1. Use vectorized operations
    pressure = np.array([100000, 85000, 70000])
    temperature = np.array([298.15, 285.15, 273.15])
    
    # Vectorized calculation (fast)
    start_time = time.time()
    result_vectorized = mm.potential_temperature(pressure, temperature)
    vectorized_time = time.time() - start_time
    
    # 2. Pre-allocate arrays
    result_preallocated = np.empty_like(pressure)
    start_time = time.time()
    for i in range(len(pressure)):
        result_preallocated[i] = mm.potential_temperature(pressure[i], temperature[i])
    preallocated_time = time.time() - start_time
    
    print(f"Vectorized: {vectorized_time:.4f}s, Pre-allocated: {preallocated_time:.4f}s")
    
    # 3. Use appropriate data types
    pressure_float32 = pressure.astype(np.float32)
    temperature_float32 = temperature.astype(np.float32)
    
    start_time = time.time()
    result_float32 = mm.potential_temperature(pressure_float32, temperature_float32)
    float32_time = time.time() - start_time
    
    print(f"Float32: {float32_time:.4f}s")
    
    return {
        'vectorized_time': vectorized_time,
        'preallocated_time': preallocated_time,
        'float32_time': float32_time
    }

# Usage
performance_metrics = optimize_performance()
```

### Integration Issues

#### Issue: Xarray Integration Problems
**Problem**: xarray integration fails or produces unexpected results.

**Solution**: Validate xarray datasets:
```python
import xarray as xr
import monet_meteo as mm

def validate_xarray_dataset(dataset):
    """Validate xarray dataset before monet-meteo operations"""
    required_coords = []
    optional_coords = ['pressure', 'level', 'altitude']
    
    # Check basic structure
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Input must be an xarray Dataset")
    
    # Check for required coordinates
    if 'time' in dataset.coords:
        required_coords.extend(['lat', 'lon'])
    
    # Validate coordinates
    for coord in required_coords:
        if coord not in dataset.coords:
            raise ValueError(f"Missing required coordinate: {coord}")
    
    # Check coordinate types
    for coord_name, coord_data in dataset.coords.items():
        if not hasattr(coord_data, 'dtype'):
            raise ValueError(f"Coordinate {coord_name} has invalid data type")
    
    # Validate data variables
    for var_name, var_data in dataset.data_vars.items():
        if not isinstance(var_data, xr.DataArray):
            raise ValueError(f"Variable {var_name} is not a DataArray")
    
    return dataset

# Usage
try:
    validated_dataset = validate_xarray_dataset(dataset)
    result = mm.xr_interpolate_vertical(validated_dataset['temperature'], 'pressure', new_pressure_levels)
except Exception as e:
    print(f"Xarray integration error: {e}")
```

#### Issue: Dask Integration Issues
**Problem**: Dask operations fail or hang.

**Solution**: Optimize dask usage:
```python
import dask.array as da
import monet_meteo as mm

def optimize_dask_operations(data, chunk_size='auto'):
    """Optimize dask operations for meteorological data"""
    
    # Create dask array with appropriate chunking
    if chunk_size == 'auto':
        # Auto-chunk based on data size
        chunk_size = (1000, 1000) if data.ndim == 2 else (1000,)
    
    dask_array = da.from_array(data, chunks=chunk_size)
    
    # Use monet-meteo's dask-optimized functions
    try:
        result = mm.xr_interpolate_with_dask(
            dask_array,
            {'x': old_coords, 'y': old_coords},
            {'x': new_coords, 'y': new_coords},
            method='linear',
            chunks=chunk_size
        )
        return result
    except Exception as e:
        print(f"Dask operation failed: {e}")
        
        # Fallback to numpy
        try:
            import warnings
            warnings.warn("Falling back to numpy processing")
            return data.interp(x=new_coords, y=new_coords)
        except Exception as e2:
            print(f"Numpy fallback also failed: {e2}")
            return None

# Usage
try:
    result = optimize_dask_operations(large_data_array)
except Exception as e:
    print(f"Dask processing error: {e}")
```

## Error Reference

### Common Errors and Their Meanings

#### ValueError
**Description**: Invalid parameter values or data.

**Common Causes**:
- Negative pressure or temperature values
- Invalid unit conversions
- Inconsistent array shapes
- Missing required coordinates

**Solutions**:
```python
# Example solutions
try:
    result = mm.potential_temperature(pressure, temperature)
except ValueError as e:
    if "negative" in str(e).lower():
        # Check for negative values
        print("Error: Input values cannot be negative")
        # Add debugging
        print(f"Pressure min: {np.min(pressure)}, Temperature min: {np.min(temperature)}")
    elif "shape" in str(e).lower():
        # Check array shapes
        print(f"Pressure shape: {pressure.shape}, Temperature shape: {temperature.shape}")
    else:
        print(f"ValueError: {e}")
```

#### TypeError
**Description**: Incorrect data types or parameter types.

**Common Causes**:
- Passing string values instead of numeric arrays
- Incorrect xarray DataArray structure
- Missing required parameters

**Solutions**:
```python
# Example solutions
try:
    result = mm.convert_pressure("1013.25", "hPa", "Pa")
except TypeError as e:
    print(f"TypeError: {e}")
    # Convert string to numeric
    pressure_value = float("1013.25")
    result = mm.convert_pressure(pressure_value, "hPa", "Pa")
```

#### MemoryError
**Description**: Insufficient memory for processing.

**Common Causes**:
- Processing very large datasets
- Inefficient data structures
- Missing chunking for large arrays

**Solutions**:
```python
# Example solutions
try:
    process_entire_dataset(large_dataset)
except MemoryError:
    print("MemoryError: Processing dataset in chunks")
    # Process in chunks
    chunk_results = []
    for chunk in np.array_split(large_dataset, 10):
        try:
            result = process_chunk(chunk)
            chunk_results.append(result)
        except Exception as e:
            print(f"Chunk processing failed: {e}")
    
    final_result = combine_results(chunk_results)
```

#### FileNotFoundError
**Description**: Input file not found or path issues.

**Common Causes**:
- Incorrect file paths
- File permissions issues
- Network file system problems

**Solutions**:
```python
# Example solutions
import os

def safe_file_load(filepath):
    """Safely load files with error handling"""
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not os.access(filepath, os.R_OK):
            raise PermissionError(f"Read permission denied: {filepath}")
        
        return mm.load_netcdf_dataset(filepath)
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
        # Check current directory
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        return None
    except PermissionError as e:
        print(f"Permission error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading file: {e}")
        return None

# Usage
dataset = safe_file_load('weather_data.nc')
```

### Error Handling Patterns

#### Comprehensive Error Handling
```python
import monet_meteo as mm
import numpy as np

def robust_atmospheric_calculations(pressure, temperature, vapor_pressure=None):
    """
    Robust atmospheric calculations with comprehensive error handling
    """
    result = {}
    errors = []
    
    try:
        # Validate inputs
        if not isinstance(pressure, (np.ndarray, list)):
            raise TypeError("Pressure must be numpy array or list")
        
        if not isinstance(temperature, (np.ndarray, list)):
            raise TypeError("Temperature must be numpy array or list")
        
        # Convert to numpy arrays
        pressure = np.asarray(pressure)
        temperature = np.asarray(temperature)
        
        # Check physical constraints
        if np.any(pressure <= 0):
            raise ValueError("Pressure must be positive")
        
        if np.any(temperature < 0):
            raise ValueError("Temperature in Kelvin cannot be negative")
        
        # Calculate potential temperature
        try:
            result['potential_temperature'] = mm.potential_temperature(pressure, temperature)
        except Exception as e:
            errors.append(f"Potential temperature calculation failed: {e}")
        
        # Calculate mixing ratio if vapor pressure provided
        if vapor_pressure is not None:
            try:
                vapor_pressure = np.asarray(vapor_pressure)
                result['mixing_ratio'] = mm.mixing_ratio(vapor_pressure, pressure)
            except Exception as e:
                errors.append(f"Mixing ratio calculation failed: {e}")
        
        # Calculate relative humidity if possible
        if 'mixing_ratio' in result:
            try:
                result['relative_humidity'] = mm.relative_humidity(temperature, temperature - 20)  # Simplified
            except Exception as e:
                errors.append(f"Relative humidity calculation failed: {e}")
        
        # Calculate altitude
        try:
            result['altitude'] = mm.pressure_to_altitude(pressure)
        except Exception as e:
            errors.append(f"Altitude calculation failed: {e}")
        
    except Exception as e:
        errors.append(f"Input validation failed: {e}")
    
    return {
        'results': result,
        'errors': errors,
        'success': len(errors) == 0
    }

# Usage
calculation_result = robust_atmospheric_calculations(pressure_data, temperature_data)
if calculation_result['success']:
    print("Calculations completed successfully")
else:
    print("Calculations completed with errors:")
    for error in calculation_result['errors']:
        print(f"  - {error}")
```

## Best Practices

### Code Organization

#### Modular Design
```python
# weather_processing.py
import monet_meteo as mm
import numpy as np
import xarray as xr
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherDataProcessor:
    """Weather data processing class"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logger
    
    def load_data(self, filepath):
        """Load weather data with error handling"""
        try:
            dataset = mm.load_netcdf_dataset(filepath)
            self.logger.info(f"Successfully loaded data from {filepath}")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    
    def validate_data(self, dataset):
        """Validate dataset structure and content"""
        required_vars = ['pressure', 'temperature']
        
        for var in required_vars:
            if var not in dataset:
                raise ValueError(f"Missing required variable: {var}")
        
        # Validate coordinate system
        try:
            mm.validate_coordinate_system(dataset)
        except ValueError as e:
            self.logger.warning(f"Coordinate system validation failed: {e}")
        
        return dataset
    
    def process_atmospheric_profile(self, dataset):
        """Process atmospheric profile data"""
        results = {}
        
        # Calculate derived quantities
        try:
            # Potential temperature
            results['potential_temperature'] = mm.potential_temperature(
                dataset['pressure'], dataset['temperature']
            )
            
            # Mixing ratio (if humidity data available)
            if 'humidity' in dataset:
                results['mixing_ratio'] = mm.mixing_ratio(
                    dataset['humidity'], dataset['pressure']
                )
            
            # Altitude
            results['altitude'] = mm.pressure_to_altitude(dataset['pressure'])
            
        except Exception as e:
            self.logger.error(f"Profile processing failed: {e}")
            raise
        
        return results
    
    def save_results(self, results, output_path):
        """Save processing results"""
        try:
            # Convert results to dataset
            dataset = xr.Dataset(results)
            
            # Save with monet-meteo
            mm.save_netcdf_dataset(dataset, output_path)
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise

# Usage
processor = WeatherDataProcessor()
dataset = processor.load_data('weather_data.nc')
validated_data = processor.validate_data(dataset)
results = processor.process_atmospheric_profile(validated_data)
processor.save_results(results, 'processed_results.nc')
```

#### Configuration Management
```python
# config.py
import yaml
from pathlib import Path

class WeatherProcessingConfig:
    """Configuration class for weather processing"""
    
    def __init__(self, config_path=None):
        self.config_path = config_path or 'config.yaml'
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            # Return default configuration
            return self.get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'processing': {
                'chunk_size': 1000,
                'interpolation_method': 'linear',
                'vertical_coordinate': 'pressure'
            },
            'units': {
                'pressure': 'Pa',
                'temperature': 'K',
                'distance': 'm',
                'wind_speed': 'm/s'
            },
            'output': {
                'format': 'netcdf',
                'compression': True,
                'chunks': None
            },
            'logging': {
                'level': 'INFO',
                'file': 'weather_processing.log'
            }
        }
    
    def save_config(self, config_path=None):
        """Save configuration to file"""
        path = config_path or self.config_path
        try:
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"Configuration saved to {path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get(self, key, default=None):
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

# Usage
config = WeatherProcessingConfig()
chunk_size = config.get('processing.chunk_size', 1000)
interpolation_method = config.get('processing.interpolation_method', 'linear')
```

### Data Validation

#### Input Data Validation
```python
# data_validation.py
import numpy as np
import xarray as xr
import monet_meteo as mm

class AtmosphericDataValidator:
    """Validator for atmospheric data"""
    
    @staticmethod
    def validate_pressure(pressure):
        """Validate pressure data"""
        if not isinstance(pressure, (np.ndarray, xr.DataArray)):
            raise TypeError("Pressure must be numpy array or xarray DataArray")
        
        pressure = np.asarray(pressure)
        
        # Check for positive values
        if np.any(pressure <= 0):
            negative_indices = np.where(pressure <= 0)[0]
            raise ValueError(f"Found {len(negative_indices)} non-positive pressure values")
        
        # Check for reasonable range
        if np.any(pressure > 110000):  # Above 1100 hPa
            high_values = np.where(pressure > 110000)[0]
            print(f"Warning: Found {len(high_values)} pressure values > 1100 hPa")
        
        if np.any(pressure < 5000):  # Below 50 hPa
            low_values = np.where(pressure < 5000)[0]
            print(f"Warning: Found {len(low_values)} pressure values < 50 hPa")
        
        return pressure
    
    @staticmethod
    def validate_temperature(temperature):
        """Validate temperature data"""
        if not isinstance(temperature, (np.ndarray, xr.DataArray)):
            raise TypeError("Temperature must be numpy array or xarray DataArray")
        
        temperature = np.asarray(temperature)
        
        # Check for negative Kelvin values
        if np.any(temperature < 0):
            negative_values = np.where(temperature < 0)[0]
            raise ValueError(f"Found {len(negative_values)} negative temperature values in Kelvin")
        
        # Check for reasonable range
        if np.any(temperature > 500):  # Above 500K
            high_values = np.where(temperature > 500)[0]
            print(f"Warning: Found {len(high_values)} temperature values > 500K")
        
        if np.any(temperature < 100):  # Below 100K
            low_values = np.where(temperature < 100)[0]
            print(f"Warning: Found {len(low_values)} temperature values < 100K")
        
        return temperature
    
    @staticmethod
    def validate_coordinates(dataset):
        """Validate coordinate system"""
        required_coords = ['lat', 'lon']
        
        for coord in required_coords:
            if coord not in dataset.coords:
                raise ValueError(f"Missing required coordinate: {coord}")
        
        # Validate coordinate ranges
        if 'lat' in dataset.coords:
            lats = dataset['lat'].values
            if np.any(lats < -90) or np.any(lats > 90):
                raise ValueError("Latitude values must be between -90 and 90 degrees")
        
        if 'lon' in dataset.coords:
            lons = dataset['lon'].values
            if np.any(lons < -180) or np.any(lons > 180):
                raise ValueError("Longitude values must be between -180 and 180 degrees")
        
        return dataset
    
    @classmethod
    def validate_dataset(cls, dataset):
        """Complete dataset validation"""
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("Input must be an xarray Dataset")
        
        # Validate coordinates
        dataset = cls.validate_coordinates(dataset)
        
        # Validate variables
        if 'pressure' in dataset:
            dataset['pressure'] = cls.validate_pressure(dataset['pressure'])
        
        if 'temperature' in dataset:
            dataset['temperature'] = cls.validate_temperature(dataset['temperature'])
        
        return dataset

# Usage
try:
    validated_dataset = AtmosphericDataValidator.validate_dataset(raw_dataset)
    print("Dataset validation passed")
except Exception as e:
    print(f"Dataset validation failed: {e}")
```

#### Data Quality Checks
```python
# data_quality.py
import numpy as np
import xarray as xr

class DataQualityChecker:
    """Check data quality and identify issues"""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.issues = []
    
    def check_missing_values(self):
        """Check for missing values"""
        issues = []
        
        for var_name, var_data in self.dataset.data_vars.items():
            if np.isnan(var_data.values).any():
                nan_count = np.isnan(var_data.values).sum()
                total_count = var_data.size
                nan_percentage = (nan_count / total_count) * 100
                
                issues.append({
                    'variable': var_name,
                    'issue': 'missing_values',
                    'count': int(nan_count),
                    'percentage': float(nan_percentage),
                    'severity': 'high' if nan_percentage > 10 else 'medium' if nan_percentage > 1 else 'low'
                })
        
        return issues
    
    def check_outliers(self, z_threshold=3):
        """Check for outlier values"""
        issues = []
        
        for var_name, var_data in self.dataset.data_vars.items():
            if var_name in ['lat', 'lon', 'time']:
                continue  # Skip coordinate variables
            
            # Calculate z-scores
            values = var_data.values.flatten()
            mean_val = np.nanmean(values)
            std_val = np.nanstd(values)
            
            if std_val > 0:  # Avoid division by zero
                z_scores = np.abs((values - mean_val) / std_val)
                outliers = np.where(z_scores > z_threshold)[0]
                
                if len(outliers) > 0:
                    outlier_values = values[outliers]
                    issues.append({
                        'variable': var_name,
                        'issue': 'outliers',
                        'count': len(outliers),
                        'values': outlier_values[:10].tolist(),  # Show first 10 outliers
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'z_threshold': z_threshold,
                        'severity': 'medium'
                    })
        
        return issues
    
    def check_consistency(self):
        """Check data consistency"""
        issues = []
        
        # Check pressure-temperature consistency
        if 'pressure' in self.dataset and 'temperature' in self.dataset:
            pressure = self.dataset['pressure'].values
            temperature = self.dataset['temperature'].values
            
            # Check for negative temperature values
            if np.any(temperature[~np.isnan(temperature)] < 0):
                issues.append({
                    'variable': 'temperature',
                    'issue': 'negative_kelvin',
                    'severity': 'high',
                    'message': 'Found negative temperature values in Kelvin'
                })
            
            # Check for unrealistic pressure-temperature relationships
            # (very rough check for potential data errors)
            valid_indices = ~(np.isnan(pressure) | np.isnan(temperature))
            valid_pressure = pressure[valid_indices]
            valid_temperature = temperature[valid_indices]
            
            # Check for extreme pressure variations with constant temperature
            if len(valid_pressure) > 10:
                pressure_std = np.std(valid_pressure)
                temp_std = np.std(valid_temperature)
                
                if pressure_std > 10000 and temp_std < 1:  # Large pressure change, small temp change
                    issues.append({
                        'variable': 'pressure-temperature',
                        'issue': 'inconsistent_variation',
                        'severity': 'medium',
                        'message': 'Large pressure variations with minimal temperature changes'
                    })
        
        return issues
    
    def run_checks(self):
        """Run all data quality checks"""
        all_issues = []
        
        # Run individual checks
        all_issues.extend(self.check_missing_values())
        all_issues.extend(self.check_outliers())
        all_issues.extend(self.check_consistency())
        
        # Sort issues by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_issues.sort(key=lambda x: severity_order.get(x.get('severity', 'medium'), 1))
        
        self.issues = all_issues
        return all_issues
    
    def generate_report(self):
        """Generate data quality report"""
        if not self.issues:
            return "Data quality check passed - no issues found"
        
        report = "Data Quality Report\n"
        report += "=" * 50 + "\n\n"
        
        # Group issues by severity
        by_severity = {'high': [], 'medium': [], 'low': []}
        for issue in self.issues:
            by_severity[issue.get('severity', 'medium')].append(issue)
        
        for severity, issues in by_severity.items():
            if issues:
                report += f"{severity.upper()} SEVERITY ISSUES:\n"
                report += "-" * 30 + "\n"
                for issue in issues:
                    report += f"Variable: {issue.get('variable', 'N/A')}\n"
                    report += f"Issue: {issue.get('issue', 'N/A')}\n"
                    report += f"Details: {issue.get('message', str(issue))}\n"
                    report += "\n"
        
        return report

# Usage
checker = DataQualityChecker(dataset)
issues = checker.run_checks()
report = checker.generate_report()
print(report)
```

## Performance Optimization

### Memory Management

#### Efficient Data Types
```python
import numpy as np
import monet_meteo as mm

def optimize_memory_usage(data_dict):
    """Optimize memory usage by using efficient data types"""
    optimized_data = {}
    
    for key, data in data_dict.items():
        if isinstance(data, np.ndarray):
            # Use float32 instead of float64 when precision allows
            if data.dtype == np.float64 and np.all(np.isfinite(data)):
                # Check if the data precision loss is acceptable
                original_range = np.max(data) - np.min(data)
                data_float32 = data.astype(np.float32)
                float32_range = np.max(data_float32) - np.min(data_float32)
                
                if abs(original_range - float32_range) / original_range < 0.001:  # < 0.1% difference
                    optimized_data[key] = data_float32
                    continue
            
            # Use int32 instead of int64 when appropriate
            if data.dtype == np.int64 and np.max(data) < 2**31-1:
                optimized_data[key] = data.astype(np.int32)
                continue
        
        optimized_data[key] = data
    
    return optimized_data

# Usage
original_data = {
    'temperature': np.random.rand(1000, 1000) * 50 + 250,  # ~250-300K
    'pressure': np.random.rand(1000, 1000) * 50000 + 50000,  # ~50-100 kPa
    'humidity': np.random.rand(1000, 1000) * 100  # 0-100%
}

optimized_data = optimize_memory_usage(original_data)
print(f"Original memory: {sum(arr.nbytes for arr in original_data.values()) / 1024**2:.1f} MB")
print(f"Optimized memory: {sum(arr.nbytes for arr in optimized_data.values()) / 1024**2:.1f} MB")
```

#### Chunked Processing
```python
import monet_meteo as mm
import xarray as xr
import numpy as np

def process_large_dataset_chunked(dataset, chunk_size=1000):
    """Process large dataset in chunks to manage memory"""
    results = []
    
    # Split dataset into chunks
    n_chunks = int(np.ceil(len(dataset.time) / chunk_size))
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(dataset.time))
        
        chunk = dataset.isel(time=slice(start_idx, end_idx))
        
        try:
            # Process chunk
            processed_chunk = {}
            
            # Interpolate to pressure levels
            if 'pressure' in chunk:
                pressure_levels = np.linspace(chunk.pressure.min(), chunk.pressure.max(), 10)
                for var in ['temperature', 'u_wind', 'v_wind']:
                    if var in chunk:
                        processed_chunk[var] = mm.xr_interpolate_vertical(
                            chunk[var], 'pressure', pressure_levels, method='log'
                        )
            
            results.append(processed_chunk)
            
        except Exception as e:
            print(f"Failed to process chunk {i}: {e}")
            continue
    
    # Combine results
    if results:
        # Create final dataset
        final_dataset = xr.Dataset()
        for var_name in results[0].keys():
            final_dataset[var_name] = xr.concat(
                [chunk[var_name] for chunk in results if var_name in chunk],
                dim='time'
            )
        
        return final_dataset
    else:
        raise RuntimeError("No chunks were successfully processed")

# Usage
try:
    result = process_large_dataset_chunked(large_dataset, chunk_size=500)
except Exception as e:
    print(f"Chunked processing failed: {e}")
```

### Computational Efficiency

#### Vectorized Operations
```python
import numpy as np
import monet_meteo as mm
import time

def compare_computation_methods():
    """Compare vectorized vs. loop-based computation"""
    
    # Generate test data
    n_profiles = 1000
    n_levels = 50
    
    pressure = np.random.rand(n_profiles, n_levels) * 50000 + 50000  # 50-100 kPa
    temperature = np.random.rand(n_profiles, n_levels) * 30 + 250  # 250-280K
    
    # Method 1: Vectorized (recommended)
    start_time = time.time()
    potential_temps_vectorized = mm.potential_temperature(pressure, temperature)
    vectorized_time = time.time() - start_time
    
    # Method 2: Loop-based (slow)
    start_time = time.time()
    potential_temps_looped = np.empty_like(pressure)
    for i in range(n_profiles):
        for j in range(n_levels):
            potential_temps_looped[i, j] = mm.potential_temperature(pressure[i, j], temperature[i, j])
    looped_time = time.time() - start_time
    
    # Method 3: List comprehension (better than nested loops)
    start_time = time.time()
    potential_temps_list = np.array([
        [mm.potential_temperature(pressure[i, j], temperature[i, j]) for j in range(n_levels)]
        for i in range(n_profiles)
    ])
    list_time = time.time() - start_time
    
    # Verify all methods give the same result
    vectorized_looped_diff = np.max(np.abs(potential_temps_vectorized - potential_temps_looped))
    vectorized_list_diff = np.max(np.abs(potential_temps_vectorized - potential_temps_list))
    
    print(f"Vectorized time: {vectorized_time:.4f}s")
    print(f"Looped time: {looped_time:.4f}s (slower by {looped_time/vectorized_time:.1f}x)")
    print(f"List comprehension time: {list_time:.4f}s (slower by {list_time/vectorized_time:.1f}x)")
    print(f"Vectorized vs looped max diff: {vectorized_looped_diff:.2e}")
    print(f"Vectorized vs list max diff: {vectorized_list_diff:.2e}")
    
    return {
        'vectorized_time': vectorized_time,
        'looped_time': looped_time,
        'list_time': list_time,
        'speedup_ratio': looped_time / vectorized_time
    }

# Usage
performance_comparison = compare_computation_methods()
```

#### Parallel Processing
```python
import monet_meteo as mm
from multiprocessing import Pool, cpu_count
import numpy as np

def process_single_profile(args):
    """Process a single atmospheric profile"""
    pressure, temperature, vapor_pressure = args
    
    try:
        # Calculate derived quantities
        potential_temp = mm.potential_temperature(pressure, temperature)
        
        if vapor_pressure is not None:
            mixing_ratio = mm.mixing_ratio(vapor_pressure, pressure)
            relative_humidity = mm.relative_humidity(temperature, temperature - 10)  # Simplified
            altitude = mm.pressure_to_altitude(pressure)
            
            return {
                'potential_temperature': potential_temp,
                'mixing_ratio': mixing_ratio,
                'relative_humidity': relative_humidity,
                'altitude': altitude
            }
        else:
            return {
                'potential_temperature': potential_temp,
                'altitude': mm.pressure_to_altitude(pressure)
            }
    
    except Exception as e:
        print(f"Profile processing failed: {e}")
        return None

def parallel_profile_processing(profiles, n_workers=None):
    """Process multiple profiles in parallel"""
    if n_workers is None:
        n_workers = min(cpu_count(), len(profiles))
    
    # Prepare arguments for multiprocessing
    args_list = []
    for profile in profiles:
        args = (
            profile['pressure'],
            profile['temperature'],
            profile.get('vapor_pressure')
        )
        args_list.append(args)
    
    # Process in parallel
    results = []
    with Pool(processes=n_workers) as pool:
        try:
            results = pool.map(process_single_profile, args_list)
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            results = [process_single_profile(args) for args in args_list]
    
    # Filter out failed results
    successful_results = [r for r in results if r is not None]
    
    return successful_results

# Usage
# Example: Generate synthetic profiles
n_profiles = 1000
profiles = []
for _ in range(n_profiles):
    pressure = np.random.rand(50) * 50000 + 50000  # 50-100 kPa
    temperature = np.random.rand(50) * 30 + 250  # 250-280K
    vapor_pressure = np.random.rand(50) * 2000 + 500  # 0.5-2.5 kPa
    
    profiles.append({
        'pressure': pressure,
        'temperature': temperature,
        'vapor_pressure': vapor_pressure
    })

# Process in parallel
results = parallel_profile_processing(profiles, n_workers=4)
print(f"Successfully processed {len(results)} out of {len(profiles)} profiles")
```

### Algorithm Optimization

#### Caching Expensive Calculations
```python
import monet_meteo as mm
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=128)
def cached_potential_temperature(pressure_tuple, temperature_tuple):
    """Cached version of potential temperature calculation"""
    pressure = np.array(pressure_tuple)
    temperature = np.array(temperature_tuple)
    return mm.potential_temperature(pressure, temperature)

def optimize_with_caching(pressure_data, temperature_data):
    """Optimize by caching expensive calculations"""
    results = []
    
    for pressure, temperature in zip(pressure_data, temperature_data):
        # Convert to tuples for caching (lists are not hashable)
        pressure_tuple = tuple(pressure.flatten())
        temperature_tuple = tuple(temperature.flatten())
        
        # Use cached calculation
        result = cached_potential_temperature(pressure_tuple, temperature_tuple)
        results.append(result)
    
    return results

# Usage
# Generate test data
pressure_data = [np.random.rand(10) * 50000 + 50000 for _ in range(100)]
temperature_data = [np.random.rand(10) * 30 + 250 for _ in range(100)]

# Process with caching
results = optimize_with_caching(pressure_data, temperature_data)

# Check cache efficiency
cache_info = cached_potential_temperature.cache_info()
print(f"Cache hits: {cache_info.hits}")
print(f"Cache misses: {cache_info.misses}")
print(f"Cache efficiency: {cache_info.hits / (cache_info.hits + cache_info.misses) * 100:.1f}%")
```

#### Precomputation
```python
import monet_meteo as mm
import numpy as np
from scipy.interpolate import interp1d

class PrecomputedAtmosphericData:
    """Precompute common atmospheric quantities"""
    
    def __init__(self, pressure_levels):
        self.pressure_levels = np.asarray(pressure_levels)
        self.precomputed_data = {}
    
    def precompute_altitude_conversion(self):
        """Precompute altitude conversion table"""
        # Create lookup table for pressure to altitude conversion
        pressures = np.logspace(1, 5, 1000)  # 10 Pa to 100 kPa
        altitudes = mm.pressure_to_altitude(pressures)
        
        # Create interpolation function
        self.pressure_to_alt_interp = interp1d(
            pressures, altitudes, 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        # Reverse mapping
        self.altitude_to_pressure_interp = interp1d(
            altitudes, pressures,
            bounds_error=False,
            fill_value='extrapolate'
        )
    
    def fast_pressure_to_altitude(self, pressure):
        """Fast pressure to altitude conversion using precomputed table"""
        if not hasattr(self, 'pressure_to_alt_interp'):
            self.precompute_altitude_conversion()
        
        return self.pressure_to_alt_interp(pressure)
    
    def fast_altitude_to_pressure(self, altitude):
        """Fast altitude to pressure conversion using precomputed table"""
        if not hasattr(self, 'altitude_to_pressure_interp'):
            self.precompute_altitude_conversion()
        
        return self.altitude_to_pressure_interp(altitude)
    
    def precompute_pressure_conversion_factors(self):
        """Precompute pressure conversion factors"""
        self.pressure_conversion_cache = {}
        
        for from_unit in ['Pa', 'hPa', 'mb', 'mmHg', 'inHg']:
            for to_unit in ['Pa', 'hPa', 'mb', 'mmHg', 'inHg']:
                if from_unit == to_unit:
                    self.pressure_conversion_cache[(from_unit, to_unit)] = 1.0
                else:
                    # Calculate conversion factor
                    test_value = 1000.0  # Standard atmospheric pressure
                    converted = mm.convert_pressure(test_value, from_unit, to_unit)
                    self.pressure_conversion_cache[(from_unit, to_unit)] = converted / test_value
    
    def fast_pressure_conversion(self, value, from_unit, to_unit):
        """Fast pressure conversion using precomputed factors"""
        if not hasattr(self, 'pressure_conversion_cache'):
            self.precompute_pressure_conversion_factors()
        
        factor = self.pressure_conversion_cache.get((from_unit, to_unit))
        if factor is None:
            # Fallback to original function
            return mm.convert_pressure(value, from_unit, to_unit)
        
        return value * factor

# Usage
# Create precomputed data handler
pressure_levels = np.logspace(1, 5, 100)  # 10 Pa to 100 kPa
handler = PrecomputedAtmosphericData(pressure_levels)

# Fast conversions
pressure_pa = 85000
altitude = handler.fast_pressure_to_altitude(pressure_pa)
print(f"Pressure {pressure_pa} Pa = altitude {altitude:.0f} m")

# Fast unit conversions
pressure_hpa = 850
pressure_pa = handler.fast_pressure_conversion(pressure_hpa, 'hPa', 'Pa')
print(f"Pressure {pressure_hpa} hPa = {pressure_pa} Pa")
```

## Debugging Techniques

### Logging and Debugging

#### Comprehensive Logging
```python
import logging
import sys
from pathlib import Path

def setup_logging(log_level='INFO', log_file=None):
    """Set up comprehensive logging"""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger('monet_meteo_debug')

# Usage
logger = setup_logging('DEBUG', 'weather_processing.log')

# Example usage in functions
def debug_atmospheric_calculations(pressure, temperature):
    """Debug atmospheric calculations with detailed logging"""
    logger.debug("Starting atmospheric calculations")
    logger.debug(f"Input shapes - pressure: {pressure.shape}, temperature: {temperature.shape}")
    logger.debug(f"Input ranges - pressure: [{pressure.min():.1f}, {pressure.max():.1f}], "
                f"temperature: [{temperature.min():.1f}, {temperature.max():.1f}]")
    
    try:
        result = mm.potential_temperature(pressure, temperature)
        logger.debug("Calculation completed successfully")
        logger.debug(f"Result range: [{result.min():.1f}, {result.max():.1f}]")
        return result
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        logger.debug(f"Error details:", exc_info=True)
        raise

# Usage
try:
    result = debug_atmospheric_calculations(pressure_data, temperature_data)
except Exception as e:
    logger.error(f"Final error: {e}")
```

#### Debug Mode for Development
```python
# debug_utils.py
import numpy as np
import monet_meteo as mm
import inspect

class DebugMode:
    """Debug mode for development and troubleshooting"""
    
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.debug_log = []
    
    def log_function_call(self, func, *args, **kwargs):
        """Log function call with inputs and outputs"""
        if not self.enabled:
            return
        
        # Get function name
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        # Log inputs
        arg_values = []
        for i, arg in enumerate(args):
            if isinstance(arg, (np.ndarray, mm.xr.DataArray)):
                arg_values.append(f"arg{i}: shape={arg.shape}, type={type(arg)}")
            else:
                arg_values.append(f"arg{i}: {arg}")
        
        kwarg_values = []
        for k, v in kwargs.items():
            if isinstance(v, (np.ndarray, mm.xr.DataArray)):
                kwarg_values.append(f"{k}: shape={v.shape}, type={type(v)}")
            else:
                kwarg_values.append(f"{k}: {v}")
        
        log_entry = {
            'function': func_name,
            'args': arg_values,
            'kwargs': kwarg_values,
            'timestamp': pd.Timestamp.now()
        }
        
        self.debug_log.append(log_entry)
        print(f"DEBUG: Calling {func_name} with args={arg_values[:3]}..., kwargs={kwarg_values[:3]}...")
    
    def log_calculation_step(self, step_name, input_data, output_data, error=None):
        """Log individual calculation steps"""
        if not self.enabled:
            return
        
        log_entry = {
            'step': step_name,
            'input_shape': input_data.shape if hasattr(input_data, 'shape') else str(type(input_data)),
            'output_shape': output_data.shape if hasattr(output_data, 'shape') else str(type(output_data)),
            'input_range': f"[{np.min(input_data):.2f}, {np.max(input_data):.2f}]" if hasattr(input_data, 'min') else "N/A",
            'output_range': f"[{np.min(output_data):.2f}, {np.max(output_data):.2f}]" if hasattr(output_data, 'min') else "N/A",
            'error': error,
            'timestamp': pd.Timestamp.now()
        }
        
        self.debug_log.append(log_entry)
        
        if error:
            print(f"DEBUG ERROR in {step_name}: {error}")
        else:
            print(f"DEBUG: {step_name} completed - input range: {log_entry['input_range']}, "
                  f"output range: {log_entry['output_range']}")
    
    def generate_debug_report(self):
        """Generate comprehensive debug report"""
        if not self.debug_log:
            return "No debug information available"
        
        report = "Debug Report\n"
        report += "=" * 50 + "\n\n"
        
        for i, entry in enumerate(self.debug_log):
            report += f"Entry {i+1} - {entry['timestamp']}\n"
            report += f"Type: {entry['type']}\n"
            
            if 'function' in entry:
                report += f"Function: {entry['function']}\n"
                report += f"Args: {entry['args']}\n"
                report += f"Kwargs: {entry['kwargs']}\n"
            
            if 'step' in entry:
                report += f"Step: {entry['step']}\n"
                report += f"Input: {entry['input_shape']}\n"
                report += f"Output: {entry['output_shape']}\n"
                report += f"Input Range: {entry['input_range']}\n"
                report += f"Output Range: {entry['output_range']}\n"
            
            if 'error' in entry and entry['error']:
                report += f"ERROR: {entry['error']}\n"
            
            report += "\n"
        
        return report
    
    def save_debug_log(self, filename):
        """Save debug log to file"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.debug_log, f, indent=2, default=str)

# Usage
debugger = DebugMode(enabled=True)

# Use as context manager for function debugging
def debug_function_call(func):
    """Decorator to debug function calls"""
    def wrapper(*args, **kwargs):
        debugger.log_function_call(func, *args, **kwargs)
        
        try:
            result = func(*args, **kwargs)
            debugger.log_calculation_step(func.__name__, args[0] if args else None, result)
            return result
        except Exception as e:
            debugger.log_calculation_step(func.__name__, args[0] if args else None, None, str(e))
            raise
    
    return wrapper

# Example usage
@debug_function_call
def calculate_potential_temperature_debug(pressure, temperature):
    """Debug version of potential temperature calculation"""
    return mm.potential_temperature(pressure, temperature)

# Test with debug mode
try:
    result = calculate_potential_temperature_debug(pressure_data, temperature_data)
    print(debugger.generate_debug_report())
except Exception as e:
    print(f"Error occurred: {e}")
    print(debugger.generate_debug_report())
```

### Profiling and Performance Analysis

#### Function Profiling
```python
import cProfile
import pstats
import io
import monet_meteo as mm
import numpy as np

def profile_atmospheric_calculations():
    """Profile atmospheric calculations to identify bottlenecks"""
    
    # Generate test data
    pressure = np.random.rand(1000, 100) * 50000 + 50000  # 50-100 kPa
    temperature = np.random.rand(1000, 100) * 30 + 250  # 250-280K
    
    # Create profiler
    pr = cProfile.Profile()
    pr.enable()
    
    # Run calculations to profile
    try:
        # Profile potential temperature calculation
        result1 = mm.potential_temperature(pressure, temperature)
        
        # Profile interpolation
        new_pressure_levels = np.linspace(50000, 100000, 50)
        result2 = mm.interpolate_vertical(
            temperature, pressure, new_pressure_levels, method='log'
        )
        
        # Profile unit conversion
        result3 = mm.convert_pressure(pressure, 'Pa', 'hPa')
        
        # Profile coordinate calculations
        result4 = mm.calculate_distance(40.7, -74.0, 41.8, -87.6)
        
    except Exception as e:
        print(f"Profiling error: {e}")
    
    pr.disable()
    
    # Print statistics
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)  # Print top 10 functions
    
    profile_stats = s.getvalue()
    
    # Save to file
    with open('profile_stats.txt', 'w') as f:
        f.write(profile_stats)
    
    print("Profiling completed. Top 10 functions:")
    print(profile_stats)
    
    return profile_stats

# Usage
profile_results = profile_atmospheric_calculations()

# Alternative: Line profiler (if line_profiler is installed)
try:
    from line_profiler import LineProfiler
    
    def profile_lines():
        """Profile specific lines of code"""
        pressure = np.random.rand(100, 50) * 50000 + 50000
        temperature = np.random.rand(100, 50) * 30 + 250
        
        lp = LineProfiler()
        
        # Profile specific function
        lp_wrapper = lp(mm.potential_temperature)
        result = lp_wrapper(pressure, temperature)
        
        lp.print_stats()
    
    profile_lines()
except ImportError:
    print("line_profiler not installed. Install with: pip install line_profiler")
```

#### Memory Profiling
```python
import tracemalloc
import monet_meteo as mm
import numpy as np
import time

def profile_memory_usage():
    """Profile memory usage during atmospheric calculations"""
    
    # Start memory profiling
    tracemalloc.start()
    
    # Generate test data
    print("Generating test data...")
    pressure = np.random.rand(1000, 1000) * 50000 + 50000
    temperature = np.random.rand(1000, 1000) * 30 + 250
    
    # Initial memory snapshot
    snapshot1 = tracemalloc.take_snapshot()
    
    # Perform calculations
    print("Performing atmospheric calculations...")
    start_time = time.time()
    
    try:
        # Memory-intensive operations
        result1 = mm.potential_temperature(pressure, temperature)
        result2 = mm.interpolate_vertical(
            temperature, pressure, np.linspace(50000, 100000, 500), method='log'
        )
        result3 = mm.convert_pressure(pressure, 'Pa', 'hPa')
        
        calculation_time = time.time() - start_time
        print(f"Calculations completed in {calculation_time:.2f} seconds")
        
        # Memory snapshot after calculations
        snapshot2 = tracemalloc.take_snapshot()
        
        # Compare memory usage
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        print("\nMemory usage differences:")
        for stat in top_stats[:10]:  # Top 10 memory differences
            print(stat)
        
        # Total memory usage
        current, peak = tracemalloc.get_traced_memory()
        print(f"\nCurrent memory usage: {current / 1024**2:.1f} MB")
        print(f"Peak memory usage: {peak / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"Memory profiling error: {e}")
    
    tracemalloc.stop()
    
    return {
        'calculation_time': calculation_time,
        'current_memory': current / 1024**2,
        'peak_memory': peak / 1024**2
    }

# Usage
memory_profile = profile_memory_usage()

# Alternative: Memory profiler (if memory_profiler is installed)
try:
    from memory_profiler import profile
    
    @profile
    def memory_intensive_function():
        """Function to profile with memory_profiler"""
        pressure = np.random.rand(500, 500) * 50000 + 50000
        temperature = np.random.rand(500, 500) * 30 + 250
        
        result = mm.potential_temperature(pressure, temperature)
        result2 = mm.interpolate_vertical(
            temperature, pressure, np.linspace(50000, 100000, 200), method='log'
        )
        
        return result, result2
    
    memory_intensive_function()
except ImportError:
    print("memory_profiler not installed. Install with: pip install memory_profiler")
```

## Data Quality Assurance

### Input Data Validation

#### Comprehensive Data Validation
```python
import numpy as np
import xarray as xr
import monet_meteo as mm
import pandas as pd

class AtmosphericDataValidator:
    """Comprehensive validator for atmospheric data"""
    
    def __init__(self, validation_rules=None):
        self.validation_rules = validation_rules or self.get_default_rules()
        self.validation_results = []
    
    def get_default_rules(self):
        """Get default validation rules"""
        return {
            'pressure': {
                'min_value': 100,  # Pa (very low pressure)
                'max_value': 110000,  # Pa (very high pressure)
                'required': True,
                'physical_constraints': ['positive']
            },
            'temperature': {
                'min_value': 50,  # K (very cold)
                'max_value': 400,  # K (very hot)
                'required': True,
                'physical_constraints': ['positive']
            },
            'u_wind': {
                'min_value': -200,  # m/s (extreme)
                'max_value': 200,  # m/s (extreme)
                'required': False
            },
            'v_wind': {
                'min_value': -200,
                'max_value': 200,
                'required': False
            },
            'humidity': {
                'min_value': 0,
                'max_value': 100,  # percentage
                'required': False
            },
            'latitude': {
                'min_value': -90,
                'max_value': 90,
                'required': True
            },
            'longitude': {
                'min_value': -180,
                'max_value': 180,
                'required': True
            }
        }
    
    def validate_data_structure(self, data):
        """Validate basic data structure"""
        if isinstance(data, (np.ndarray, list)):
            return {'type': 'array', 'valid': True}
        elif isinstance(data, xr.Dataset):
            return {'type': 'xarray_dataset', 'valid': True, 'variables': list(data.data_vars.keys())}
        elif isinstance(data, xr.DataArray):
            return {'type': 'xarray_dataarray', 'valid': True, 'shape': data.shape}
        else:
            return {'type': 'unknown', 'valid': False}
    
    def validate_range(self, values, var_name):
        """Validate value ranges"""
        rules = self.validation_rules.get(var_name, {})
        min_val = rules.get('min_value')
        max_val = rules.get('max_value')
        
        issues = []
        
        if min_val is not None and np.any(values < min_val):
            below_min = np.sum(values < min_val)
            issues.append({
                'type': 'below_minimum',
                'count': int(below_min),
                'percentage': float(below_min / len(values) * 100),
                'min_value': min_val,
                'actual_min': float(np.min(values))
            })
        
        if max_val is not None and np.any(values > max_val):
            above_max = np.sum(values > max_val)
            issues.append({
                'type': 'above_maximum',
                'count': int(above_max),
                'percentage': float(above_max / len(values) * 100),
                'max_value': max_val,
                'actual_max': float(np.max(values))
            })
        
        return issues
    
    def validate_physical_constraints(self, values, var_name):
        """Validate physical constraints"""
        rules = self.validation_rules.get(var_name, {})
        constraints = rules.get('physical_constraints', [])
        
        issues = []
        
        for constraint in constraints:
            if constraint == 'positive' and np.any(values <= 0):
                non_positive = np.sum(values <= 0)
                issues.append({
                    'type': 'physical_constraint',
                    'constraint': 'positive',
                    'count': int(non_positive),
                    'percentage': float(non_positive / len(values) * 100)
                })
        
        return issues
    
    def validate_coordinate_consistency(self, data):
        """Validate coordinate consistency"""
        issues = []
        
        if isinstance(data, xr.Dataset):
            # Check for consistent coordinate shapes
            for var_name in data.data_vars:
                if var_name in data.coords:
                    # Variable shouldn't be the same as coordinate
                    issues.append({
                        'type': 'coordinate_inconsistency',
                        'variable': var_name,
                        'issue': f"Variable {var_name} is also a coordinate"
                    })
        
        return issues
    
    def validate_data_completeness(self, data):
        """Validate data completeness"""
        issues = []
        
        if isinstance(data, xr.Dataset):
            # Check for missing required variables
            required_vars = [var for var, rules in self.validation_rules.items() 
                           if rules.get('required', False)]
            
            missing_vars = [var for var in required_vars if var not in data.data_vars]
            
            if missing_vars:
                issues.append({
                    'type': 'missing_required_variables',
                    'variables': missing_vars
                })
            
            # Check for empty data arrays
            for var_name in data.data_vars:
                if data[var_name].size == 0:
                    issues.append({
                        'type': 'empty_data',
                        'variable': var_name
                    })
        
        return issues
    
    def validate_dataset(self, data):
        """Complete dataset validation"""
        results = {
            'data_structure': self.validate_data_structure(data),
            'validation_issues': [],
            'summary': {}
        }
        
        # Validate data completeness
        completeness_issues = self.validate_data_completeness(data)
        results['validation_issues'].extend(completeness_issues)
        
        # Validate coordinate consistency
        coord_issues = self.validate_coordinate_consistency(data)
        results['validation_issues'].extend(coord_issues)
        
        # Validate individual variables
        if isinstance(data, xr.Dataset):
            variables_to_check = list(data.data_vars.keys())
        elif isinstance(data, xr.DataArray):
            variables_to_check = [data.name] if data.name else ['data']
        else:
            variables_to_check = ['values']
        
        for var_name in variables_to_check:
            if isinstance(data, xr.Dataset):
                values = data[var_name].values
            elif isinstance(data, xr.DataArray):
                values = data.values
            else:
                values = np.asarray(data)
            
            # Skip coordinate variables
            if var_name in ['lat', 'lon', 'time', 'pressure', 'level']:
                continue
            
            # Validate ranges
            range_issues = self.validate_range(values, var_name)
            results['validation_issues'].extend([
                {**issue, 'variable': var_name} for issue in range_issues
            ])
            
            # Validate physical constraints
            constraint_issues = self.validate_physical_constraints(values, var_name)
            results['validation_issues'].extend([
                {**issue, 'variable': var_name} for issue in constraint_issues
            ])
        
        # Generate summary
        total_issues = len(results['validation_issues'])
        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        
        for issue in results['validation_issues']:
            severity = issue.get('severity', 'medium')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        results['summary'] = {
            'total_issues': total_issues,
            'severity_counts': severity_counts,
            'validation_passed': total_issues == 0,
            'data_type': results['data_structure']['type']
        }
        
        self.validation_results = results
        return results
    
    def generate_validation_report(self):
        """Generate validation report"""
        if not hasattr(self, 'validation_results'):
            return "No validation results available"
        
        results = self.validation_results
        report = "Data Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        # Data structure summary
        report += f"Data Type: {results['data_structure']['type']}\n"
        if isinstance(results['data_structure'], dict) and 'variables' in results['data_structure']:
            report += f"Variables: {', '.join(results['data_structure']['variables'])}\n"
        report += "\n"
        
        # Validation summary
        summary = results['summary']
        report += f"Total Issues: {summary['total_issues']}\n"
        report += f"Validation Status: {'PASSED' if summary['validation_passed'] else 'FAILED'}\n"
        report += f"Severity Distribution: {summary['severity_counts']}\n\n"
        
        # Detailed issues
        if summary['total_issues'] > 0:
            report += "Detailed Issues:\n"
            report += "-" * 30 + "\n"
            
            for issue in results['validation_issues']:
                report += f"\nVariable: {issue.get('variable', 'N/A')}\n"
                report += f"Type: {issue['type']}\n"
                
                if issue['type'] in ['below_minimum', 'above_maximum']:
                    report += f"Count: {issue['count']} ({issue['percentage']:.1f}%)\n"
                    if 'min_value' in issue:
                        report += f"Minimum allowed: {issue['min_value']}\n"
                        report += f"Actual minimum: {issue['actual_min']}\n"
                    if 'max_value' in issue:
                        report += f"Maximum allowed: {issue['max_value']}\n"
                        report += f"Actual maximum: {issue['actual_max']}\n"
                
                elif issue['type'] == 'physical_constraint':
                    report += f"Constraint: {issue['constraint']}\n"
                    report += f"Violations: {issue['count']} ({issue['percentage']:.1f}%)\n"
                
                elif issue['type'] == 'missing_required_variables':
                    report += f"Missing variables: {issue['variables']}\n"
                
                elif issue['type'] == 'empty_data':
                    report += f"Variable {issue['variable']} contains no data\n"
        
        return report

# Usage
validator = AtmosphericDataValidator()

# Validate dataset
validation_results = validator.validate_dataset(weather_dataset)
print(validator.generate_validation_report())
```

### Data Cleaning and Preprocessing

#### Automated Data Cleaning
```python
import numpy as np
import xarray as xr
import monet_meteo as mm

class AtmosphericDataCleaner:
    """Clean and preprocess atmospheric data"""
    
    def __init__(self):
        self.cleaning_log = []
    
    def remove_outliers_iqr(self, data, factor=1.5):
        """Remove outliers using Interquartile Range method"""
        if isinstance(data, xr.DataArray):
            values = data.values
        else:
            values = np.asarray(data)
        
        # Calculate IQR
        q1 = np.nanpercentile(values, 25)
        q3 = np.nanpercentile(values, 75)
        iqr = q3 - q1
        
        # Define outlier bounds
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Identify outliers
        outliers = (values < lower_bound) | (values > upper_bound)
        outlier_count = np.sum(outliers)
        outlier_percentage = (outlier_count / len(values)) * 100
        
        # Remove outliers
        cleaned_data = np.where(outliers, np.nan, values)
        
        log_entry = {
            'method': 'IQR_outlier_removal',
            'factor': factor,
            'outlier_count': int(outlier_count),
            'outlier_percentage': float(outlier_percentage),
            'original_range': [float(np.nanmin(values)), float(np.nanmax(values))],
            'cleaned_range': [float(np.nanmin(cleaned_data)), float(np.nanmax(cleaned_data))]
        }
        
        self.cleaning_log.append(log_entry)
        
        return cleaned_data
    
    def interpolate_missing_values(self, data, method='linear', max_gap=5):
        """Interpolate missing values with gap limiting"""
        if isinstance(data, xr.DataArray):
            # For xarray DataArrays
            if len(data.dims) == 1:
                # 1D interpolation
                return data.interpolate_na(dim=data.dims[0], method=method, limit=max_gap)
            else:
                # For multi-dimensional, interpolate along first dimension
                return data.interpolate_na(dim=data.dims[0], method=method, limit=max_gap)
        else:
            # For numpy arrays
            data = np.asarray(data)
            if data.ndim == 1:
                # 1D interpolation
                return self._interpolate_1d_numpy(data, method, max_gap)
            else:
                # For multi-dimensional, interpolate along last axis
                return np.apply_along_axis(
                    lambda x: self._interpolate_1d_numpy(x, method, max_gap), 
                    -1, 
                    data
                )
    
    def _interpolate_1d_numpy(self, data, method='linear', max_gap=5):
        """Internal 1D numpy interpolation"""
        # Find non-NaN indices
        valid_indices = np.where(~np.isnan(data))[0]
        
        if len(valid_indices) < 2:
            return data  # Not enough data points to interpolate
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        
        try:
            interp_func = interp1d(
                valid_indices, data[valid_indices],
                kind=method,
                bounds_error=False,
                fill_value=np.nan
            )
            
            # Interpolate
            interpolated = interp_func(np.arange(len(data)))
            
            # Limit interpolation to max_gap
            gaps = np.diff(valid_indices)
            large_gaps = np.where(gaps > max_gap)[0]
            
            if len(large_gaps) > 0:
                # Set values beyond max_gap to NaN
                for gap_pos in large_gaps:
                    start_idx = valid_indices[gap_pos]
                    end_idx = valid_indices[gap_pos + 1]
                    interpolated[start_idx:end_idx+1] = np.nan
            
            return interpolated
        
        except Exception:
            # Fallback to linear interpolation if method fails
            try:
                interp_func = interp1d(
                    valid_indices, data[valid_indices],
                    kind='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                return interp_func(np.arange(len(data)))
            except:
                return data  # Return original if interpolation fails
    
    def smooth_data(self, data, window_size=3, method='moving_average'):
        """Apply smoothing to data"""
        if isinstance(data, xr.DataArray):
            values = data.values
        else:
            values = np.asarray(data)
        
        if values.ndim == 1:
            smoothed = self._smooth_1d(values, window_size, method)
        else:
            smoothed = np.apply_along_axis(
                lambda x: self._smooth_1d(x, window_size, method), 
                -1, 
                values
            )
        
        log_entry = {
            'method': f'smoothing_{method}',
            'window_size': window_size,
            'original_shape': values.shape,
            'smoothed_shape': smoothed.shape
        }
        
        self.cleaning_log.append(log_entry)
        
        return smoothed
    
    def _smooth_1d(self, data, window_size, method):
        """Internal 1D smoothing"""
        if window_size <= 1:
            return data
        
        # Pad the data
        padded = np.pad(data, (window_size//2, window_size//2), mode='edge')
        
        if method == 'moving_average':
            kernel = np.ones(window_size) / window_size
            return np.convolve(padded, kernel, mode='valid')
        elif method == 'gaussian':
            # Simple Gaussian smoothing
            x = np.arange(-window_size//2, window_size//2 + 1)
            kernel = np.exp(-x**2 / (2 * (window_size/3)**2))
            kernel = kernel / np.sum(kernel)
            return np.convolve(padded, kernel, mode='valid')
        else:
            return data
    
    def clean_dataset(self, dataset, cleaning_config=None):
        """Clean entire dataset"""
        if cleaning_config is None:
            cleaning_config = {
                'outlier_removal': {'enabled': True, 'factor': 1.5},
                'interpolation': {'enabled': True, 'method': 'linear', 'max_gap': 5},
                'smoothing': {'enabled': False, 'window_size': 3, 'method': 'moving_average'}
            }
        
        cleaned_data = {}
        
        for var_name in dataset.data_vars:
            original_data = dataset[var_name].values.copy()
            cleaned_values = original_data.copy()
            
            # Apply cleaning steps
            if cleaning_config['outlier_removal']['enabled']:
                cleaned_values = self.remove_outliers_iqr(
                    cleaned_values, 
                    factor=cleaning_config['outlier_removal']['factor']
                )
            
            if cleaning_config['interpolation']['enabled']:
                cleaned_values = self.interpolate_missing_values(
                    cleaned_values,
                    method=cleaning_config['interpolation']['method'],
                    max_gap=cleaning_config['interpolation']['max_gap']
                )
            
            if cleaning_config['smoothing']['enabled']:
                cleaned_values = self.smooth_data(
                    cleaned_values,
                    window_size=cleaning_config['smoothing']['window_size'],
                    method=cleaning_config['smoothing']['method']
                )
            
            cleaned_data[var_name] = (dataset[var_name].dims, cleaned_values)
        
        # Create cleaned dataset
        cleaned_dataset = xr.Dataset(cleaned_data, coords=dataset.coords)
        
        # Copy attributes
        for attr_name, attr_value in dataset.attrs.items():
            cleaned_dataset.attrs[attr_name] = attr_value
        
        return cleaned_dataset
    
    def generate_cleaning_report(self):
        """Generate cleaning report"""
        if not self.cleaning_log:
            return "No cleaning operations performed"
        
        report = "Data Cleaning Report\n"
        report += "=" * 50 + "\n\n"
        
        for i, operation in enumerate(self.cleaning_log):
            report += f"Operation {i+1}: {operation['method']}\n"
            
            if 'outlier_count' in operation:
                report += f"  Outliers removed: {operation['outlier_count']} "
                report += f"({operation['outlier_percentage']:.1f}%)\n"
                report += f"  Original range: [{operation['original_range'][0]:.2f}, "
                report += f"{operation['original_range'][1]:.2f}]\n"
                report += f"  Cleaned range: [{operation['cleaned_range'][0]:.2f}, "
                report += f"{operation['cleaned_range'][1]:.2f}]\n"
            
            elif 'window_size' in operation:
                report += f"  Window size: {operation['window_size']}\n"
                report += f"  Method: {operation['method']}\n"
            
            report += "\n"
        
        return report

# Usage
cleaner = AtmosphericDataCleaner()

# Clean dataset
cleaned_dataset = cleaner.clean_dataset(weather_dataset)
print(cleaner.generate_cleaning_report())
```

## Community Support

### Getting Help

#### Issue Reporting
```python
def create_issue_template():
    """Create a structured issue template"""
    template = {
        'title': "Brief description of the issue",
        'environment': {
            'monet_meteo_version': "monet_meteo.__version__",
            'python_version': "sys.version",
            'operating_system': "platform.system()",
            'dependencies': {
                'numpy': "np.__version__",
                'xarray': "xr.__version__",
                'scipy': "scipy.__version__",
                'pandas': "pd.__version__"
            }
        },
        'problem_description': {
            'expected_behavior': "What you expected to happen",
            'actual_behavior': "What actually happened",
            'steps_to_reproduce': [
                "Step 1: ...",
                "Step 2: ...",
                "Step 3: ..."
            ],
            'code_example': """```python
# Minimal reproducible code example
import monet_meteo as mm
import numpy as np

# Example data
pressure = np.array([...])
temperature = np.array([...])

# Problematic code
result = mm.some_function(pressure, temperature)
```"""
        },
        'error_information': {
            'full_traceback': """Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  ...
ValueError: ...""",
            'error_type': "ValueError",
            'error_message': "Brief error description"
        },
        'data_information': {
            'data_shape': "pressure.shape, temperature.shape",
            'data_range': "pressure.min(), pressure.max(), temperature.min(), temperature.max()",
            'data_sample': "pressure[:5], temperature[:5]"
        },
        'suggested_fix': "Your suggested solution or workaround (if any)"
    }
    
    return template

# Usage
issue_template = create_issue_template()
print("Issue template created. Please fill in the details before reporting.")
```

#### Debug Information Collection
```python
import sys
import platform
import numpy as np
import xarray as xr
import monet_meteo as mm
import pandas as pd
from datetime import datetime

def collect_system_info():
    """Collect system information for bug reports"""
    info = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'python': {
            'version': sys.version,
            'implementation': sys.implementation.name if hasattr(sys, 'implementation') else 'N/A',
            'executable': sys.executable
        },
        'packages': {
            'monet_meteo': mm.__version__,
            'numpy': np.__version__,
            'xarray': xr.__version__,
            'pandas': pd.__version__,
            'scipy': scipy.__version__ if 'scipy' in sys.modules else 'Not installed'
        }
    }
    
    return info

def collect_environment_info():
    """Collect environment information for troubleshooting"""
    info = collect_system_info()
    
    # Add environment variables
    info['environment_variables'] = {
        'PYTHONPATH': sys.path,
        'HOME': platform.uname().pw_dir if hasattr(platform.uname(), 'pw_dir') else 'N/A'
    }
    
    return info

# Usage
system_info = collect_system_info()
print("System Information:")
for category, data in system_info.items():
    print(f"\n{category.upper()}:")
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {data}")
```

### Contributing Guidelines

#### Code Quality Standards
```python
# coding_standards.py
"""
Coding standards for monet-meteo contributions
"""

def python_style_check():
    """Check Python code style compliance"""
    # PEP 8 compliance guidelines:
    # - Use 4 spaces for indentation
    # - Limit lines to 79 characters
    # - Use descriptive variable and function names
    # - Include docstrings for all public functions
    # - Follow naming conventions:
    #   * functions and variables: snake_case
    #   * classes: PascalCase
    #   * constants: UPPER_CASE
    
    style_guidelines = {
        'indentation': '4 spaces',
        'line_length': '79 characters',
        'naming_conventions': {
            'functions': 'snake_case',
            'classes': 'PascalCase',
            'constants': 'UPPER_CASE'
        },
        'documentation': {
            'public_functions': 'docstrings required',
            'classes': 'docstrings required',
            'complex_functions': 'inline comments'
        },
        'imports': {
            'standard_library': 'first',
            'third_party': 'second',
            'local': 'third',
            'sorting': 'alphabetical'
        }
    }
    
    return style_guidelines

def unit_testing_standards():
    """Unit testing standards"""
    testing_standards = {
        'test_naming': 'test_function_name()',
        'test_structure': {
            'setup': 'Arrange test data',
            'execution': 'Call function under test',
            'assertion': 'Check expected results'
        },
        'test_coverage': {
            'public_functions': 'target 90% coverage',
            'edge_cases': 'test boundary conditions',
            'error_handling': 'test exception cases'
        },
        'testing_framework': 'pytest',
        'assertion_helpers': {
            'numerical': 'np.testing.assert_allclose',
            'arrays': 'np.testing.assert_array_equal',
            'types': 'isinstance()'
        }
    }
    
    return testing_standards

def documentation_standards():
    """Documentation standards"""
    doc_standards = {
        'docstring_format': 'Google style',
        'docstring_structure': {
            'one_line_summary': 'Required',
            'description': 'Detailed explanation',
            'parameters': 'Args section',
            'returns': 'Returns section',
            'raises': 'Raises section (if applicable)',
            'examples': '>>> examples section'
        },
        'api_documentation': {
            'modules': 'Document all public modules',
            'functions': 'Document all public functions',
            'classes': 'Document all public classes',
            'constants': 'Document all public constants'
        },
        'examples': {
            'code_examples': 'Include working examples',
            'edge_cases': 'Show common usage patterns',
            'error_handling': 'Demonstrate proper error handling'
        }
    }
    
    return doc_standards

# Usage
style_guidelines = python_style_check()
testing_standards = unit_testing_standards()
doc_standards = documentation_standards()

print("Coding Standards:")
print(f"Style: {style_guidelines}")
print(f"Testing: {testing_standards}")
print(f"Documentation: {doc_standards}")
```

### Resources and Links

#### Helpful Resources
```python
# resources.py
"""
Helpful resources for monet-meteo users and contributors
"""

def get_helpful_resources():
    """Get list of helpful resources"""
    resources = {
        'documentation': {
            'official_docs': 'https://monet-meteo.readthedocs.io/',
            'api_reference': 'https://monet-meteo.readthedocs.io/api/index.html',
            'user_guide': 'https://monet-meteo.readthedocs.io/userguide.html'
        },
        'community': {
            'github_repo': 'https://github.com/noaa-arlab/monet-meteo',
            'issues': 'https://github.com/noaa-arlab/monet-meteo/issues',
            'discussions': 'https://github.com/noaa-arlab/monet-meteo/discussions',
            'stack_overflow': 'https://stackoverflow.com/questions/tagged/monet-meteo'
        },
        'learning': {
            'meteorology_basics': 'https://glossary.ametsoc.org/',
            'numpy_tutorial': 'https://numpy.org/doc/stable/user/quickstart.html',
            'xarray_tutorial': 'https://docs.xarray.dev/en/stable/user-guide/index.html',
            'python_for_science': 'https://scipy-lectures.org/'
        },
        'development': {
            'contributing_guide': 'https://monet-meteo.readthedocs.io/advanced/contributing.html',
            'code_of_conduct': 'https://github.com/noaa-arlab/monet-meteo/blob/main/CODE_OF_CONDUCT.md',
            'license': 'https://github.com/noaa-arlab/monet-meteo/blob/main/LICENSE'
        }
    }
    
    return resources

def get_common_issues():
    """Get list of common issues and solutions"""
    common_issues = [
        {
            'issue': 'ImportError: No module named monet_meteo',
            'cause': 'Package not installed or not in Python path',
            'solution': [
                'Check if package is installed: pip show monet-meteo',
                'Install package: pip install monet-meteo',
                'Check Python path: sys.path'
            ]
        },
        {
            'issue': 'MemoryError when processing large datasets',
            'cause': 'Insufficient memory for large arrays',
            'solution': [
                'Process data in chunks using chunk_size parameter',
                'Use dask arrays for out-of-core computation',
                'Reduce data precision (float32 instead of float64)',
                'Increase available memory or use machine with more RAM'
            ]
        },
        {
            'issue': 'ValueError: Input values must be positive',
            'cause': 'Negative or zero values passed to functions expecting positive values',
            'solution': [
                'Check input data for negative values',
                'Validate data before processing',
                'Handle missing/invalid data appropriately',
                'Use absolute values if applicable'
            ]
        },
        {
            'issue': 'Interpolation produces NaN values',
            'cause': 'Extrapolation beyond data bounds or insufficient data points',
            'solution': [
                'Check bounds of interpolation',
                'Use fill_value parameter for extrapolation',
                'Ensure sufficient data points for interpolation method',
                'Use different interpolation method'
            ]
        }
    ]
    
    return common_issues

# Usage
resources = get_helpful_resources()
common_issues = get_common_issues()

print("Helpful Resources:")
for category, links in resources.items():
    print(f"\n{category.upper()}:")
    for name, url in links.items():
        print(f"  {name}: {url}")

print("\nCommon Issues and Solutions:")
for issue in common_issues:
    print(f"\nIssue: {issue['issue']}")
    print(f"Solution: {issue['solution']}")
```

---

**Final Note**: This troubleshooting and best practices guide provides comprehensive coverage of common issues and optimization strategies for the monet-meteo library. If you encounter issues not covered here, please check the [official documentation](https://monet-meteo.readthedocs.io/) or [submit an issue](https://github.com/noaa-arlab/monet-meteo/issues) with detailed information about your problem.

Remember to:
- Always validate input data before processing
- Use appropriate data types and chunking for large datasets
- Follow the coding standards when contributing
- Provide detailed information when reporting issues
- Leverage the community resources and documentation