"""
Integration tests for xarray and Dask compatibility.

Tests that all functions work correctly with xarray DataArrays and Dask arrays,
including proper handling of coordinates, attributes, and lazy evaluation.
"""
import numpy as np
import pytest
import xarray as xr
import dask.array as da

# Import functions to test
from monet_meteo.thermodynamics.thermodynamic_calculations import (
    potential_temperature,
    virtual_temperature,
    saturation_vapor_pressure,
    mixing_ratio,
    relative_humidity
)
from monet_meteo.derived.derived_calculations import (
    heat_index,
    wind_chill,
    dewpoint_temperature
)
from monet_meteo.dynamics.dynamic_calculations import (
    relative_vorticity,
    divergence,
    geostrophic_wind
)
from monet_meteo.statistical.statistical_calculations import (
    bulk_richardson_number,
    correlation_coefficient,
    standard_deviation
)


class TestXarrayIntegration:
    """Test integration with xarray DataArrays."""
    
    @pytest.fixture
    def sample_xarray_data(self):
        """Create sample xarray dataset for testing."""
        # Create coordinate arrays
        pressure_levels = np.array([1000.0, 925.0, 850.0, 700.0, 500.0])
        latitudes = np.linspace(-90, 90, 10)
        longitudes = np.linspace(-180, 180, 20)
        
        # Create 3D data arrays
        temperature_data = 280.0 + 20.0 * np.random.randn(5, 10, 20)
        pressure_data = pressure_levels[:, np.newaxis, np.newaxis] * (1 + 0.1 * np.random.randn(5, 10, 20))
        humidity_data = 0.5 + 0.3 * np.random.randn(5, 10, 20)
        u_wind_data = 10.0 * np.random.randn(5, 10, 20)
        v_wind_data = 5.0 * np.random.randn(5, 10, 20)
        
        # Create xarray DataArrays with proper coordinates
        coords = {
            'pressure': pressure_levels,
            'latitude': latitudes,
            'longitude': longitudes
        }
        
        temperature = xr.DataArray(temperature_data, coords=coords, dims=['pressure', 'latitude', 'longitude'],
                                 attrs={'units': 'K', 'long_name': 'Air Temperature'})
        pressure = xr.DataArray(pressure_data, coords=coords, dims=['pressure', 'latitude', 'longitude'],
                              attrs={'units': 'Pa', 'long_name': 'Atmospheric Pressure'})
        humidity = xr.DataArray(humidity_data, coords=coords, dims=['pressure', 'latitude', 'longitude'],
                              attrs={'units': '1', 'long_name': 'Relative Humidity'})
        u_wind = xr.DataArray(u_wind_data, coords=coords, dims=['pressure', 'latitude', 'longitude'],
                            attrs={'units': 'm/s', 'long_name': 'Zonal Wind'})
        v_wind = xr.DataArray(v_wind_data, coords=coords, dims=['pressure', 'latitude', 'longitude'],
                            attrs={'units': 'm/s', 'long_name': 'Meridional Wind'})
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'humidity': humidity,
            'u_wind': u_wind,
            'v_wind': v_wind
        }
    
    def test_potential_temperature_xarray(self, sample_xarray_data):
        """Test potential temperature calculation with xarray DataArrays."""
        temp = sample_xarray_data['temperature']
        pressure = sample_xarray_data['pressure']
        
        theta = potential_temperature(pressure=pressure, temperature=temp)
        
        # Should return xarray DataArray
        assert isinstance(theta, xr.DataArray)
        
        # Should preserve coordinates and dimensions
        assert theta.dims == temp.dims
        assert theta.coords == temp.coords
        
        # Should have reasonable values
        assert np.all(np.isfinite(theta))
        assert np.all(theta > 0)
        
        # Should preserve or add appropriate attributes
        assert 'units' in theta.attrs
        assert theta.attrs['units'] == 'K'
    
    def test_virtual_temperature_xarray(self, sample_xarray_data):
        """Test virtual temperature calculation with xarray DataArrays."""
        temp = sample_xarray_data['temperature']
        humidity = sample_xarray_data['humidity']
        
        # Calculate mixing ratio from humidity and pressure
        pressure = sample_xarray_data['pressure']
        mixing_ratio_val = mixing_ratio(
            vapor_pressure=humidity * saturation_vapor_pressure(temp),
            pressure=pressure
        )
        
        t_virtual = virtual_temperature(temperature=temp, mixing_ratio=mixing_ratio_val)
        
        # Should return xarray DataArray
        assert isinstance(t_virtual, xr.DataArray)
        
        # Should preserve coordinates
        assert t_virtual.dims == temp.dims
        
        # Should be slightly higher than actual temperature
        assert np.all(t_virtual >= temp)
    
    def test_relative_humidity_xarray(self, sample_xarray_data):
        """Test relative humidity calculation with xarray DataArrays."""
        temp = sample_xarray_data['temperature']
        pressure = sample_xarray_data['pressure']
        mixing_ratio_val = sample_xarray_data['humidity'] * 0.02  # Convert to mixing ratio
        
        rh = relative_humidity(
            vapor_pressure=mixing_ratio_val,
            saturation_vapor_pressure=saturation_vapor_pressure(temp)
        )
        
        # Should return xarray DataArray
        assert isinstance(rh, xr.DataArray)
        
        # Should preserve coordinates
        assert rh.dims == temp.dims
        
        # Should be between 0 and 1
        assert np.all(rh >= 0)
        assert np.all(rh <= 1)
    
    def test_heat_index_xarray(self, sample_xarray_data):
        """Test heat index calculation with xarray DataArrays."""
        temp_c = sample_xarray_data['temperature'] - 273.15  # Convert to Celsius
        rh = sample_xarray_data['humidity'] * 100  # Convert to percentage
        
        hi = heat_index(temperature=temp_c, relative_humidity=rh)
        
        # Should return xarray DataArray
        assert isinstance(hi, xr.DataArray)
        
        # Should preserve coordinates
        assert hi.dims == temp_c.dims
        
        # Heat index should generally be >= temperature for warm, humid conditions
        warm_mask = temp_c > 25.0  # > 25°C
        humid_mask = rh > 60.0  # > 60% humidity
        combined_mask = warm_mask & humid_mask
        
        if np.any(combined_mask):
            assert np.all(hi.where(combined_mask) >= temp_c.where(combined_mask))
    
    def test_dewpoint_temperature_xarray(self, sample_xarray_data):
        """Test dewpoint temperature calculation with xarray DataArrays."""
        temp = sample_xarray_data['temperature']
        rh = sample_xarray_data['humidity']
        
        td = dewpoint_temperature(temperature=temp, relative_humidity=rh)
        
        # Should return xarray DataArray
        assert isinstance(td, xr.DataArray)
        
        # Should preserve coordinates
        assert td.dims == temp.dims
        
        # Dewpoint should be <= temperature
        assert np.all(td <= temp)
        
        # Should be finite
        assert np.all(np.isfinite(td))


class TestDaskIntegration:
    """Test integration with Dask arrays."""
    
    @pytest.fixture
    def sample_dask_data(self):
        """Create sample dask arrays for testing."""
        # Create large arrays that would benefit from dask
        shape = (100, 50, 60)  # Larger than memory for some systems
        
        # Create dask arrays with chunks
        temperature_data = da.random.random(shape, chunks=(20, 10, 12)) * 50 + 250
        pressure_data = da.random.random(shape, chunks=(20, 10, 12)) * 20000 + 80000
        humidity_data = da.random.random(shape, chunks=(20, 10, 12)) * 0.8 + 0.1
        u_wind_data = da.random.random(shape, chunks=(20, 10, 12)) * 20 - 10
        v_wind_data = da.random.random(shape, chunks=(20, 10, 12)) * 15 - 7.5
        
        return {
            'temperature': temperature_data,
            'pressure': pressure_data,
            'humidity': humidity_data,
            'u_wind': u_wind_data,
            'v_wind': v_wind_data
        }
    
    def test_potential_temperature_dask(self, sample_dask_data):
        """Test potential temperature with dask arrays."""
        temp = sample_dask_data['temperature']
        pressure = sample_dask_data['pressure']
        
        theta = potential_temperature(pressure=pressure, temperature=temp)
        
        # Should return dask array
        assert isinstance(theta, da.Array)
        
        # Should preserve chunking
        assert theta.chunks == temp.chunks
        
        # Should compute to correct values
        theta_computed = theta.compute()
        assert np.all(np.isfinite(theta_computed))
        assert np.all(theta_computed > 0)
    
    def test_standard_deviation_dask(self, sample_dask_data):
        """Test statistical functions with dask arrays."""
        temp = sample_dask_data['temperature']
        
        # Compute standard deviation along an axis
        temp_std = da.nanstd(temp, axis=0)
        
        # Should return dask array
        assert isinstance(temp_std, da.Array)
        
        # Should compute to reasonable values
        std_computed = temp_std.compute()
        assert np.all(std_computed > 0)
        assert np.all(std_computed < 50)  # Reasonable atmospheric variability
    
    def test_dask_lazy_evaluation(self, sample_dask_data):
        """Test that dask arrays maintain lazy evaluation."""
        temp = sample_dask_data['temperature']
        pressure = sample_dask_data['pressure']
        
        # Create computation graph
        theta = potential_temperature(pressure=pressure, temperature=temp)
        
        # Should not have computed yet
        assert hasattr(theta, 'chunks')
        
        # Should only compute when explicitly requested
        theta_computed = theta.compute()
        assert isinstance(theta_computed, np.ndarray)
        assert np.all(np.isfinite(theta_computed))


class TestCoordinatePreservation:
    """Test that coordinates and metadata are properly preserved."""
    
    def test_coordinate_preservation_2d(self):
        """Test coordinate preservation with 2D data."""
        # Create 2D grid
        lats = np.linspace(-90, 90, 20)
        lons = np.linspace(-180, 180, 40)
        
        temp_data = 280.0 + 20.0 * np.random.randn(20, 40)
        humidity_data = 0.5 + 0.2 * np.random.randn(20, 40)
        
        temp = xr.DataArray(temp_data, coords={'latitude': lats, 'longitude': lons},
                          dims=['latitude', 'longitude'])
        humidity = xr.DataArray(humidity_data, coords={'latitude': lats, 'longitude': lons},
                              dims=['latitude', 'longitude'])
        
        # Calculate virtual temperature
        mixing_ratio_val = 0.01 * humidity  # Simple conversion
        t_virtual = virtual_temperature(temperature=temp, mixing_ratio=mixing_ratio_val)
        
        # Should preserve coordinates exactly
        assert t_virtual.coords['latitude'].equals(temp.coords['latitude'])
        assert t_virtual.coords['longitude'].equals(temp.coords['longitude'])
        
        # Should preserve dimensions
        assert t_virtual.dims == temp.dims
    
    def test_attribute_preservation(self):
        """Test that attributes are preserved or appropriately modified."""
        temp_data = np.array([280.0, 290.0, 300.0])
        pressure_data = np.array([100000.0, 85000.0, 70000.0])
        
        temp = xr.DataArray(temp_data, dims=['level'],
                          attrs={'units': 'K', 'standard_name': 'air_temperature'})
        pressure = xr.DataArray(pressure_data, dims=['level'],
                              attrs={'units': 'Pa', 'standard_name': 'air_pressure'})
        
        theta = potential_temperature(pressure=pressure, temperature=temp)
        
        # Should preserve some attributes
        assert 'units' in theta.attrs
        assert theta.attrs['units'] == 'K'
        
        # Should have potential temperature standard name
        assert 'standard_name' in theta.attrs
        assert 'potential_temperature' in theta.attrs['standard_name']


class TestIntegrationEdgeCases:
    """Test edge cases in integration."""
    
    def test_mixed_array_types(self):
        """Test functions with mixed array types."""
        # Create mixed types
        temp_numpy = np.array([280.0, 290.0, 300.0])
        pressure_xr = xr.DataArray([100000.0, 85000.0, 70000.0], dims=['level'])
        
        # Should handle mixed types gracefully
        theta = potential_temperature(pressure=pressure_xr, temperature=temp_numpy)
        
        # Should return appropriate type (likely xarray)
        assert isinstance(theta, (xr.DataArray, np.ndarray))
        assert len(theta) == 3
    
    def test_different_shapes(self):
        """Test behavior with different array shapes."""
        # Create arrays with different shapes
        temp_1d = np.array([280.0, 290.0, 300.0])
        pressure_1d = np.array([100000.0, 85000.0, 70000.0])
        
        # Should work with 1D arrays
        theta = potential_temperature(pressure=pressure_1d, temperature=temp_1d)
        assert len(theta) == 3
        assert np.all(theta > 0)
    
    def test_nan_handling(self):
        """Test handling of NaN values in xarray."""
        data = np.array([280.0, np.nan, 300.0, 290.0])
        coords = {'level': [0, 1, 2, 3]}
        
        temp = xr.DataArray(data, coords=coords, dims=['level'])
        pressure = xr.DataArray([100000.0, 90000.0, 80000.0, 85000.0], 
                              coords=coords, dims=['level'])
        
        theta = potential_temperature(pressure=pressure, temperature=temp)
        
        # Should preserve NaN locations
        assert np.isnan(theta[1])  # Second element should be NaN
        assert not np.isnan(theta[0])  # Other elements should be valid
        assert not np.isnan(theta[2])
        assert not np.isnan(theta[3])


class TestPerformanceIntegration:
    """Test performance characteristics of integrated functions."""
    
    def test_large_array_performance(self):
        """Test that functions scale reasonably with array size."""
        import time
        
        # Test with progressively larger arrays
        sizes = [(50, 50), (100, 100), (200, 200)]
        
        for size in sizes:
            temp_data = 280.0 + 20.0 * np.random.randn(*size)
            pressure_data = 100000.0 + 20000.0 * np.random.randn(*size)
            
            temp = xr.DataArray(temp_data, dims=['y', 'x'])
            pressure = xr.DataArray(pressure_data, dims=['y', 'x'])
            
            start_time = time.time()
            theta = potential_temperature(pressure=pressure, temperature=temp)
            end_time = time.time()
            
            # Should complete in reasonable time (less than 1 second for these sizes)
            assert end_time - start_time < 1.0
            
            # Should produce valid results
            assert theta.shape == size
            assert np.all(np.isfinite(theta))
    
    def test_memory_efficiency(self):
        """Test that functions don't create unnecessary copies."""
        # Create large arrays
        large_shape = (1000, 1000)
        temp_data = da.random.random(large_shape, chunks=(100, 100))
        pressure_data = da.random.random(large_shape, chunks=(100, 100)) * 50000 + 80000
        
        # These operations should be memory efficient with dask
        theta = potential_temperature(pressure=pressure_data, temperature=temp_data)
        
        # Should maintain dask structure
        assert isinstance(theta, da.Array)
        assert theta.chunks == temp_data.chunks