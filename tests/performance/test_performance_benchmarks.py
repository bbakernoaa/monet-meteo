"""
Performance benchmarks for atmospheric calculations.

Tests performance characteristics and scalability of meteorological functions
with different array sizes and data types.
"""
import time
import numpy as np
import pytest
import xarray as xr

# Import functions to benchmark
from monet_meteo.thermodynamics.thermodynamic_calculations import (
    potential_temperature,
    virtual_temperature,
    saturation_vapor_pressure,
    mixing_ratio,
    relative_humidity,
    moist_lapse_rate,
    dry_lapse_rate
)
from monet_meteo.derived.derived_calculations import (
    heat_index,
    wind_chill,
    dewpoint_temperature
)
from monet_meteo.dynamics.dynamic_calculations import (
    relative_vorticity,
    divergence,
    geostrophic_wind,
    absolute_vorticity,
    potential_vorticity
)
from monet_meteo.statistical.statistical_calculations import (
    bulk_richardson_number,
    standard_deviation,
    correlation_coefficient
)


class TestPerformanceBenchmarks:
    """Performance benchmarks for meteorological calculations."""
    
    @pytest.mark.parametrize("array_size", [
        (100, 100),      # Small
        (500, 500),      # Medium  
        (1000, 1000),    # Large
        (2000, 2000),    # Very large
    ])
    def test_potential_temperature_performance(self, array_size):
        """Benchmark potential temperature calculation performance."""
        # Create test data
        pressure = np.random.uniform(80000, 105000, array_size)
        temperature = np.random.uniform(250, 320, array_size)
        
        # Time the calculation
        start_time = time.time()
        theta = potential_temperature(pressure=pressure, temperature=temperature)
        end_time = time.time()
        
        computation_time = end_time - start_time
        
        # Verify correctness
        assert theta.shape == array_size
        assert np.all(np.isfinite(theta))
        assert np.all(theta > 0)
        
        # Performance assertions (should complete within reasonable time)
        # Allow up to 5 seconds even for largest arrays
        assert computation_time < 5.0
        
        # Print performance info (pytest will capture this)
        print(f"Potential temperature {array_size}: {computation_time:.4f}s")
    
    @pytest.mark.parametrize("array_size", [
        (50, 100, 100),    # 3D small
        (100, 200, 200),   # 3D medium
    ])
    def test_3d_thermodynamic_performance(self, array_size):
        """Benchmark 3D thermodynamic calculations."""
        # Create 3D test data
        pressure = np.random.uniform(50000, 105000, array_size)
        temperature = np.random.uniform(220, 300, array_size)
        humidity = np.random.uniform(0.1, 0.9, array_size)
        
        # Test multiple functions
        functions = [
            ("saturation_vapor_pressure", lambda: saturation_vapor_pressure(temperature)),
            ("mixing_ratio", lambda: mixing_ratio(
                vapor_pressure=humidity * saturation_vapor_pressure(temperature),
                pressure=pressure
            )),
            ("virtual_temperature", lambda: virtual_temperature(
                temperature=temperature,
                mixing_ratio=humidity * 0.02
            )),
            ("moist_lapse_rate", lambda: moist_lapse_rate(temperature, pressure)),
        ]
        
        for name, func in functions:
            start_time = time.time()
            result = func()
            end_time = time.time()
            
            computation_time = end_time - start_time
            
            # Verify correctness
            assert result.shape == array_size
            assert np.all(np.isfinite(result))
            
            # Performance check
            assert computation_time < 10.0  # Should complete within 10 seconds
            
            print(f"{name} {array_size}: {computation_time:.4f}s")
    
    def test_dry_vs_moist_lapse_rate_performance(self):
        """Compare performance of dry vs moist lapse rate calculations."""
        array_size = (1000, 1000)
        temperature = np.random.uniform(250, 320, array_size)
        pressure = np.random.uniform(80000, 105000, array_size)
        
        # Time dry lapse rate (should be very fast)
        start_time = time.time()
        dry_lr = dry_lapse_rate()
        dry_time = time.time() - start_time
        
        # Time moist lapse rate (should be slower due to calculations)
        start_time = time.time()
        moist_lr = moist_lapse_rate(temperature, pressure)
        moist_time = time.time() - start_time
        
        # Verify results
        assert isinstance(dry_lr, float)
        assert moist_lr.shape == array_size
        assert np.all(np.isfinite(moist_lr))
        
        # Moist lapse rate should take longer than dry
        # (though this might not always be true due to timing precision)
        print(f"Dry lapse rate: {dry_time:.6f}s")
        print(f"Moist lapse rate: {moist_time:.4f}s")
        
        # Both should be fast
        assert dry_time < 0.01
        assert moist_time < 2.0
    
    @pytest.mark.parametrize("grid_size", [100, 200, 500, 1000])
    def test_dynamic_calculations_performance(self, grid_size):
        """Benchmark dynamic meteorological calculations."""
        # Create 2D wind field
        u = np.random.uniform(-20, 20, (grid_size, grid_size))
        v = np.random.uniform(-20, 20, (grid_size, grid_size))
        dx = dy = 10000.0  # 10 km grid spacing
        
        # Test vorticity and divergence
        start_time = time.time()
        zeta = relative_vorticity(u, v, dx, dy)
        div = divergence(u, v, dx, dy)
        vorticity_time = time.time() - start_time
        
        # Verify results
        assert zeta.shape == (grid_size, grid_size)
        assert div.shape == (grid_size, grid_size)
        assert np.all(np.isfinite(zeta))
        assert np.all(np.isfinite(div))
        
        # Performance check
        assert vorticity_time < 5.0
        
        print(f"Vorticity/Divergence {grid_size}x{grid_size}: {vorticity_time:.4f}s")
    
    def test_geostrophic_wind_performance(self):
        """Benchmark geostrophic wind calculation."""
        # Create geopotential height field
        nx, ny = 500, 500
        height = np.random.uniform(9.8e4, 1.02e5, (ny, nx))  # 980-1020 hPa in m^2/s^2
        dx = dy = 20000.0  # 20 km grid spacing
        
        # Latitude array
        latitudes = np.linspace(-90, 90, ny)
        latitude = np.tile(latitudes[:, np.newaxis], (1, nx))
        
        start_time = time.time()
        ug, vg = geostrophic_wind(height, dx, dy, latitude)
        end_time = time.time()
        
        computation_time = end_time - start_time
        
        # Verify results
        assert ug.shape == (ny, nx)
        assert vg.shape == (ny, nx)
        assert np.all(np.isfinite(ug))
        assert np.all(np.isfinite(vg))
        
        # Performance check
        assert computation_time < 5.0
        
        print(f"Geostrophic wind {ny}x{nx}: {computation_time:.4f}s")
    
    @pytest.mark.parametrize("array_size", [
        (1000,),         # 1D large
        (500, 500),      # 2D medium
        (100, 200, 200), # 3D
    ])
    def test_statistical_functions_performance(self, array_size):
        """Benchmark statistical function performance."""
        # Create test data
        data1 = np.random.randn(*array_size)
        data2 = np.random.randn(*array_size)
        
        # Test standard deviation
        start_time = time.time()
        std = standard_deviation(data1)
        std_time = time.time() - start_time
        
        # Test correlation coefficient
        start_time = time.time()
        corr = correlation_coefficient(data1.flatten(), data2.flatten())
        corr_time = time.time() - start_time
        
        # Verify results
        assert np.isfinite(std)
        assert std > 0
        assert np.isfinite(corr)
        assert abs(corr) <= 1
        
        # Performance checks
        assert std_time < 2.0
        assert corr_time < 2.0
        
        print(f"Standard deviation {array_size}: {std_time:.4f}s")
        print(f"Correlation {array_size}: {corr_time:.4f}s")


class TestMemoryEfficiency:
    """Test memory efficiency of calculations."""
    
    def test_large_array_memory_usage(self):
        """Test that large arrays don't cause memory issues."""
        # Create large arrays that approach memory limits
        array_size = (2000, 2000)  # ~32 MB for float64
        pressure = np.random.uniform(80000, 105000, array_size)
        temperature = np.random.uniform(250, 320, array_size)
        
        # These should complete without excessive memory usage
        theta = potential_temperature(pressure=pressure, temperature=temperature)
        assert theta.shape == array_size
        assert np.all(np.isfinite(theta))
        
        # Test multiple operations
       svp = saturation_vapor_pressure(temperature)
        mix_ratio = mixing_ratio(
            vapor_pressure=0.5 * svp,
            pressure=pressure
        )
        t_virtual = virtual_temperature(temperature=temperature, mixing_ratio=mix_ratio)
        
        # All should be finite and reasonable
        assert np.all(np.isfinite(theta))
        assert np.all(np.isfinite(svp))
        assert np.all(np.isfinite(mix_ratio))
        assert np.all(np.isfinite(t_virtual))
    
    def test_3d_array_efficiency(self):
        """Test 3D array processing efficiency."""
        # Create 3D atmospheric profile
        nz, ny, nx = 50, 200, 200  # 2 million points
        pressure = np.random.uniform(50000, 105000, (nz, ny, nx))
        temperature = np.random.uniform(220, 300, (nz, ny, nx))
        
        # Process each level
        start_time = time.time()
        for i in range(nz):
            theta = potential_temperature(
                pressure=pressure[i, :, :],
                temperature=temperature[i, :, :]
            )
            # Just process, don't store all results
            assert theta.shape == (ny, nx)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Should complete in reasonable time
        assert total_time < 30.0  # 30 seconds for all levels
        
        print(f"3D processing {nz} levels of {ny}x{nx}: {total_time:.4f}s")


class TestScalability:
    """Test how functions scale with input size."""
    
    def test_scaling_law_verification(self):
        """Verify that computation time scales appropriately with array size."""
        sizes = [100, 200, 400, 800]
        times = []
        
        for size in sizes:
            pressure = np.random.uniform(80000, 105000, (size, size))
            temperature = np.random.uniform(250, 320, (size, size))
            
            start_time = time.time()
            theta = potential_temperature(pressure=pressure, temperature=temperature)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Time should generally increase with size (though not necessarily linearly)
        # due to cache effects, vectorization, etc.
        assert times[-1] > times[0]  # Largest should take more time than smallest
        
        # But should scale reasonably (not exponentially)
        # Allow up to quadratic scaling: time ~ size^2
        for i in range(1, len(sizes)):
            size_ratio = (sizes[i] / sizes[0]) ** 2
            time_ratio = times[i] / times[0]
            assert time_ratio < size_ratio * 10  # Allow some overhead
        
        print("Scaling test results:")
        for size, t in zip(sizes, times):
            print(f"  {size}x{size}: {t:.4f}s")
    
    def test_parallel_processing_benefit(self):
        """Test if there are benefits from potential parallel processing."""
        # Test with different array shapes to see if there are optimal shapes
        shapes = [
            (1000, 100),   # Tall and narrow
            (100, 1000),   # Short and wide
            (316, 316),    # Approximately square (same total size)
        ]
        
        for shape in shapes:
            pressure = np.random.uniform(80000, 105000, shape)
            temperature = np.random.uniform(250, 320, shape)
            
            start_time = time.time()
            theta = potential_temperature(pressure=pressure, temperature=temperature)
            end_time = time.time()
            
            computation_time = end_time - start_time
            
            # Should complete reasonably fast regardless of shape
            assert computation_time < 3.0
            
            print(f"Shape {shape}: {computation_time:.4f}s")


class TestOptimizationOpportunities:
    """Identify potential optimization opportunities."""
    
    def test_vectorization_effectiveness(self):
        """Test effectiveness of vectorized operations."""
        # Compare different ways of computing the same thing
        size = 1000
        pressure = np.random.uniform(80000, 105000, size)
        temperature = np.random.uniform(250, 320, size)
        
        # Standard vectorized approach
        start_time = time.time()
        theta_vectorized = potential_temperature(pressure=pressure, temperature=temperature)
        vectorized_time = time.time() - start_time
        
        # Simulate loop-based approach (what we're avoiding)
        start_time = time.time()
        theta_looped = np.zeros(size)
        for i in range(size):
            theta_looped[i] = potential_temperature(
                pressure=pressure[i], 
                temperature=temperature[i]
            )
        looped_time = time.time() - start_time
        
        # Vectorized should be faster
        # (This might not always be true for small arrays due to overhead)
        print(f"Vectorized: {vectorized_time:.6f}s")
        print(f"Looped: {looped_time:.6f}s")
        
        # Verify results are the same
        assert np.allclose(theta_vectorized, theta_looped, rtol=1e-10)
    
    def test_function_call_overhead(self):
        """Test function call overhead for repeated operations."""
        size = 100
        pressure = np.random.uniform(80000, 105000, size)
        temperature = np.random.uniform(250, 320, size)
        
        # Test calling function multiple times vs single call
        n_calls = 100
        
        start_time = time.time()
        for _ in range(n_calls):
            theta = potential_temperature(pressure=pressure, temperature=temperature)
        multiple_calls_time = time.time() - start_time
        
        # Test single call with repeated data
        pressure_repeated = np.tile(pressure, n_calls)
        temperature_repeated = np.tile(temperature, n_calls)
        
        start_time = time.time()
        theta_repeated = potential_temperature(
            pressure=pressure_repeated, 
            temperature=temperature_repeated
        )
        single_call_time = time.time() - start_time
        
        print(f"Multiple calls: {multiple_calls_time:.4f}s")
        print(f"Single call: {single_call_time:.4f}s")
        
        # Verify results are equivalent
        theta_reshaped = theta_repeated.reshape(n_calls, size)
        for i in range(n_calls):
            assert np.allclose(theta_reshaped[i], theta, rtol=1e-10)


class TestRealWorldScenarios:
    """Test performance in realistic meteorological scenarios."""
    
    def test_global_model_grid_performance(self):
        """Test performance on typical global model grid sizes."""
        # Typical global model resolutions
        resolutions = [
            ("T42", 42, 84),      # Low resolution
            ("T85", 85, 170),     # Medium resolution  
            ("T213", 213, 426),   # High resolution
        ]
        
        for name, nlat, nlon in resolutions:
            # Create typical atmospheric fields
            pressure = np.random.uniform(80000, 105000, (nlat, nlon))
            temperature = 280.0 + 50.0 * np.random.randn(nlat, nlon)
            humidity = np.random.uniform(0.1, 0.9, (nlat, nlon))
            
            start_time = time.time()
            
            # Typical workflow: calculate multiple diagnostics
            theta = potential_temperature(pressure=pressure, temperature=temperature)
            svp = saturation_vapor_pressure(temperature)
            mix_ratio = mixing_ratio(
                vapor_pressure=humidity * svp,
                pressure=pressure
            )
            t_virtual = virtual_temperature(temperature=temperature, mixing_ratio=mix_ratio)
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            # Should complete reasonably fast for operational use
            if name == "T42":
                assert computation_time < 1.0
            elif name == "T85":
                assert computation_time < 5.0
            elif name == "T213":
                assert computation_time < 20.0
            
            print(f"{name} ({nlat}x{nlon}): {computation_time:.4f}s")
    
    def test_high_frequency_data_processing(self):
        """Test ability to process high-frequency data."""
        # Simulate processing data at different temporal frequencies
        frequencies = [
            ("Hourly", 24),       # Daily
            ("3-hourly", 8),      # 3-hourly
            ("Sub-hourly", 60),   # Minute resolution
        ]
        
        array_size = (500, 500)  # Moderate spatial resolution
        
        for name, n_times in frequencies:
            # Create time series of atmospheric fields
            pressures = np.random.uniform(80000, 105000, (n_times, *array_size))
            temperatures = np.random.uniform(250, 320, (n_times, *array_size))
            
            start_time = time.time()
            
            # Process each time step
            for t in range(n_times):
                theta = potential_temperature(
                    pressure=pressures[t], 
                    temperature=temperatures[t]
                )
                assert np.all(np.isfinite(theta))
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should scale reasonably with time steps
            print(f"{name} ({n_times} time steps): {total_time:.4f}s")
            
            # Time per time step should be reasonable
            time_per_step = total_time / n_times
            assert time_per_step < 0.1  # Less than 100ms per time step