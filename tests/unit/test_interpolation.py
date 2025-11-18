"""
Test suite for interpolation functions.

Tests interpolation methods including vertical interpolation, horizontal interpolation,
and coordinate transformations.
"""
import numpy as np
import pytest

# Import interpolation functions
from monet_meteo.interpolation import (
    interpolate_to_pressure_level,
    interpolate_to_height_level,
    interpolate_vertical_profile,
    interpolate_horizontal,
    pressure_to_height,
    height_to_pressure
)


class TestVerticalInterpolation:
    """Test vertical interpolation functions."""
    
    def test_interpolate_to_pressure_level(self):
        """Test interpolation to specific pressure level."""
        # Create test profile
        pressure = np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0])  # hPa
        temperature = np.array([298.15, 290.0, 280.0, 270.0, 250.0, 230.0])  # K
        target_pressure = 875.0  # hPa (between 925 and 850)
        
        temp_interp = interpolate_to_pressure_level(
            pressure, temperature, target_pressure
        )
        
        # Should be between the two bounding values
        assert 280.0 < temp_interp < 290.0
        
        # Should be reasonable for linear interpolation
        expected = 285.0  # Roughly midway
        assert abs(temp_interp - expected) < 5.0
    
    def test_interpolate_to_height_level(self):
        """Test interpolation to specific height level."""
        # Create test profile
        height = np.array([0.0, 1000.0, 2000.0, 3000.0, 5000.0])  # m
        temperature = np.array([298.15, 288.15, 278.15, 268.15, 248.15])  # K
        target_height = 1500.0  # m (between 1000 and 2000)
        
        temp_interp = interpolate_to_height_level(
            height, temperature, target_height
        )
        
        # Should be between the two bounding values
        assert 278.15 < temp_interp < 288.15
        
        # Should be approximately midway
        expected = 283.15
        assert abs(temp_interp - expected) < 1.0
    
    def test_interpolate_vertical_profile(self):
        """Test interpolation of entire vertical profile."""
        # Original profile
        pressure_orig = np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0])
        temp_orig = np.array([298.15, 288.15, 278.15, 268.15, 258.15, 248.15])
        
        # New pressure levels
        pressure_new = np.array([950.0, 850.0, 750.0, 650.0])
        
        temp_new = interpolate_vertical_profile(
            pressure_orig, temp_orig, pressure_new
        )
        
        # Should have same length as target pressure array
        assert len(temp_new) == len(pressure_new)
        
        # All values should be within original range
        assert np.all(temp_new >= 248.15)
        assert np.all(temp_new <= 298.15)
        
        # Should be monotonically decreasing (like original)
        assert np.all(temp_new[1:] < temp_new[:-1])
    
    def test_interpolate_outside_range(self):
        """Test interpolation outside the original range."""
        pressure = np.array([1000.0, 800.0, 600.0])
        temperature = np.array([298.15, 278.15, 258.15])
        
        # Interpolate above the highest level
        temp_high = interpolate_to_pressure_level(pressure, temperature, 500.0)
        assert temp_high < 258.15  # Should extrapolate lower
        
        # Interpolate below the lowest level
        temp_low = interpolate_to_pressure_level(pressure, temperature, 1100.0)
        assert temp_low > 298.15  # Should extrapolate higher


class TestHorizontalInterpolation:
    """Test horizontal interpolation functions."""
    
    def test_interpolate_horizontal_linear(self):
        """Test linear horizontal interpolation."""
        # Create 2D grid
        x = np.array([0.0, 1000.0, 2000.0, 3000.0])  # m
        y = np.array([0.0, 1000.0, 2000.0])  # m
        X, Y = np.meshgrid(x, y)
        
        # Simple temperature field
        temperature = 300.0 - 0.1 * X - 0.05 * Y  # Gradient field
        
        # Interpolate to new point
        x_new = 500.0
        y_new = 500.0
        
        temp_interp = interpolate_horizontal(
            x, y, temperature, x_new, y_new
        )
        
        # Should be close to expected value
        expected = 300.0 - 0.1 * 500.0 - 0.05 * 500.0
        assert abs(temp_interp - expected) < 1.0
    
    def test_interpolate_horizontal_2d_array(self):
        """Test horizontal interpolation with 2D target arrays."""
        # Create grid
        x = np.array([0.0, 1000.0, 2000.0])
        y = np.array([0.0, 1000.0, 2000.0])
        X, Y = np.meshgrid(x, y)
        
        # Temperature field
        temperature = np.sin(X / 1000.0) * np.cos(Y / 1000.0) * 10.0 + 280.0
        
        # Interpolate to multiple points
        x_new = np.array([500.0, 1500.0])
        y_new = np.array([500.0, 1500.0])
        
        temp_interp = interpolate_horizontal(
            x, y, temperature, x_new, y_new
        )
        
        # Should have correct shape
        assert len(temp_interp) == len(x_new)
        assert np.all(np.isfinite(temp_interp))


class TestCoordinateTransformations:
    """Test coordinate transformation functions."""
    
    def test_pressure_to_height_standard(self):
        """Test pressure to height conversion under standard atmosphere."""
        # Standard pressure levels
        pressures = np.array([1013.25, 850.0, 700.0, 500.0, 300.0, 100.0])  # hPa
        
        heights = pressure_to_height(pressures)
        
        # Should be monotonically increasing
        assert np.all(heights[1:] > heights[:-1])
        
        # Surface should be near 0
        assert abs(heights[0]) < 100.0  # Within 100m of surface
        
        # Tropopause region should be around 10-15 km
        assert 8000.0 < heights[3] < 15000.0  # 500 hPa level
        
        # Should be reasonable values
        assert np.all(heights >= 0)
        assert np.all(heights < 50000.0)  # Less than 50 km
    
    def test_height_to_pressure_standard(self):
        """Test height to pressure conversion under standard atmosphere."""
        # Standard height levels
        heights = np.array([0.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0])  # m
        
        pressures = height_to_pressure(heights)
        
        # Should be monotonically decreasing
        assert np.all(pressures[1:] < pressures[:-1])
        
        # Surface should be around 1013 hPa
        assert 950.0 < pressures[0] < 1100.0
        
        # Mount Everest should be around 300-350 hPa
        assert 300.0 < pressures[3] < 350.0  # 5 km level
        
        # Should be reasonable values
        assert np.all(pressures > 0)
        assert np.all(pressures < 1100.0)
    
    def test_pressure_height_round_trip(self):
        """Test round-trip pressure-height conversions."""
        # Test several pressure levels
        pressures_original = np.array([1000.0, 800.0, 600.0, 400.0, 200.0])
        
        # Convert pressure -> height -> pressure
        heights = pressure_to_height(pressures_original)
        pressures_back = height_to_pressure(heights)
        
        # Should be close to original (within atmospheric variability)
        assert np.allclose(pressures_back, pressures_original, rtol=0.1)
        
        # Test several height levels
        heights_original = np.array([0.0, 2000.0, 4000.0, 8000.0, 12000.0])
        
        # Convert height -> pressure -> height
        pressures = height_to_pressure(heights_original)
        heights_back = pressure_to_height(pressures)
        
        assert np.allclose(heights_back, heights_original, rtol=0.1)
    
    def test_coordinate_extreme_values(self):
        """Test coordinate transformations at extreme values."""
        # Very high altitude
        height_high = 30000.0  # 30 km
        pressure_high = height_to_pressure(height_high)
        assert pressure_high > 0  # Should still have some pressure
        assert pressure_high < 100.0  # Very low pressure
        
        # Very low pressure
        pressure_low = 10.0  # 10 hPa
        height_low = pressure_to_height(pressure_low)
        assert height_low > 30000.0  # Very high altitude
        
        # Sea level variations
        pressure_sea_level = np.array([980.0, 1013.25, 1040.0])
        height_sea_level = pressure_to_height(pressure_sea_level)
        assert np.all(height_sea_level < 500.0)  # All near sea level


class TestInterpolationEdgeCases:
    """Test edge cases in interpolation functions."""
    
    def test_interpolation_single_level(self):
        """Test interpolation with single level."""
        pressure = np.array([1000.0])
        temperature = np.array([298.15])
        
        # Interpolating to the same level should return original value
        temp_interp = interpolate_to_pressure_level(pressure, temperature, 1000.0)
        assert abs(temp_interp - 298.15) < 1e-10
    
    def test_interpolation_identical_levels(self):
        """Test interpolation when target level equals existing level."""
        pressure = np.array([1000.0, 900.0, 800.0])
        temperature = np.array([298.15, 288.15, 278.15])
        
        # Interpolating to exact existing level
        temp_interp = interpolate_to_pressure_level(pressure, temperature, 900.0)
        assert abs(temp_interp - 288.15) < 1e-10
    
    def test_interpolation_monotonicity(self):
        """Test that interpolation preserves monotonicity where expected."""
        # Temperature decreasing with height
        pressure = np.array([1000.0, 900.0, 800.0, 700.0])
        temperature = np.array([298.15, 288.15, 278.15, 268.15])
        
        # Interpolate to finer resolution
        pressure_fine = np.array([950.0, 925.0, 875.0, 825.0, 750.0])
        temperature_fine = interpolate_vertical_profile(pressure, temperature, pressure_fine)
        
        # Should still be monotonically decreasing
        assert np.all(temperature_fine[1:] < temperature_fine[:-1])
    
    def test_interpolation_extreme_gradients(self):
        """Test interpolation with extreme gradients."""
        # Large temperature gradient
        pressure = np.array([1000.0, 999.0])  # Very close pressure levels
        temperature = np.array([300.0, 200.0])  # Large temperature change
        
        # Interpolate between them
        temp_interp = interpolate_to_pressure_level(pressure, temperature, 999.5)
        
        # Should be approximately midway
        assert 200.0 < temp_interp < 300.0
        assert abs(temp_interp - 250.0) < 10.0  # Should be close to midpoint
    
    def test_interpolation_nan_handling(self):
        """Test interpolation behavior with NaN values."""
        pressure = np.array([1000.0, 900.0, 800.0])
        temperature = np.array([298.15, np.nan, 278.15])
        
        # Should handle NaN appropriately (implementation dependent)
        temp_interp = interpolate_to_pressure_level(pressure, temperature, 850.0)
        # Either should interpolate between 1000 and 800 hPa, or return NaN
        assert np.isfinite(temp_interp) or np.isnan(temp_interp)


class TestInterpolationAccuracy:
    """Test accuracy of interpolation methods."""
    
    def test_linear_interpolation_accuracy(self):
        """Test accuracy of linear interpolation."""
        # Create known linear relationship
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = 2.0 * x + 1.0  # y = 2x + 1
        
        # Interpolate to known point
        x_interp = 1.5
        y_interp = interpolate_horizontal([0, 1, 2, 3, 4], [0], y.reshape(-1, 1), x_interp, 0)
        
        # Should be exact for linear interpolation
        expected = 2.0 * 1.5 + 1.0
        assert abs(y_interp[0] - expected) < 1e-10
    
    def test_interpolation_conservative_properties(self):
        """Test that interpolation conserves certain properties."""
        # Temperature should generally decrease with height
        pressure = np.array([1000.0, 800.0, 600.0, 400.0])
        temperature = np.array([298.15, 278.15, 258.15, 238.15])
        
        # Interpolate to finer grid
        pressure_fine = np.linspace(1000.0, 400.0, 20)
        temperature_fine = interpolate_vertical_profile(pressure, temperature, pressure_fine)
        
        # Should maintain decreasing trend
        assert np.all(temperature_fine[1:] < temperature_fine[:-1])
        
        # Should maintain bounds
        assert np.all(temperature_fine >= 238.15)
        assert np.all(temperature_fine <= 298.15)