"""
Test suite for dynamic meteorological calculations.

Tests dynamic calculations including vorticity, divergence, geostrophic wind,
gradient wind, potential vorticity, and other dynamic parameters.
"""
import numpy as np
import pytest

# Import the dynamic calculation functions
from monet_meteo.dynamics.dynamic_calculations import (
    relative_vorticity,
    absolute_vorticity,
    divergence,
    geostrophic_wind,
    gradient_wind,
    potential_vorticity,
    coriolis_parameter,
    vertical_velocity_pressure,
    omega_to_w
)


class TestCoriolisParameter:
    """Test Coriolis parameter calculations."""
    
    def test_coriolis_standard_latitudes(self):
        """Test Coriolis parameter at standard latitudes."""
        latitudes = np.array([0.0, 30.0, 45.0, 60.0, 90.0])  # degrees
        lat_rad = np.radians(latitudes)
        
        f = coriolis_parameter(lat_rad)
        
        # Should be zero at equator
        assert abs(f[0]) < 1e-10
        
        # Should increase with latitude
        assert f[1] > 0  # 30°
        assert f[2] > f[1]  # 45° > 30°
        assert f[3] > f[2]  # 60° > 45°
        assert f[4] > f[3]  # 90° > 60°
        
        # Should be approximately 1.46e-4 at 45°
        expected_45 = 2 * 7.292e-5 * np.sin(np.radians(45.0))
        assert abs(f[2] - expected_45) < 1e-6
    
    def test_coriolis_negative_latitudes(self):
        """Test Coriolis parameter at negative latitudes (southern hemisphere)."""
        lat_rad = np.radians(-45.0)  # -45°
        
        f = coriolis_parameter(lat_rad)
        
        # Should be negative in southern hemisphere
        assert f < 0
        
        # Should have same magnitude as northern hemisphere
        f_positive = coriolis_parameter(np.radians(45.0))
        assert abs(f) == abs(f_positive)
    
    def test_coriolis_extreme_latitudes(self):
        """Test Coriolis parameter at extreme latitudes."""
        # At poles
        f_north_pole = coriolis_parameter(np.pi/2)  # 90°
        f_south_pole = coriolis_parameter(-np.pi/2)  # -90°
        
        # Should be maximum magnitude at poles
        expected_max = 2 * 7.292e-5
        assert abs(f_north_pole - expected_max) < 1e-6
        assert abs(f_south_pole + expected_max) < 1e-6


class TestRelativeVorticity:
    """Test relative vorticity calculations."""
    
    def test_vorticity_solid_rotation(self):
        """Test vorticity for solid body rotation."""
        # Create a simple grid
        nx, ny = 10, 10
        x = np.linspace(0, 1000000, nx)  # 1000 km
        y = np.linspace(0, 1000000, ny)  # 1000 km
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Solid body rotation: u = -omega * y, v = omega * x
        omega = 1e-4  # rad/s
        x_grid, y_grid = np.meshgrid(x, y)
        u = -omega * y_grid
        v = omega * x_grid
        
        zeta = relative_vorticity(u, v, dx, dy)
        
        # Should be constant and equal to 2*omega for solid body rotation
        expected = 2 * omega
        assert np.allclose(zeta, expected, rtol=1e-2)
    
    def test_vorticity_shear_flow(self):
        """Test vorticity for shear flow."""
        # Create a simple grid
        nx, ny = 10, 10
        x = np.linspace(0, 100000, nx)  # 100 km
        y = np.linspace(0, 100000, ny)  # 100 km
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Shear flow: u = 0, v = shear * y
        shear = 1e-4  # s^-1
        y_grid = np.zeros((ny, nx))
        for i in range(ny):
            y_grid[i, :] = y[i]
        
        u = np.zeros((ny, nx))
        v = shear * y_grid
        
        zeta = relative_vorticity(u, v, dx, dy)
        
        # Should be equal to -shear (dv/dx = 0, du/dy = 0)
        # Actually should be zero for this flow
        assert np.allclose(zeta, 0, atol=1e-6)
    
    def test_vorticity_extreme_values(self):
        """Test vorticity calculations with extreme values."""
        # Very small grid spacing
        dx, dy = 1.0, 1.0  # 1 meter
        
        # Create simple velocity field
        u = np.ones((5, 5)) * 10.0  # 10 m/s
        v = np.ones((5, 5)) * 5.0   # 5 m/s
        
        zeta = relative_vorticity(u, v, dx, dy)
        
        # Should be finite and reasonable
        assert np.all(np.isfinite(zeta))
        assert np.all(np.abs(zeta) < 1e6)  # Less than 1e6 s^-1


class TestDivergence:
    """Test divergence calculations."""
    
    def test_divergence_convergence_flow(self):
        """Test divergence for converging/diverging flow."""
        # Create a simple grid
        nx, ny = 10, 10
        x = np.linspace(-100000, 100000, nx)  # ±100 km
        y = np.linspace(-100000, 100000, ny)  # ±100 km
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Divergent flow: u = x, v = y
        x_grid, y_grid = np.meshgrid(x, y)
        u = x_grid * 1e-5  # Scale to get reasonable velocities
        v = y_grid * 1e-5
        
        div = divergence(u, v, dx, dy)
        
        # Should be positive and constant for this flow
        assert np.all(div > 0)
        assert np.allclose(div, div[0, 0], rtol=1e-2)
    
    def test_divergence_solenoidal_flow(self):
        """Test divergence for solenoidal (divergence-free) flow."""
        # Create a simple grid
        nx, ny = 10, 10
        x = np.linspace(0, 100000, nx)
        y = np.linspace(0, 100000, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Rotational flow: u = -y, v = x
        x_grid, y_grid = np.meshgrid(x, y)
        u = -y_grid * 1e-5
        v = x_grid * 1e-5
        
        div = divergence(u, v, dx, dy)
        
        # Should be approximately zero for solenoidal flow
        assert np.allclose(div, 0, atol=1e-6)
    
    def test_divergence_extreme_values(self):
        """Test divergence with extreme values."""
        # Very large velocities
        u = np.ones((5, 5)) * 100.0  # 100 m/s
        v = np.ones((5, 5)) * 100.0  # 100 m/s
        dx, dy = 1000.0, 1000.0  # 1 km spacing
        
        div = divergence(u, v, dx, dy)
        
        # Should be finite
        assert np.all(np.isfinite(div))
        assert np.all(np.abs(div) < 1e3)  # Less than 1000 s^-1


class TestGeostrophicWind:
    """Test geostrophic wind calculations."""
    
    def test_geostrophic_wind_constant_pressure(self):
        """Test geostrophic wind with constant pressure gradient."""
        # Create a simple grid
        nx, ny = 10, 10
        x = np.linspace(0, 1000000, nx)
        y = np.linspace(0, 1000000, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Constant geopotential height gradient
        height = np.zeros((ny, nx))
        for i in range(ny):
            height[i, :] = 100.0 * i  # 100 m^2/s^2 per grid point
        
        # Constant latitude
        latitude = np.ones((ny, nx)) * np.radians(45.0)
        
        ug, vg = geostrophic_wind(height, dx, dy, latitude)
        
        # Should have meridional wind only (v-component)
        assert np.allclose(ug, 0, atol=1e-10)
        assert np.all(vg > 0)  # Should be positive
        
        # Magnitude should be reasonable
        f = coriolis_parameter(np.radians(45.0))
        expected_vg = -(9.81 / f) * (100.0 / dy)
        assert np.allclose(vg, expected_vg, rtol=1e-2)
    
    def test_geostrophic_wind_cyclostrophic_balance(self):
        """Test geostrophic wind in cyclostrophic balance."""
        # Create circular height field
        nx, ny = 20, 20
        x = np.linspace(-500000, 500000, nx)
        y = np.linspace(-500000, 500000, ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        x_grid, y_grid = np.meshgrid(x, y)
        r_squared = x_grid**2 + y_grid**2
        
        # Circular height field: higher in center
        height = 1000.0 * np.exp(-r_squared / (2 * 200000**2))
        
        # Constant latitude
        latitude = np.ones((ny, nx)) * np.radians(45.0)
        
        ug, vg = geostrophic_wind(height, dx, dy, latitude)
        
        # Should have rotational wind pattern
        assert not np.allclose(ug, 0, atol=1e-10)
        assert not np.allclose(vg, 0, atol=1e-10)
        
        # Wind should be strongest at some distance from center
        max_speed = np.sqrt(np.max(ug**2 + vg**2))
        assert max_speed > 0
        assert max_speed < 100.0  # Should be reasonable (< 100 m/s)


class TestAbsoluteVorticity:
    """Test absolute vorticity calculations."""
    
    def test_absolute_vorticity_standard(self):
        """Test absolute vorticity calculation."""
        # Create a simple grid
        nx, ny = 10, 10
        dx, dy = 10000.0, 10000.0  # 10 km spacing
        
        # Simple velocity field
        u = np.ones((ny, nx)) * 10.0  # 10 m/s
        v = np.ones((ny, nx)) * 5.0   # 5 m/s
        
        # Latitude array
        latitudes = np.ones((ny, nx)) * np.radians(45.0)
        
        zeta_a = absolute_vorticity(u, v, dx, dy, latitudes)
        
        # Should be finite and reasonable
        assert np.all(np.isfinite(zeta_a))
        assert np.all(np.abs(zeta_a) < 1e-2)  # Less than 0.01 s^-1
    
    def test_absolute_vorticity_conservation(self):
        """Test that absolute vorticity includes planetary vorticity."""
        # At 45°N
        lat_rad = np.radians(45.0)
        f = coriolis_parameter(lat_rad)
        
        # Zero relative vorticity
        u = np.ones((5, 5)) * 10.0
        v = np.ones((5, 5)) * 10.0
        dx, dy = 10000.0, 10000.0
        
        zeta_a = absolute_vorticity(u, v, dx, dy, lat_rad)
        
        # Should be approximately equal to f (planetary vorticity)
        # since relative vorticity should be small for uniform flow
        assert np.allclose(zeta_a, f, atol=1e-5)


class TestPotentialVorticity:
    """Test potential vorticity calculations."""
    
    def test_potential_vorticity_standard(self):
        """Test potential vorticity calculation."""
        # Create a simple grid
        nx, ny, nz = 5, 5, 3
        dx, dy = 100000.0, 100000.0  # 100 km spacing
        
        # Simple velocity fields
        u = np.ones((nz, ny, nx)) * 10.0  # 10 m/s
        v = np.ones((nz, ny, nx)) * 5.0   # 5 m/s
        
        # Latitude
        latitude = np.ones((ny, nx)) * np.radians(45.0)
        
        # Potential temperature
        theta = np.ones((nz, ny, nx)) * 300.0  # 300 K
        
        # Vertical coordinate (pressure or height)
        p = np.array([85000.0, 70000.0, 50000.0])  # Pa
        
        pv = potential_vorticity(u, v, latitude, theta, p, dx, dy)
        
        # Should be finite and reasonable
        assert np.all(np.isfinite(pv))
        assert np.all(pv > 0)  # Should be positive for normal atmosphere
        
        # Typical PV values are around 1e-6 to 1e-5 K m^2 kg^-1 s^-1
        assert np.all(pv < 1e-3)  # Less than 1e-3
    
    def test_potential_vorticity_units(self):
        """Test potential vorticity units and scaling."""
        # Standard atmosphere values
        u = np.ones((3, 5, 5)) * 20.0  # 20 m/s
        v = np.ones((3, 5, 5)) * 0.0   # No meridional wind
        latitude = np.ones((5, 5)) * np.radians(45.0)
        theta = np.linspace(300.0, 350.0, 3)[:, np.newaxis, np.newaxis]  # K
        p = np.array([85000.0, 70000.0, 50000.0])  # Pa
        dx, dy = 500000.0, 500000.0  # 500 km
        
        pv = potential_vorticity(u, v, latitude, theta, p, dx, dy)
        
        # Should have typical atmospheric PV magnitudes
        assert np.all(pv > 1e-7)  # Greater than 1e-7
        assert np.all(pv < 1e-4)  # Less than 1e-4


class TestVerticalVelocity:
    """Test vertical velocity conversions."""
    
    def test_omega_to_w_conversion(self):
        """Test omega to geometric vertical velocity conversion."""
        # Create test data
        omega = np.ones((5, 5)) * 1.0  # Pa/s
        pressure = np.ones((5, 5)) * 85000.0  # Pa
        temperature = np.ones((5, 5)) * 280.0  # K
        
        w = omega_to_w(omega, pressure, temperature)
        
        # Should be finite and negative (for positive omega)
        assert np.all(np.isfinite(w))
        assert np.all(w < 0)  # Positive omega should give negative w
        
        # Magnitude should be reasonable
        assert np.all(np.abs(w) < 10.0)  # Less than 10 m/s
    
    def test_omega_to_w_with_mixing_ratio(self):
        """Test omega to w conversion with mixing ratio."""
        omega = np.ones((3, 3)) * 0.5  # Pa/s
        pressure = np.ones((3, 3)) * 70000.0  # Pa
        temperature = np.ones((3, 3)) * 290.0  # K
        mixing_ratio = np.ones((3, 3)) * 0.01  # 10 g/kg
        
        w = omega_to_w(omega, pressure, temperature, mixing_ratio)
        
        # Should be finite
        assert np.all(np.isfinite(w))
        
        # Should be different from case without mixing ratio
        w_no_moisture = omega_to_w(omega, pressure, temperature)
        assert not np.allclose(w, w_no_moisture, rtol=1e-2)
    
    def test_vertical_velocity_extreme_conditions(self):
        """Test vertical velocity conversion at extreme conditions."""
        # Very cold conditions
        w_cold = omega_to_w(1.0, 50000.0, 220.0)
        assert np.isfinite(w_cold)
        assert w_cold < 0
        
        # Very warm conditions
        w_warm = omega_to_w(1.0, 50000.0, 320.0)
        assert np.isfinite(w_warm)
        assert w_warm < 0
        
        # Should be different due to density differences
        assert abs(w_warm) > abs(w_cold)  # Warmer air is less dense


class TestGradientWind:
    """Test gradient wind calculations."""
    
    def test_gradient_wind_circular_flow(self):
        """Test gradient wind for circular flow."""
        # Radius of curvature
        radius = 1000000.0  # 1000 km
        
        # Pressure gradient force
        dp_dr = 0.001  # Pa/m
        
        # Density
        density = 1.2  # kg/m^3
        
        # Coriolis parameter
        f = coriolis_parameter(np.radians(45.0))
        
        vg = gradient_wind(radius, dp_dr, density, f)
        
        # Should be finite and positive
        assert np.isfinite(vg)
        assert vg > 0
        
        # Should be reasonable for atmospheric conditions
        assert vg < 100.0  # Less than 100 m/s
    
    def test_gradient_wind_geostrophic_limit(self):
        """Test that gradient wind approaches geostrophic wind for large radius."""
        large_radius = 1e8  # Very large radius
        dp_dr = 0.001
        density = 1.2
        f = coriolis_parameter(np.radians(45.0))
        
        vg = gradient_wind(large_radius, dp_dr, density, f)
        
        # Should approach geostrophic balance: vg ≈ -1/(ρf) * dp/dr
        vg_geostrophic = -dp_dr / (density * f)
        
        assert abs(vg - vg_geostrophic) < 0.1
    
    def test_gradient_wind_cyclostrophic_limit(self):
        """Test cyclostrophic wind limit (small radius)."""
        small_radius = 1000.0  # Small radius
        dp_dr = 0.01  # Larger pressure gradient
        density = 1.2
        f = coriolis_parameter(np.radians(45.0))
        
        vg = gradient_wind(small_radius, dp_dr, density, f)
        
        # Should be finite and positive
        assert np.isfinite(vg)
        assert vg > 0
        
        # Should be larger than geostrophic wind due to centrifugal force
        vg_geostrophic = -dp_dr / (density * f)
        assert vg > abs(vg_geostrophic)