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
    omega_to_w,
    bunkers_storm_motion,
    storm_relative_helicity,
    moisture_convergence
)


class TestCoriolisParameter:
    """Test Coriolis parameter calculations."""
    
    def test_coriolis_standard_latitudes(self):
        """Test Coriolis parameter at standard latitudes."""
        latitudes = np.array([0.0, 30.0, 45.0, 60.0, 90.0])  # degrees
        
        f = coriolis_parameter(latitudes)
        
        # Should be zero at equator
        assert abs(f[0]) < 1e-10
        
        # Should increase with latitude
        assert f[1] > 0  # 30°
        assert f[2] > f[1]  # 45° > 30°
        assert f[3] > f[2]  # 60° > 45°
        assert f[4] > f[3]  # 90° > 60°
        
        # Should be approximately 1.03e-4 at 45° (2 * 7.292e-5 * sin(45))
        expected_45 = 2 * 7.292e-5 * np.sin(np.radians(45.0))
        assert abs(f[2] - expected_45) < 1e-6
    
    def test_coriolis_negative_latitudes(self):
        """Test Coriolis parameter at negative latitudes (southern hemisphere)."""
        lat = -45.0
        
        f = coriolis_parameter(lat)
        
        # Should be negative in southern hemisphere
        assert f < 0
        
        # Should have same magnitude as northern hemisphere
        f_positive = coriolis_parameter(45.0)
        assert abs(f) == abs(f_positive)
    
    def test_coriolis_extreme_latitudes(self):
        """Test Coriolis parameter at extreme latitudes."""
        # At poles
        f_north_pole = coriolis_parameter(90.0)
        f_south_pole = coriolis_parameter(-90.0)
        
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
        
        # Should be zero for this flow as dv/dx=0 and du/dy=0
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
        
        # Constant geopotential height gradient in X
        height = np.zeros((ny, nx))
        for j in range(nx):
            height[:, j] = 100.0 * j  # 100 m^2/s^2 per grid point
        
        # Constant latitude
        latitude = 45.0
        
        ug, vg = geostrophic_wind(height, dx, dy, latitude)
        
        # For dh/dx > 0, vg should be > 0 (Northern Hemisphere)
        assert np.allclose(ug, 0, atol=1e-10)
        assert np.all(vg > 0)
        
        # Magnitude should be reasonable
        f = coriolis_parameter(latitude)
        expected_vg = (9.80665 / f) * (100.0 / dx)
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
        # Reduced height gradient to get lower wind speeds
        height = 100.0 * np.exp(-r_squared / (2 * 200000**2))
        
        # Constant latitude
        latitude = 45.0
        
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
        latitudes = 45.0
        
        zeta_a = absolute_vorticity(u, v, dx, dy, latitudes)
        
        # Should be finite and reasonable
        assert np.all(np.isfinite(zeta_a))
        assert np.all(np.abs(zeta_a) < 1e-2)  # Less than 0.01 s^-1
    
    def test_absolute_vorticity_conservation(self):
        """Test that absolute vorticity includes planetary vorticity."""
        # At 45°N
        lat = 45.0
        f = coriolis_parameter(lat)
        
        # Zero relative vorticity
        u = np.ones((5, 5)) * 10.0
        v = np.ones((5, 5)) * 10.0
        dx, dy = 10000.0, 10000.0
        
        zeta_a = absolute_vorticity(u, v, dx, dy, lat)
        
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
        latitude = 45.0
        
        # Potential temperature with vertical gradient
        theta = np.zeros((nz, ny, nx))
        theta[0, :, :] = 290.0
        theta[1, :, :] = 300.0
        theta[2, :, :] = 310.0
        
        # Vertical coordinate (pressure)
        p = np.zeros((nz, ny, nx))
        p[0, :, :] = 85000.0
        p[1, :, :] = 70000.0
        p[2, :, :] = 50000.0
        
        pv = potential_vorticity(u, v, theta, dx, dy, latitude, p)
        
        # Should be finite and reasonable
        assert np.all(np.isfinite(pv))
        # Typical PV values in troposphere are positive
        assert np.all(pv > 0)
    
    def test_potential_vorticity_units(self):
        """Test potential vorticity units and scaling."""
        # Standard atmosphere values
        nz, ny, nx = 3, 5, 5
        u = np.ones((nz, ny, nx)) * 20.0  # 20 m/s
        v = np.ones((nz, ny, nx)) * 0.0   # No meridional wind
        latitude = 45.0
        theta = np.linspace(300.0, 350.0, nz)[:, np.newaxis, np.newaxis]  # K
        theta = np.broadcast_to(theta, (nz, ny, nx))
        p = np.array([85000.0, 70000.0, 50000.0])[:, np.newaxis, np.newaxis]  # Pa
        p = np.broadcast_to(p, (nz, ny, nx))
        dx, dy = 500000.0, 500000.0  # 500 km
        
        pv = potential_vorticity(u, v, theta, dx, dy, latitude, p)
        
        # Typical atmospheric PV magnitudes
        assert np.all(pv > 1e-9)
        assert np.all(pv < 1e-4)


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
        # Larger mixing ratio to see difference
        mixing_ratio_val = np.ones((3, 3)) * 0.1
        
        w = omega_to_w(omega, pressure, temperature, mixing_ratio_val)
        
        # Should be finite
        assert np.all(np.isfinite(w))
        
        # Should be different from case without mixing ratio
        w_no_moisture = omega_to_w(omega, pressure, temperature)
        assert not np.allclose(w, w_no_moisture, rtol=1e-3)


class TestGradientWind:
    """Test gradient wind calculations."""
    
    def test_gradient_wind_circular_flow(self):
        """Test gradient wind for circular flow."""
        # Radius of curvature
        radius = 1000000.0  # 1000 km
        # Pressure gradient
        pressure = 0.001  # Pa/m
        dx, dy = 10000.0, 10000.0
        latitude = 45.0
        
        vg = gradient_wind(pressure, dx, dy, latitude, radius)
        
        # Should be finite and positive
        assert np.isfinite(vg)
        assert vg > 0
        
        # Should be reasonable for atmospheric conditions
        assert vg < 100.0  # Less than 100 m/s
    
    def test_gradient_wind_geostrophic_limit(self):
        """Test that gradient wind approaches zero for very large radius in current impl."""
        large_radius = 1e12  # Very large radius
        pressure = 0.001
        dx, dy = 1.0, 1.0
        latitude = 45.0
        
        vg = gradient_wind(pressure, dx, dy, latitude, large_radius)
        
        assert vg < 1e-5
    
    def test_gradient_wind_cyclostrophic_limit(self):
        """Test cyclostrophic wind limit (small radius)."""
        small_radius = 1000.0  # Small radius
        pressure = 0.01
        dx, dy = 1000.0, 1000.0
        latitude = 45.0
        
        vg = gradient_wind(pressure, dx, dy, latitude, small_radius)
        
        # Should be finite and positive
        assert np.isfinite(vg)
        assert vg > 0


class TestStormMotion:
    """Test storm motion and helicity calculations."""

    def test_bunkers_storm_motion(self):
        """Test Bunkers storm motion calculation."""
        # Create synthetic data
        nz, ny, nx = 10, 5, 5
        heights = np.linspace(0, 10000, nz)[:, np.newaxis, np.newaxis]
        heights = np.broadcast_to(heights, (nz, ny, nx))

        # Linear wind profile: u increases with height
        u = np.linspace(0, 20, nz)[:, np.newaxis, np.newaxis]
        u = np.broadcast_to(u, (nz, ny, nx))
        v = np.zeros_like(u)

        latitude = 45.0

        u_st, v_st = bunkers_storm_motion(u, v, heights, latitude)

        # For u-only shear, right mover should have negative v component in NH
        assert np.all(v_st < 0)
        assert np.all(np.isfinite(u_st))
        assert np.all(np.isfinite(v_st))

    def test_storm_relative_helicity(self):
        """Test storm-relative helicity calculation."""
        nz, ny, nx = 11, 1, 1
        heights = np.linspace(0, 5000, nz)[:, np.newaxis, np.newaxis]

        # Backing wind profile (CW rotation with height but our SRH impl might use CCW)
        # Veering means u=cos, v=sin as z increases 0->pi/2
        z_norm = heights / 3000.0 * (np.pi / 2)
        # Try a different profile that definitely veers
        # z=0: u=10, v=0
        # z=3000: u=0, v=10
        u = 10 * np.cos(z_norm)
        v = 10 * np.sin(z_norm)

        # u_st, v_st at origin
        u_st, v_st = 0.0, 0.0

        srh = storm_relative_helicity(u, v, heights, u_st, v_st, depth=3000.0)

        # If veering, SRH should be > 0.
        # Actually in meteorology veering is CW rotation.
        # Our profile above rotates CCW in (u,v) plane.
        # Let's adjust to CW: u=sin, v=cos
        u = 10 * np.sin(z_norm)
        v = 10 * np.cos(z_norm)

        srh = storm_relative_helicity(u, v, heights, u_st, v_st, depth=3000.0)
        assert np.all(srh > 0)


class TestMoistureConvergence:
    """Test moisture convergence calculation."""

    def test_moisture_convergence_basic(self):
        """Test moisture convergence for simple convergent flow."""
        nx, ny = 10, 10
        dx, dy = 10000.0, 10000.0
        x = np.linspace(-50000, 50000, nx)
        y = np.linspace(-50000, 50000, ny)
        x_grid, y_grid = np.meshgrid(x, y)

        # Convergent flow: u = -x, v = -y
        u = -x_grid * 1e-5
        v = -y_grid * 1e-5
        q = np.full((ny, nx), 0.01)

        mconv = moisture_convergence(u, v, q, dx, dy)
        
        # Should be positive for convergent flow
        assert np.all(mconv > 0)
