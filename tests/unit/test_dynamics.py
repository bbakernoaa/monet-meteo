"""
Test suite for dynamic meteorological calculations.
"""

import numpy as np

from monet_meteo.dynamics.dynamic_calculations import (
    relative_vorticity,
    geostrophic_wind,
    gradient_wind,
    potential_vorticity,
    coriolis_parameter,
    omega_to_w,
)


class TestCoriolisParameter:
    def test_coriolis_standard_latitudes(self):
        latitudes = np.array([0.0, 30.0, 45.0, 60.0, 90.0])
        f = coriolis_parameter(latitudes)
        assert abs(f[0]) < 1e-10
        assert f[1] > 0
        assert f[2] > f[1]
        expected_45 = 2 * 7.2921159e-5 * np.sin(np.radians(45.0))
        assert abs(f[2] - expected_45) < 1e-6


class TestRelativeVorticity:
    def test_vorticity_solid_rotation(self):
        nx, ny = 10, 10
        x = np.linspace(0, 1e6, nx)
        y = np.linspace(0, 1e6, ny)
        dx, dy = x[1] - x[0], y[1] - y[0]
        x_grid, y_grid = np.meshgrid(x, y)  # axis 0 is y, axis 1 is x
        omega = 1e-4
        u = -omega * y_grid
        v = omega * x_grid
        zeta = relative_vorticity(u, v, dx, dy)
        assert np.allclose(zeta, 2 * omega, rtol=1e-2)


class TestGeostrophicWind:
    def test_geostrophic_wind_constant_pressure(self):
        nx, ny = 10, 10
        dx, dy = 1e5, 1e5
        # Height varying with y (axis 0)
        height = np.zeros((ny, nx))
        for i in range(ny):
            height[i, :] = 100.0 * i
        latitude = np.radians(45.0)
        ug, vg = geostrophic_wind(height, dx, dy, latitude)
        # Gradient in y -> wind in x (ug)
        assert np.all(np.abs(ug) > 0)
        assert np.allclose(vg, 0, atol=1e-10)


class TestPotentialVorticity:
    def test_potential_vorticity_standard(self):
        nz, ny, nx = 3, 5, 5
        u = np.ones((nz, ny, nx)) * 10.0
        v = np.ones((nz, ny, nx)) * 5.0
        latitude = np.radians(45.0)
        # Theta MUST vary with height
        theta = np.linspace(300, 310, nz)[:, np.newaxis, np.newaxis] * np.ones(
            (nz, ny, nx)
        )
        p = np.array([100000.0, 85000.0, 70000.0])
        dx, dy = 1e5, 1e5
        pv = potential_vorticity(u, v, latitude, theta, p, dx, dy)
        assert np.all(pv > 0)


class TestGradientWind:
    def test_gradient_wind_geostrophic_limit(self):
        radius = 1e8
        dp_dr = 0.001
        density = 1.2
        f = coriolis_parameter(45.0)
        vg = gradient_wind(radius, dp_dr, density, f)
        vg_geostrophic = -dp_dr / (density * f)
        assert abs(vg - vg_geostrophic) < 1.0


class TestVerticalVelocity:
    def test_omega_to_w_with_mixing_ratio(self):
        omega = 0.5
        pressure = 70000.0
        temperature = 290.0
        w1 = omega_to_w(omega, pressure, temperature, 0.0)
        w2 = omega_to_w(omega, pressure, temperature, 0.01)
        assert abs(w1 - w2) > 1e-4
