"""
Test suite for statistical and micrometeorological functions.

Tests statistical calculations, turbulence parameters, flux calculations,
and other micrometeorological functions.
"""

import numpy as np

# Import statistical calculation functions
from monet_meteo.statistical.statistical_calculations import (
    bulk_richardson_number,
    correlation_coefficient,
    covariance,
    friction_velocity,
    latent_heat_flux,
    momentum_flux,
    monin_obukhov_length,
    sensible_heat_flux,
    standard_deviation,
    turbulence_kinetic_energy,
)


class TestBulkRichardsonNumber:
    """Test bulk Richardson number calculations."""

    def test_bulk_richardson_standard_conditions(self):
        """Test bulk Richardson number at standard conditions."""
        # Typical atmospheric conditions
        u_wind = 10.0  # m/s
        v_wind = 5.0  # m/s
        potential_temperature = np.array([298.0, 300.0])  # K
        height = np.array([10.0, 100.0])  # m

        ri = bulk_richardson_number(u_wind, v_wind, potential_temperature, height)

        # Should be finite and reasonable
        assert np.all(np.isfinite(ri))
        assert np.all(ri > 0)  # Typically positive in stable conditions

    def test_bulk_richardson_stability_cases(self):
        """Test bulk Richardson number for different stability cases."""
        u_wind = 5.0
        v_wind = 0.0

        # Stable case (temperature increases with height)
        theta_stable = np.array([295.0, 300.0])
        height = np.array([10.0, 100.0])
        ri_stable = bulk_richardson_number(u_wind, v_wind, theta_stable, height)

        assert np.all(ri_stable > 0)

        # Unstable case (temperature decreases with height)
        theta_unstable = np.array([300.0, 295.0])
        ri_unstable = bulk_richardson_number(u_wind, v_wind, theta_unstable, height)

        assert np.all(ri_unstable < 0)

    def test_bulk_richardson_wind_speed_dependence(self):
        """Test Richardson number dependence on wind speed."""
        potential_temperature = np.array([298.0, 300.0])
        height = np.array([10.0, 100.0])

        # Low wind speed
        ri_low = bulk_richardson_number(2.0, 0.0, potential_temperature, height)

        # High wind speed
        ri_high = bulk_richardson_number(20.0, 0.0, potential_temperature, height)

        # High wind should give smaller Richardson number (more mechanical turbulence)
        assert np.all(ri_high < ri_low)

    def test_bulk_richardson_height_dependence(self):
        """Test Richardson number dependence on height difference."""
        u_wind = 10.0
        v_wind = 0.0
        potential_temperature = np.array([298.0, 300.0])

        # Small height difference
        height_small = np.array([10.0, 20.0])
        ri_small = bulk_richardson_number(u_wind, v_wind, potential_temperature, height_small)

        # Large height difference
        height_large = np.array([10.0, 200.0])
        ri_large = bulk_richardson_number(u_wind, v_wind, potential_temperature, height_large)

        # Larger height difference should give larger Richardson number
        assert np.all(ri_large > ri_small)


class TestTurbulenceParameters:
    """Test turbulence parameter calculations."""

    def test_friction_velocity_standard(self):
        """Test friction velocity calculation."""
        wind_speed = 10.0  # m/s
        height = 10.0
        surface_roughness = 0.01  # m
        stability_parameter = 0.0  # Neutral conditions

        u_star = friction_velocity(wind_speed, height, surface_roughness, stability_parameter)

        # Should be positive and less than wind speed
        assert u_star > 0
        assert u_star < wind_speed

    def test_obukhov_length_standard(self):
        """Test Obukhov length calculation."""
        friction_velocity_val = 0.5  # m/s
        sensible_heat_flux = 200.0  # W/m^2
        potential_temperature = 298.0  # K

        l_obukhov = monin_obukhov_length(friction_velocity_val, potential_temperature, 1.2, 1004.0, sensible_heat_flux)

        # Should be finite and reasonable
        assert np.isfinite(l_obukhov)
        assert l_obukhov < 0


class TestHeatFluxes:
    """Test heat flux calculations."""

    def test_sensible_heat_flux_standard(self):
        """Test sensible heat flux calculation."""
        air_temperature = 298.0  # K
        surface_temperature = 300.0  # K
        r_a = 50.0

        shf = sensible_heat_flux(air_temperature, surface_temperature, r_a)

        assert np.isfinite(shf)
        assert shf > 0

    def test_latent_heat_flux_standard(self):
        """Test latent heat flux calculation."""
        vapor_pressure_air = 1500.0  # Pa
        vapor_pressure_surface = 2000.0  # Pa
        aerodynamic_resistance = 50.0  # s/m

        lef = latent_heat_flux(vapor_pressure_air, vapor_pressure_surface, aerodynamic_resistance)

        assert np.isfinite(lef)
        assert lef > 0


class TestStatisticalFunctions:
    """Test basic statistical functions."""

    def test_standard_deviation_basic(self):
        """Test standard deviation calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        std = standard_deviation(data)
        assert np.isfinite(std)
        assert std > 0

    def test_correlation_coefficient_basic(self):
        """Test correlation coefficient calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        corr = correlation_coefficient(x, y)
        assert abs(corr - 1.0) < 1e-10

    def test_covariance_basic(self):
        """Test covariance calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        cov = covariance(x, y)
        assert cov > 0


class TestTurbulenceKineticEnergy:
    """Test turbulence kinetic energy calculations."""

    def test_turbulence_kinetic_energy_basic(self):
        """Test TKE calculation from velocity components."""
        u_prime = np.array([1.0, -0.5, 0.8, -1.2, 0.3])
        v_prime = np.array([0.5, -0.8, 1.1, -0.2, 0.9])
        w_prime = np.array([0.2, -0.3, 0.1, -0.4, 0.6])
        tke = turbulence_kinetic_energy(u_prime, v_prime, w_prime)
        assert tke > 0


class TestMomentumFlux:
    """Test momentum flux calculations."""

    def test_momentum_flux_basic(self):
        """Test momentum flux calculation."""
        u_prime = np.array([1.0, -0.5, 0.8, -1.2, 0.3])
        w_prime = np.array([0.2, -0.3, 0.1, -0.4, 0.6])
        air_density = 1.225
        tau = momentum_flux(u_prime, w_prime, air_density)
        assert np.isfinite(tau)
