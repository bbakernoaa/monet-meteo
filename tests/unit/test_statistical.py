"""
Test suite for statistical and micrometeorological functions.

Tests statistical calculations, turbulence parameters, flux calculations,
and other micrometeorological functions.
"""
import numpy as np
import pytest

# Import statistical calculation functions
from monet_meteo.statistical.statistical_calculations import (
    bulk_richardson_number,
    friction_velocity,
    obukhov_length,
    stability_parameter,
    sensible_heat_flux,
    latent_heat_flux,
    momentum_flux,
    turbulence_kinetic_energy,
    standard_deviation,
    correlation_coefficient,
    covariance
)


class TestBulkRichardsonNumber:
    """Test bulk Richardson number calculations."""
    
    def test_bulk_richardson_standard_conditions(self):
        """Test bulk Richardson number at standard conditions."""
        # Typical atmospheric conditions
        u_wind = 10.0  # m/s
        v_wind = 5.0   # m/s
        potential_temperature = np.array([298.0, 300.0])  # K
        height = np.array([10.0, 100.0])  # m
        
        ri = bulk_richardson_number(u_wind, v_wind, potential_temperature, height)
        
        # Should be finite and reasonable
        assert np.isfinite(ri)
        assert ri > 0  # Typically positive in stable conditions
        
        # Magnitude should be reasonable for atmospheric flows
        assert ri < 10.0
    
    def test_bulk_richardson_stability_cases(self):
        """Test bulk Richardson number for different stability cases."""
        u_wind = 5.0
        v_wind = 0.0
        
        # Stable case (temperature increases with height)
        theta_stable = np.array([295.0, 300.0])
        height = np.array([10.0, 100.0])
        ri_stable = bulk_richardson_number(u_wind, v_wind, theta_stable, height)
        
        assert ri_stable > 0
        
        # Unstable case (temperature decreases with height)
        theta_unstable = np.array([300.0, 295.0])
        ri_unstable = bulk_richardson_number(u_wind, v_wind, theta_unstable, height)
        
        assert ri_unstable < 0
        
        # Neutral case (no temperature gradient)
        theta_neutral = np.array([295.0, 295.0])
        ri_neutral = bulk_richardson_number(u_wind, v_wind, theta_neutral, height)
        
        assert abs(ri_neutral) < 0.01  # Should be close to zero
    
    def test_bulk_richardson_wind_speed_dependence(self):
        """Test Richardson number dependence on wind speed."""
        potential_temperature = np.array([298.0, 300.0])
        height = np.array([10.0, 100.0])
        
        # Low wind speed
        ri_low = bulk_richardson_number(2.0, 0.0, potential_temperature, height)
        
        # High wind speed
        ri_high = bulk_richardson_number(20.0, 0.0, potential_temperature, height)
        
        # High wind should give smaller Richardson number (more mechanical turbulence)
        assert ri_high < ri_low
    
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
        assert ri_large > ri_small


class TestTurbulenceParameters:
    """Test turbulence parameter calculations."""
    
    def test_friction_velocity_standard(self):
        """Test friction velocity calculation."""
        wind_speed = 10.0  # m/s
        surface_roughness = 0.01  # m
        stability_parameter = 0.0  # Neutral conditions
        
        u_star = friction_velocity(wind_speed, surface_roughness, stability_parameter)
        
        # Should be positive and less than wind speed
        assert u_star > 0
        assert u_star < wind_speed
        
        # Should be reasonable for typical conditions
        assert u_star > 0.1  # Greater than 0.1 m/s
        assert u_star < 2.0  # Less than 2 m/s
    
    def test_friction_velocity_stability_effects(self):
        """Test friction velocity under different stability conditions."""
        wind_speed = 10.0
        surface_roughness = 0.01
        
        # Neutral conditions
        u_star_neutral = friction_velocity(wind_speed, surface_roughness, 0.0)
        
        # Stable conditions (positive stability parameter)
        u_star_stable = friction_velocity(wind_speed, surface_roughness, 0.1)
        
        # Unstable conditions (negative stability parameter)
        u_star_unstable = friction_velocity(wind_speed, surface_roughness, -0.1)
        
        # Stable should reduce friction velocity, unstable should increase it
        assert u_star_stable < u_star_neutral
        assert u_star_unstable > u_star_neutral
    
    def test_obukhov_length_standard(self):
        """Test Obukhov length calculation."""
        friction_velocity = 0.5  # m/s
        sensible_heat_flux = 200.0  # W/m^2
        potential_temperature = 298.0  # K
        
        l_obukhov = obukhov_length(friction_velocity, sensible_heat_flux, potential_temperature)
        
        # Should be finite and reasonable
        assert np.isfinite(l_obukhov)
        assert abs(l_obukhov) > 1.0  # Greater than 1 meter
        assert abs(l_obukhov) < 1000.0  # Less than 1 km
    
    def test_obukhov_length_stability_interpretation(self):
        """Test Obukhov length stability interpretation."""
        friction_velocity = 0.5
        potential_temperature = 298.0
        
        # Positive heat flux (unstable)
        l_unstable = obukhov_length(friction_velocity, 200.0, potential_temperature)
        assert l_unstable < 0  # Negative for unstable conditions
        
        # Negative heat flux (stable)
        l_stable = obukhov_length(friction_velocity, -100.0, potential_temperature)
        assert l_stable > 0  # Positive for stable conditions
        
        # Zero heat flux (neutral)
        l_neutral = obukhov_length(friction_velocity, 0.0, potential_temperature)
        assert abs(l_neutral) > 1e6  # Very large (approaches infinity)


class TestHeatFluxes:
    """Test heat flux calculations."""
    
    def test_sensible_heat_flux_standard(self):
        """Test sensible heat flux calculation."""
        air_temperature = 298.0  # K
        surface_temperature = 300.0  # K
        wind_speed = 5.0  # m/s
        stability_parameter = 0.0  # Neutral
        
        shf = sensible_heat_flux(air_temperature, surface_temperature, wind_speed, stability_parameter)
        
        # Should be finite and reasonable
        assert np.isfinite(shf)
        assert abs(shf) < 1000.0  # Less than 1000 W/m^2
        
        # Surface warmer than air should give positive flux (from surface to air)
        assert shf > 0
    
    def test_sensible_heat_flux_temperature_difference(self):
        """Test sensible heat flux dependence on temperature difference."""
        wind_speed = 5.0
        stability_parameter = 0.0
        
        # Surface warmer than air
        shf_warm = sensible_heat_flux(298.0, 300.0, wind_speed, stability_parameter)
        assert shf_warm > 0
        
        # Air warmer than surface
        shf_cold = sensible_heat_flux(300.0, 298.0, wind_speed, stability_parameter)
        assert shf_cold < 0
        
        # Same magnitude but opposite sign
        assert abs(shf_warm - abs(shf_cold)) < 10.0
    
    def test_latent_heat_flux_standard(self):
        """Test latent heat flux calculation."""
        vapor_pressure_air = 1500.0  # Pa
        vapor_pressure_surface = 2000.0  # Pa
        aerodynamic_resistance = 50.0  # s/m
        
        lef = latent_heat_flux(vapor_pressure_air, vapor_pressure_surface, aerodynamic_resistance)
        
        # Should be finite and positive (surface wetter than air)
        assert np.isfinite(lef)
        assert lef > 0
        assert lef < 500.0  # Reasonable atmospheric magnitude
    
    def test_latent_heat_flux_gradient_dependence(self):
        """Test latent heat flux dependence on vapor pressure gradient."""
        aerodynamic_resistance = 50.0
        
        # Large gradient
        lef_large = latent_heat_flux(1000.0, 3000.0, aerodynamic_resistance)
        
        # Small gradient
        lef_small = latent_heat_flux(1900.0, 2000.0, aerodynamic_resistance)
        
        # Large gradient should give larger flux
        assert lef_large > lef_small


class TestStatisticalFunctions:
    """Test basic statistical functions."""
    
    def test_standard_deviation_basic(self):
        """Test standard deviation calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        std = standard_deviation(data)
        
        # Should be finite and positive
        assert np.isfinite(std)
        assert std > 0
        
        # Known value for this dataset
        expected = np.std(data, ddof=1)  # Sample standard deviation
        assert abs(std - expected) < 1e-10
    
    def test_standard_deviation_edge_cases(self):
        """Test standard deviation edge cases."""
        # Constant data
        data_constant = np.array([5.0, 5.0, 5.0, 5.0])
        std_constant = standard_deviation(data_constant)
        assert abs(std_constant) < 1e-10
        
        # Single value
        data_single = np.array([5.0])
        std_single = standard_deviation(data_single)
        assert std_single == 0.0
        
        # Large values
        data_large = np.array([1e6, 2e6, 3e6])
        std_large = standard_deviation(data_large)
        assert np.isfinite(std_large)
        assert std_large > 1e5
    
    def test_correlation_coefficient_basic(self):
        """Test correlation coefficient calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])  # Perfect positive correlation
        
        corr = correlation_coefficient(x, y)
        
        # Should be close to 1 for perfect correlation
        assert abs(corr - 1.0) < 1e-10
    
    def test_correlation_coefficient_negative(self):
        """Test negative correlation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 8.0, 6.0, 4.0, 2.0])  # Perfect negative correlation
        
        corr = correlation_coefficient(x, y)
        
        # Should be close to -1
        assert abs(corr - (-1.0)) < 1e-10
    
    def test_correlation_coefficient_no_correlation(self):
        """Test zero correlation."""
        np.random.seed(42)
        x = np.random.randn(100)
        y = np.random.randn(100)
        
        corr = correlation_coefficient(x, y)
        
        # Should be small (but not necessarily zero due to random variation)
        assert abs(corr) < 0.5
    
    def test_covariance_basic(self):
        """Test covariance calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        
        cov = covariance(x, y)
        
        # Should be positive for positively correlated data
        assert cov > 0
        
        # Should match numpy calculation
        expected = np.cov(x, y, ddof=1)[0, 1]
        assert abs(cov - expected) < 1e-10
    
    def test_covariance_unity(self):
        """Test covariance of a variable with itself."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        cov = covariance(x, x)
        
        # Should equal variance
        var = standard_deviation(x) ** 2
        assert abs(cov - var) < 1e-10


class TestTurbulenceKineticEnergy:
    """Test turbulence kinetic energy calculations."""
    
    def test_turbulence_kinetic_energy_basic(self):
        """Test TKE calculation from velocity components."""
        u_prime = np.array([1.0, -0.5, 0.8, -1.2, 0.3])
        v_prime = np.array([0.5, -0.8, 1.1, -0.2, 0.9])
        w_prime = np.array([0.2, -0.3, 0.1, -0.4, 0.6])
        
        tke = turbulence_kinetic_energy(u_prime, v_prime, w_prime)
        
        # Should be positive
        assert tke > 0
        
        # Should be finite
        assert np.isfinite(tke)
        
        # Should be reasonable for typical turbulence intensities
        assert tke < 10.0  # Less than 10 m²/s²
    
    def test_turbulence_kinetic_energy_zero_mean(self):
        """Test TKE with zero-mean fluctuations."""
        # Create fluctuations with zero mean
        u_prime = np.array([1.0, -1.0, 0.5, -0.5, 0.0])
        v_prime = np.array([0.8, -0.8, 0.2, -0.2, 0.0])
        w_prime = np.array([0.3, -0.3, 0.1, -0.1, 0.0])
        
        tke = turbulence_kinetic_energy(u_prime, v_prime, w_prime)
        
        # Should be positive (variance is always positive)
        assert tke > 0
        
        # Known calculation
        expected = 0.5 * (np.mean(u_prime**2) + np.mean(v_prime**2) + np.mean(w_prime**2))
        assert abs(tke - expected) < 1e-10


class TestMomentumFlux:
    """Test momentum flux calculations."""
    
    def test_momentum_flux_basic(self):
        """Test momentum flux calculation."""
        u_prime = np.array([1.0, -0.5, 0.8, -1.2, 0.3])
        w_prime = np.array([0.2, -0.3, 0.1, -0.4, 0.6])
        air_density = 1.225  # kg/m³
        
        tau = momentum_flux(u_prime, w_prime, air_density)
        
        # Should be finite
        assert np.isfinite(tau)
        
        # Should be reasonable atmospheric magnitude
        assert abs(tau) < 10.0  # Less than 10 N/m²
    
    def test_momentum_flux_density_dependence(self):
        """Test momentum flux dependence on air density."""
        u_prime = np.array([1.0, -0.5, 0.8, -1.2, 0.3])
        w_prime = np.array([0.2, -0.3, 0.1, -0.4, 0.6])
        
        # Sea level density
        tau_sea_level = momentum_flux(u_prime, w_prime, 1.225)
        
        # High altitude density
        tau_high_alt = momentum_flux(u_prime, w_prime, 0.8)
        
        # Higher density should give larger flux
        assert abs(tau_sea_level) > abs(tau_high_alt)


class TestStatisticalEdgeCases:
    """Test edge cases in statistical functions."""
    
    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        empty_array = np.array([])
        
        # Standard deviation of empty array
        std_empty = standard_deviation(empty_array)
        assert np.isnan(std_empty)
        
        # Correlation of empty arrays
        corr_empty = correlation_coefficient(empty_array, empty_array)
        assert np.isnan(corr_empty)
    
    def test_identical_arrays(self):
        """Test behavior with identical arrays."""
        x = np.array([1.0, 1.0, 1.0])
        y = np.array([1.0, 1.0, 1.0])
        
        # Standard deviation should be zero
        assert standard_deviation(x) == 0.0
        
        # Correlation should be undefined (division by zero)
        corr = correlation_coefficient(x, y)
        assert np.isnan(corr)
    
    def test_large_arrays(self):
        """Test behavior with large arrays."""
        np.random.seed(42)
        n = 10000
        x = np.random.randn(n)
        y = 2 * x + 0.1 * np.random.randn(n)  # Correlated with noise
        
        # Should handle large arrays efficiently
        corr = correlation_coefficient(x, y)
        assert 0.8 < corr < 1.0  # Should be highly correlated
        
        std = standard_deviation(x)
        assert 0.9 < std < 1.1  # Should be close to 1 for standard normal
    
    def test_extreme_values(self):
        """Test behavior with extreme values."""
        # Very large values
        x_large = np.array([1e10, 2e10, 3e10])
        std_large = standard_deviation(x_large)
        assert np.isfinite(std_large)
        assert std_large > 1e9
        
        # Very small values
        x_small = np.array([1e-10, 2e-10, 3e-10])
        std_small = standard_deviation(x_small)
        assert np.isfinite(std_small)
        assert std_small > 1e-11