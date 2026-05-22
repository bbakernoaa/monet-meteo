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
    monin_obukhov_length,
    stability_parameter,
    sensible_heat_flux,
    latent_heat_flux,
    friction_velocity_from_wind
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
        roughness_length = 0.01  # m
        stability_param = 0.0  # Neutral conditions
        
        u_star = friction_velocity_from_wind(wind_speed, height, roughness_length, stability_param)
        
        # Should be positive and less than wind speed
        assert u_star > 0
        assert u_star < wind_speed
    
    def test_monin_obukhov_length_standard(self):
        """Test Monin-Obukhov length calculation."""
        u_star = 0.5  # m/s
        temperature = 298.15
        air_density = 1.2
        specific_heat = 1004.0
        sensible_heat_flux_val = 200.0  # W/m^2
        
        L = monin_obukhov_length(u_star, temperature, air_density, specific_heat, sensible_heat_flux_val)
        
        # Should be finite and reasonable
        assert np.isfinite(L)
        assert L < 0  # Unstable for positive SHF
    
class TestHeatFluxes:
    """Test heat flux calculations."""
    
    def test_sensible_heat_flux_standard(self):
        """Test sensible heat flux calculation."""
        air_temperature = 298.0  # K
        surface_temperature = 300.0  # K
        r_a = 50.0 # Aerodynamic resistance
        
        shf = sensible_heat_flux(air_temperature, surface_temperature, r_a)
        
        assert np.isfinite(shf)
        assert shf > 0
    
    def test_latent_heat_flux_standard(self):
        """Test latent heat flux calculation."""
        vapor_pressure_air = 1500.0  # Pa
        vapor_pressure_surface = 2000.0  # Pa
        r_a = 50.0
        
        lef = latent_heat_flux(vapor_pressure_air, vapor_pressure_surface, r_a)
        
        assert np.isfinite(lef)
        assert lef > 0


class TestStatisticalFunctions:
    """Test basic statistical functions."""
    
    def test_standard_deviation_basic(self):
        """Test standard deviation calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        std = np.std(data, ddof=1)
        
        # Should be finite and positive
        assert np.isfinite(std)
        assert std > 0
