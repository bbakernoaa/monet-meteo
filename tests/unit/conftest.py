"""
Test configuration and fixtures for unit tests.
"""
import numpy as np
import pytest
import xarray as xr

@pytest.fixture
def sample_pressure():
    """Sample pressure values in hPa."""
    return np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 300.0, 200.0, 100.0])

@pytest.fixture
def sample_temperature():
    """Sample temperature values in K."""
    return np.array([298.15, 290.0, 280.0, 270.0, 250.0, 230.0, 220.0, 210.0])

@pytest.fixture
def sample_humidity():
    """Sample relative humidity values (0-1)."""
    return np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

@pytest.fixture
def sample_wind_u():
    """Sample u-wind component in m/s."""
    return np.array([10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])

@pytest.fixture
def sample_wind_v():
    """Sample v-wind component in m/s."""
    return np.array([5.0, 8.0, 12.0, 18.0, 22.0, 25.0, 28.0, 30.0])

@pytest.fixture
def sample_latitude():
    """Sample latitude values in degrees."""
    return np.array([30.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0])

@pytest.fixture
def sample_height():
    """Sample height values in meters."""
    return np.array([0.0, 100.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

@pytest.fixture
def sample_xarray_data(sample_pressure, sample_temperature, sample_humidity):
    """Sample xarray DataArray for testing."""
    coords = {'pressure': sample_pressure, 'level': range(len(sample_pressure))}
    data_vars = {
        'temperature': (['level'], sample_temperature),
        'relative_humidity': (['level'], sample_humidity),
        'pressure': (['level'], sample_pressure)
    }
    return xr.Dataset(data_vars, coords=coords)

@pytest.fixture
def scalar_test_values():
    """Scalar test values for edge case testing."""
    return {
        'pressure': 1013.25,  # hPa
        'temperature': 288.15,  # K (15°C)
        'humidity': 0.5,  # 50%
        'mixing_ratio': 0.01,  # kg/kg
        'dewpoint': 278.15,  # K (-5°C)
        'wind_u': 10.0,  # m/s
        'wind_v': 5.0,  # m/s
        'latitude': 45.0,  # degrees
        'height': 100.0  # m
    }

@pytest.fixture
def extreme_values():
    """Extreme values for testing edge cases."""
    return {
        'very_cold': 180.0,  # K (-93°C)
        'very_hot': 330.0,  # K (57°C)
        'very_low_pressure': 100.0,  # hPa
        'very_high_pressure': 1100.0,  # hPa
        'very_humid': 0.99,  # 99%
        'very_dry': 0.01,  # 1%
        'very_strong_wind': 50.0,  # m/s
        'very_weak_wind': 0.1  # m/s
    }

# Mathematical constants for testing
@pytest.fixture
def expected_constants():
    """Expected constant values for validation."""
    return {
        'R_d': 287.04,  # J kg⁻¹ K⁻¹
        'R_v': 461.5,   # J kg⁻¹ K⁻¹
        'c_pd': 1004.0, # J kg⁻¹ K⁻¹
        'g': 9.8065,    # m s⁻²
        'Omega': 7.292e-5  # s⁻¹
    }