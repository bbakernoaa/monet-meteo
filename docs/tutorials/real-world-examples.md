# Real-World Meteorological Data Processing Examples

This tutorial demonstrates practical applications of Monet-Meteo for real-world meteorological data processing scenarios. Each example addresses common tasks faced by meteorologists, climate scientists, and atmospheric researchers.

## 🎯 Tutorial Overview

### Learning Objectives
- Process real-world meteorological datasets from various sources
- Apply Monet-Meteo to common meteorological analysis tasks
- Handle real-world data quality issues and gaps
- Implement operational meteorological workflows
- Create reproducible analysis pipelines

### Prerequisites
- Monet-Meteo installed
- Familiarity with meteorological data formats
- Basic understanding of atmospheric science concepts

## 🌡️ Weather Station Data Processing

### Example 1: Automated Weather Station Quality Control

```python
import monet_meteo as mm
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def quality_control_weather_data(df, station_metadata):
    """
    Perform quality control on weather station data
    
    Parameters
    ----------
    df : pandas.DataFrame
        Weather station data with columns:
        ['timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed', 'wind_dir']
    station_metadata : dict
        Station metadata including location, elevation, etc.
        
    Returns
    -------
    pandas.DataFrame
        Quality-controlled data
    """
    # Apply physical limits
    physical_limits = {
        'temperature': (-60, 60),  # °C
        'humidity': (0, 100),      # %
        'pressure': (800, 1100),  # hPa
        'wind_speed': (0, 150),    # m/s
        'wind_dir': (0, 360)      # degrees
    }
    
    # Remove values outside physical limits
    for variable, (min_val, max_val) in physical_limits.items():
        if variable in df.columns:
            df = df[(df[variable] >= min_val) & (df[variable] <= max_val)]
    
    # Remove duplicate timestamps
    df = df.drop_duplicates(subset=['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Handle missing data
    df = df.interpolate(method='linear', limit=3)  # Linear interpolation for gaps ≤ 3 readings
    
    # Calculate derived variables for validation
    df['potential_temperature'] = mm.potential_temperature(
        df['pressure'], 
        mm.convert_temperature(df['temperature'], 'C', 'K')
    )
    
    df['dewpoint'] = mm.dewpoint_from_relative_humidity(
        mm.convert_temperature(df['temperature'], 'C', 'K'),
        df['humidity'] / 100
    )
    
    # Flag potential errors using thermodynamic consistency
    # Check if dewpoint > temperature (physically impossible)
    df['thermo_consistency'] = df['dewpoint'] <= df['temperature']
    
    # Check for rapid temperature changes (> 10°C in 1 hour)
    df['temp_change'] = df['temperature'].diff().abs()
    df['rapid_change'] = df['temp_change'] > 10
    
    return df

# Example usage
# Load weather station data
# station_data = pd.read_csv('weather_station_12345.csv')
# metadata = {'elevation': 150, 'latitude': 40.7, 'longitude': -74.0}
# qc_data = quality_control_weather_data(station_data, metadata)
```

### Example 2: Automated Surface Observing System (ASOS) Processing

```python
def process_asos_data(file_path):
    """
    Process ASOS METAR data using Monet-Meteo
    
    Parameters
    ----------
    file_path : str
        Path to ASOS data file
        
    Returns
    -------
    pandas.DataFrame
        Processed meteorological data
    """
    # Load raw ASOS data
    df = pd.read_csv(file_path)
    
    # Parse METAR codes (simplified example)
    def parse_temperature(temp_str):
        """Parse temperature from METAR code"""
        if isinstance(temp_str, str):
            if temp_str.startswith('M'):  # Negative temperature
                return -float(temp_str[1:]) / 10
            else:
                return float(temp_str) / 10
        return temp_str
    
    def parse_wind(wind_str):
        """Parse wind speed and direction"""
        if isinstance(wind_str, str):
            parts = wind_str.split('G') if 'G' in wind_str else [wind_str]
            speed = float(parts[0]) / 10 if parts[0] else 0
            return speed
        return wind_str
    
    # Apply parsing
    df['temperature_c'] = df['temperature'].apply(parse_temperature)
    df['wind_speed_ms'] = df['wind'].apply(parse_wind)
    
    # Convert units
    df['pressure_hpa'] = df['pressure']
    df['relative_humidity'] = df['humidity']
    
    # Calculate missing values using Monet-Meteo
    df['potential_temperature'] = mm.potential_temperature(
        df['pressure_hpa'],
        mm.convert_temperature(df['temperature_c'], 'C', 'K')
    )
    
    df['dewpoint_c'] = mm.convert_temperature(
        mm.dewpoint_from_relative_humidity(
            mm.convert_temperature(df['temperature_c'], 'C', 'K'),
            df['relative_humidity'] / 100
        ),
        'K', 'C'
    )
    
    # Calculate heat index and wind chill
    df['heat_index'] = mm.heat_index(
        mm.convert_temperature(df['temperature_c'], 'C', 'K'),
        df['relative_humidity'] / 100
    )
    
    df['wind_chill'] = mm.wind_chill(
        mm.convert_temperature(df['temperature_c'], 'C', 'K'),
        df['wind_speed_ms']
    )
    
    return df

# Example usage
# asos_data = process_asos_data('asos_data.csv')
```

## 🌊 Upper Air Data Processing

### Example 3: Radiosonde Sounding Analysis

```python
def analyze_radiosonde_sounding(sounding_data):
    """
    Analyze radiosonde sounding data
    
    Parameters
    ----------
    sounding_data : pandas.DataFrame
        Sounding data with columns:
        ['pressure', 'height', 'temperature', 'dewpoint', 'wind_speed', 'wind_dir']
        
    Returns
    -------
    dict
        Sounding analysis results
    """
    # Sort by pressure (descending)
    sounding_data = sounding_data.sort_values('pressure', ascending=False)
    
    # Calculate atmospheric layers
    sounding_data['layer_thickness'] = sounding_data['height'].diff()
    sounding_data['temp_gradient'] = sounding_data['temperature'].diff() / sounding_data['layer_thickness']
    
    # Identify inversions
    inversions = []
    for i in range(1, len(sounding_data)):
        if sounding_data.iloc[i]['temp_gradient'] > 0 and sounding_data.iloc[i-1]['temp_gradient'] > 0:
            inversion_start = sounding_data.iloc[i-1]
            inversion_end = sounding_data.iloc[i]
            inversions.append({
                'pressure_start': inversion_start['pressure'],
                'pressure_end': inversion_end['pressure'],
                'height_start': inversion_start['height'],
                'height_end': inversion_end['height'],
                'strength': inversion_end['temperature'] - inversion_start['temperature']
            })
    
    # Calculate CAPE and CIN (simplified)
    sounding_data['virtual_temp'] = mm.virtual_temperature(
        sounding_data['temperature'],
        mm.mixing_ratio(
            mm.saturation_vapor_pressure(sounding_data['temperature']),
            sounding_data['pressure'] * 100
        )
    )
    
    # Calculate mixing ratio
    sounding_data['mixing_ratio'] = mm.mixing_ratio(
        mm.saturation_vapor_pressure(sounding_data['temperature']) * 
        np.exp((17.67 * (sounding_data['dewpoint'] - 273.15)) / 
               (sounding_data['dewpoint'] - 29.65)),
        sounding_data['pressure'] * 100
    )
    
    # Calculate lifting condensation level
    surface_data = sounding_data.iloc[0]
    lcl_height = mm.lifting_condensation_level(
        surface_data['temperature'],
        mm.convert_temperature(surface_data['dewpoint'], 'C', 'K')
    )
    
    # Find LCL pressure
    lcl_pressure_row = sounding_data[
        (sounding_data['height'] >= lcl_height) & 
        (sounding_data['height'] <= lcl_height + 100)
    ].iloc[0] if len(sounding_data[
        (sounding_data['height'] >= lcl_height) & 
        (sounding_data['height'] <= lcl_height + 100)
    ]) > 0 else surface_data
    
    lcl_pressure = lcl_pressure_row['pressure']
    
    return {
        'inversions': inversions,
        'lcl_height': lcl_height,
        'lcl_pressure': lcl_pressure,
        'surface_conditions': surface_data.to_dict(),
        'sounding_profile': sounding_data.to_dict('records')
    }

# Example usage
# sounding = pd.read_csv('radiosonde_profile.csv')
# analysis = analyze_radiosonde_sounding(sounding)
```

## 🛰️ Satellite Data Processing

### Example 4: Satellite-derived Atmospheric Products

```python
def process_satellite_atmospheric_data(satellite_data):
    """
    Process satellite atmospheric data using Monet-Meteo
    
    Parameters
    ----------
    satellite_data : xarray.Dataset
        Satellite data with atmospheric variables
        
    Returns
    -------
    xarray.Dataset
        Processed atmospheric data
    """
    # Convert satellite brightness temperature to physical temperature
    # Assuming sat_data contains brightness temperature
    satellite_data['surface_temperature'] = mm.convert_temperature(
        satellite_data['brightness_temp'], 'K', 'C'
    )
    
    # Calculate atmospheric moisture profiles
    satellite_data['vapor_pressure'] = mm.saturation_vapor_pressure(
        satellite_data['air_temperature']
    )
    
    satellite_data['relative_humidity'] = mm.relative_humidity(
        satellite_data['vapor_pressure'],
        mm.saturation_vapor_pressure(satellite_data['air_temperature'])
    )
    
    # Calculate atmospheric stability from satellite observations
    satellite_data['stability_parameter'] = mm.stability_parameter(
        satellite_data['height'],
        satellite_data['obukhov_length']
    )
    
    # Calculate surface energy balance components
    satellite_data['longwave_radiation'] = mm.stephan_boltzmann(
        satellite_data['surface_temperature'] + 273.15
    )
    
    # Calculate sensible heat flux
    satellite_data['sensible_heat_flux'] = mm.sensible_heat_flux(
        satellite_data['air_temperature'],
        satellite_data['surface_temperature'] + 273.15,
        satellite_data['aerodynamic_resistance']
    )
    
    return satellite_data

# Example usage
# sat_data = xr.open_dataset('satellite_atmospheric.nc')
# processed_sat = process_satellite_atmospheric_data(sat_data)
```

## 🌪️ Severe Weather Analysis

### Example 5: Tornado and Severe Thunderstorm Analysis

```python
def analyze_severe_weather_conditions(observation_data):
    """
    Analyze conditions for severe weather development
    
    Parameters
    ----------
    observation_data : pandas.DataFrame
        Surface and upper air observations
        
    Returns
    -------
    dict
        Severe weather analysis results
    """
    # Calculate atmospheric instability indices
    def calculate_sherwood_index(temp_850, temp_500, temp_700):
        """Calculate Sherwood Index for severe weather potential"""
        return (temp_850 - temp_500) / (temp_700 - temp_500)
    
    def calculate_showalter_index(temp_850, temp_700, temp_500, dewpoint_850):
        """Calculate Showalter Index"""
        # Simplified calculation
        temp_700_850 = temp_850 - (temp_850 - temp_700) * 2/3
        dewpoint_700 = dewpoint_850 - 20  # Simplified
        return (temp_700_850 - temp_500) - (dewpoint_700 - temp_500)
    
    # Calculate CAPE (simplified)
    surface_temp = observation_data[observation_data['pressure'] == 1000]['temperature'].iloc[0]
    surface_dewpoint = observation_data[observation_data['pressure'] == 1000]['dewpoint'].iloc[0]
    
    lcl_pressure = mm.calculate_lcl_pressure(surface_temp, surface_dewpoint)
    cape = mm.calculate_cape(observation_data, lcl_pressure)
    
    # Calculate bulk shear
    u_wind_500 = observation_data[observation_data['pressure'] == 500]['u_wind'].iloc[0]
    v_wind_500 = observation_data[observation_data['pressure'] == 500]['v_wind'].iloc[0]
    u_wind_850 = observation_data[observation_data['pressure'] == 850]['u_wind'].iloc[0]
    v_wind_850 = observation_data[observation_data['pressure'] == 850]['v_wind'].iloc[0]
    
    bulk_shear = np.sqrt((u_wind_500 - u_wind_850)**2 + (v_wind_500 - v_wind_850)**2)
    
    # Calculate storm-relative helicity
    srh = mm.calculate_storm_relative_helicity(
        observation_data['u_wind'],
        observation_data['v_wind'],
        observation_data['height']
    )
    
    return {
        'cape': cape,
        'bulk_shear': bulk_shear,
        'storm_relative_helicity': srh,
        'lcl_pressure': lcl_pressure,
        'instability_indices': {
            'sherwood_index': calculate_sherwood_index(
                observation_data[observation_data['pressure'] == 850]['temperature'].iloc[0],
                observation_data[observation_data['pressure'] == 500]['temperature'].iloc[0],
                observation_data[observation_data['pressure'] == 700]['temperature'].iloc[0]
            ),
            'showalter_index': calculate_showalter_index(
                observation_data[observation_data['pressure'] == 850]['temperature'].iloc[0],
                observation_data[observation_data['pressure'] == 700]['temperature'].iloc[0],
                observation_data[observation_data['pressure'] == 500]['temperature'].iloc[0],
                observation_data[observation_data['pressure'] == 850]['dewpoint'].iloc[0]
            )
        },
        'severe_weather_risk': assess_severe_weather_risk(cape, bulk_shear, srh)
    }

def assess_severe_weather_risk(cape, bulk_shear, srh):
    """Assess severe weather risk based on indices"""
    risk_level = 'None'
    
    if cape > 2500 and bulk_shear > 20:
        risk_level = 'High'
    elif cape > 1000 and bulk_shear > 15:
        risk_level = 'Moderate'
    elif cape > 500 and bulk_shear > 10:
        risk_level = 'Low'
    
    return risk_level

# Example usage
# severe_data = pd.read_csv('severe_weather_obs.csv')
# analysis = analyze_severe_weather_conditions(severe_data)
```

## 🌊 Marine Meteorological Analysis

### Example 6: Marine Weather Analysis

```python
def analyze_marine_conditions(ocean_data, atmospheric_data):
    """
    Analyze marine meteorological conditions
    
    Parameters
    ----------
    ocean_data : pandas.DataFrame
        Oceanographic data
    atmospheric_data : pandas.DataFrame
        Atmospheric data over ocean
        
    Returns
    -------
    dict
        Marine meteorological analysis
    """
    # Calculate air-sea temperature difference
    air_sea_diff = atmospheric_data['temperature'] - ocean_data['sea_surface_temp']
    
    # Calculate sensible heat flux over ocean
    sensible_heat_flux = mm.sensible_heat_flux(
        mm.convert_temperature(atmospheric_data['temperature'], 'C', 'K'),
        mm.convert_temperature(ocean_data['sea_surface_temp'], 'C', 'K'),
        atmospheric_data['aerodynamic_resistance']
    )
    
    # Calculate latent heat flux over ocean
    latent_heat_flux = mm.latent_heat_flux(
        mm.saturation_vapor_pressure(mm.convert_temperature(atmospheric_data['temperature'], 'C', 'K')) * 
        atmospheric_data['relative_humidity'] / 100,
        mm.saturation_vapor_pressure(mm.convert_temperature(ocean_data['sea_surface_temp'], 'C', 'K')),
        atmospheric_data['aerodynamic_resistance']
    )
    
    # Calculate marine atmospheric stability
    marine_stability = mm.bulk_richardson_number(
        atmospheric_data['u_wind'],
        atmospheric_data['v_wind'],
        atmospheric_data['potential_temperature'],
        atmospheric_data['height']
    )
    
    # Calculate wave growth potential
    wave_growth_potential = calculate_wave_growth_potential(
        atmospheric_data['wind_speed'],
        atmospheric_data['fetch_length']
    )
    
    return {
        'air_sea_temperature_difference': air_sea_diff,
        'sensible_heat_flux': sensible_heat_flux,
        'latent_heat_flux': latent_heat_flux,
        'marine_stability': marine_stability,
        'wave_growth_potential': wave_growth_potential,
        'sea_state': classify_sea_state(atmospheric_data['wind_speed'])
    }

def calculate_wave_growth_potential(wind_speed, fetch_length):
    """Calculate wave growth potential based on wind speed and fetch"""
    # Simplified wave growth calculation
    return wind_speed * np.sqrt(fetch_length) / 100

def classify_sea_state(wind_speed):
    """Classify sea state based on wind speed"""
    if wind_speed < 0.5:
        return 'Calm'
    elif wind_speed < 1.5:
        return 'Light air'
    elif wind_speed < 3.3:
        return 'Light breeze'
    elif wind_speed < 5.5:
        return 'Gentle breeze'
    elif wind_speed < 7.9:
        return 'Moderate breeze'
    elif wind_speed < 10.7:
        return 'Fresh breeze'
    elif wind_speed < 13.9:
        return 'Strong breeze'
    elif wind_speed < 17.2:
        return 'Near gale'
    elif wind_speed < 20.8:
        return 'Gale'
    elif wind_speed < 24.5:
        return 'Strong gale'
    elif wind_speed < 28.5:
        return 'Storm'
    elif wind_speed < 32.7:
        return 'Violent storm'
    else:
        return 'Hurricane'

# Example usage
# ocean_data = pd.read_csv('ocean_conditions.csv')
# atmospheric_data = pd.read_csv('marine_atmospheric.csv')
# marine_analysis = analyze_marine_conditions(ocean_data, atmospheric_data)
```

## 🏔️ Mountain Meteorology

### Example 7: Alpine Meteorological Analysis

```

def analyze_alpine_meteorology(met_data, topography):
    """
    Analyze alpine meteorological conditions
    
    Parameters
    ----------
    met_data : pandas.DataFrame
        Meteorological observations
    topography : dict
        Topographic information including elevation, slope, aspect
        
    Returns
    -------
    dict
        Alpine meteorological analysis
    """
    # Calculate atmospheric pressure at different elevations
    surface_pressure = met_data['pressure'].iloc[0]
    pressure_at_elevation = mm.pressure_to_altitude(
        met_data['elevation'].values, 
        surface_pressure
    )
    
    # Calculate temperature lapse rates
    temperature_lapse_rate = mm.dry_lapse_rate()
    moist_lapse_rate = mm.moist_lapse_rate(
        met_data['temperature'],
        met_data['pressure'] * 100
    )
    
    # Calculate wind loading on slopes
    wind_loading = calculate_wind_loading(
        met_data['wind_speed'],
        met_data['wind_dir'],
        topography['slope'],
        topography['aspect']
    )
    
    # Calculate avalanche conditions
    snow_stability = assess_avalanche_conditions(
        met_data['temperature'],
        met_data['wind_speed'],
        met_data['snow_depth'],
        topography['slope']
    )
    
    # Calculate atmospheric stability in valleys
    valley_stability = mm.bulk_richardson_number(
        met_data['u_wind'],
        met_data['v_wind'],
        met_data['potential_temperature'],
        met_data['height']
    )
    
    return {
        'pressure_at_elevation': pressure_at_elevation,
        'temperature_lapse_rates': {
            'dry': temperature_lapse_rate,
            'moist': moist_lapse_rate
        },
        'wind_loading': wind_loading,
        'avalanche_risk': snow_stability,
        'valley_stability': valley_stability,
        'mountain_breeze': detect_mountain_breeze(met_data)
    }

def calculate_wind_loading(wind_speed, wind_dir, slope, aspect):
    """Calculate wind loading on mountain slopes"""
    # Simplified wind loading calculation
    # In reality, this would involve complex 3D flow calculations
    return wind_speed * np.cos(np.radians(wind_dir - aspect)) * np.sin(np.radians(slope))

def assess_avalanche_conditions(temperature, wind_speed, snow_depth, slope):
    """Assess avalanche conditions based on meteorological factors"""
    risk_level = 'Low'
    
    # Temperature factors
    if temperature > 0 and snow_depth > 30:
        risk_level = 'High'
    elif temperature > -5 and wind_speed > 15:
        risk_level = 'Moderate'
    elif slope > 35:
        risk_level = 'Moderate' if risk_level == 'Low' else risk_level
    
    return risk_level

def detect_mountain_breeze(met_data):
    """Detect mountain breeze patterns"""
    # Analyze diurnal wind patterns
    met_data['hour'] = pd.to_datetime(met_data['timestamp']).dt.hour
    diurnal_pattern = met_data.groupby('hour')['wind_speed'].mean()
    
    # Simple mountain breeze detection
    if diurnal_pattern.max() - diurnal_pattern.min() > 5:
        return 'Mountain breeze detected'
    else:
        return 'No significant mountain breeze'

# Example usage
# alpine_data = pd.read_csv('alpine_meteo.csv')
# topo_data = {'elevation': 2500, 'slope': 30, 'aspect': 180}
# alpine_analysis = analyze_alpine_meteorology(alpine_data, topo_data)
```

## 🏭 Industrial Meteorology

### Example 8: Air Quality and Dispersion Analysis

```python
def analyze_air_dispersion(meteorological_data, emission_data):
    """
    Analyze atmospheric dispersion of pollutants
    
    Parameters
    ----------
    meteorological_data : pandas.DataFrame
        Meteorological observations
    emission_data : pandas.DataFrame
        Emission source data
        
    Returns
    -------
    dict
        Air dispersion analysis results
    """
    # Calculate atmospheric stability class
    stability_class = determine_stability_class(
        meteorological_data['wind_speed'],
        meteorological_data['solar_radiation'],
        meteorological_data['cloud_cover']
    )
    
    # Calculate mixing height
    mixing_height = mm.atmospheric_boundary_layer_height(
        meteorological_data['surface_temperature'],
        meteorological_data['potential_temperature_gradient'],
        meteorological_data['wind_speed'],
        meteorological_data['height']
    )
    
    # Calculate dispersion parameters
    dispersion_params = calculate_dispersion_parameters(
        meteorological_data['wind_speed'],
        stability_class,
        emission_data['stack_height'],
        emission_data['exit_velocity']
    )
    
    # Calculate plume rise
    plume_rise = calculate_plume_rise(
        emission_data['heat_flux'],
        emission_data['stack_height'],
        meteorological_data['wind_speed'],
        stability_class
    )
    
    # Calculate ground-level concentrations (simplified)
    ground_concentration = calculate_ground_concentration(
        emission_data['emission_rate'],
        dispersion_params['sigma_y'],
        dispersion_params['sigma_z'],
        plume_rise,
        mixing_height
    )
    
    return {
        'stability_class': stability_class,
        'mixing_height': mixing_height,
        'dispersion_parameters': dispersion_params,
        'plume_rise': plume_rise,
        'ground_concentration': ground_concentration,
        'air_quality_impact': assess_air_quality_impact(ground_concentration)
    }

def determine_stability_class(wind_speed, solar_radiation, cloud_cover):
    """Determine Pasquill-Gifford stability class"""
    # Simplified stability determination
    if wind_speed < 2:
        if solar_radiation > 300:
            return 'A'  # Very unstable
        else:
            return 'F'  # Very stable
    elif wind_speed < 5:
        if solar_radiation > 300:
            return 'B'  # Moderately unstable
        else:
            return 'E'  # Slightly stable
    elif wind_speed < 6:
        return 'C'  # Slightly unstable
    else:
        return 'D'  # Neutral

def calculate_dispersion_parameters(wind_speed, stability_class, stack_height, exit_velocity):
    """Calculate atmospheric dispersion parameters"""
    # Simplified dispersion parameter calculation
    # In reality, these would be based on Pasquill-Gifford equations
    sigma_y = 0.1 * wind_speed * stack_height
    sigma_z = 0.05 * wind_speed * stack_height
    
    return {
        'sigma_y': sigma_y,
        'sigma_z': sigma_z,
        'wind_speed': wind_speed
    }

def calculate_plume_rise(heat_flux, stack_height, wind_speed, stability_class):
    """Calculate plume rise using Briggs formula"""
    if heat_flux > 0:
        # Simplified plume rise calculation
        delta_T = 50  # Temperature difference (simplified)
        plume_rise = (1.6 * (heat_flux / (wind_speed * stack_height))**(1/3)) * stack_height
    else:
        plume_rise = 0
    
    return plume_rise

def calculate_ground_concentration(emission_rate, sigma_y, sigma_z, plume_rise, mixing_height):
    """Calculate ground-level concentration (simplified)"""
    # Simplified Gaussian plume model
    x = 1000  # Distance from source (m)
    y = 0      # Crosswind distance (m)
    z = 0      # Height above ground (m)
    
    concentration = (emission_rate / (2 * np.pi * sigma_y * sigma_z * wind_speed)) * \
                   np.exp(-0.5 * (y / sigma_y)**2) * \
                   (np.exp(-0.5 * ((z - plume_rise) / sigma_z)**2) + 
                    np.exp(-0.5 * ((z + plume_rise) / sigma_z)**2))
    
    return concentration

def assess_air_quality_impact(concentration):
    """Assess air quality impact based on concentration"""
    if concentration < 10:
        return 'Good'
    elif concentration < 50:
        return 'Moderate'
    elif concentration < 100:
        return 'Unhealthy for sensitive groups'
    elif concentration < 150:
        return 'Unhealthy'
    else:
        return 'Very unhealthy'

# Example usage
# met_data = pd.read_csv('meteorological_obs.csv')
# emission_data = pd.read_csv('emission_sources.csv')
# dispersion_analysis = analyze_air_dispersion(met_data, emission_data)
```

## 📊 Operational Meteorological Workflows

### Example 9: Automated Weather Forecast Verification

```python
def verify_weather_forecasts(observed_data, forecast_data):
    """
    Verify weather forecast accuracy using meteorological statistics
    
    Parameters
    ----------
    observed_data : pandas.DataFrame
        Observed meteorological data
    forecast_data : pandas.DataFrame
        Forecast meteorological data
        
    Returns
    -------
    dict
        Forecast verification statistics
    """
    # Calculate verification metrics for temperature
    temp_bias = np.mean(forecast_data['temperature'] - observed_data['temperature'])
    temp_mae = np.mean(np.abs(forecast_data['temperature'] - observed_data['temperature']))
    temp_rmse = np.sqrt(np.mean((forecast_data['temperature'] - observed_data['temperature'])**2))
    
    # Calculate verification metrics for precipitation
    precipitation_threshold = 0.1  # mm
    hits = np.sum((forecast_data['precipitation'] > precipitation_threshold) & 
                  (observed_data['precipitation'] > precipitation_threshold))
    misses = np.sum((forecast_data['precipitation'] <= precipitation_threshold) & 
                    (observed_data['precipitation'] > precipitation_threshold))
    false_alarms = np.sum((forecast_data['precipitation'] > precipitation_threshold) & 
                          (observed_data['precipitation'] <= precipitation_threshold))
    
    # Calculate categorical statistics
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0  # Probability of detection
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0  # False alarm ratio
    ets = (hits - hits * (hits + false_alarms) / (hits + misses + false_alarms)) / \
          (hits + misses + false_alarms - hits * (hits + false_alarms) / (hits + misses + false_alarms)) if \
          (hits + misses + false_alarms) > 0 else 0  # Equitable threat score
    
    # Calculate wind forecast accuracy
    wind_speed_error = np.sqrt((forecast_data['u_wind'] - observed_data['u_wind'])**2 + 
                              (forecast_data['v_wind'] - observed_data['v_wind'])**2)
    wind_speed_rmse = np.sqrt(np.mean(wind_speed_error**2))
    
    # Calculate categorical forecast statistics for severe weather
    severe_threshold = 30  # m/s
    severe_hits = np.sum((forecast_data['wind_speed'] > severe_threshold) & 
                        (observed_data['wind_speed'] > severe_threshold))
    severe_misses = np.sum((forecast_data['wind_speed'] <= severe_threshold) & 
                          (observed_data['wind_speed'] > severe_threshold))
    severe_false_alarms = np.sum((forecast_data['wind_speed'] > severe_threshold) & 
                               (observed_data['wind_speed'] <= severe_threshold))
    
    return {
        'temperature_verification': {
            'bias': temp_bias,
            'mae': temp_mae,
            'rmse': temp_rmse
        },
        'precipitation_verification': {
            'pod': pod,
            'far': far,
            'ets': ets,
            'hits': hits,
            'misses': misses,
            'false_alarms': false_alarms
        },
        'wind_verification': {
            'speed_rmse': wind_speed_rmse
        },
        'severe_weather_verification': {
            'hits': severe_hits,
            'misses': severe_misses,
            'false_alarms': severe_false_alarms,
            'pod_severe': severe_hits / (severe_hits + severe_misses) if (severe_hits + severe_misses) > 0 else 0
        }
    }

# Example usage
# observed = pd.read_csv('observed_weather.csv')
# forecast = pd.read_csv('forecast_weather.csv')
# verification_stats = verify_weather_forecasts(observed, forecast)
```

## 🎯 Best Practices for Real-World Applications

### Data Quality Management
1. **Implement rigorous quality control**
2. **Handle missing data appropriately**
3. **Validate against physical constraints**
4. **Document data processing steps**

### Computational Efficiency
1. **Use appropriate chunking for large datasets**
2. **Implement parallel processing where possible**
3. **Cache intermediate results**
4. **Optimize memory usage**

### Operational Considerations
1. **Handle real-time data streams**
2. **Implement automated quality checks**
3. **Create reproducible workflows**
4. **Build error handling and recovery mechanisms**

### Scientific Rigor
1. **Use established meteorological methods**
2. **Validate results with independent methods**
3. **Document assumptions and limitations**
4. **Perform uncertainty analysis**

## 🚀 Next Steps

- Implement these examples with real data
- Customize for specific applications
- Build automated processing pipelines
- Integrate with operational systems

## 📚 Additional Resources

- [WMO Guidelines for Meteorological Data](https://library.wmo.int/)
- [American Meteorological Society Resources](https://www.ametsoc.org/)
- [National Weather Service Training](https://www.weather.gov/training/)