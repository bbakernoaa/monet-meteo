# Derived Parameters Module

The derived parameters module provides functions for calculating commonly used derived meteorological parameters including heat index, wind chill, and comfort indices.

## Functions

### `heat_index(temperature, relative_humidity, pressure=None)`
Calculate heat index, which is the apparent temperature felt by the human body due to humidity.

```python
mm.heat_index(temperature, relative_humidity, pressure=None)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `relative_humidity` (float, numpy.ndarray, or xarray.DataArray): Relative humidity as dimensionless value (0-1)
- `pressure` (float, numpy.ndarray, or xarray.DataArray, optional): Atmospheric pressure in Pa, default is standard pressure

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Heat index in Kelvin

**Example:**
```python
temperature = 308.15  # K (35°C)
relative_humidity = 0.7  # 70%
heat_index = mm.heat_index(temperature, relative_humidity)
print(f"Heat index: {heat_index:.2f} K ({mm.convert_temperature(heat_index, 'K', 'C'):.1f}°C)")
```

### `wind_chill(temperature, wind_speed, pressure=None)`
Calculate wind chill temperature, which is the apparent temperature felt due to wind effects.

```python
mm.wind_chill(temperature, wind_speed, pressure=None)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `wind_speed` (float, numpy.ndarray, or xarray.DataArray): Wind speed at 10m height in m/s
- `pressure` (float, numpy.ndarray, or xarray.DataArray, optional): Atmospheric pressure in Pa, default is standard pressure

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Wind chill temperature in Kelvin

**Example:**
```python
temperature = 263.15  # K (-10°C)
wind_speed = 10.0  # m/s
wind_chill = mm.wind_chill(temperature, wind_speed)
print(f"Wind chill: {wind_chill:.2f} K ({mm.convert_temperature(wind_chill, 'K', 'C'):.1f}°C)")
```

### `lifting_condensation_level(temperature, dewpoint)`
Calculate lifting condensation level (LCL) height.

```python
mm.lifting_condensation_level(temperature, dewpoint)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Surface air temperature in Kelvin
- `dewpoint` (float, numpy.ndarray, or xarray.DataArray): Surface dewpoint temperature in Kelvin

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: LCL height in meters

**Example:**
```python
temperature = 298.15  # K (25°C)
dewpoint = 288.15  # K (15°C)
lcl_height = mm.lifting_condensation_level(temperature, dewpoint)
print(f"Lifting condensation level: {lcl_height:.1f} m")
```

### `wet_bulb_temperature(temperature, pressure, relative_humidity)`
Calculate wet bulb temperature using Stull (2011) approximation.

```python
mm.wet_bulb_temperature(temperature, pressure, relative_humidity)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Total atmospheric pressure in Pa
- `relative_humidity` (float, numpy.ndarray, or xarray.DataArray): Relative humidity as dimensionless value (0-1)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Wet bulb temperature in Kelvin

**Example:**
```python
temperature = 298.15  # K (25°C)
pressure = 101325  # Pa
relative_humidity = 0.6  # 60%
wetbulb = mm.wet_bulb_temperature(temperature, pressure, relative_humidity)
print(f"Wet bulb temperature: {wetbulb:.2f} K ({mm.convert_temperature(wetbulb, 'K', 'C'):.1f}°C)")
```

### `dewpoint_temperature(temperature, relative_humidity)`
Calculate dewpoint temperature from temperature and relative humidity.

```python
mm.dewpoint_temperature(temperature, relative_humidity)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `relative_humidity` (float, numpy.ndarray, or xarray.DataArray): Relative humidity as dimensionless value (0-1)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Dewpoint temperature in Kelvin

**Example:**
```python
temperature = 293.15  # K (20°C)
relative_humidity = 0.7  # 70%
dewpoint = mm.dewpoint_temperature(temperature, relative_humidity)
print(f"Dewpoint temperature: {dewpoint:.2f} K ({mm.convert_temperature(dewpoint, 'K', 'C'):.1f}°C)")
```

### `saturation_vapor_pressure(temperature)`
Calculate saturation vapor pressure using the Clausius-Clapeyron equation.

```python
mm.saturation_vapor_pressure(temperature)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Saturation vapor pressure in Pascal

**Example:**
```python
temperature = 293.15  # K (20°C)
sat_vapor_pressure = mm.saturation_vapor_pressure(temperature)
print(f"Saturation vapor pressure: {sat_vapor_pressure:.2f} Pa")
```

### `actual_vapor_pressure(temperature, relative_humidity)`
Calculate actual vapor pressure from temperature and relative humidity.

```python
mm.actual_vapor_pressure(temperature, relative_humidity)
```

**Parameters:**
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature in Kelvin
- `relative_humidity` (float, numpy.ndarray, or xarray.DataArray): Relative humidity as dimensionless value (0-1)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Actual vapor pressure in Pascal

**Example:**
```python
temperature = 293.15  # K (20°C)
relative_humidity = 0.7  # 70%
actual_vapor_pressure = mm.actual_vapor_pressure(temperature, relative_humidity)
print(f"Actual vapor pressure: {actual_vapor_pressure:.2f} Pa")
```

## Usage Patterns

### Human Comfort Assessment
```python
import monet_meteo as mm
import numpy as np

def assess_human_comfort(temperature_k, relative_humidity, wind_speed_mps):
    """
    Comprehensive human comfort assessment using derived parameters
    """
    # Convert to more common units for comfort assessment
    temperature_c = mm.convert_temperature(temperature_k, 'K', 'C')
    
    # Calculate comfort indices
    heat_index_k = mm.heat_index(temperature_k, relative_humidity)
    heat_index_c = mm.convert_temperature(heat_index_k, 'K', 'C')
    
    wind_chill_k = mm.wind_chill(temperature_k, wind_speed_mps)
    wind_chill_c = mm.convert_temperature(wind_chill_k, 'K', 'C')
    
    # Calculate dewpoint for humidity assessment
    dewpoint_k = mm.dewpoint_temperature(temperature_k, relative_humidity)
    dewpoint_c = mm.convert_temperature(dewpoint_k, 'K', 'C')
    
    # Classify comfort levels
    comfort_assessment = classify_comfort_level(temperature_c, relative_humidity, wind_speed_mps)
    
    return {
        'temperature_c': temperature_c,
        'heat_index_c': heat_index_c,
        'wind_chill_c': wind_chill_c,
        'dewpoint_c': dewpoint_c,
        'relative_humidity': relative_humidity,
        'wind_speed_mps': wind_speed_mps,
        'comfort_assessment': comfort_assessment
    }

def classify_comfort_level(temp_c, rh, wind_speed_mps):
    """Classify comfort level based on temperature, humidity, and wind"""
    if temp_c > 35:
        comfort = "Extreme Heat - Dangerous"
    elif temp_c > 30:
        if rh > 0.7:
            comfort = "Very Hot & Humid - Uncomfortable"
        else:
            comfort = "Very Hot - Uncomfortable"
    elif temp_c > 25:
        if rh > 0.7:
            comfort = "Warm & Humid - Slightly Uncomfortable"
        else:
            comfort = "Warm - Comfortable"
    elif temp_c > 20:
        comfort = "Comfortable"
    elif temp_c > 15:
        comfort = "Cool - Comfortable"
    elif temp_c > 10:
        if wind_speed_mps > 10:
            comfort = "Cold - Windy"
        else:
            comfort = "Cool"
    elif temp_c > 0:
        comfort = "Cold"
    else:
        comfort = "Freezing"
    
    if wind_speed_mps > 15:
        comfort += " - Very Windy"
    elif wind_speed_mps > 10:
        comfort += " - Windy"
    
    return comfort

# Example usage
comfort = assess_human_comfort(308.15, 0.7, 5.0)  # 35°C, 70% RH, 5 m/s wind
print(f"Temperature: {comfort['temperature_c']:.1f}°C")
print(f"Heat index: {comfort['heat_index_c']:.1f}°C")
print(f"Comfort level: {comfort['comfort_assessment']}")
```

### Atmospheric Moisture Analysis
```python
def analyze_atmospheric_moisture(temperature_k, pressure_pa):
    """
    Analyze atmospheric moisture characteristics
    """
    # Calculate saturation vapor pressure
    sat_vapor_pressure = mm.saturation_vapor_pressure(temperature_k)
    
    # Calculate dewpoint for different relative humidity values
    rh_values = np.array([0.3, 0.5, 0.7, 0.9, 1.0])  # 30%, 50%, 70%, 90%, 100%
    dewpoints = [mm.dewpoint_temperature(temperature_k, rh) for rh in rh_values]
    
    # Calculate lifting condensation level
    surface_dewpoint = dewpoints[-1]  # 100% RH
    lcl_height = mm.lifting_condensation_level(temperature_k, surface_dewpoint)
    
    # Calculate mixing ratios
    mixing_ratios = []
    for rh in rh_values:
        actual_vapor_pressure = mm.actual_vapor_pressure(temperature_k, rh)
        mixing_ratio = mm.mixing_ratio(actual_vapor_pressure, pressure_pa)
        mixing_ratios.append(mixing_ratio)
    
    return {
        'saturation_vapor_pressure': sat_vapor_pressure,
        'dewpoints': dict(zip([f'rh_{int(rh*100)}%' for rh in rh_values], dewpoints)),
        'lcl_height': lcl_height,
        'mixing_ratios': dict(zip([f'rh_{int(rh*100)}%' for rh in rh_values], mixing_ratios))
    }

# Example usage
moisture_analysis = analyze_atmospheric_moisture(293.15, 101325)
print(f"Saturation vapor pressure: {moisture_analysis['saturation_vapor_pressure']:.1f} Pa")
print(f"LCL height: {moisture_analysis['lcl_height']:.1f} m")
```

### Weather Phenomenon Prediction
```python
def predict_weather_phenomena(temperature_k, relative_humidity, pressure_pa):
    """
    Predict weather phenomena based on atmospheric conditions
    """
    # Calculate derived parameters
    dewpoint_k = mm.dewpoint_temperature(temperature_k, relative_humidity)
    wetbulb_k = mm.wet_bulb_temperature(temperature_k, pressure_pa, relative_humidity)
    lcl_height = mm.lifting_condensation_level(temperature_k, dewpoint_k)
    
    # Convert to Celsius for easier interpretation
    temperature_c = mm.convert_temperature(temperature_k, 'K', 'C')
    dewpoint_c = mm.convert_temperature(dewpoint_k, 'K', 'C')
    wetbulb_c = mm.convert_temperature(wetbulb_k, 'K', 'C')
    
    # Predict weather phenomena
    predictions = []
    
    # Fog potential
    if wetbulb_c - temperature_c < 2.0 and relative_humidity > 0.9:
        predictions.append("High fog potential")
    elif wetbulb_c - temperature_c < 3.0 and relative_humidity > 0.8:
        predictions.append("Moderate fog potential")
    
    # Dew/frost potential
    if temperature_c > 0 and relative_humidity > 0.95:
        predictions.append("High dew potential")
    elif temperature_c <= 0 and relative_humidity > 0.95:
        predictions.append("High frost potential")
    
    # Precipitation potential (simplified)
    if lcl_height < 1000:  # LCL close to surface
        predictions.append("Low-level clouds likely")
    if lcl_height < 500:
        predictions.append("Potential for light precipitation")
    
    # Visibility assessment
    visibility = assess_visibility(temperature_c, dewpoint_c)
    if visibility < 1000:
        predictions.append(f"Low visibility: {visibility:.0f} m")
    
    return {
        'temperature_c': temperature_c,
        'dewpoint_c': dewpoint_c,
        'wetbulb_c': wetbulb_c,
        'relative_humidity': relative_humidity,
        'lcl_height': lcl_height,
        'predictions': predictions
    }

def assess_visibility(temp_c, dewpoint_c):
    """Assess visibility based on temperature-dewpoint spread"""
    spread = temp_c - dewpoint_c
    
    if spread < 1.0:
        return 200  # Very poor
    elif spread < 2.0:
        return 500  # Poor
    elif spread < 3.0:
        return 1000  # Moderate
    elif spread < 5.0:
        return 2000  # Good
    else:
        return 5000  # Very good

# Example usage
weather_predictions = predict_weather_phenomena(288.15, 0.95, 102000)
print(f"Temperature: {weather_predictions['temperature_c']:.1f}°C")
print(f"Dewpoint: {weather_predictions['dewpoint_c']:.1f}°C")
print(f"Predictions: {weather_predictions['predictions']}")
```

### Aviation Weather Assessment
```python
def assessaviation_weather(temperature_k, relative_humidity, wind_speed_mps, pressure_pa):
    """
    Assess aviation weather conditions
    """
    # Calculate aviation-relevant parameters
    dewpoint_k = mm.dewpoint_temperature(temperature_k, relative_humidity)
    wetbulb_k = mm.wet_bulb_temperature(temperature_k, pressure_pa, relative_humidity)
    
    # Calculate density altitude (simplified)
    density_altitude = calculate_density_altitude(temperature_k, pressure_pa)
    
    # Calculate icing potential
    icing_potential = assess_icing_potential(temperature_k, relative_humidity)
    
    # Calculate turbulence potential
    turbulence_potential = assess_turbulence_potential(wind_speed_mps, temperature_k)
    
    return {
        'density_altitude': density_altitude,
        'icing_potential': icing_potential,
        'turbulence_potential': turbulence_potential,
        'recommendations': generate_aviation_recommendations(
            temperature_k, relative_humidity, wind_speed_mps, icing_potential, turbulence_potential
        )
    }

def calculate_density_altitude(temperature_k, pressure_pa):
    """Calculate simplified density altitude"""
    # Simplified calculation
    standard_pressure = 101325  # Pa
    standard_temp = 288.15  # K
    
    altitude_factor = (pressure_pa / standard_pressure) ** 0.234969
    density_altitude = (temperature_k / standard_temp - 1) / 0.0065 * 1000 + \
                      (standard_temp / temperature_k - 1) / 0.0065 * 1000 * (1 - altitude_factor)
    
    return density_altitude

def assess_icing_potential(temperature_k, relative_humidity):
    """Assess aircraft icing potential"""
    temp_c = mm.convert_temperature(temperature_k, 'K', 'C')
    
    if temp_c < -10:
        return "Low"
    elif temp_c < 0 and relative_humidity > 0.8:
        return "Moderate"
    elif temp_c < 2 and relative_humidity > 0.9:
        return "High"
    else:
        return "Low"

def assess_turbulence_potential(wind_speed_mps, temperature_k):
    """Assess turbulence potential based on wind speed"""
    if wind_speed_mps < 5:
        return "Low"
    elif wind_speed_mps < 10:
        return "Moderate"
    elif wind_speed_mps < 15:
        return "High"
    else:
        return "Severe"

def generate_aviation_recommendations(temp_k, rh, wind_speed, icing, turbulence):
    """Generate aviation recommendations"""
    recommendations = []
    
    if icing == "High":
        recommendations.append("Severe icing potential - avoid flight")
    elif icing == "Moderate":
        recommendations.append("Icing possible - use anti-ice")
    
    if turbulence == "High" or turbulence == "Severe":
        recommendations.append(f"Significant turbulence expected - {turbulence} level")
    
    if wind_speed > 20:
        recommendations.append("Strong winds - consider crosswind limitations")
    
    if not recommendations:
        recommendations.append("Conditions appear favorable for flight")
    
    return recommendations

# Example usage
aviation_weather = assess_aviation_weather(273.15, 0.9, 12.0, 101325)
print(f"Density altitude: {aviation_weather['density_altitude']:.0f} ft")
print(f"Icing potential: {aviation_weather['icing_potential']}")
print(f"Turbulence potential: {aviation_weather['turbulence_potential']}")
print(f"Recommendations: {aviation_weather['recommendations']}")
```

## Error Handling

The functions include input validation to ensure physical合理性:

- Temperature values must be above absolute zero
- Relative humidity values must be between 0 and 1
- Wind speed values should be non-negative
- Pressure values must be positive

### Common Errors
```python
# Error: Temperature below absolute zero
try:
    mm.heat_index(-274, 0.5)
except ValueError as e:
    print(f"Error: {e}")

# Error: Relative humidity > 1
try:
    mm.wind_chill(293.15, 5.0, 0.7, 1.5)
except ValueError as e:
    print(f"Error: {e}")

# Error: Negative wind speed
try:
    mm.wind_chill(293.15, -5.0, 0.5)
except ValueError as e:
    print(f"Error: {e}")

# Error: Temperature-dewpoint inversion
try:
    mm.lifting_condensation_level(288.15, 293.15)
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Vectorization
All functions support numpy arrays for efficient vectorized operations:

```python
import numpy as np

# Vectorized calculation for multiple conditions
temperatures = np.array([293.15, 298.15, 303.15, 308.15])
relative_humidity = 0.6
heat_indices = mm.heat_index(temperatures, relative_humidity)
```

### Memory Efficiency
For large datasets, process in chunks:

```python
def process_large_weather_dataset(data_chunk):
    """Process a chunk of weather data"""
    result = {}
    result['heat_index'] = mm.heat_index(
        data_chunk['temperature'],
        data_chunk['relative_humidity']
    )
    result['dewpoint'] = mm.dewpoint_temperature(
        data_chunk['temperature'],
        data_chunk['relative_humidity']
    )
    result['wetbulb'] = mm.wet_bulb_temperature(
        data_chunk['temperature'],
        data_chunk['pressure'],
        data_chunk['relative_humidity']
    )
    return result
```

## References

- Rothfusz, L.P. (1990). The Heat Index Equation. National Weather Service Technical Attachment, SR 90-23.
- Osczevski, R., & Bluestein, M. (2005). The New Wind Chill Equivalent Temperature Chart. Bulletin of the American Meteorological Society, 86(10), 1453-1458.
- Stull, R. (2011). Wet-Bulb Temperature from Relative Humidity and Air Temperature. Journal of Applied Meteorology and Climatology, 50(11), 2267-2269.
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/

## See Also

- [Thermodynamics Module](../thermodynamics.md) - Fundamental thermodynamic calculations
- [Statistical Analysis](../statistical.md) - Statistical and micrometeorological functions
- [Unit Conversions](../units.md) - Meteorological unit conversion utilities
- [Data Models](../models.md) - Structured data models for atmospheric data
- [Meteorological Functions](../meteo.md) - Core meteorological calculations