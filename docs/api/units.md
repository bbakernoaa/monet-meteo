# Unit Conversions Module

The units module provides comprehensive utilities for converting between different meteorological units. All functions support numpy arrays and xarray DataArrays for efficient vectorized operations.

## Functions

## Pressure Conversions

### `convert_pressure(value, from_unit, to_unit)`
Convert pressure between different units.

```python
mm.convert_pressure(value, from_unit, to_unit)
```

**Parameters:**
- `value` (float, numpy.ndarray, or xarray.DataArray): Pressure value(s) to convert
- `from_unit` (str): Source unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')
- `to_unit` (str): Target unit ('Pa', 'hPa', 'mb', 'mmHg', 'inHg', 'atm')

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Converted pressure value(s)

**Examples:**
```python
# Convert 1013.25 hPa to Pa
pressure_pa = mm.convert_pressure(1013.25, 'hPa', 'Pa')
print(pressure_pa)  # 101325.0

# Convert 29.92 inHg to hPa
pressure_hpa = mm.convert_pressure(29.92, 'inHg', 'hPa')
print(pressure_hpa)  # 1013.25

# Convert 760 mmHg to atm
pressure_atm = mm.convert_pressure(760, 'mmHg', 'atm')
print(pressure_atm)  # 0.999132
```

## Temperature Conversions

### `convert_temperature(value, from_unit, to_unit)`
Convert temperature between different units.

```python
mm.convert_temperature(value, from_unit, to_unit)
```

**Parameters:**
- `value` (float, numpy.ndarray, or xarray.DataArray): Temperature value(s) to convert
- `from_unit` (str): Source unit ('K', 'C', 'F')
- `to_unit` (str): Target unit ('K', 'C', 'F')

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Converted temperature value(s)

**Examples:**
```python
# Convert 25°C to Kelvin
temp_k = mm.convert_temperature(25, 'C', 'K')
print(temp_k)  # 298.15

# Convert 77°F to Celsius
temp_c = mm.convert_temperature(77, 'F', 'C')
print(temp_c)  # 25.0

# Convert 0°C to Fahrenheit
temp_f = mm.convert_temperature(0, 'C', 'F')
print(temp_f)  # 32.0
```

## Distance Conversions

### `convert_distance(value, from_unit, to_unit)`
Convert distance between different units.

```python
mm.convert_distance(value, from_unit, to_unit)
```

**Parameters:**
- `value` (float, numpy.ndarray, or xarray.DataArray): Distance value(s) to convert
- `from_unit` (str): Source unit ('m', 'km', 'ft', 'mi', 'nm')
- `to_unit` (str): Target unit ('m', 'km', 'ft', 'mi', 'nm')

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Converted distance value(s)

**Examples:**
```python
# Convert 1 km to meters
distance_m = mm.convert_distance(1, 'km', 'm')
print(distance_m)  # 1000.0

# Convert 5000 ft to miles
distance_mi = mm.convert_distance(5000, 'ft', 'mi')
print(distance_mi)  # 0.94697

# Convert 100 nautical miles to km
distance_km = mm.convert_distance(100, 'nm', 'km')
print(distance_km)  # 185.2
```

## Wind Speed Conversions

### `convert_wind_speed(value, from_unit, to_unit)`
Convert wind speed between different units.

```python
mm.convert_wind_speed(value, from_unit, to_unit)
```

**Parameters:**
- `value` (float, numpy.ndarray, or xarray.DataArray): Wind speed value(s) to convert
- `from_unit` (str): Source unit ('m/s', 'knots', 'km/h', 'mph')
- `to_unit` (str): Target unit ('m/s', 'knots', 'km/h', 'mph')

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Converted wind speed value(s)

**Examples:**
```python
# Convert 10 m/s to knots
wind_knots = mm.convert_wind_speed(10, 'm/s', 'knots')
print(wind_knots)  # 19.4384

# Convert 20 knots to km/h
wind_kmh = mm.convert_wind_speed(20, 'knots', 'km/h')
print(wind_kmh)  # 37.028

# Convert 50 mph to m/s
wind_mps = mm.convert_wind_speed(50, 'mph', 'm/s')
print(wind_mps)  # 22.352
```

## Moisture Conversions

### `convert_mixing_ratio(value, from_unit, to_unit)`
Convert mixing ratio between different units.

```python
mm.convert_mixing_ratio(value, from_unit, to_unit)
```

**Parameters:**
- `value` (float, numpy.ndarray, or xarray.DataArray): Mixing ratio value(s) to convert
- `from_unit` (str): Source unit ('kg/kg', 'g/kg', 'ppm', 'ppt')
- `to_unit` (str): Target unit ('kg/kg', 'g/kg', 'ppm', 'ppt')

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Converted mixing ratio value(s)

**Examples:**
```python
# Convert 8 g/kg to kg/kg
mixing_ratio_kgkg = mm.convert_mixing_ratio(8, 'g/kg', 'kg/kg')
print(mixing_ratio_kgkg)  # 0.008

# Convert 0.008 kg/kg to ppm
mixing_ratio_ppm = mm.convert_mixing_ratio(0.008, 'kg/kg', 'ppm')
print(mixing_ratio_ppm)  # 8000.0
```

### `convert_specific_humidity(value, from_unit, to_unit)`
Convert specific humidity between different units.

```python
mm.convert_specific_humidity(value, from_unit, to_unit)
```

**Parameters:**
- `value` (float, numpy.ndarray, or xarray.DataArray): Specific humidity value(s) to convert
- `from_unit` (str): Source unit ('kg/kg', 'g/kg', 'g/m3', 'mg/m3')
- `to_unit` (str): Target unit ('kg/kg', 'g/kg', 'g/m3', 'mg/m3')

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Converted specific humidity value(s)

**Examples:**
```python
# Convert 15 g/m3 to kg/kg
specific_humidity_kgkg = mm.convert_specific_humidity(15, 'g/m3', 'kg/kg')
print(specific_humidity_kgkg)  # 0.015
```

## Concentration Conversions

### `convert_concentration(value, from_unit, to_unit, molecular_weight=28.97)`
Convert concentration between different units.

```python
mm.convert_concentration(value, from_unit, to_unit, molecular_weight=28.97)
```

**Parameters:**
- `value` (float, numpy.ndarray, or xarray.DataArray): Concentration value(s) to convert
- `from_unit` (str): Source unit ('ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol')
- `to_unit` (str): Target unit ('ppm', 'ppb', 'ppt', 'ug/m3', 'ng/m3', 'mol/mol')
- `molecular_weight` (float, optional): Molecular weight of species in g/mol (default 28.97 for dry air)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Converted concentration value(s)

**Examples:**
```python
# Convert 400 ppm to ppb
co2_ppb = mm.convert_concentration(400, 'ppm', 'ppb')
print(co2_ppb)  # 400000.0

# Convert 1000 ppb to ppm
co2_ppm = mm.convert_concentration(1000, 'ppb', 'ppm')
print(co2_ppm)  # 1.0
```

## Usage Patterns

### Basic Unit Conversion
```python
import monet_meteo as mm
import numpy as np

# Single value conversion
pressure = 1013.25  # hPa
pressure_pa = mm.convert_pressure(pressure, 'hPa', 'Pa')

# Vectorized conversion for arrays
pressures = np.array([1013.25, 1020.0, 1000.0])
pressures_pa = mm.convert_pressure(pressures, 'hPa', 'Pa')

print(f"Pressure: {pressure} hPa = {pressure_pa} Pa")
```

### Temperature Unit Chain
```python
# Convert from Fahrenheit to Celsius to Kelvin
temp_f = 32  # °F
temp_c = mm.convert_temperature(temp_f, 'F', 'C')
temp_k = mm.convert_temperature(temp_c, 'C', 'K')

print(f"{temp_f}°F = {temp_c:.1f}°C = {temp_k:.1f}K")
```

### Wind Speed Conversion for Aviation
```python
# Convert various wind speed units for aviation
wind_mps = 15.0  # m/s
wind_knots = mm.convert_wind_speed(wind_mps, 'm/s', 'knots')
wind_kmh = mm.convert_wind_speed(wind_mps, 'm/s', 'km/h')
wind_mph = mm.convert_wind_speed(wind_mps, 'm/s', 'mph')

print(f"Wind speed: {wind_mps} m/s = {wind_knots:.1f} knots = {wind_kmh:.1f} km/h = {wind_mph:.1f} mph")
```

### Moisture Unit Conversions
```python
# Convert between moisture units
mixing_ratio = 0.008  # kg/kg
mixing_ratio_gkg = mm.convert_mixing_ratio(mixing_ratio, 'kg/kg', 'g/kg')
mixing_ratio_ppm = mm.convert_mixing_ratio(mixing_ratio, 'kg/kg', 'ppm')

print(f"Mixing ratio: {mixing_ratio} kg/kg = {mixing_ratio_gkg} g/kg = {mixing_ratio_ppm} ppm")
```

### Air Quality Unit Conversions
```python
# Convert air quality measurements
co2_ppm = 400  # ppm
co2_ppb = mm.convert_concentration(co2_ppm, 'ppm', 'ppb')
co2_molmol = mm.convert_concentration(co2_ppm, 'ppm', 'mol/mol')

print(f"CO2: {co2_ppm} ppm = {co2_ppb} ppb = {co2_molmol} mol/mol")
```

## Advanced Applications

### Meteorological Data Processing Pipeline
```python
def process_meteorological_data(data):
    """
    Process meteorological data with unit conversions
    """
    processed = {}
    
    # Convert pressure to consistent units
    processed['pressure_pa'] = mm.convert_pressure(
        data['pressure'], data['pressure_unit'], 'Pa'
    )
    
    # Convert temperatures to Kelvin
    processed['temperature_k'] = mm.convert_temperature(
        data['temperature'], data['temperature_unit'], 'K'
    )
    
    # Convert wind speeds to m/s
    processed['wind_speed_ms'] = mm.convert_wind_speed(
        data['wind_speed'], data['wind_speed_unit'], 'm/s'
    )
    
    # Convert mixing ratios to consistent units
    if 'mixing_ratio' in data:
        processed['mixing_ratio_kgkg'] = mm.convert_mixing_ratio(
            data['mixing_ratio'], data['mixing_ratio_unit'], 'kg/kg'
        )
    
    return processed

# Example usage
# meteorological_data = {
#     'pressure': 1013.25,
#     'pressure_unit': 'hPa',
#     'temperature': 25,
#     'temperature_unit': 'C',
#     'wind_speed': 10,
#     'wind_speed_unit': 'm/s',
#     'mixing_ratio': 8,
#     'mixing_ratio_unit': 'g/kg'
# }
# processed_data = process_meteorological_data(meteorological_data)
```

### Climate Data Unit Standardization
```python
def standardize_climate_units(dataset):
    """
    Standardize units in climate dataset for analysis
    """
    # Convert all pressures to Pa
    if 'pressure' in dataset:
        dataset['pressure_pa'] = mm.convert_pressure(
            dataset['pressure'], dataset.attrs.get('pressure_unit', 'hPa'), 'Pa'
        )
    
    # Convert all temperatures to Kelvin
    for temp_var in ['temperature', 'dewpoint', 'surface_temperature']:
        if temp_var in dataset:
            dataset[f'{temp_var}_k'] = mm.convert_temperature(
                dataset[temp_var], dataset.attrs.get(f'{temp_var}_unit', 'C'), 'K'
            )
    
    # Convert all wind speeds to m/s
    for wind_var in ['wind_speed', 'u_wind', 'v_wind']:
        if wind_var in dataset:
            dataset[f'{wind_var}_ms'] = mm.convert_wind_speed(
                dataset[wind_var], dataset.attrs.get(f'{wind_var}_unit', 'm/s'), 'm/s'
            )
    
    return dataset

# Example usage
# standardized_dataset = standardize_climate_units(climate_dataset)
```

### Unit Conversion for Model Input
```python
def prepare_model_input(observation_data, model_requirements):
    """
    Convert observation data to model input units
    """
    model_input = {}
    
    # Pressure conversion
    if 'pressure' in observation_data:
        model_input['pressure'] = mm.convert_pressure(
            observation_data['pressure'],
            observation_data.get('pressure_unit', 'hPa'),
            model_requirements['pressure_unit']
        )
    
    # Temperature conversion
    if 'temperature' in observation_data:
        model_input['temperature'] = mm.convert_temperature(
            observation_data['temperature'],
            observation_data.get('temperature_unit', 'C'),
            model_requirements['temperature_unit']
        )
    
    # Wind speed conversion
    if 'wind_speed' in observation_data:
        model_input['wind_speed'] = mm.convert_wind_speed(
            observation_data['wind_speed'],
            observation_data.get('wind_speed_unit', 'm/s'),
            model_requirements['wind_speed_unit']
        )
    
    return model_input

# Example usage
# obs_data = {'pressure': 1013.25, 'temperature': 25, 'wind_speed': 10}
# model_req = {'pressure_unit': 'Pa', 'temperature_unit': 'K', 'wind_speed_unit': 'm/s'}
# model_input = prepare_model_input(obs_data, model_req)
```

## Error Handling

All functions include comprehensive input validation:

### Common Errors
```python
# Error: Negative pressure
try:
    mm.convert_pressure(-100, 'hPa', 'Pa')
except ValueError as e:
    print(f"Error: {e}")

# Error: Temperature below absolute zero
try:
    mm.convert_temperature(-274, 'C', 'K')
except ValueError as e:
    print(f"Error: {e}")

# Error: Unsupported unit
try:
    mm.convert_pressure(1013.25, 'hPa', 'bar')
except ValueError as e:
    print(f"Error: {e}")

# Error: Mixing ratio cannot be negative
try:
    mm.convert_mixing_ratio(-0.001, 'kg/kg', 'g/kg')
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Vectorization
All functions support numpy arrays for efficient vectorized operations:

```python
import numpy as np

# Vectorized conversion for large datasets
temperatures = np.random.uniform(-40, 50, 10000)  # Random temperatures in °C
temperatures_k = mm.convert_temperature(temperatures, 'C', 'K')
```

### Memory Efficiency
For large datasets, process in chunks:

```python
def process_large_unit_conversion(data_chunk, conversion_func, from_unit, to_unit):
    """Process a chunk of data with unit conversion"""
    return conversion_func(data_chunk, from_unit, to_unit)

# Example usage with large dataset
# chunk_size = 1000
# for i in range(0, len(large_dataset), chunk_size):
#     chunk = large_dataset[i:i+chunk_size]
#     converted_chunk = process_large_unit_conversion(
#         chunk, mm.convert_pressure, 'hPa', 'Pa'
#     )
```

## Supported Units

### Pressure Units
- `Pa`: Pascal (SI unit)
- `hPa`: Hectopascal (1 hPa = 100 Pa)
- `mb`: Millibar (1 mb = 1 hPa = 100 Pa)
- `mmHg`: Millimeters of mercury
- `inHg`: Inches of mercury
- `atm`: Standard atmosphere (1 atm = 101325 Pa)

### Temperature Units
- `K`: Kelvin (SI base unit)
- `C`: Degrees Celsius
- `F`: Degrees Fahrenheit

### Distance Units
- `m`: Meters (SI unit)
- `km`: Kilometers
- `ft`: Feet
- `mi`: Miles
- `nm`: Nautical miles

### Wind Speed Units
- `m/s`: Meters per second (SI unit)
- `knots` or `kt`: Knots (nautical miles per hour)
- `km/h` or `kmh`: Kilometers per hour
- `mph`: Miles per hour

### Moisture Units
- `kg/kg`: Kilograms of water vapor per kilogram of air (mixing ratio)
- `g/kg`: Grams of water vapor per kilogram of air
- `ppm`: Parts per million by mass
- `ppt`: Parts per trillion by mass

### Concentration Units
- `ppm`: Parts per million
- `ppb`: Parts per billion
- `ppt`: Parts per trillion
- `ug/m3`: Micrograms per cubic meter
- `ng/m3`: Nanograms per cubic meter
- `mol/mol`: Moles per mole

## References

- World Meteorological Organization (WMO). (2018). Guide to Meteorological Instruments and Methods of Observation.
- International Bureau of Weights and Measures (BIPM). (2019). The International System of Units (SI).
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/

## See Also

- [Thermodynamics Module](thermodynamics.md) - Thermodynamic variable calculations
- [Statistical Analysis](statistical.md) - Statistical and micrometeorological functions
- [Dynamic Calculations](dynamics.md) - Dynamic meteorology functions
- [Data Models](models.md) - Structured data models for atmospheric data