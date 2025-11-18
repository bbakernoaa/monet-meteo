# Dynamic Calculations Module

The dynamics module provides functions for calculating dynamic meteorological variables including vorticity, divergence, wind components, and atmospheric stability parameters.

## Functions

### `absolute_vorticity(u_wind, v_wind, lat, dx, dy)`
Calculate absolute vorticity including planetary vorticity.

```python
mm.absolute_vorticity(u_wind, v_wind, lat, dx, dy)
```

**Parameters:**
- `u_wind` (float, numpy.ndarray, or xarray.DataArray): Eastward wind component (m/s)
- `v_wind` (float, numpy.ndarray, or xarray.DataArray): Northward wind component (m/s)
- `lat` (float, numpy.ndarray, or xarray.DataArray): Latitude in degrees
- `dx` (float): Grid spacing in x-direction (m)
- `dy` (float): Grid spacing in y-direction (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Absolute vorticity (s⁻¹)

**Example:**
```python
u = 10.0  # m/s
v = 5.0   # m/s
lat = 45.0  # degrees
dx = 100000  # m (100 km)
dy = 100000  # m (100 km)
abs_vort = mm.absolute_vorticity(u, v, lat, dx, dy)
print(f"Absolute vorticity: {abs_vort:.2e} s⁻¹")
```

### `relative_vorticity(u_wind, v_wind, dx, dy)`
Calculate relative vorticity from wind components.

```python
mm.relative_vorticity(u_wind, v_wind, dx, dy)
```

**Parameters:**
- `u_wind` (float, numpy.ndarray, or xarray.DataArray): Eastward wind component (m/s)
- `v_wind` (float, numpy.ndarray, or xarray.DataArray): Northward wind component (m/s)
- `dx` (float): Grid spacing in x-direction (m)
- `dy` (float): Grid spacing in y-direction (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Relative vorticity (s⁻¹)

**Example:**
```python
u = np.array([[10, 12], [8, 9]])  # m/s
v = np.array([[5, 6], [4, 5]])   # m/s
dx = 50000  # m (50 km)
dy = 50000  # m (50 km)
rel_vort = mm.relative_vorticity(u, v, dx, dy)
```

### `divergence(u_wind, v_wind, dx, dy)`
Calculate horizontal divergence from wind components.

```python
mm.divergence(u_wind, v_wind, dx, dy)
```

**Parameters:**
- `u_wind` (float, numpy.ndarray, or xarray.DataArray): Eastward wind component (m/s)
- `v_wind` (float, numpy.ndarray, or xarray.DataArray): Northward wind component (m/s)
- `dx` (float): Grid spacing in x-direction (m)
- `dy` (float): Grid spacing in y-direction (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Divergence (s⁻¹)

**Example:**
```python
u = 10.0  # m/s
v = 5.0   # m/s
dx = 100000  # m (100 km)
dy = 100000  # m (100 km)
div = mm.divergence(u, v, dx, dy)
print(f"Divergence: {div:.2e} s⁻¹")
```

### `geostrophic_wind(pressure, lat, dx, dy)`
Calculate geostrophic wind from pressure gradients.

```python
mm.geostrophic_wind(pressure, lat, dx, dy)
```

**Parameters:**
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Atmospheric pressure (Pa)
- `lat` (float, numpy.ndarray, or xarray.DataArray): Latitude in degrees
- `dx` (float): Grid spacing in x-direction (m)
- `dy` (float): Grid spacing in y-direction (m)

**Returns:**
- tuple: (u_geo, v_geo) where u_geo and v_geo are geostrophic wind components (m/s)

**Example:**
```python
pressure = np.array([[101325, 101200], [101400, 101325]])  # Pa
lat = 45.0  # degrees
dx = 50000  # m (50 km)
dy = 50000  # m (50 km)
u_geo, v_geo = mm.geostrophic_wind(pressure, lat, dx, dy)
print(f"Geostrophic wind: u={u_geo:.2f} m/s, v={v_geo:.2f} m/s")
```

### `gradient_wind(pressure, lat, curvature, dx, dy)`
Calculate gradient wind including curvature effects.

```python
mm.gradient_wind(pressure, lat, curvature, dx, dy)
```

**Parameters:**
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Atmospheric pressure (Pa)
- `lat` (float, numpy.ndarray, or xarray.DataArray): Latitude in degrees
- `curvature` (float, numpy.ndarray, or xarray.DataArray): Radius of curvature (m)
- `dx` (float): Grid spacing in x-direction (m)
- `dy` (float): Grid spacing in y-direction (m)

**Returns:**
- tuple: (u_grad, v_grad) where u_grad and v_grad are gradient wind components (m/s)

**Example:**
```python
pressure = 101325  # Pa
lat = 45.0  # degrees
curvature = 50000  # m (50 km radius)
dx = 50000  # m (50 km)
dy = 50000  # m (50 km)
u_grad, v_grad = mm.gradient_wind(pressure, lat, curvature, dx, dy)
```

### `potential_vorticity(potential_temperature, pressure, lat, dx, dy)`
Calculate potential vorticity.

```python
mm.potential_vorticity(potential_temperature, pressure, lat, dx, dy)
```

**Parameters:**
- `potential_temperature` (float, numpy.ndarray, or xarray.DataArray): Potential temperature (K)
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Atmospheric pressure (Pa)
- `lat` (float, numpy.ndarray, or xarray.DataArray): Latitude in degrees
- `dx` (float): Grid spacing in x-direction (m)
- `dy` (float): Grid spacing in y-direction (m)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Potential vorticity (PVU, 1 K m² kg⁻¹ s⁻¹)

**Example:**
```python
theta = 300.0  # K
pressure = 50000  # Pa (500 hPa)
lat = 45.0  # degrees
dx = 50000  # m (50 km)
dy = 50000  # m (50 km)
pv = mm.potential_vorticity(theta, pressure, lat, dx, dy)
print(f"Potential vorticity: {pv:.2f} PVU")
```

### `vertical_velocity_pressure(u_wind, v_wind, pressure_levels, dx, dy)`
Calculate vertical velocity from horizontal divergence.

```python
mm.vertical_velocity_pressure(u_wind, v_wind, pressure_levels, dx, dy)
```

**Parameters:**
- `u_wind` (float, numpy.ndarray, or xarray.DataArray): Eastward wind component (m/s)
- `v_wind` (float, numpy.ndarray, or xarray.DataArray): Northward wind component (m/s)
- `pressure_levels` (array-like): Pressure levels (Pa)
- `dx` (float): Grid spacing in x-direction (m)
- `dy` (float): Grid spacing in y-direction (m)

**Returns:**
- numpy.ndarray or xarray.DataArray: Vertical velocity (Pa/s)

**Example:**
```python
u = np.array([[10, 12], [8, 9]])  # m/s
v = np.array([[5, 6], [4, 5]])   # m/s
pressure_levels = np.array([100000, 85000, 70000])  # Pa
dx = 50000  # m (50 km)
dy = 50000  # m (50 km)
omega = mm.vertical_velocity_pressure(u, v, pressure_levels, dx, dy)
```

### `omega_to_w(omega, pressure, temperature)`
Convert omega (pressure vertical velocity) to w (height vertical velocity).

```python
mm.omega_to_w(omega, pressure, temperature)
```

**Parameters:**
- `omega` (float, numpy.ndarray, or xarray.DataArray): Pressure vertical velocity (Pa/s)
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Atmospheric pressure (Pa)
- `temperature` (float, numpy.ndarray, or xarray.DataArray): Air temperature (K)

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Height vertical velocity (m/s)

**Example:**
```python
omega = -1.0  # Pa/s
pressure = 85000  # Pa
temperature = 280.0  # K
w = mm.omega_to_w(omega, pressure, temperature)
print(f"Vertical velocity: {w:.4f} m/s")
```

### `coriolis_parameter(lat)`
Calculate Coriolis parameter.

```python
mm.coriolis_parameter(lat)
```

**Parameters:**
- `lat` (float, numpy.ndarray, or xarray.DataArray): Latitude in degrees

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Coriolis parameter (s⁻¹)

**Example:**
```python
lat = 45.0  # degrees
f = mm.coriolis_parameter(lat)
print(f"Coriolis parameter: {f:.2e} s⁻¹")
```

## Constants

The dynamics module uses physical constants defined in [`monet_meteo.constants`](../constants.md):

- `g`: Acceleration due to gravity (9.80665 m s⁻²)
- `Omega`: Earth's rotation rate (7.292×10⁻⁵ s⁻¹)
- `R_earth`: Earth's radius (6.371×10⁶ m)

## Usage Patterns

### Basic Dynamic Calculations
```python
import monet_meteo as mm
import numpy as np

# Define wind field
u = np.array([[10, 12, 15], [8, 10, 12], [5, 8, 10]])  # m/s
v = np.array([[5, 6, 8], [4, 5, 7], [2, 4, 6]])   # m/s
lat = 45.0  # degrees
dx = dy = 50000  # m (50 km)

# Calculate vorticity
rel_vort = mm.relative_vorticity(u, v, dx, dy)
abs_vort = mm.absolute_vorticity(u, v, lat, dx, dy)

# Calculate divergence
div = mm.divergence(u, v, dx, dy)

print(f"Relative vorticity: {rel_vort}")
print(f"Absolute vorticity: {abs_vort}")
print(f"Divergence: {div}")
```

### Pressure-based Wind Calculations
```python
# Define pressure field
pressure = np.array([[101325, 101200, 101150],
                    [101400, 101325, 101250],
                    [101475, 101400, 101325]])  # Pa

# Calculate geostrophic wind
u_geo, v_geo = mm.geostrophic_wind(pressure, lat, dx, dy)

# Calculate gradient wind with curvature
curvature = 100000  # m (100 km radius)
u_grad, v_grad = mm.gradient_wind(pressure, lat, curvature, dx, dy)

print(f"Geostrophic wind: u={u_geo:.2f} m/s, v={v_geo:.2f} m/s")
print(f"Gradient wind: u={u_grad:.2f} m/s, v={v_grad:.2f} m/s")
```

### Vertical Velocity Calculation
```python
# Define 3D wind field
u_3d = np.array([[10, 12, 15], [8, 10, 12], [5, 8, 10]])  # m/s
v_3d = np.array([[5, 6, 8], [4, 5, 7], [2, 4, 6]])   # m/s
pressure_levels = np.array([100000, 85000, 70000])  # Pa

# Calculate vertical velocity
omega = mm.vertical_velocity_pressure(u_3d, v_3d, pressure_levels, dx, dy)

# Convert to height coordinate
temperature = 280.0  # K
w = mm.omega_to_w(omega, pressure_levels[1], temperature)

print(f"Vertical velocity (Pa/s): {omega}")
print(f"Vertical velocity (m/s): {w}")
```

### Potential Vorticity Analysis
```python
# Define atmospheric profile
theta = 300.0  # K
pressure = 50000  # Pa (500 hPa)
lat_range = np.linspace(30, 60, 10)  # degrees

# Calculate potential vorticity
pv = mm.potential_vorticity(theta, pressure, lat_range, dx, dy)

print(f"Potential vorticity range: {pv.min():.2f} - {pv.max():.2f} PVU")
```

## Advanced Applications

### Hurricane Analysis
```python
def analyze_hurricane_structure(wind_field, pressure_field, lat, dx, dy):
    """
    Analyze hurricane structure using dynamic parameters
    """
    # Calculate vorticity and divergence
    vorticity = mm.relative_vorticity(wind_field['u'], wind_field['v'], dx, dy)
    divergence = mm.divergence(wind_field['u'], wind_field['v'], dx, dy)
    
    # Calculate pressure gradient wind
    u_pg, v_pg = mm.geostrophic_wind(pressure_field, lat, dx, dy)
    
    # Calculate eye characteristics
    max_vorticity_idx = np.unravel_index(np.argmax(vorticity), vorticity.shape)
    eye_center = (max_vorticity_idx[0] * dx, max_vorticity_idx[1] * dy)
    
    return {
        'vorticity': vorticity,
        'divergence': divergence,
        'pressure_gradient_wind': (u_pg, v_pg),
        'eye_center': eye_center,
        'max_vorticity': vorticity[max_vorticity_idx]
    }

# Example usage
# hurricane_analysis = analyze_hurricane_structure(hurricane_winds, hurricane_pressure, lat, dx, dy)
```

### Frontal Analysis
```python
def identify_fronts(pressure_field, wind_field, lat, dx, dy):
    """
    Identify frontal zones based on dynamic parameters
    """
    # Calculate geostrophic wind
    u_geo, v_geo = mm.geostrophic_wind(pressure_field, lat, dx, dy)
    
    # Calculate vorticity along frontal boundaries
    vorticity = mm.relative_vorticity(wind_field['u'], wind_field['v'], dx, dy)
    
    # Identify regions of strong vorticity gradient (frontal zones)
    vorticity_gradient = np.gradient(vorticity)
    frontal_strength = np.sqrt(vorticity_gradient[0]**2 + vorticity_gradient[1]**2)
    
    return {
        'geostrophic_wind': (u_geo, v_geo),
        'vorticity': vorticity,
        'frontal_strength': frontal_strength,
        'frontal_zones': frontal_strength > np.percentile(frontal_strength, 90)
    }

# Example usage
# frontal_zones = identify_fronts(pressure_data, wind_data, lat, dx, dy)
```

### Atmospheric Wave Analysis
```python
def analyze_atmospheric_winds(u_wind, v_wind, pressure_levels, lat, dx, dy):
    """
    Analyze atmospheric wave patterns
    """
    # Calculate vertical velocity from divergence
    omega = mm.vertical_velocity_pressure(u_wind, v_wind, pressure_levels, dx, dy)
    
    # Convert to height coordinates
    w = mm.omega_to_w(omega, pressure_levels, 250.0)  # Assume 250K
    
    # Calculate vorticity at different levels
    vorticity_3d = []
    for i in range(len(pressure_levels)):
        vort = mm.relative_vorticity(u_wind[i], v_wind[i], dx, dy)
        vorticity_3d.append(vort)
    
    return {
        'vertical_velocity': w,
        'vorticity_3d': np.array(vorticity_3d),
        'omega': omega
    }

# Example usage
# wave_analysis = analyze_atmospheric_winds(u_3d, v_3d, pressure_levels, lat, dx, dy)
```

## Error Handling

The functions include input validation for meteorological合理性:

- Wind speeds should be physically reasonable
- Grid spacing must be positive
- Latitude values should be within valid range (-90 to 90 degrees)
- Pressure values must be positive

### Common Errors
```python
# Error: Negative grid spacing
try:
    mm.relative_vorticity(u, v, -50000, 50000)
except ValueError as e:
    print(f"Error: {e}")

# Error: Invalid latitude
try:
    mm.coriolis_parameter(95.0)
except ValueError as e:
    print(f"Error: {e}")

# Error: Inconsistent array shapes
try:
    u = np.array([10, 12, 15])
    v = np.array([5, 6])
    mm.relative_vorticity(u, v, 50000, 50000)
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Vectorization
All functions support numpy arrays for efficient vectorized operations:

```python
# Vectorized calculation for multiple grid points
lats = np.linspace(30, 60, 10)
f_values = mm.coriolis_parameter(lats)
```

### Memory Efficiency
For large datasets, process in chunks:

```python
def process_large_wind_dataset(data_chunk, dx, dy):
    """Process a chunk of wind data"""
    result = {}
    result['vorticity'] = mm.relative_vorticity(
        data_chunk['u_wind'], 
        data_chunk['v_wind'], 
        dx, dy
    )
    result['divergence'] = mm.divergence(
        data_chunk['u_wind'], 
        data_chunk['v_wind'], 
        dx, dy
    )
    return result
```

## References

- Holton, J.R., & Hakim, G.J. (2013). An Introduction to Dynamic Meteorology. Academic Press.
- Bluestein, H.B. (1993). Synoptic-Dynamic Meteorology in Midlatitudes. Oxford University Press.
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/

## See Also

- [Thermodynamics Module](../thermodynamics.md) - Thermodynamic variable calculations
- [Statistical Analysis](../statistical.md) - Statistical and micrometeorological functions
- [Unit Conversions](../units.md) - Meteorological unit conversion utilities
- [Data Models](../models.md) - Structured data models for atmospheric data
- [Xarray Integration](../io.md) - Integration with xarray for gridded data