# Coordinates Module

The coordinates module provides utilities for coordinate transformations, geographic calculations, and atmospheric coordinate system conversions. These functions are essential for meteorological data processing, atmospheric modeling, and geographic analysis.

## Functions

## Geographic Coordinates

### `latlon_to_cartesian(lat, lon, elevation=0.0)`
Convert latitude/longitude coordinates to Cartesian coordinates (ECEF).

```python
mm.latlon_to_cartesian(lat, lon, elevation=0.0)
```

**Parameters:**
- `lat` (float, numpy.ndarray, or xarray.DataArray): Latitude in degrees
- `lon` (float, numpy.ndarray, or xarray.DataArray): Longitude in degrees
- `elevation` (float, numpy.ndarray, or xarray.DataArray, optional): Elevation in meters, default 0.0

**Returns:**
- tuple: (x, y, z) Cartesian coordinates in meters

**Example:**
```python
import monet_meteo as mm

# Convert single point
x, y, z = mm.latlon_to_cartesian(40.7128, -74.0060, 10.0)
print(f"Cartesian coordinates: x={x:.2f}, y={y:.2f}, z={z:.2f} m")

# Convert multiple points
lats = [40.7128, 51.5074, 35.6762]
lons = [-74.0060, -0.1278, 139.6503]
elevations = [10.0, 35.0, 40.0]

cartesian_coords = [mm.latlon_to_cartesian(lat, lon, elev) 
                   for lat, lon, elev in zip(lats, lons, elevations)]
```

### `cartesian_to_latlon(x, y, z)`
Convert Cartesian coordinates to latitude/longitude.

```python
mm.cartesian_to_latlon(x, y, z)
```

**Parameters:**
- `x` (float, numpy.ndarray, or xarray.DataArray): X coordinate in meters
- `y` (float, numpy.ndarray, or xarray.DataArray): Y coordinate in meters
- `z` (float, numpy.ndarray, or xarray.DataArray): Z coordinate in meters

**Returns:**
- tuple: (lat, lon) coordinates in degrees

**Example:**
```python
# Convert back to geographic coordinates
lat, lon = mm.cartesian_to_latlon(x, y, z)
print(f"Geographic coordinates: lat={lat:.4f}°, lon={lon:.4f}°")
```

## Distance and Bearing Calculations

### `calculate_distance(lat1, lon1, lat2, lon2, method='haversine')`
Calculate distance between two geographic points.

```python
mm.calculate_distance(lat1, lon1, lat2, lon2, method='haversine')
```

**Parameters:**
- `lat1, lon1` (float, numpy.ndarray, or xarray.DataArray): Latitude and longitude of first point in degrees
- `lat2, lon2` (float, numpy.ndarray, or xarray.DataArray): Latitude and longitude of second point in degrees
- `method` (str, optional): Distance calculation method ('haversine' or 'vincenty'), default 'haversine'

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Distance in meters

**Example:**
```python
# Calculate distance between two cities
new_york = (40.7128, -74.0060)
london = (51.5074, -0.1278)
distance = mm.calculate_distance(*new_york, *london)
print(f"Distance: {distance/1000:.1f} km")

# Calculate distances between multiple points
points = [(40.7128, -74.0060), (51.5074, -0.1278), (35.6762, 139.6503)]
distances = []
for i in range(len(points)):
    for j in range(i+1, len(points)):
        dist = mm.calculate_distance(*points[i], *points[j])
        distances.append(dist)
```

### `bearing(lat1, lon1, lat2, lon2)`
Calculate the initial bearing (forward azimuth) between two points.

```python
mm.bearing(lat1, lon1, lat2, lon2)
```

**Parameters:**
- `lat1, lon1` (float, numpy.ndarray, or xarray.DataArray): Latitude and longitude of first point in degrees
- `lat2, lon2` (float, numpy.ndarray, or xarray.DataArray): Latitude and longitude of second point in degrees

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Bearing in degrees (0-360)

**Example:**
```python
# Calculate bearing from New York to London
bearing = mm.bearing(*new_york, *london)
print(f"Bearing: {bearing:.1f} degrees")
```

### `destination_point(lat, lon, distance, bearing)`
Calculate destination point given starting point, distance, and bearing.

```python
mm.destination_point(lat, lon, distance, bearing)
```

**Parameters:**
- `lat, lon` (float, numpy.ndarray, or xarray.DataArray): Starting latitude and longitude in degrees
- `distance` (float, numpy.ndarray, or xarray.DataArray): Distance in meters
- `bearing` (float, numpy.ndarray, or xarray.DataArray): Bearing in degrees (0-360)

**Returns:**
- tuple: (dest_lat, dest_lon) destination coordinates in degrees

**Example:**
```python
# Calculate destination point
dest_lat, dest_lon = mm.destination_point(*new_york, 500000, 45)
print(f"Destination: {dest_lat:.4f}°, {dest_lon:.4f}°")
```

## Coordinate System Transformations

### `rotate_wind_components(u_wind, v_wind, lat, lon, rotation_method='grid')`
Rotate wind components from geographic to local coordinate system.

```python
mm.rotate_wind_components(u_wind, v_wind, lat, lon, rotation_method='grid')
```

**Parameters:**
- `u_wind` (float, numpy.ndarray, or xarray.DataArray): Eastward wind component (m/s)
- `v_wind` (float, numpy.ndarray, or xarray.DataArray): Northward wind component (m/s)
- `lat` (float, numpy.ndarray, or xarray.DataArray): Latitude in degrees
- `lon` (float, numpy.ndarray, or xarray.DataArray): Longitude in degrees
- `rotation_method` (str, optional): Rotation method ('grid' or 'simple'), default 'grid'

**Returns:**
- tuple: (u_rotated, v_rotated) rotated wind components

**Example:**
```python
# Rotate wind components to local coordinate system
u_rot, v_rot = mm.rotate_wind_components(10, 5, 40, -74)
print(f"Rotated wind: u={u_rot:.2f} m/s, v={v_rot:.2f} m/s")
```

### `pressure_to_sigma(pressure, surface_pressure, top_pressure=10000)`
Convert pressure coordinates to sigma coordinates.

```python
mm.pressure_to_sigma(pressure, surface_pressure, top_pressure=10000)
```

**Parameters:**
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Pressure levels (Pa)
- `surface_pressure` (float, numpy.ndarray, or xarray.DataArray): Surface pressure (Pa)
- `top_pressure` (float, optional): Top pressure level (Pa), default 10000

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Sigma coordinates (0-1)

**Example:**
```python
# Convert pressure to sigma coordinates
sigma = mm.pressure_to_sigma(50000, 101325, top_pressure=20000)
print(f"Sigma coordinate: {sigma:.3f}")
```

### `sigma_to_pressure(sigma, surface_pressure, top_pressure=10000)`
Convert sigma coordinates to pressure coordinates.

```python
mm.sigma_to_pressure(sigma, surface_pressure, top_pressure=10000)
```

**Parameters:**
- `sigma` (float, numpy.ndarray, or xarray.DataArray): Sigma coordinates (0-1)
- `surface_pressure` (float, numpy.ndarray, or xarray.DataArray): Surface pressure (Pa)
- `top_pressure` (float, optional): Top pressure level (Pa), default 10000

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Pressure levels (Pa)

**Example:**
```python
# Convert sigma to pressure coordinates
pressure = mm.sigma_to_pressure(0.5, 101325, top_pressure=20000)
print(f"Pressure: {pressure:.0f} Pa")
```

## Grid Calculations

### `calculate_grid_spacing(lat, lon, grid_resolution_deg)`
Calculate grid spacing in meters for a regular grid.

```python
mm.calculate_grid_spacing(lat, lon, grid_resolution_deg)
```

**Parameters:**
- `lat` (float): Latitude in degrees
- `lon` (float): Longitude in degrees
- `grid_resolution_deg` (float): Grid resolution in degrees

**Returns:**
- tuple: (dx, dy) grid spacing in meters

**Example:**
```python
# Calculate grid spacing at 40°N for 0.1° resolution
dx, dy = mm.calculate_grid_spacing(40, -74, 0.1)
print(f"Grid spacing: dx={dx:.0f} m, dy={dy:.0f} m")
```

### `calculate_grid_area(lat, lon_resolution, lon_resolution)`
Calculate grid cell area for regular latitude/longitude grid.

```python
mm.calculate_grid_area(lat, lon_resolution, lon_resolution)
```

**Parameters:**
- `lat` (float): Latitude in degrees
- `lon_resolution` (float): Longitude resolution in degrees
- `lat_resolution` (float): Latitude resolution in degrees

**Returns:**
- float: Grid cell area in square meters

**Example:**
```python
# Calculate area of 0.1° x 0.1° grid cell at 40°N
area = mm.calculate_grid_area(40, 0.1, 0.1)
print(f"Grid cell area: {area:.0f} m²")
```

## Vertical Coordinate Conversions

### `pressure_to_altitude(pressure, temperature=288.15, lapse_rate=0.0065)`
Convert pressure to altitude using barometric formula.

```python
mm.pressure_to_altitude(pressure, temperature=288.15, lapse_rate=0.0065)
```

**Parameters:**
- `pressure` (float, numpy.ndarray, or xarray.DataArray): Pressure in Pa
- `temperature` (float, optional): Reference temperature in K, default 288.15
- `lapse_rate` (float, optional): Temperature lapse rate in K/m, default 0.0065

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Altitude in meters

**Example:**
```python
# Convert pressure to altitude
altitude = mm.pressure_to_altitude(85000, temperature=288.15, lapse_rate=0.0065)
print(f"Altitude: {altitude:.0f} m")

# Convert multiple pressure levels
pressure_levels = np.array([101325, 85000, 70000, 50000, 30000])
altitudes = mm.pressure_to_altitude(pressure_levels)
```

### `altitude_to_pressure(altitude, temperature=288.15, lapse_rate=0.0065)`
Convert altitude to pressure using barometric formula.

```python
mm.altitude_to_pressure(altitude, temperature=288.15, lapse_rate=0.0065)
```

**Parameters:**
- `altitude` (float, numpy.ndarray, or xarray.DataArray): Altitude in meters
- `temperature` (float, optional): Reference temperature in K, default 288.15
- `lapse_rate` (float, optional): Temperature lapse rate in K/m, default 0.0065

**Returns:**
- float, numpy.ndarray, or xarray.DataArray: Pressure in Pa

**Example:**
```python
# Convert altitude to pressure
pressure = mm.altitude_to_pressure(1000, temperature=288.15, lapse_rate=0.0065)
print(f"Pressure: {pressure:.0f} Pa")
```

## Usage Patterns

### Basic Geographic Calculations
```python
import monet_meteo as mm
import numpy as np

def calculate_airport_distances(airports):
    """
    Calculate distances between airports
    """
    distance_matrix = np.zeros((len(airports), len(airports)))
    
    for i, (name1, (lat1, lon1)) in enumerate(airports.items()):
        for j, (name2, (lat2, lon2)) in enumerate(airports.items()):
            if i != j:
                distance = mm.calculate_distance(lat1, lon1, lat2, lon2)
                distance_matrix[i, j] = distance / 1000  # Convert to km
    
    return distance_matrix

# Example usage
airports = {
    'JFK': (40.6413, -73.7781),
    'LAX': (33.9416, -118.4085),
    'ORD': (41.9796, -87.9045),
    'DFW': (32.8998, -97.0403)
}

distance_matrix = calculate_airport_distances(airports)
print(f"Distance matrix (km):\n{distance_matrix}")
```

### Wind Component Rotation
```python
def analyze_wind_field_at_location(u_wind, v_wind, lat_grid, lon_grid):
    """
    Analyze wind field at specific location with proper rotation
    """
    # Find grid point closest to target location
    target_lat, target_lon = 40.7128, -74.0060  # New York
    
    # Find nearest grid point
    lat_idx = np.argmin(np.abs(lat_grid - target_lat))
    lon_idx = np.argmin(np.abs(lon_grid - target_lon))
    
    # Get wind components at that point
    u_at_point = u_wind[lat_idx, lon_idx]
    v_at_point = v_wind[lat_idx, lon_idx]
    
    # Rotate to local coordinate system
    u_local, v_local = mm.rotate_wind_components(u_at_point, v_at_point, target_lat, target_lon)
    
    return {
        'original_wind': (u_at_point, v_at_point),
        'local_wind': (u_local, v_local),
        'wind_speed': np.sqrt(u_at_point**2 + v_at_point**2),
        'wind_direction': np.degrees(np.arctan2(-u_at_point, -v_at_point)) % 360
    }

# Example usage
# wind_analysis = analyze_wind_field_at_location(u_grid, v_grid, lat_grid, lon_grid)
```

### Vertical Coordinate Processing
```python
def convert_vertical_coordinates(pressure_data, surface_pressure, target_coords='sigma'):
    """
    Convert between pressure and sigma coordinates
    """
    if target_coords == 'sigma':
        # Convert pressure to sigma
        sigma_coords = mm.pressure_to_sigma(
            pressure_data, surface_pressure, top_pressure=10000
        )
        return sigma_coords
    elif target_coords == 'pressure':
        # Convert sigma to pressure
        pressure_coords = mm.sigma_to_pressure(
            pressure_data, surface_pressure, top_pressure=10000
        )
        return pressure_coords
    else:
        raise ValueError("target_coords must be 'pressure' or 'sigma'")

# Example usage
# sigma_levels = convert_vertical_coordinates(pressure_levels, surface_pressure, 'sigma')
```

### Grid Cell Analysis
```python
def analyze_grid_properties(lat_grid, lon_grid):
    """
    Analyze grid cell properties
    """
    grid_properties = {}
    
    # Calculate grid spacing
    lat_spacing = np.mean(np.diff(lat_grid[0, :]))
    lon_spacing = np.mean(np.diff(lon_grid[:, 0]))
    
    dx, dy = mm.calculate_grid_spacing(
        np.mean(lat_grid), np.mean(lon_grid), lat_spacing
    )
    
    # Calculate areas
    areas = []
    for i in range(lat_grid.shape[0] - 1):
        for j in range(lon_grid.shape[1] - 1):
            lat = lat_grid[i, j]
            area = mm.calculate_grid_area(lat, lon_spacing, lat_spacing)
            areas.append(area)
    
    grid_properties.update({
        'grid_spacing_meters': {'dx': dx, 'dy': dy},
        'grid_spacing_degrees': {'dlat': lat_spacing, 'dlon': lon_spacing},
        'mean_area_sqm': np.mean(areas),
        'total_area_sqm': np.sum(areas)
    })
    
    return grid_properties

# Example usage
# grid_props = analyze_grid_properties(latitude_grid, longitude_grid)
```

## Advanced Applications

### Weather Route Planning
```python
def plan_weather_route(start_coords, end_coords, weather_data, waypoints=None):
    """
    Plan route considering weather conditions
    """
    # Calculate direct route distance and bearing
    direct_distance = mm.calculate_distance(*start_coords, *end_coords)
    direct_bearing = mm.bearing(*start_coords, *end_coords)
    
    # Generate waypoints if not provided
    if waypoints is None:
        # Simple waypoint generation
        num_waypoints = 5
        waypoints = []
        for i in range(1, num_waypoints):
            fraction = i / num_waypoints
            # Simple interpolation (in reality, you'd use great circle navigation)
            lat = start_coords[0] + fraction * (end_coords[0] - start_coords[0])
            lon = start_coords[1] + fraction * (end_coords[1] - start_coords[1])
            waypoints.append((lat, lon))
    
    # Analyze weather along route
    route_analysis = {
        'start': start_coords,
        'end': end_coords,
        'waypoints': waypoints,
        'direct_distance': direct_distance,
        'direct_bearing': direct_bearing,
        'legs': []
    }
    
    # Analyze each leg of the journey
    for i, (waypoint1, waypoint2) in enumerate(zip([start_coords] + waypoints, waypoints + [end_coords])):
        leg_distance = mm.calculate_distance(*waypoint1, *waypoint2)
        leg_bearing = mm.bearing(*waypoint1, *waypoint2)
        
        # Get weather conditions along leg (simplified)
        weather_conditions = get_weather_along_leg(waypoint1, waypoint2, weather_data)
        
        route_analysis['legs'].append({
            'start': waypoint1,
            'end': waypoint2,
            'distance': leg_distance,
            'bearing': leg_bearing,
            'weather': weather_conditions
        })
    
    return route_analysis

def get_weather_along_leg(start, end, weather_data):
    """Get weather conditions along a route leg"""
    # Simplified weather analysis
    # In practice, you'd interpolate weather data along the great circle path
    return {
        'wind_speed': 10.0,  # m/s
        'wind_direction': 270,  # degrees
        'precipitation': 0.0,  # mm
        'visibility': 10000  # m
    }

# Example usage
# route = plan_weather_route(new_york, london, weather_data)
```

### Atmospheric Profile Reanalysis
```python
def reanalyze_atmospheric_profile(observation_location, model_data):
    """
    Reanalyze atmospheric model data at observation location
    """
    # Get model grid coordinates
    model_lats = model_data['latitude']
    model_lons = model_data['longitude']
    
    # Find nearest model grid point
    obs_lat, obs_lon = observation_location
    
    lat_idx = np.argmin(np.abs(model_lats - obs_lat))
    lon_idx = np.argmin(np.abs(model_lons - obs_lon))
    
    # Extract model data at observation location
    model_profile = {
        'pressure': model_data['pressure'][:, lat_idx, lon_idx],
        'temperature': model_data['temperature'][:, lat_idx, lon_idx],
        'u_wind': model_data['u_wind'][:, lat_idx, lon_idx],
        'v_wind': model_data['v_wind'][:, lat_idx, lon_idx]
    }
    
    # Convert pressure to altitude for easier comparison
    model_altitudes = mm.pressure_to_altitude(model_profile['pressure'])
    
    # Rotate wind components to local coordinate system
    local_u, local_v = mm.rotate_wind_components(
        model_profile['u_wind'], 
        model_profile['v_wind'], 
        obs_lat, obs_lon
    )
    
    return {
        'location': observation_location,
        'model_profile': model_profile,
        'altitudes': model_altitudes,
        'rotated_wind': {'u': local_u, 'v': local_v},
        'grid_point': (lat_idx, lon_idx)
    }

# Example usage
# reanalysis = reanalyze_atmospheric_profile(observation_location, model_data)
```

### Climate Grid Processing
```python
def process_climate_grid(climate_data, output_resolution=0.5):
    """
    Process climate model data for analysis at different resolution
    """
    # Get original grid
    original_lats = climate_data['latitude']
    original_lons = climate_data['longitude']
    
    # Create output grid
    new_lats = np.arange(-90, 90 + output_resolution, output_resolution)
    new_lons = np.arange(-180, 180 + output_resolution, output_resolution)
    
    # Calculate grid properties
    grid_spacing = mm.calculate_grid_spacing(
        np.mean(new_lats), np.mean(new_lons), output_resolution
    )
    
    # Process each variable
    processed_data = {}
    for variable in ['temperature', 'precipitation', 'wind_speed']:
        if variable in climate_data:
            # Interpolate to new grid (simplified - in practice use proper interpolation)
            interpolated_data = interpolate_to_new_grid(
                climate_data[variable], original_lats, original_lons, 
                new_lats, new_lons
            )
            processed_data[variable] = interpolated_data
    
    # Calculate derived properties
    processed_data['grid_area'] = mm.calculate_grid_area(
        np.mean(new_lats), output_resolution, output_resolution
    )
    
    return {
        'processed_data': processed_data,
        'grid_specification': {
            'latitude': new_lats,
            'longitude': new_lons,
            'resolution_degrees': output_resolution,
            'grid_spacing_meters': grid_spacing,
            'grid_area_sqm': processed_data['grid_area']
        }
    }

def interpolate_to_new_grid(data, old_lats, old_lons, new_lats, new_lons):
    """Interpolate data to new grid (simplified implementation)"""
    # This is a placeholder - in practice use proper interpolation
    # like scipy.interpolate.griddata or xarray.interp
    return np.random.rand(len(new_lats), len(new_lons))  # Placeholder

# Example usage
# climate_processed = process_climate_grid(climate_model_data, output_resolution=0.5)
```

## Error Handling

The coordinates module includes comprehensive error handling:

### Common Errors
```python
# Error: Invalid latitude
try:
    mm.latlon_to_cartesian(91, -74)  # Latitude > 90
except ValueError as e:
    print(f"Error: {e}")

# Error: Negative distance
try:
    mm.calculate_distance(40, -74, 41, -73, -1000)  # Negative distance
except ValueError as e:
    print(f"Error: {e}")

# Error: Invalid bearing
try:
    mm.destination_point(40, -74, 1000, 361)  # Bearing > 360
except ValueError as e:
    print(f"Error: {e}")

# Error: Inconsistent array shapes
try:
    lats = np.array([40, 41])
    lons = np.array([-74])
    mm.calculate_distance(lats, lons, lats, lons)
except ValueError as e:
    print(f"Error: {e}")
```

## Performance Considerations

### Vectorized Operations
All functions support numpy arrays for efficient vectorized operations:

```python
import numpy as np

# Vectorized distance calculation for multiple points
lats1 = np.array([40.7, 51.5, 35.7])
lons1 = np.array([-74.0, -0.1, 139.7])
lats2 = np.array([34.1, 34.1, 34.1])
lons2 = np.array([-118.2, -118.2, -118.2])

distances = mm.calculate_distance(lats1, lons1, lats2, lons2)
print(f"Distances: {distances/1000:.1f} km")
```

### Memory Efficiency
For large datasets, process in chunks:

```python
def process_large_coordinate_dataset(data_chunk):
    """Process a chunk of coordinate data"""
    result = {}
    result['distances'] = mm.calculate_distance(
        data_chunk['lat1'], data_chunk['lon1'],
        data_chunk['lat2'], data_chunk['lon2']
    )
    result['bearings'] = mm.bearing(
        data_chunk['lat1'], data_chunk['lon1'],
        data_chunk['lat2'], data_chunk['lon2']
    )
    return result
```

## Constants

The coordinates module uses physical constants defined in [`monet_meteo.constants`](../constants.md):

- `R_earth`: Earth's radius (6.371×10⁶ m)
- `Omega`: Earth's rotation rate (7.292×10⁻⁵ s⁻¹)

## References

- Snyder, J.P. (1987). Map Projections - A Working Manual. U.S. Geological Survey.
- Vincenty, T. (1975). Direct and Inverse Solutions of Geodesics on the Ellipsoid. Survey Review.
- American Meteorological Society (2023). Glossary of Meteorology. https://glossary.ametsoc.org/

## See Also

- [Thermodynamics Module](thermodynamics.md) - Thermodynamic variable calculations
- [Dynamic Calculations](dynamics.md) - Dynamic meteorology functions
- [Statistical Analysis](statistical.md) - Statistical and micrometeorological functions
- [Unit Conversions](units.md) - Meteorological unit conversion utilities
- [Data Models](models.md) - Structured data models for atmospheric data
- [Interpolation](interpolation.md) - Data interpolation methods