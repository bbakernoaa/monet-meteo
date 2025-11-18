# Installation Guide

This guide provides detailed instructions for installing Monet-Meteo and its dependencies.

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux
- Memory: Minimum 512 MB RAM (2 GB recommended for large datasets)

### Python Dependencies
Required dependencies will be automatically installed:
- `numpy` >= 1.20.0
- `typing_extensions` >= 3.7.0

### Optional Dependencies
These dependencies enable additional features:
- `xarray` >= 0.18.0 - For advanced data handling and netCDF support
- `dask` >= 2021.0.0 - For parallel processing and out-of-core computations
- `pint` >= 0.17.0 - For unit management (optional, but recommended)
- `matplotlib` >= 3.3.0 - For plotting and visualization
- `cartopy` >= 0.19.0 - For geographic mapping
- `netCDF4` >= 1.5.0 - For netCDF file support

## 🚀 Installation Methods

### Method 1: pip Installation (Recommended)

#### Stable Release
```bash
pip install monet-meteo
```

#### Development Version
```bash
pip install git+https://github.com/noaa-arl/monet-meteo.git
```

#### Specific Version
```bash
pip install monet-meteo==0.0.1
```

### Method 2: Conda Installation

If you use conda, you can install from conda-forge:
```bash
conda install -c conda-forge monet-meteo
```

### Method 3: From Source

For development or custom builds:

```bash
# Clone the repository
git clone https://github.com/noaa-arl/monet-meteo.git
cd monet-meteo

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,docs]"
```

## 🔧 Post-Installation Verification

After installation, verify that Monet-Meteo is working correctly:

```python
# Test basic import
import monet_meteo as mm
print(f"Monet-Meteo version: {mm.__version__}")

# Test basic functionality
pressure = 850  # hPa
temperature = 288.15  # K
theta = mm.potential_temperature(pressure, temperature)
print(f"Potential temperature test: {theta:.2f} K")

print("✅ Monet-Meteo installation successful!")
```

## 📦 Optional Dependencies Installation

### For Scientific Computing
```bash
pip install xarray dask netCDF4
```

### For Data Visualization
```bash
pip install matplotlib cartopy
```

### For Unit Management
```bash
pip install pint
```

### Complete Environment Setup
```bash
# Create a new conda environment
conda create -n monet-meteo-env python=3.9
conda activate monet-meteo-env

# Install all dependencies
pip install monet-meteo xarray dask netCDF4 matplotlib cartopy pint
```

## 🏗️ Building from Source

### Prerequisites
- Python 3.8+
- pip
- setuptools

### Build Steps
```bash
# Clone the repository
git clone https://github.com/noaa-arl/monet-meteo.git
cd monet-meteo

# Install build dependencies
pip install build wheel

# Build the package
python -m build

# Install the built package
pip install dist/monet_meteo-0.0.1-py3-none-any.whl
```

### Development Setup
```bash
# Clone the repository
git clone https://github.com/noaa-arl/monet-meteo.git
cd monet-meteo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,docs,test]"
```

## 🐛 Troubleshooting Installation

### Common Issues

#### Issue: Permission Errors
```bash
# Use user installation
pip install --user monet-meteo

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install monet-meteo
```

#### Issue: NumPy Version Conflicts
```bash
# Upgrade NumPy
pip install --upgrade numpy

# Or create fresh environment
conda create -n monet-meteo numpy=1.21
```

#### Issue: xarray/dask Installation Problems
```bash
# Install conda-forge version
conda install -c conda-forge xarray dask

# Or install with specific versions
pip install xarray==0.19.0 dask==2021.0.0
```

#### Issue: Windows Installation Problems
```bash
# Use conda instead of pip
conda install -c conda-forge monet-meteo

# Or install with pre-compiled wheels
pip install --only-binary=:all: monet-meteo
```

### Environment Configuration

#### Virtual Environment Setup
```bash
# Create virtual environment
python -m venv monet-meteo-env

# Activate on Windows
.\monet-meteo-env\Scripts\activate

# Activate on Unix/Linux/Mac
source monet-meteo-env/bin/activate

# Install package
pip install monet-meteo
```

#### Conda Environment Setup
```bash
# Create conda environment
conda create -n monet-meteo python=3.9

# Activate environment
conda activate monet-meteo

# Install dependencies
conda install -c conda-forge numpy xarray dask matplotlib
pip install monet-meteo
```

#### Docker Setup
```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "monet_meteo"]
```

### Testing Installation

Run the test suite to verify installation:

```python
# Run unit tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/unit/test_thermodynamics.py

# Run with coverage
python -m pytest --cov=monet_meteo tests/
```

### Checking Dependencies

```python
# Check installed dependencies
import pkg_resources
installed_packages = [d.project_name for d in pkg_resources.working_set]
print("Monet-Meteo dependencies:")
for dep in ['numpy', 'xarray', 'dask', 'matplotlib']:
    if dep in installed_packages:
        print(f"✅ {dep}")
    else:
        print(f"❌ {dep}")
```

## 📊 Performance Considerations

### Memory Usage
- Minimum: 512 MB RAM for basic operations
- Recommended: 2 GB+ for large datasets
- Optimal: 8 GB+ for climate model data processing

### CPU Requirements
- Minimum: 1 CPU core
- Recommended: 2+ cores for parallel processing
- Optimal: 4+ cores for dask operations

### Storage Requirements
- Library: ~5 MB
- Documentation: ~10 MB
- Test data: ~50 MB
- Sample datasets: ~100 MB+

## 🔄 Updating Monet-Meteo

### Update pip installation
```bash
pip install --upgrade monet-meteo
```

### Update conda installation
```bash
conda update -c conda-forge monet-meteo
```

### Development version update
```bash
cd monet-meteo
git pull
pip install --upgrade -e .
```

## 📚 Additional Resources

- [Official Documentation](https://monet-meteo.readthedocs.io/)
- [GitHub Repository](https://github.com/noaa-arl/monet-meteo)
- [Issue Tracker](https://github.com/noaa-arl/monet-meteo/issues)
- [Discussion Forum](https://github.com/noaa-arl/monet-meteo/discussions)