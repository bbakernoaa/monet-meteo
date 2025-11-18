# PyPI Publication Guide for monet-meteo

This document provides instructions for publishing the monet-meteo package to PyPI.

## Prerequisites

1. Ensure you have a PyPI account and are added as a maintainer for the monet-meteo package
2. Install required tools:
   ```bash
   pip install build twine
   ```

## Pre-Publication Checklist

- [ ] All tests pass (`pytest tests/`)
- [ ] Code is formatted with Black (`black monet_meteo/ tests/`)
- [ ] Code passes linting (`flake8 monet_meteo/ tests/`)
- [ ] Type checking passes (`mypy monet_meteo/`)
- [ ] Documentation is up to date
- [ ] Changelog is updated with release notes
- [ ] Version number is updated in the appropriate place
- [ ] All new features are properly tested

## Building the Package

1. Clean any previous builds:
   ```bash
   rm -rf dist/ build/ *.egg-info/
   ```

2. Build the package:
   ```bash
   python -m build
   ```
   
   This creates both source distribution (`.tar.gz`) and wheel (`.whl`) files in the `dist/` directory.

3. Verify the built package:
   ```bash
   twine check dist/*
   ```

## Testing the Package

Before publishing to the public PyPI, test on TestPyPI:

1. Upload to TestPyPI:
   ```bash
   twine upload --repository testpypi dist/*
   ```

2. Test installation from TestPyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ monet-meteo
   ```

3. Verify the installation works as expected

## Publishing to PyPI

### Automatic Publication (Recommended)

The preferred method is using the GitHub Actions workflow:
1. Create a GitHub release with a tag
2. The workflow automatically builds and publishes to PyPI

### Manual Publication

If needed, you can publish manually:

1. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

## Post-Publication Verification

1. Verify the package is available on PyPI
2. Test installation:
   ```bash
   pip install monet-meteo
   ```
3. Verify functionality:
   ```python
   import monet_meteo
   print(monet_meteo.__version__)
   ```

## Troubleshooting

### Common Issues

- **Invalid distribution format**: Ensure you're using `python -m build` instead of `python setup.py sdist bdist_wheel`
- **Missing files**: Check your `MANIFEST.in` file includes all necessary files
- **Metadata errors**: Verify your `pyproject.toml` has correct metadata
- **Authentication**: Make sure you're authenticated with PyPI

### Testing Distribution Locally

You can test the distribution locally before publishing:

```bash
pip install dist/monet_meteo-*.tar.gz
```

or

```bash
pip install dist/monet_meteo-*.whl
```

## Security Best Practices

- Never commit API tokens or credentials to the repository
- Use environment variables or secure credential storage for PyPI tokens
- Use trusted publishing with GitHub Actions when possible
- Regularly rotate PyPI API tokens