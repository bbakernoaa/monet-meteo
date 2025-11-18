# Release Process for monet-meteo

This document outlines the process for releasing new versions of the monet-meteo package.

## Prerequisites

- Ensure all tests pass
- Update the changelog with changes for the new version
- Ensure documentation is up to date
- Have PyPI credentials ready

## Version Numbering

This project follows Semantic Versioning (SemVer) with the format `MAJOR.MINOR.PATCH`:

- MAJOR version: Incompatible API changes
- MINOR version: Functionality added in a backward-compatible manner
- PATCH version: Backward-compatible bug fixes

## Release Steps

### 1. Prepare the Release

1. Create a new branch for the release:
   ```bash
   git checkout -b release-vX.Y.Z
   ```

2. Update the changelog with the new version and date

3. Update any documentation as needed

4. Run all tests to ensure everything works:
   ```bash
   pytest tests/
   ```

5. Commit the changes:
   ```bash
   git add .
   git commit -m "Prepare for release X.Y.Z"
   ```

### 2. Create and Tag the Release

1. Push the release branch:
   ```bash
   git push origin release-vX.Y.Z
   ```

2. Create a tag for the release:
   ```bash
   git tag -a vX.Y.Z -m "Version X.Y.Z"
   ```

3. Push the tag:
   ```bash
   git push origin vX.Y.Z
   ```

### 3. Create GitHub Release

1. Go to the repository's "Releases" page on GitHub
2. Click "Draft a new release"
3. Select the tag you just pushed
4. Set the title to "Version X.Y.Z"
5. Copy the changelog entries for this version into the description
6. Click "Publish release"

### 4. Build and Publish to PyPI

The PyPI publication is handled automatically by the GitHub Actions workflow when a release is published on GitHub.

1. The GitHub Actions workflow will:
   - Checkout the tagged commit
   - Build the package distribution
   - Run all tests
   - Publish to PyPI

2. Monitor the workflow to ensure it completes successfully

### 5. Verify the Release

1. Check that the new version is available on PyPI:
   ```bash
   pip index versions monet-meteo
   ```

2. Test the installation of the new version:
   ```bash
   pip install --upgrade monet-meteo
   ```

3. Verify the version in Python:
   ```python
   import monet_meteo
   print(monet_meteo.__version__)
   ```

## Post-Release Steps

1. Update the main branch with the release changes:
   ```bash
   git checkout main
   git merge release-vX.Y.Z
   git push origin main
   ```

2. Create a new development branch if needed:
   ```bash
   git checkout -b develop
   ```

3. Update the version in development to the next expected version with a `.dev` suffix

## Hotfix Releases

For urgent bug fixes:

1. Create a hotfix branch from the release tag:
   ```bash
   git checkout -b hotfix-vX.Y.Z vX.Y.Z
   ```

2. Apply the necessary fixes

3. Follow the standard release process from step 2 onwards