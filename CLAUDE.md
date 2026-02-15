# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

btrack is a Python library for multi-object tracking using Bayesian inference, specifically designed for cell tracking in time-lapse microscopy data. The core tracking algorithm is implemented in C++ with Python bindings.

## Architecture

### Hybrid C++/Python Implementation
- **C++ Core**: Performance-critical tracking algorithms in `btrack/src/` with headers in `btrack/include/`
- **Python Wrapper**: Main API in `btrack/core.py` with ctypes bindings via `libwrapper.py`
- **Build System**: Uses Makefile for C++ compilation, setuptools for Python packaging

### Key Components
- `BayesianTracker` (core.py): Main tracking class that wraps C++ implementation
- `btypes.py`: Python type definitions and data structures
- `models.py`: Motion and observation models for tracking
- `config.py`: Configuration management
- `optimise/`: Hypothesis generation and optimization algorithms
- `io/`: Import/export functionality for various data formats
- `napari/`: Napari plugin integration

### Data Flow
1. Input data → `PyTrackObject` instances (btypes.py)
2. Tracking algorithm (C++ core) → Tracklets
3. Hypothesis generation and optimization → Final tracks
4. Export via `io/` modules to various formats

## Development Commands

### Building
```bash
# Build C++ components
./build.sh

# Alternative manual build
make clean && make
```

### Testing
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests (same as CI)
pytest --cov --cov-report=xml

# Run specific test file
pytest tests/test_tracker.py

# Test across Python versions locally (optional)
tox run
```

### Code Quality
```bash
# Format code (black)
black btrack/

# Lint with ruff
ruff check btrack/

# Run pre-commit hooks
pre-commit run --all-files
```

### CI/CD Requirements

#### Python Version Support (2025)
- **Supported**: Python 3.9, 3.10, 3.11, 3.12, 3.13
- **CI Testing**: Currently tests Python 3.10, 3.11, 3.12 in GitHub Actions
- **Tox Configuration**: Includes Python 3.10, 3.11, 3.12, 3.13

#### GitHub Actions Architecture
- **Test workflow**: Cross-platform testing with cross-compiled Windows libraries
- **Deploy workflow**: Parallel library builds → single deploy job
- **Reusable action**: `.github/actions/build-btrack/` for consistent builds
- **Cross-compilation**: Windows DLLs built on Linux with mingw-w64
- **Platform Testing**: Ubuntu/macOS native builds, Windows uses cross-compiled libraries

#### Deployment Pipeline
1. **Build libraries**: Platform-specific C++ libraries (Linux native, macOS native, Windows cross-compiled)
2. **Deploy**: Single job that collects all platform libraries, builds packages, tests on TestPyPI, and deploys to PyPI

#### Secrets and Configuration
- `TESTPYPI_SECRET`: API token for TestPyPI deployment
- `ADD_TO_PROJECT_PAT`: Personal access token for project board automation
- PyPI deployment uses trusted publishing (no API token required)

#### Workflow Features
- **Cross-platform builds**: Native Linux/macOS, cross-compiled Windows libraries
- **Artifact handling**: Libraries built once, reused across test jobs
- **Direct testing**: pytest runs directly in CI (no tox wrapper)
- **Parallel execution**: Build-and-test runs in parallel across platforms
- **Trusted publishing**: PyPI deployment without API tokens

### Installation Modes
```bash
# Development install
pip install -e ".[dev]"

# With napari integration
pip install -e ".[napari]"

# Documentation dependencies
pip install -e ".[docs]"
```

## Working with C++ Code

### Dependencies
- Eigen library (automatically cloned by build.sh to `btrack/include/eigen/`)
- C++17 standard required (changed from C++11 in v0.5.1+)

### Key C++ Files
- `tracker.h/.cc`: Main tracking algorithm implementation
- `hypothesis.h/.cc`: Track hypothesis generation and scoring
- `manager.h/.cc`: Memory and object lifecycle management
- `wrapper.cc`: Python C API interface

### Platform-Specific Builds
The Makefile handles cross-platform compilation:
- Linux: g++ with shared library (.so)
- macOS: clang++ with universal binary support (.dylib)
- Windows: MinGW cross-compilation (.dll)

## Configuration System

Tracking behavior is controlled via configuration objects defined in `config.py`. Key configuration categories:
- Motion models (constant velocity, random walk, etc.)
- Hypothesis generation parameters
- Optimization settings
- Volume constraints

## Testing Structure

- Unit tests in `tests/` directory
- Test data in `tests/_test_data/`
- Napari-specific tests in `tests/napari/`
- Uses pytest with fixtures defined in `conftest.py`

## Common Development Patterns

### Adding New Motion Models
1. Define model in `models.py`
2. Add corresponding C++ implementation in relevant source files
3. Update configuration options in `config.py`
4. Add tests covering the new functionality

### Extending Export Formats
1. Create new exporter in `io/` module
2. Register with `export_delegator`
3. Add format detection logic
4. Include appropriate tests

### Working with the Python-C++ Interface
- Use `libwrapper.py` for ctypes function signatures
- Follow existing patterns for numpy array handling
- Be careful with memory management across language boundary

## Release Process

### Creating a Release
1. Update version in `pyproject.toml` (setuptools_scm handles versioning)
2. Create and push a version tag: `git tag v1.2.3 && git push origin v1.2.3`
3. GitHub Actions will automatically:
   - Build platform-specific libraries
   - Run full test suite
   - Deploy to TestPyPI
   - Verify installation across platforms
   - Deploy to PyPI (with manual approval)

### Manual Testing
Before creating a release tag, manually test:
```bash
# Build and test locally
./build.sh
pytest

# Check packaging
python -m build
pytest --cov

# Verify C++ compilation on target platforms
make clean && make
```
