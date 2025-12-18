"""Shared pytest fixtures for dustapprox tests."""

import pytest
import numpy as np
import astropy.units as u
from pathlib import Path


@pytest.fixture
def sample_wavelengths():
    """Sample wavelength array in angstroms."""
    return np.array([1000, 2000, 3000, 5000, 10000, 20000]) * u.angstrom


@pytest.fixture
def sample_wavelengths_micron():
    """Sample wavelength array in microns."""
    return np.array([0.1, 0.2, 0.3, 0.5, 1.0, 2.0]) * u.micron


@pytest.fixture
def sample_flux():
    """Sample flux array."""
    return np.array([1e-10, 2e-10, 3e-10, 4e-10, 3e-10, 2e-10])


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_stellar_params():
    """Sample stellar parameters for testing."""
    return {
        "teff": 5777.0,  # K (Sun-like)
        "logg": 4.44,
        "feh": 0.0,
        "alpha": 0.0,
    }


@pytest.fixture
def sample_extinction_params():
    """Sample extinction parameters for testing."""
    return {
        "A0": 1.0,  # mag
        "R0": 3.1,  # standard Rv
    }


@pytest.fixture
def small_test_dataframe():
    """Create a small test DataFrame for model testing."""
    import pandas as pd
    
    data = {
        "teff": [5000, 6000, 7000],
        "logg": [4.0, 4.5, 5.0],
        "feh": [0.0, -0.5, 0.5],
        "A0": [0.5, 1.0, 1.5],
        "alpha": [0.0, 0.0, 0.0],
        "R0": [3.1, 3.1, 3.1],
        "passband": ["GAIA_GAIA3.G", "GAIA_GAIA3.G", "GAIA_GAIA3.G"],
        "mag0": [10.0, 9.5, 9.0],
        "mag": [10.5, 10.5, 10.5],
        "Ax": [0.5, 1.0, 1.5],
    }
    return pd.DataFrame(data)
