"""Tests for dustapprox.io.svo module."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from dustapprox.io.svo import (
    spectra_file_reader,
    get_svo_spectrum_units,
    get_svo_passbands,
    SVOSpectrum,
)


class TestSpectraFileReader:
    """Test the spectra_file_reader function."""

    def test_read_basic_spectrum_file(self, tmp_path):
        """Test reading a basic SVO spectrum file."""
        # Create a minimal SVO spectrum file
        spectrum_content = """# Kurucz ODFNEW /NOVER (2003)
# teff = 5000 K (temperature)
# logg = 4.5 log(cm/s2) (gravity)
# meta = 0.0 (metallicity)
# vtur = 2 km/s (microturbulence)
# alpha = 0.0 (alpha)
# column 1: WAVELENGTH (ANGSTROM), Wavelength in Angstrom
# column 2: FLUX (ERG/CM2/S/A), Flux in erg/cm2/s/A
1000.0 1.0e-10
1100.0 1.5e-10
1200.0 2.0e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        assert isinstance(data, dict)
        assert "teff" in data
        assert data["teff"]["value"] == 5000.0
        assert data["logg"]["value"] == 4.5
        assert data["feh"]["value"] == 0.0  # meta renamed to feh
        assert data["vtur"]["value"] == 2.0
        assert "columns" in data
        assert "WAVELENGTH" in data["columns"]
        assert "FLUX" in data["columns"]
        assert isinstance(data["data"], pd.DataFrame)
        assert len(data["data"]) == 3

    def test_read_spectrum_extracts_parameters(self, tmp_path):
        """Test extraction of stellar parameters."""
        spectrum_content = """# Test spectrum
# teff = 6000 K (temperature)
# logg = 3.2 log(g)
# meta = -0.5 (metallicity)
# alpha = 0.3 (alpha)
# column 1: WAVELENGTH (ANGSTROM), test
# column 2: FLUX (ERG/CM2/S/A), test
500.0 1e-12
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        assert data["teff"]["value"] == 6000.0
        assert data["logg"]["value"] == 3.2
        assert data["feh"]["value"] == -0.5
        assert data["alpha"]["value"] == 0.3
        assert data["teff"]["unit"] == "K"
        assert data["teff"]["description"] == "temperature"

    def test_read_spectrum_erg_unit_standardization(self, tmp_path):
        """Test that ERG/CM2/S/A units are standardized."""
        spectrum_content = """# Test
# teff = 5000 K
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        # Should be standardized to lowercase with slashes
        assert data["columns"]["FLUX"]["unit"] == "erg/cm2/s/Angstrom"

    def test_read_spectrum_with_comments(self, tmp_path):
        """Test that comment lines in data are ignored."""
        spectrum_content = """# Kurucz
# teff = 5500 K
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
# This is a comment in data
1000.0 1e-10
1100.0 1.5e-10
# Another comment
1200.0 2e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        # Should ignore comment lines in data section
        assert len(data["data"]) == 3

    def test_read_spectrum_empty_header_lines(self, tmp_path):
        """Test that empty lines in header don't break parsing."""
        spectrum_content = """# Kurucz
# teff = 5000 K (temperature)
# logg = 4.5 log(cm/s2) (gravity)

# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        assert data["teff"]["value"] == 5000.0
        assert data["logg"]["value"] == 4.5

    def test_read_spectrum_column_description(self, tmp_path):
        """Test extraction of column descriptions."""
        spectrum_content = """# Test
# teff = 5000 K
# column 1: WAVELENGTH (ANGSTROM), Wavelength in Angstrom units
# column 2: FLUX (ERG/CM2/S/A), Flux in erg/cm2/s/A units
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        assert data["columns"]["WAVELENGTH"]["description"] == "Wavelength in Angstrom units"
        assert data["columns"]["FLUX"]["description"] == "Flux in erg/cm2/s/A units"


class TestGetSVOSpectrumUnits:
    """Test the get_svo_spectrum_units function."""

    def test_get_units_standard_format(self, tmp_path):
        """Test getting units from standard SVO spectrum."""
        spectrum_content = """# Test
# teff = 5000 K
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))
        lamb_unit, flux_unit = get_svo_spectrum_units(data)

        # Should return Quantity objects
        assert hasattr(lamb_unit, "unit")
        assert hasattr(flux_unit, "unit")
        # Units should contain wavelength and flux info
        assert "Angstrom" in str(lamb_unit.unit) or "angstrom" in str(lamb_unit.unit).lower()

    def test_get_units_lowercase_fallback(self, tmp_path):
        """Test that lowercase units are handled as fallback."""
        spectrum_content = """# Test
# teff = 5000 K
# column 1: WAVELENGTH (angstrom), wavelength
# column 2: FLUX (erg/cm2/s/a), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))
        # Manually adjust to lowercase to test fallback
        data["columns"]["WAVELENGTH"]["unit"] = "angstrom"
        data["columns"]["FLUX"]["unit"] = "erg/cm2/s/a"

        lamb_unit, flux_unit = get_svo_spectrum_units(data)

        assert hasattr(lamb_unit, "unit")
        assert hasattr(flux_unit, "unit")


class TestGetSVOPassbands:
    """Test the get_svo_passbands function."""

    @patch("dustapprox.io.svo.get_pyphot_filter")
    def test_get_single_passband_string(self, mock_get_filter):
        """Test getting a single passband as string."""
        mock_filter = Mock()
        mock_filter.name = "GAIA/GAIA3.G"
        mock_get_filter.return_value = mock_filter

        result = get_svo_passbands("GAIA/GAIA3.G")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == mock_filter
        mock_get_filter.assert_called_once_with("GAIA/GAIA3.G")

    @patch("dustapprox.io.svo.get_pyphot_filter")
    def test_get_multiple_passbands_list(self, mock_get_filter):
        """Test getting multiple passbands as list."""
        mock_filters = [Mock(name=f"filter_{i}") for i in range(3)]
        mock_get_filter.side_effect = mock_filters

        identifiers = ["GAIA/GAIA3.G", "GAIA/GAIA3.Gbp", "GAIA/GAIA3.Grp"]
        result = get_svo_passbands(identifiers)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(f in result for f in mock_filters)
        assert mock_get_filter.call_count == 3

    @patch("dustapprox.io.svo.get_pyphot_filter")
    def test_get_passbands_calls_pyphot(self, mock_get_filter):
        """Test that get_pyphot_filter is called correctly."""
        mock_filter = Mock()
        mock_get_filter.return_value = mock_filter

        get_svo_passbands(["2MASS/2MASS.J", "2MASS/2MASS.H"])

        assert mock_get_filter.call_count == 2
        calls = mock_get_filter.call_args_list
        assert calls[0][0][0] == "2MASS/2MASS.J"
        assert calls[1][0][0] == "2MASS/2MASS.H"


class TestSVOSpectrum:
    """Test the SVOSpectrum class."""

    def test_svo_spectrum_initialization(self, tmp_path):
        """Test SVOSpectrum initialization."""
        spectrum_content = """# Kurucz
# teff = 5000 K (temperature)
# logg = 4.5 log(cm/s2) (gravity)
# meta = 0.0 (metallicity)
# vtur = 2 km/s (microturbulence)
# alpha = 0.0 (alpha)
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1.0e-10
1100.0 1.5e-10
1200.0 2.0e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        spectrum = SVOSpectrum(str(fname))

        assert spectrum.filename == str(fname)
        assert hasattr(spectrum, "units")
        assert len(spectrum.units) == 2
        assert hasattr(spectrum, "λ")
        assert hasattr(spectrum, "flux")
        assert hasattr(spectrum, "meta")
        assert isinstance(spectrum.meta, dict)

    def test_svo_spectrum_meta_extraction(self, tmp_path):
        """Test that metadata is correctly extracted."""
        spectrum_content = """# Kurucz
# teff = 5500 K (temperature)
# logg = 4.2 log(cm/s2) (gravity)
# meta = -0.3 (metallicity)
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        spectrum = SVOSpectrum(str(fname))

        # Check metadata extraction
        assert "teff" in spectrum.meta
        assert spectrum.meta["teff"] == 5500.0
        assert "logg" in spectrum.meta
        assert spectrum.meta["logg"] == 4.2
        assert "feh" in spectrum.meta  # meta renamed to feh
        assert spectrum.meta["feh"] == -0.3

    def test_svo_spectrum_wavelength_array(self, tmp_path):
        """Test wavelength array handling."""
        spectrum_content = """# Test
# teff = 5000 K
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
1100.0 1.5e-10
1200.0 2e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        spectrum = SVOSpectrum(str(fname))

        assert len(spectrum.λ) == 3
        assert np.allclose(spectrum.λ.value[:], [1000.0, 1100.0, 1200.0])

    def test_svo_spectrum_flux_array(self, tmp_path):
        """Test flux array handling."""
        spectrum_content = """# Test
# teff = 5000 K
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1.0e-10
1100.0 1.5e-10
1200.0 2.0e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        spectrum = SVOSpectrum(str(fname))

        assert len(spectrum.flux) == 3
        assert np.allclose(spectrum.flux.value[:], [1.0e-10, 1.5e-10, 2.0e-10])

    def test_svo_spectrum_units_tuple(self, tmp_path):
        """Test that units are stored as tuple."""
        spectrum_content = """# Test
# teff = 5000 K
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        spectrum = SVOSpectrum(str(fname))

        # units is set in __init__ as a tuple
        assert hasattr(spectrum, "units")
        assert isinstance(spectrum.units, tuple)


class TestSVOEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file_raises_error(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            spectra_file_reader("/nonexistent/path/spectrum.txt")

    def test_svo_spectrum_nonexistent_file_raises_error(self):
        """Test that SVOSpectrum with nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            SVOSpectrum("/nonexistent/path/spectrum.txt")

    def test_read_spectrum_minimal_header(self, tmp_path):
        """Test reading spectrum with minimal header info."""
        spectrum_content = """# Test
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        # Should still work with minimal header
        assert "data" in data
        assert isinstance(data["data"], pd.DataFrame)

    def test_read_spectrum_many_parameters(self, tmp_path):
        """Test reading spectrum with many parameters."""
        spectrum_content = """# Full header
# teff = 5000 K (temperature)
# logg = 4.5 log(cm/s2) (gravity)
# meta = 0.0 (metallicity)
# vtur = 2 km/s (microturbulence)
# alpha = 0.3 (alpha)
# lh = 1.5 (param1)
# custom1 = 10.0 (custom_unit1)
# custom2 = 20.5 (custom_unit2)
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        # All parameters should be extracted
        assert data["teff"]["value"] == 5000.0
        assert data["logg"]["value"] == 4.5
        assert data["vtur"]["value"] == 2.0
        assert data["alpha"]["value"] == 0.3
        assert data["lh"]["value"] == 1.5
        assert data["custom1"]["value"] == 10.0
        assert data["custom2"]["value"] == 20.5

    def test_get_svo_passbands_empty_list(self):
        """Test behavior with empty filter list."""
        result = get_svo_passbands([])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_spectrum_with_whitespace_variations(self, tmp_path):
        """Test spectrum file with varying whitespace."""
        spectrum_content = """# Test spectrum with whitespace
# teff = 5000 K (temperature)
# logg = 4.5 log(cm/s2) (gravity)
# meta = 0.0 (metallicity)
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0    1e-10
1100.0     1.5e-10
1200.0  2e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        data = spectra_file_reader(str(fname))

        # Should handle whitespace robustly
        assert isinstance(data["data"], pd.DataFrame)
        assert len(data["data"]) == 3


class TestSVOIntegration:
    """Integration tests for SVO module."""

    def test_spectra_file_reader_to_spectrum_roundtrip(self, tmp_path):
        """Test that data from reader matches SVOSpectrum."""
        spectrum_content = """# Kurucz
# teff = 5000 K (temperature)
# logg = 4.5 log(cm/s2) (gravity)
# meta = 0.0 (metallicity)
# column 1: WAVELENGTH (ANGSTROM), wavelength
# column 2: FLUX (ERG/CM2/S/A), flux
1000.0 1.0e-10
1100.0 1.5e-10
1200.0 2.0e-10
"""
        fname = tmp_path / "spectrum.txt"
        fname.write_text(spectrum_content)

        # Get data via reader
        data = spectra_file_reader(str(fname))
        # Get data via SVOSpectrum
        spectrum = SVOSpectrum(str(fname))

        # Wavelength should match
        assert np.allclose(data["data"]["WAVELENGTH"].values, spectrum.λ.value)
        # Flux should match
        assert np.allclose(data["data"]["FLUX"].values, spectrum.flux.value)
        # Metadata should match
        assert spectrum.meta["teff"] == data["teff"]["value"]
        assert spectrum.meta["logg"] == data["logg"]["value"]

    @patch("dustapprox.io.svo.get_pyphot_filter")
    def test_get_svo_passbands_tuple_input(self, mock_get_filter):
        """Test that get_svo_passbands works with tuple input."""
        mock_filters = [Mock(name=f"f_{i}") for i in range(2)]
        mock_get_filter.side_effect = mock_filters

        result = get_svo_passbands(("id1", "id2"))

        assert len(result) == 2
        assert mock_get_filter.call_count == 2
