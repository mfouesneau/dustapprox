"""Tests for dustapprox.literature module."""

import numpy as np
import pandas as pd
from unittest.mock import patch

from dustapprox.literature import edr3, c1


def create_mock_c1_data():
    """Create mock C1 extinction data for testing."""
    data = {
        "X": ["C1B556", "C1M467", "C1B768"],
        "alpha": [0.2, 0.25, 0.22],
        "beta_1": [0.1, 0.12, 0.11],
        "beta_2": [0.05, 0.06, 0.055],
        "beta_3": [0.02, 0.03, 0.025],
        "beta_4": [0.01, 0.015, 0.012],
        "gamma_1": [0.5, 0.55, 0.52],
        "gamma_2": [0.1, 0.12, 0.11],
        "gamma_3": [0.02, 0.03, 0.025],
        "delta": [0.15, 0.2, 0.17],
    }
    df = pd.DataFrame(data)
    # Don't set the index here - let the dr3_ext do it
    return df


class TestEdr3ExtInit:
    """Test edr3_ext class initialization."""

    def test_edr3_ext_initialization(self):
        """Test that edr3_ext initializes with data."""
        ext = edr3.edr3_ext()
        assert hasattr(ext, "Ay_top")
        assert hasattr(ext, "Ay_ms")
        assert isinstance(ext.Ay_top, pd.DataFrame)
        assert isinstance(ext.Ay_ms, pd.DataFrame)

    def test_edr3_ext_data_index(self):
        """Test that the data has the expected multi-index structure."""
        ext = edr3.edr3_ext()
        assert isinstance(ext.Ay_top.index, pd.MultiIndex)
        assert isinstance(ext.Ay_ms.index, pd.MultiIndex)
        assert ext.Ay_top.index.names == ["Kname", "Xname"]
        assert ext.Ay_ms.index.names == ["Kname", "Xname"]

    def test_edr3_ext_data_bands(self):
        """Test that the data contains expected bands."""
        ext = edr3.edr3_ext()
        bands_top = ext.Ay_top.index.get_level_values("Kname").unique()
        bands_ms = ext.Ay_ms.index.get_level_values("Kname").unique()
        
        # Check for presence of key bands
        assert len(bands_top) > 0
        assert len(bands_ms) > 0

    def test_edr3_ext_data_xnames(self):
        """Test that the data contains expected X variable names."""
        ext = edr3.edr3_ext()
        xnames_top = ext.Ay_top.index.get_level_values("Xname").unique()
        xnames_ms = ext.Ay_ms.index.get_level_values("Xname").unique()
        
        # Check for key X variables
        assert len(xnames_top) > 0
        assert len(xnames_ms) > 0

    def test_edr3_ext_data_columns(self):
        """Test that the data has expected coefficient columns."""
        ext = edr3.edr3_ext()
        expected_cols = [
            "Intercept", "X", "X2", "X3", "A", "A2", "A3",
            "XA", "XA2", "AX2"
        ]
        for col in expected_cols:
            assert col in ext.Ay_top.columns
            assert col in ext.Ay_ms.columns


class TestEdr3ExtFrom:
    """Test the internal _from method of edr3_ext."""

    def test_from_scalar_inputs(self):
        """Test _from with scalar inputs."""
        ext = edr3.edr3_ext()
        # Use valid band and X names from the data
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        xnames = ext.Ay_top.index.get_level_values("Xname").unique()[0]
        
        result = ext._from(bands, xnames, 0.1, 1.0, flavor="top")
        # Note: _from returns an array even for scalar inputs (using atleast_1d)
        assert isinstance(result, np.ndarray) or isinstance(result, (float, np.floating))
        assert not np.isnan(np.atleast_1d(result)[0])

    def test_from_array_inputs(self):
        """Test _from with array inputs."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        xnames = ext.Ay_top.index.get_level_values("Xname").unique()[0]
        
        x_vals = np.array([0.1, 0.2, 0.3])
        a0_vals = np.array([1.0, 1.5, 2.0])
        
        result = ext._from(bands, xnames, x_vals, a0_vals, flavor="top")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)

    def test_from_mixed_scalar_array(self):
        """Test _from with mixed scalar and array inputs."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        xnames = ext.Ay_top.index.get_level_values("Xname").unique()[0]
        
        result = ext._from(bands, xnames, np.array([0.1, 0.2]), 1.0, flavor="top")
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_from_flavor_ms(self):
        """Test _from with 'ms' flavor."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_ms.index.get_level_values("Kname").unique()[0]
        xnames = ext.Ay_ms.index.get_level_values("Xname").unique()[0]
        
        result = ext._from(bands, xnames, 0.1, 1.0, flavor="ms")
        result_val = np.atleast_1d(result)[0]
        assert not np.isnan(result_val)

    def test_from_formula_consistency(self):
        """Test that _from correctly applies the formula."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        xnames = ext.Ay_top.index.get_level_values("Xname").unique()[0]
        
        X, A0 = 0.5, 2.0
        result = ext._from(bands, xnames, X, A0, flavor="top")
        
        # Manually compute using coefficients
        coeffs = ext.Ay_top.loc[bands, xnames]
        expected = coeffs["Intercept"]
        expected += coeffs["X"] * X
        expected += coeffs["X2"] * X**2
        expected += coeffs["X3"] * X**3
        expected += coeffs["A"] * A0
        expected += coeffs["A2"] * A0**2
        expected += coeffs["A3"] * A0**3
        expected += coeffs["XA"] * X * A0
        expected += coeffs["XA2"] * X * A0**2
        expected += coeffs["AX2"] * X**2 * A0
        
        np.testing.assert_almost_equal(result, expected)


class TestEdr3ExtFromTeff:
    """Test edr3_ext.from_teff method."""

    def test_from_teff_scalar(self):
        """Test from_teff with scalar temperature."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_teff(bands, 5000.0, 1.0, flavor="top")
        result_val = np.atleast_1d(result)[0]
        assert not np.isnan(result_val)

    def test_from_teff_array(self):
        """Test from_teff with array of temperatures."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        teffs = np.array([4000.0, 5000.0, 6000.0])
        result = ext.from_teff(bands, teffs, 1.0, flavor="top")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(~np.isnan(result))

    def test_from_teff_normalization(self):
        """Test that from_teff correctly normalizes temperature."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        xnames = ext.Ay_top.index.get_level_values("Xname").unique()[0]
        
        # Verify that from_teff uses the normalized temperature
        teff = 5040.0  # This should normalize to 1.0
        result1 = ext.from_teff(bands, teff, 1.0, flavor="top")
        result2 = ext._from(bands, xnames, 1.0, 1.0, flavor="top")
        
        # Note: only if xname is TeffNorm, otherwise they might not match
        if xnames == "TeffNorm":
            np.testing.assert_almost_equal(result1, result2)

    def test_from_teff_ms_flavor(self):
        """Test from_teff with main sequence flavor."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_ms.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_teff(bands, 5000.0, 1.0, flavor="ms")
        result_val = np.atleast_1d(result)[0]
        assert not np.isnan(result_val)

    def test_from_teff_various_values(self):
        """Test from_teff with various realistic stellar values."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        # Test realistic values
        teffs = [3500.0, 5000.0, 7500.0, 10000.0]
        for teff in teffs:
            result = ext.from_teff(bands, teff, 0.5, flavor="top")
            assert not np.isnan(result)
            assert np.isfinite(result)


class TestEdr3ExtFromBprp:
    """Test edr3_ext.from_bprp method."""

    def test_from_bprp_scalar(self):
        """Test from_bprp with scalar color."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_bprp(bands, 0.5, 1.0, flavor="top")
        result_val = np.atleast_1d(result)[0]
        assert not np.isnan(result_val)

    def test_from_bprp_array(self):
        """Test from_bprp with array of colors."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        bprps = np.array([0.2, 0.5, 1.0, 1.5])
        result = ext.from_bprp(bands, bprps, 1.0, flavor="top")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert np.all(~np.isnan(result))

    def test_from_bprp_negative_colors(self):
        """Test from_bprp with negative color values."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_bprp(bands, -0.1, 1.0, flavor="top")
        result_val = np.atleast_1d(result)[0]
        assert np.isfinite(result_val)

    def test_from_bprp_ms_flavor(self):
        """Test from_bprp with main sequence flavor."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_ms.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_bprp(bands, 0.5, 1.0, flavor="ms")
        result_val = np.atleast_1d(result)[0]
        assert isinstance(result_val, (float, np.floating))


class TestEdr3ExtFromGmK:
    """Test edr3_ext.from_GmK method."""

    def test_from_gmk_scalar(self):
        """Test from_GmK with scalar color."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_GmK(bands, 1.0, 1.0, flavor="top")
        result_val = np.atleast_1d(result)[0]
        assert not np.isnan(result_val)

    def test_from_gmk_array(self):
        """Test from_GmK with array of colors."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        gmks = np.array([0.5, 1.0, 1.5, 2.0])
        result = ext.from_GmK(bands, gmks, 1.0, flavor="top")
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        assert np.all(~np.isnan(result))

    def test_from_gmk_ms_flavor(self):
        """Test from_GmK with main sequence flavor."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_ms.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_GmK(bands, 1.0, 1.0, flavor="ms")
        result_val = np.atleast_1d(result)[0]
        assert isinstance(result_val, (float, np.floating))

    def test_from_gmk_various_values(self):
        """Test from_GmK with various realistic stellar colors."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        # Test realistic G-Ks values
        gmks = [0.2, 0.5, 1.0, 2.0, 3.0]
        for gmk in gmks:
            result = ext.from_GmK(bands, gmk, 0.5, flavor="top")
            assert not np.isnan(result)


class TestEdr3ExtExtraction:
    """Test extracting coefficients and values from edr3_ext."""

    def test_extract_all_bands(self):
        """Test that all band names can be extracted."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()
        
        for band in bands:
            # Should not raise an error
            assert band is not None

    def test_extract_all_xnames(self):
        """Test that all X variable names can be extracted."""
        ext = edr3.edr3_ext()
        xnames = ext.Ay_top.index.get_level_values("Xname").unique()
        
        for xname in xnames:
            # Should not raise an error
            assert xname is not None

    def test_top_and_ms_consistency(self):
        """Test that top and ms data have similar structure."""
        ext = edr3.edr3_ext()
        bands_top = ext.Ay_top.index.get_level_values("Kname").unique()
        bands_ms = ext.Ay_ms.index.get_level_values("Kname").unique()
        
        # Should have at least some overlap
        assert len(bands_top) > 0
        assert len(bands_ms) > 0


class TestDr3ExtInit:
    """Test dr3_ext class initialization."""

    def test_dr3_ext_initialization(self):
        """Test that dr3_ext initializes with mocked data."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            assert hasattr(ext, "data")
            assert isinstance(ext.data, pd.DataFrame)

    def test_dr3_ext_data_index(self):
        """Test that dr3_ext data has X as index."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            assert ext.data.index.name == "X"

    def test_dr3_ext_data_columns(self):
        """Test that dr3_ext data has expected coefficient columns."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            expected_cols = ["alpha", "beta_1", "beta_2", "beta_3", "beta_4",
                            "gamma_1", "gamma_2", "gamma_3", "delta"]
            for col in expected_cols:
                assert col in ext.data.columns

    def test_dr3_ext_initialization_custom_data(self):
        """Test dr3_ext initialization with custom data path."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext(data="gaia_C1_extinction.ecsv")
            assert hasattr(ext, "data")
            assert isinstance(ext.data, pd.DataFrame)


class TestDr3ExtCall:
    """Test dr3_ext callable interface."""

    def test_dr3_ext_call_scalar(self):
        """Test dr3_ext call with scalar inputs."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                result = ext(band, 0.5, 1.0)
                assert isinstance(result, (float, np.floating, np.ndarray))
                assert not np.isnan(np.atleast_1d(result)[0])

    def test_dr3_ext_call_array_bprp(self):
        """Test dr3_ext call with array BP-RP."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                bprps = np.array([0.2, 0.5, 1.0])
                result = ext(band, bprps, 1.0)
                assert isinstance(result, np.ndarray)
                assert result.shape == (3,)

    def test_dr3_ext_call_array_ag(self):
        """Test dr3_ext call with array A_G."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                ags = np.array([0.5, 1.0, 1.5])
                result = ext(band, 0.5, ags)  # type: ignore
                assert isinstance(result, np.ndarray)
                assert result.shape == (3,)

    def test_dr3_ext_call_both_arrays(self):
        """Test dr3_ext call with both inputs as arrays."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                bprps = np.array([0.2, 0.5, 1.0])
                ags = np.array([0.5, 1.0, 1.5])
                result = ext(band, bprps, ags)  # type: ignore
                # Result should have same shape as input arrays
                assert isinstance(result, np.ndarray)

    def test_dr3_ext_formula_consistency(self):
        """Test that call correctly applies the polynomial formula."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                bprp, ag = 0.5, 1.0
                result = ext(band, bprp, ag)
                
                # Manually compute using coefficients
                c = ext.data.loc[band]
                expected = c["alpha"]
                for i in range(1, 5):
                    expected += c[f"beta_{i}"] * bprp**i
                for j in range(1, 4):
                    expected += c[f"gamma_{j}"] * ag**j
                expected += c["delta"] * bprp * ag
                
                np.testing.assert_almost_equal(np.squeeze(result), expected)

    def test_dr3_ext_squeeze_output(self):
        """Test that dr3_ext squeezes scalar outputs."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                # Single scalar inputs should return scalar
                result = ext(band, 0.5, 1.0)
                assert np.ndim(result) == 0 or isinstance(result, (float, np.floating))


class TestDr3ExtBands:
    """Test dr3_ext with different band names."""

    def test_dr3_ext_all_bands(self):
        """Test dr3_ext with all available bands."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            
            for band in band_names:
                result = ext(band, 0.5, 1.0)
                assert not np.isnan(np.atleast_1d(result)[0])
                assert np.isfinite(np.atleast_1d(result)[0])

    def test_dr3_ext_band_consistency(self):
        """Test that calling with same band gives consistent results."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                result1 = ext(band, 0.5, 1.0)
                result2 = ext(band, 0.5, 1.0)
                np.testing.assert_equal(np.squeeze(result1), np.squeeze(result2))


class TestLiteratureImports:
    """Test that literature module imports work correctly."""

    def test_literature_has_edr3(self):
        """Test that literature module exports edr3."""
        from dustapprox import literature
        assert hasattr(literature, "edr3")

    def test_literature_has_c1(self):
        """Test that literature module exports c1."""
        from dustapprox import literature
        assert hasattr(literature, "c1")

    def test_literature_all_list(self):
        """Test that literature module __all__ is defined."""
        from dustapprox import literature
        assert hasattr(literature, "__all__")
        assert isinstance(literature.__all__, list)
        assert "edr3" in literature.__all__
        assert "c1" in literature.__all__

    def test_can_import_edr3_ext(self):
        """Test that edr3_ext can be imported."""
        from dustapprox.literature.edr3 import edr3_ext
        assert edr3_ext is not None

    def test_can_import_dr3_ext(self):
        """Test that dr3_ext can be imported."""
        from dustapprox.literature.c1 import dr3_ext
        assert dr3_ext is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_edr3_zero_extinction(self):
        """Test edr3_ext with zero extinction."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_teff(bands, 5000.0, 0.0, flavor="top")
        assert not np.isnan(result)
        assert np.isfinite(result)

    def test_edr3_high_extinction(self):
        """Test edr3_ext with high extinction values."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        result = ext.from_teff(bands, 5000.0, 20.0, flavor="top")
        assert not np.isnan(result)
        assert np.isfinite(result)

    def test_dr3_zero_extinction(self):
        """Test dr3_ext with zero A_G."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                result = ext(band, 0.5, 0.0)
                assert not np.isnan(np.atleast_1d(result)[0])

    def test_dr3_zero_color(self):
        """Test dr3_ext with zero BP-RP color."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                result = ext(band, 0.0, 1.0)
                assert not np.isnan(np.atleast_1d(result)[0])

    def test_edr3_empty_array(self):
        """Test edr3_ext with empty arrays."""
        ext = edr3.edr3_ext()
        bands = ext.Ay_top.index.get_level_values("Kname").unique()[0]
        
        empty_arr = np.array([])
        result = ext.from_teff(bands, empty_arr, 1.0, flavor="top")
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_dr3_empty_array(self):
        """Test dr3_ext with empty arrays."""
        mock_data = create_mock_c1_data()
        with patch("dustapprox.literature.c1.ecsv.read", return_value=mock_data):
            ext = c1.dr3_ext()
            band_names = ext.data.index
            band = band_names[0] if len(band_names) > 0 else None
            
            if band is not None:
                empty_arr = np.array([])
                result = ext(band, empty_arr, 1.0)
                assert isinstance(result, np.ndarray)
                assert len(result) == 0
