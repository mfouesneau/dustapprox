"""Tests for dustapprox.legacy_extinction module."""

import numpy as np
import pytest
import warnings
import astropy.units as u

from dustapprox.legacy_extinction import (
    ExtinctionLaw,
    CCM89,
    F99,
    _val_in_unit,
    _warning_on_one_line,
)


class TestWarningFormatter:
    """Test the _warning_on_one_line function."""

    def test_warning_format(self):
        """Test that warning format is correct."""
        result = _warning_on_one_line(
            "Test message",
            UserWarning,
            "test_file.py",
            42,
        )
        assert "test_file.py:42" in result
        assert "UserWarning" in result
        assert "Test message" in result

    def test_warning_with_different_category(self):
        """Test warning with different exception category."""
        result = _warning_on_one_line(
            "Another message",
            RuntimeWarning,
            "another_file.py",
            100,
        )
        assert "RuntimeWarning" in result
        assert "another_file.py:100" in result


class TestValInUnit:
    """Test the _val_in_unit helper function."""

    def test_val_in_unit_with_quantity(self):
        """Test _val_in_unit with a Quantity object."""
        value = 5000.0 * u.angstrom
        result = _val_in_unit("test", value, "angstrom")
        assert result.unit == u.angstrom
        assert result.value == 5000.0

    def test_val_in_unit_with_unit_conversion(self):
        """Test _val_in_unit converts to desired unit."""
        value = 0.5 * u.micron
        result = _val_in_unit("test", value, "angstrom")
        assert result.unit == u.angstrom
        np.testing.assert_almost_equal(result.value, 5000.0)

    def test_val_in_unit_without_units_warning(self):
        """Test _val_in_unit warns when no units provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _val_in_unit("test_var", 1000.0, "angstrom")
            assert len(w) == 1
            assert "does not have explicit units" in str(w[0].message)
            assert "Assuming" in str(w[0].message)
            assert result.value == 1000.0


class TestExtinctionLawBase:
    """Test the ExtinctionLaw base class."""

    def test_extinction_law_init(self):
        """Test ExtinctionLaw initialization."""
        law = ExtinctionLaw()
        assert law.name == "None"

    def test_extinction_law_repr(self):
        """Test ExtinctionLaw repr."""
        law = ExtinctionLaw()
        repr_str = repr(law)
        assert "None" in repr_str

    def test_extinction_law_call_not_implemented(self):
        """Test that ExtinctionLaw.__call__ raises NotImplementedError."""
        law = ExtinctionLaw()
        with pytest.raises(NotImplementedError):
            law(5000.0 * u.angstrom)

    def test_extinction_law_isvalid(self):
        """Test that ExtinctionLaw.isvalid returns True by default."""
        law = ExtinctionLaw()
        assert law.isvalid() is True


class TestCCM89Init:
    """Test CCM89 initialization."""

    def test_ccm89_init(self):
        """Test CCM89 initialization."""
        ccm = CCM89()
        assert ccm.name == "CCM89"
        assert hasattr(ccm, "long_name")
        assert ccm.long_name == "Cardelli, Clayton, & Mathis (1989)"

    def test_ccm89_repr(self):
        """Test CCM89 repr."""
        ccm = CCM89()
        repr_str = repr(ccm)
        assert "CCM89" in repr_str

    def test_ccm89_is_extinction_law(self):
        """Test that CCM89 is instance of ExtinctionLaw."""
        ccm = CCM89()
        assert isinstance(ccm, ExtinctionLaw)


class TestCCM89Call:
    """Test CCM89 calling interface."""

    def test_ccm89_scalar_input(self):
        """Test CCM89 with scalar wavelength."""
        ccm = CCM89()
        result = ccm(5000.0 * u.angstrom)
        # Scalar inputs are converted to array internally
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] > 0

    def test_ccm89_array_input(self):
        """Test CCM89 with array of wavelengths."""
        ccm = CCM89()
        wavelengths = np.array([3000.0, 5000.0, 10000.0]) * u.angstrom
        result = ccm(wavelengths)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_ccm89_without_units(self):
        """Test CCM89 without explicit units (warns)."""
        ccm = CCM89()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = ccm(5000.0)  # No units
            assert isinstance(result, (float, np.floating, np.ndarray))

    def test_ccm89_with_av(self):
        """Test CCM89 with custom Av."""
        ccm = CCM89()
        result1 = ccm(5000.0 * u.angstrom, Av=1.0)
        result2 = ccm(5000.0 * u.angstrom, Av=2.0)
        np.testing.assert_almost_equal(result2, 2.0 * result1)

    def test_ccm89_with_rv(self):
        """Test CCM89 with custom Rv."""
        ccm = CCM89()
        result1 = ccm(5000.0 * u.angstrom, Rv=2.5)
        result2 = ccm(5000.0 * u.angstrom, Rv=3.1)
        assert result1 != result2

    def test_ccm89_alambda_true(self):
        """Test CCM89 with Alambda=True."""
        ccm = CCM89()
        result = ccm(5000.0 * u.angstrom, Alambda=True)
        assert isinstance(result, np.ndarray)
        assert result[0] > 0

    def test_ccm89_alambda_false(self):
        """Test CCM89 with Alambda=False."""
        ccm = CCM89()
        result_false = ccm(5000.0 * u.angstrom, Alambda=False)
        assert isinstance(result_false, np.ndarray)
        assert result_false[0] > 0

    def test_ccm89_infrared_region(self):
        """Test CCM89 in infrared region (0.3 < x < 1.1)."""
        ccm = CCM89()
        # x = 1e4/lambda, for x=0.5, lambda = 20000 angstrom
        wavelength = 20000.0 * u.angstrom
        result = ccm(wavelength)
        assert result > 0

    def test_ccm89_optical_nir_region(self):
        """Test CCM89 in optical/NIR region (1.1 <= x <= 3.3)."""
        ccm = CCM89()
        # x = 1.5, lambda = 6666.67 angstrom
        wavelength = 6666.67 * u.angstrom
        result = ccm(wavelength)
        assert result > 0

    def test_ccm89_uv_region(self):
        """Test CCM89 in UV region (3.3 < x <= 8.0)."""
        ccm = CCM89()
        # x = 5, lambda = 2000 angstrom
        wavelength = 2000.0 * u.angstrom
        result = ccm(wavelength)
        assert result > 0

    def test_ccm89_far_uv_region(self):
        """Test CCM89 in far UV region (8.0 < x <= 10.0)."""
        ccm = CCM89()
        # x = 9, lambda = 1111.11 angstrom
        wavelength = 1111.11 * u.angstrom
        result = ccm(wavelength)
        assert result > 0

    def test_ccm89_out_of_range_short(self):
        """Test CCM89 with wavelength too short (x > 10.0)."""
        ccm = CCM89()
        # x > 10, lambda < 1000 angstrom
        wavelength = 900.0 * u.angstrom
        result = ccm(wavelength)
        # Should return 0 for out of range
        assert result == 0.0

    def test_ccm89_out_of_range_long(self):
        """Test CCM89 with wavelength too long (x < 0.3)."""
        ccm = CCM89()
        # x < 0.3, lambda > 33333 angstrom
        wavelength = 100000.0 * u.angstrom
        result = ccm(wavelength)
        # Should return 0 for out of range
        assert result == 0.0

    def test_ccm89_multiple_wavelengths_mixed_regions(self):
        """Test CCM89 with multiple wavelengths across different regions."""
        ccm = CCM89()
        wavelengths = (
            np.array(
                [
                    1000.0,  # Far UV
                    2000.0,  # UV
                    5000.0,  # Optical
                    10000.0,  # NIR
                    50000.0,  # Infrared (out of range)
                ]
            )
            * u.angstrom
        )
        results = ccm(wavelengths)
        assert isinstance(results, np.ndarray)
        assert results.shape == (5,)
        # Some may be 0 for out-of-range
        assert results[0] > 0  # 1000 A is in range
        assert results[1] > 0  # 2000 A is in range
        assert results[2] > 0  # 5000 A is in range
        assert results[3] > 0  # 10000 A is in range

    def test_ccm89_unit_conversion(self):
        """Test CCM89 with different wavelength units."""
        ccm = CCM89()
        # 5000 angstrom = 0.5 micron
        result1 = ccm(5000.0 * u.angstrom)
        result2 = ccm(0.5 * u.micron)
        np.testing.assert_almost_equal(result1, result2)


class TestF99Init:
    """Test F99 initialization."""

    def test_f99_init(self):
        """Test F99 initialization."""
        f99 = F99()
        assert f99.name == "F99"
        assert hasattr(f99, "long_name")
        assert f99.long_name == "Fitzpatrick (1999)"

    def test_f99_repr(self):
        """Test F99 repr."""
        f99 = F99()
        repr_str = repr(f99)
        assert "F99" in repr_str

    def test_f99_is_extinction_law(self):
        """Test that F99 is instance of ExtinctionLaw."""
        f99 = F99()
        assert isinstance(f99, ExtinctionLaw)


class TestF99Call:
    """Test F99 calling interface."""

    def test_f99_scalar_input(self):
        """Test F99 with scalar wavelength."""
        f99 = F99()
        result = f99(5000.0 * u.angstrom)
        # Scalar inputs are converted to array internally
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)
        assert result[0] > 0

    def test_f99_array_input(self):
        """Test F99 with array of wavelengths."""
        f99 = F99()
        wavelengths = np.array([3000.0, 5000.0, 10000.0]) * u.angstrom
        result = f99(wavelengths)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.all(result > 0)

    def test_f99_without_units(self):
        """Test F99 without explicit units (warns)."""
        f99 = F99()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = f99(5000.0)  # No units
            assert isinstance(result, (float, np.floating, np.ndarray))

    def test_f99_with_av(self):
        """Test F99 with custom Av."""
        f99 = F99()
        result1 = f99(5000.0 * u.angstrom, Av=1.0)
        result2 = f99(5000.0 * u.angstrom, Av=2.0)
        np.testing.assert_almost_equal(result2, 2.0 * result1)

    def test_f99_with_rv(self):
        """Test F99 with custom Rv."""
        f99 = F99()
        result1 = f99(5000.0 * u.angstrom, Rv=2.5)
        result2 = f99(5000.0 * u.angstrom, Rv=3.1)
        assert result1 != result2

    def test_f99_alambda_true(self):
        """Test F99 with Alambda=True."""
        f99 = F99()
        result = f99(5000.0 * u.angstrom, Alambda=True)
        assert isinstance(result, np.ndarray)
        assert result[0] > 0

    def test_f99_alambda_false(self):
        """Test F99 with Alambda=False."""
        f99 = F99()
        result_false = f99(5000.0 * u.angstrom, Alambda=False)
        assert isinstance(result_false, np.ndarray)
        assert result_false[0] > 0

    def test_f99_uv_region(self):
        """Test F99 in UV region."""
        f99 = F99()
        wavelength = 2000.0 * u.angstrom
        result = f99(wavelength)
        assert result > 0

    def test_f99_optical_nir_region(self):
        """Test F99 in optical/NIR region."""
        f99 = F99()
        wavelength = 5000.0 * u.angstrom
        result = f99(wavelength)
        assert result > 0

    def test_f99_infrared_region(self):
        """Test F99 in infrared region."""
        f99 = F99()
        wavelength = 10000.0 * u.angstrom
        result = f99(wavelength)
        assert result > 0

    def test_f99_multiple_wavelengths_mixed_regions(self):
        """Test F99 with multiple wavelengths across different regions."""
        f99 = F99()
        wavelengths = (
            np.array(
                [
                    1500.0,  # UV
                    3000.0,  # UV
                    5000.0,  # Optical
                    10000.0,  # NIR
                ]
            )
            * u.angstrom
        )
        results = f99(wavelengths)
        assert isinstance(results, np.ndarray)
        assert results.shape == (4,)
        assert np.all(results > 0)

    def test_f99_unit_conversion(self):
        """Test F99 with different wavelength units."""
        f99 = F99()
        result1 = f99(5000.0 * u.angstrom)
        result2 = f99(0.5 * u.micron)
        np.testing.assert_almost_equal(result1, result2)

    def test_f99_different_rv_values(self):
        """Test F99 with various Rv values."""
        f99 = F99()
        wavelength = 5000.0 * u.angstrom
        rv_values = [2.0, 2.5, 3.1, 3.5, 4.0]
        results = [f99(wavelength, Rv=rv) for rv in rv_values]
        # All should be positive
        assert np.all(np.array(results) > 0)


class TestCurveComparison:
    """Test comparisons between different curves."""

    def test_ccm89_vs_f99_optical(self):
        """Test that CCM89 and F99 differ in optical region."""
        ccm = CCM89()
        f99 = F99()
        wavelength = 5000.0 * u.angstrom
        ccm_result = ccm(wavelength)
        f99_result = f99(wavelength)
        # Different curves should give different values
        assert ccm_result != f99_result

    def test_both_curves_positive(self):
        """Test that both curves give positive extinction."""
        ccm = CCM89()
        f99 = F99()
        wavelengths = np.array([2000.0, 5000.0, 10000.0]) * u.angstrom

        ccm_results = ccm(wavelengths)
        f99_results = f99(wavelengths)

        assert np.all(ccm_results > 0)
        assert np.all(f99_results > 0)

    def test_both_curves_scale_with_av(self):
        """Test that both curves scale linearly with Av."""
        ccm = CCM89()
        f99 = F99()
        wavelength = 5000.0 * u.angstrom

        ccm1 = ccm(wavelength, Av=1.0)
        ccm2 = ccm(wavelength, Av=2.0)
        f99_1 = f99(wavelength, Av=1.0)
        f99_2 = f99(wavelength, Av=2.0)

        np.testing.assert_almost_equal(ccm2, 2.0 * ccm1)
        np.testing.assert_almost_equal(f99_2, 2.0 * f99_1)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_ccm89_single_point_array(self):
        """Test CCM89 with single-element array."""
        ccm = CCM89()
        wavelengths = np.array([5000.0]) * u.angstrom
        result = ccm(wavelengths)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_f99_single_point_array(self):
        """Test F99 with single-element array."""
        f99 = F99()
        wavelengths = np.array([5000.0]) * u.angstrom
        result = f99(wavelengths)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_ccm89_large_wavelength_range(self):
        """Test CCM89 with large range of wavelengths."""
        ccm = CCM89()
        wavelengths = np.logspace(3, 5, 100) * u.angstrom
        result = ccm(wavelengths)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert np.all(result >= 0)

    def test_f99_large_wavelength_range(self):
        """Test F99 with large range of wavelengths."""
        f99 = F99()
        wavelengths = np.logspace(3, 5, 100) * u.angstrom
        result = f99(wavelengths)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert np.all(result > 0)

    def test_ccm89_different_Rv_positive_extinction(self):
        """Test CCM89 gives positive extinction for range of Rv values."""
        ccm = CCM89()
        wavelength = 5000.0 * u.angstrom
        for rv in [2.0, 2.5, 3.1, 3.5, 4.0]:
            result = ccm(wavelength, Rv=rv)
            assert result > 0

    def test_f99_different_Rv_positive_extinction(self):
        """Test F99 gives positive extinction for range of Rv values."""
        f99 = F99()
        wavelength = 5000.0 * u.angstrom
        for rv in [2.0, 2.5, 3.1, 3.5, 4.0]:
            result = f99(wavelength, Rv=rv)
            assert result > 0


class TestModuleExports:
    """Test module exports."""

    def test_module_all(self):
        """Test that __all__ includes expected classes."""
        from dustapprox import legacy_extinction

        assert hasattr(legacy_extinction, "__all__")
        assert "ExtinctionLaw" in legacy_extinction.__all__
        assert "CCM89" in legacy_extinction.__all__
        assert "F99" in legacy_extinction.__all__

    def test_import_extinction_law(self):
        """Test importing ExtinctionLaw."""
        from dustapprox.legacy_extinction import ExtinctionLaw

        assert ExtinctionLaw is not None

    def test_import_ccm89(self):
        """Test importing CCM89."""
        from dustapprox.legacy_extinction import CCM89

        assert CCM89 is not None

    def test_import_f99(self):
        """Test importing F99."""
        from dustapprox.legacy_extinction import F99

        assert F99 is not None
