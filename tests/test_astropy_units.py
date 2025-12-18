"""Tests for dustapprox.astropy_units module."""

import pytest
import numpy as np
import warnings
from astropy.units import Quantity
import astropy.units as u

from dustapprox.astropy_units import (
    Unit,
    has_unit,
    val_in_unit,
)


class TestCustomUnits:
    """Test custom unit definitions."""

    def test_flam_unit_defined(self):
        """Test that flam unit is properly defined."""
        unit = Unit("flam")
        assert unit is not None
        # Check equivalency
        expected = u.erg / u.s / u.angstrom / u.cm**2
        assert unit.is_equivalent(expected)

    def test_fnu_unit_defined(self):
        """Test that fnu unit is properly defined."""
        unit = Unit("fnu")
        assert unit is not None
        expected = u.erg / u.s / u.Hz / u.cm**2
        assert unit.is_equivalent(expected)

    def test_photflam_unit_defined(self):
        """Test that photflam unit is properly defined."""
        unit = Unit("photflam")
        assert unit is not None
        expected = u.photon / u.s / u.angstrom / u.cm**2
        assert unit.is_equivalent(expected)

    def test_photfnu_unit_defined(self):
        """Test that photfnu unit is properly defined."""
        unit = Unit("photfnu")
        assert unit is not None
        expected = u.photon / u.s / u.Hz / u.cm**2
        assert unit.is_equivalent(expected)

    def test_angstroms_alias(self):
        """Test that angstroms is an alias for angstrom."""
        unit = Unit("angstroms")
        assert unit.is_equivalent(u.angstrom)

    def test_lsun_alias(self):
        """Test that lsun is an alias for Lsun."""
        unit = Unit("lsun")
        assert unit.is_equivalent(u.Lsun)

    def test_ergs_alias(self):
        """Test that ergs is an alias for erg."""
        unit = Unit("ergs")
        assert unit.is_equivalent(u.erg)


class TestHasUnit:
    """Test has_unit function."""

    def test_quantity_has_unit(self):
        """Test that Quantity objects are detected as having units."""
        val = 5.0 * u.meter
        assert has_unit(val)

    def test_quantity_with_units_attr(self):
        """Test objects with 'units' attribute."""

        class MockWithUnits:
            units = u.meter

        obj = MockWithUnits()
        assert has_unit(obj)

    def test_quantity_with_unit_attr(self):
        """Test objects with 'unit' attribute."""

        class MockWithUnit:
            unit = u.meter

        obj = MockWithUnit()
        assert has_unit(obj)

    def test_plain_number_no_unit(self):
        """Test that plain numbers don't have units."""
        assert not has_unit(5.0)
        assert not has_unit(42)

    def test_numpy_array_no_unit(self):
        """Test that numpy arrays without units are detected."""
        arr = np.array([1, 2, 3])
        assert not has_unit(arr)

    def test_list_no_unit(self):
        """Test that lists don't have units."""
        assert not has_unit([1, 2, 3])


class TestValInUnit:
    """Test val_in_unit function."""

    def test_value_with_correct_unit(self):
        """Test conversion when value already has the correct unit."""
        val = 5.0 * u.degree
        result = val_in_unit("test_var", val, "degree")
        assert isinstance(result, Quantity)
        assert result.value == 5.0
        assert result.unit == u.degree

    def test_value_with_convertible_unit(self):
        """Test conversion when value has convertible unit."""
        val = 1.0 * u.meter
        result = val_in_unit("test_var", val, "cm")
        assert isinstance(result, Quantity)
        assert result.value == 100.0
        assert result.unit == u.cm

    def test_value_without_unit_raises_warning(self):
        """Test that unitless values raise a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = val_in_unit("my_var", 5.0, "meter")

            # Check warning was raised
            assert len(w) == 1
            assert "does not have explicit units" in str(w[0].message)
            assert "my_var" in str(w[0].message)
            assert "meter" in str(w[0].message)

            # Check result
            assert isinstance(result, Quantity)
            assert result.value == 5.0
            assert result.unit == u.meter

    def test_numpy_array_without_unit(self):
        """Test array values without units."""
        arr = np.array([1, 2, 3])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = val_in_unit("test_array", arr, "meter")

            assert len(w) == 1
            assert isinstance(result, Quantity)
            assert np.array_equal(result.value, arr)
            assert result.unit == u.meter

    def test_unit_conversion_preserves_value(self):
        """Test that unit conversion is done correctly."""
        val = 1.0 * u.km
        result = val_in_unit("distance", val, "meter")
        assert result.value == 1000.0
        assert result.unit == u.meter

    def test_incompatible_units_raises_error(self):
        """Test that incompatible unit conversion raises error."""
        val = 5.0 * u.meter
        with pytest.raises(u.UnitConversionError):
            val_in_unit("test", val, "second")

    def test_custom_unit_conversion(self):
        """Test conversion with custom units."""
        val = 1.0 * Unit("angstroms")
        result = val_in_unit("wavelength", val, "angstrom")
        assert result.value == 1.0
        assert result.unit is not None
        assert result.unit.is_equivalent(u.angstrom)

    def test_flam_unit_usage(self):
        """Test usage of custom flam unit."""
        val = 1e-15 * Unit("flam")
        result = val_in_unit("flux", val, "flam")
        assert isinstance(result, Quantity)
        assert result.value == 1e-15


class TestEdgeCases:
    """Test edge cases and potential bugs."""

    def test_zero_value_with_unit(self):
        """Test that zero values with units work correctly."""
        val = 0.0 * u.meter
        result = val_in_unit("zero", val, "meter")
        assert result.value == 0.0
        assert result.unit == u.meter

    def test_negative_value_with_unit(self):
        """Test that negative values with units work correctly."""
        val = -5.0 * u.degree
        result = val_in_unit("angle", val, "degree")
        assert result.value == -5.0
        assert result.unit == u.degree

    def test_very_large_number(self):
        """Test with very large numbers."""
        val = 1e100 * u.meter
        result = val_in_unit("huge", val, "meter")
        assert result.value == 1e100
        assert result.unit == u.meter

    def test_very_small_number(self):
        """Test with very small numbers."""
        val = 1e-100 * u.meter
        result = val_in_unit("tiny", val, "meter")
        assert result.value == 1e-100
        assert result.unit == u.meter

    def test_nan_value_with_unit(self):
        """Test NaN values with units."""
        val = np.nan * u.meter
        result = val_in_unit("nan_val", val, "meter")
        assert np.isnan(result.value)
        assert result.unit == u.meter

    def test_inf_value_with_unit(self):
        """Test infinity values with units."""
        val = np.inf * u.meter
        result = val_in_unit("inf_val", val, "meter")
        assert np.isinf(result.value)
        assert result.unit == u.meter

    def test_unit_string_variations(self):
        """Test different string representations of units."""
        val = 1.0 * u.m

        # Different valid string formats
        result1 = val_in_unit("test", val, "m")
        result2 = val_in_unit("test", val, "meter")
        assert result1.unit is not None
        assert result2.unit is not None
        assert result1.unit.is_equivalent(u.meter)
        assert result2.unit.is_equivalent(u.meter)
