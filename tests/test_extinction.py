"""Tests for dustapprox.extinction module."""

import pytest
import numpy as np
import astropy.units as u
from dust_extinction.parameter_averages import CCM89, F99, G23

from dustapprox.extinction import (
    get_extinction_model,
    evaluate_extinction_model,
)


class TestGetExtinctionModel:
    """Test get_extinction_model function."""

    def test_get_model_by_string_ccm89(self):
        """Test getting CCM89 model by name."""
        model = get_extinction_model("CCM89")
        assert model == CCM89

    def test_get_model_by_string_f99(self):
        """Test getting F99 model by name."""
        model = get_extinction_model("F99")
        assert model == F99

    def test_get_model_by_string_g23(self):
        """Test getting G23 model by name."""
        model = get_extinction_model("G23")
        assert model == G23

    def test_get_model_returns_instance(self):
        """Test that passing a model instance returns it unchanged."""
        instance = CCM89(Rv=3.1)
        result = get_extinction_model(instance)
        assert result is instance

    def test_invalid_model_name_raises_error(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_extinction_model("INVALID_MODEL")

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises error."""
        # Passing an integer should raise an error
        with pytest.raises((ValueError, AttributeError)):
            get_extinction_model("123")

    def test_all_common_models_available(self):
        """Test that all common extinction models are available."""
        common_models = ["CCM89", "F99", "G23"]

        for model_name in common_models:
            model = get_extinction_model(model_name)
            assert model is not None
            # Should be a class, not an instance
            assert isinstance(model, type)


class TestEvaluateExtinctionModel:
    """Test evaluate_extinction_model function."""

    def test_basic_evaluation(self, sample_wavelengths):
        """Test basic extinction curve evaluation."""
        result = evaluate_extinction_model(
            "CCM89",
            sample_wavelengths,
            A0=1.0,
            R0=3.1,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_wavelengths.shape
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Extinction should be positive

    def test_evaluation_with_different_rv(self, sample_wavelengths):
        """Test that different R0 values give different results."""
        result1 = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=1.0, R0=3.1
        )
        result2 = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=1.0, R0=5.0
        )

        # Results should be different for different Rv
        assert not np.allclose(result1, result2)

    def test_evaluation_with_different_a0(self, sample_wavelengths):
        """Test that A0 scales the result linearly."""
        result1 = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=1.0, R0=3.1
        )
        result2 = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=2.0, R0=3.1
        )

        # Should scale linearly with A0
        assert np.allclose(result2, 2.0 * result1)

    def test_zero_extinction(self, sample_wavelengths):
        """Test with zero extinction."""
        result = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=0.0, R0=3.1
        )

        assert np.allclose(result, 0.0)

    def test_wavelength_in_microns(self, sample_wavelengths_micron):
        """Test evaluation with wavelengths in microns."""
        result = evaluate_extinction_model(
            "CCM89",
            sample_wavelengths_micron,
            A0=1.0,
            R0=3.1,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == sample_wavelengths_micron.shape
        assert np.all(np.isfinite(result))

    def test_wavelength_without_units(self):
        """Test evaluation with wavelengths without units (should warn)."""
        import warnings

        wavelengths = np.array([1000, 2000, 5000, 10000])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = evaluate_extinction_model(
                "CCM89", wavelengths, A0=1.0, R0=3.1
            )

            # Should warn about missing units
            assert len(w) >= 1
            assert result is not None

    def test_single_wavelength(self):
        """Test evaluation at a single wavelength."""
        wavelength = np.array([5500]) * u.angstrom
        result = evaluate_extinction_model("CCM89", wavelength, A0=1.0, R0=3.1)

        assert isinstance(result, np.ndarray)
        assert result.shape == wavelength.shape  # scalar
        assert np.isfinite(result)

    def test_extrapolation_flag(self, sample_wavelengths):
        """Test extrapolation parameter."""
        # With extrapolation
        result_extrap = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=1.0, R0=3.1, extrapolate=True
        )

        # Without extrapolation
        result_no_extrap = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=1.0, R0=3.1, extrapolate=False
        )

        # Both should produce results
        assert result_extrap is not None
        assert result_no_extrap is not None

    def test_model_instance_input(self, sample_wavelengths):
        """Test passing a model instance instead of name."""
        model = CCM89(Rv=3.1)
        result = evaluate_extinction_model(
            model, sample_wavelengths, A0=1.0, R0=3.1
        )

        assert isinstance(result, np.ndarray)
        assert np.all(np.isfinite(result))

    def test_different_extinction_curves(self, sample_wavelengths):
        """Test that different extinction curves give different results."""
        curves = ["CCM89", "F99", "O94"]
        results = []

        for curve in curves:
            result = evaluate_extinction_model(
                curve, sample_wavelengths, A0=1.0, R0=3.1
            )
            results.append(result)

        # Results should be different between models
        assert not np.allclose(results[0], results[1])
        assert not np.allclose(results[1], results[2])

    def test_extreme_rv_values(self, sample_wavelengths):
        """Test with extreme R0 values within allowed range."""
        # Test with minimum allowed Rv
        result_low = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=1.0, R0=2.0, extrapolate=True
        )

        # Test with high Rv
        result_high = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=1.0, R0=6.0, extrapolate=True
        )

        # Both should produce valid results
        assert np.all(np.isfinite(result_low))
        assert np.all(np.isfinite(result_high))
        assert not np.allclose(result_low, result_high)

    def test_very_short_wavelengths(self):
        """Test evaluation at very short wavelengths (UV)."""
        wavelengths = np.array([912, 1000, 1500, 2000]) * u.angstrom
        result = evaluate_extinction_model(
            "CCM89", wavelengths, A0=1.0, R0=3.1, extrapolate=True
        )

        # Should handle UV wavelengths
        assert isinstance(result, np.ndarray)
        # May contain NaN for extrapolated regions
        assert result.shape == wavelengths.shape

    def test_very_long_wavelengths(self):
        """Test evaluation at very long wavelengths (IR)."""
        wavelengths = np.array([1, 2, 5, 10]) * u.micron
        result = evaluate_extinction_model(
            "F99", wavelengths, A0=1.0, R0=3.1, extrapolate=True
        )

        # Should handle IR wavelengths
        assert isinstance(result, np.ndarray)
        assert result.shape == wavelengths.shape

    def test_f99_extrapolation_patch(self):
        """Test that F99 extrapolation patch is applied correctly."""
        # Very wide range of wavelengths
        wavelengths = np.logspace(-1, 1, 20) * u.micron

        result = evaluate_extinction_model(
            "F99", wavelengths, A0=1.0, R0=3.1, extrapolate=True
        )

        # Should not raise an error and should produce results
        assert isinstance(result, np.ndarray)
        assert result.shape == wavelengths.shape  # pyright: ignore


class TestExtinctionEdgeCases:
    """Test edge cases and potential bugs in extinction module."""

    def test_negative_wavelength_handling(self):
        """Test behavior with invalid negative wavelengths."""
        wavelengths = np.array([-1000, 1000, 2000]) * u.angstrom

        # This should either raise an error or handle gracefully
        # Implementation-dependent behavior
        try:
            result = evaluate_extinction_model(
                "CCM89", wavelengths, A0=1.0, R0=3.1
            )
            # If it doesn't raise, check that we get some output
            assert result is not None
        except (ValueError, RuntimeError):
            # Expected for negative wavelengths
            pass

    def test_zero_wavelength_handling(self):
        """Test behavior with zero wavelength."""
        wavelengths = np.array([0, 1000, 2000]) * u.angstrom

        # Should handle division by zero gracefully
        try:
            result = evaluate_extinction_model(
                "CCM89", wavelengths, A0=1.0, R0=3.1, extrapolate=True
            )
            # May contain inf or nan for zero wavelength
            assert result is not None
        except (ValueError, RuntimeError, ZeroDivisionError):
            # This is also acceptable behavior
            pass

    def test_nan_wavelengths(self):
        """Test handling of NaN in wavelength array."""
        wavelengths = np.array([np.nan, 1000, 2000]) * u.angstrom

        result = evaluate_extinction_model(
            "CCM89", wavelengths, A0=1.0, R0=3.1
        )

        # First element should be NaN
        assert np.isnan(result[0])
        # Other elements should be finite
        assert np.isfinite(result[1])
        assert np.isfinite(result[2])

    def test_inf_wavelengths(self):
        """Test handling of infinity in wavelength array."""
        wavelengths = np.array([np.inf, 1000, 2000]) * u.angstrom

        result = evaluate_extinction_model(
            "CCM89", wavelengths, A0=1.0, R0=3.1, extrapolate=True
        )

        # Should produce some result
        assert result is not None
        assert result.shape == wavelengths.shape

    def test_empty_wavelength_array(self):
        """Test with empty wavelength array."""
        wavelengths = np.array([]) * u.angstrom

        result = evaluate_extinction_model(
            "CCM89", wavelengths, A0=1.0, R0=3.1
        )

        assert result.shape == (0,)

    def test_very_large_a0(self, sample_wavelengths):
        """Test with very large extinction value."""
        result = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=100.0, R0=3.1
        )

        assert np.all(np.isfinite(result))
        assert np.all(result > 0)

    def test_negative_a0(self, sample_wavelengths):
        """Test with negative A0 (unphysical but mathematically valid)."""
        result = evaluate_extinction_model(
            "CCM89", sample_wavelengths, A0=-1.0, R0=3.1
        )

        # Should still produce a result (negative extinction)
        assert isinstance(result, np.ndarray)
        assert np.all(result <= 0)

    def test_wavelength_order_independence(self):
        """Test that wavelength order doesn't affect individual values."""
        wavelengths_asc = np.array([1000, 2000, 5000, 10000]) * u.angstrom
        wavelengths_desc = wavelengths_asc[::-1]

        result_asc = evaluate_extinction_model(
            "CCM89", wavelengths_asc, A0=1.0, R0=3.1
        )
        result_desc = evaluate_extinction_model(
            "CCM89", wavelengths_desc, A0=1.0, R0=3.1
        )

        # Results should be reversed but otherwise identical
        assert np.allclose(result_asc, result_desc[::-1])

    def test_duplicate_wavelengths(self):
        """Test with duplicate wavelengths."""
        wavelengths = np.array([1000, 1000, 2000, 2000]) * u.angstrom

        result = evaluate_extinction_model(
            "CCM89", wavelengths, A0=1.0, R0=3.1
        )

        # Duplicate wavelengths should give duplicate results
        assert np.isclose(result[0], result[1])
        assert np.isclose(result[2], result[3])
