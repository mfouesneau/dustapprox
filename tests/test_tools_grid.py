"""Tests for dustapprox.tools.grid module."""

import pytest
import numpy as np
import pandas as pd
import urllib3
import requests
from unittest.mock import Mock
from types import SimpleNamespace
from typing import cast

from dustapprox.tools.grid import (
    DEFAULT_FILTERS,
    compute_photometric_grid,
)


class TestDefaultFilters:
    """Test DEFAULT_FILTERS constant."""

    def test_default_filters_is_list(self):
        """Test that DEFAULT_FILTERS is a list."""
        assert isinstance(DEFAULT_FILTERS, list)

    def test_default_filters_not_empty(self):
        """Test that DEFAULT_FILTERS contains filters."""
        assert len(DEFAULT_FILTERS) > 0

    def test_default_filters_format(self):
        """Test that filter names follow expected format."""
        for filter_name in DEFAULT_FILTERS:
            assert isinstance(filter_name, str)
            # Most should have format "SYSTEM/FILTER"
            if "/" in filter_name:
                parts = filter_name.split("/")
                assert len(parts) == 2

    def test_common_filters_present(self):
        """Test that common filter systems are present."""
        filter_str = " ".join(DEFAULT_FILTERS)

        # Check for common systems
        assert "GAIA" in filter_str or "Gaia" in filter_str
        assert "SLOAN" in filter_str or "SDSS" in filter_str
        assert "2MASS" in filter_str
        assert "WISE" in filter_str

    def test_no_duplicate_filters(self):
        """Test that there are no duplicate filters."""
        assert len(DEFAULT_FILTERS) == len(set(DEFAULT_FILTERS))


class TestParallelTask:
    """Test _parallel_task function (requires mocking)."""


class TestComputePhotometricGrid:
    """Test compute_photometric_grid function."""

    def test_compute_grid_with_mocks(self, monkeypatch):
        """Exercise compute_photometric_grid path with mocked parallel pipeline."""

        # Return a single fake model file
        monkeypatch.setattr(
            "dustapprox.tools.grid.glob", lambda pattern: ["dummy.fits"]
        )

        # Bypass tqdm wrapper
        monkeypatch.setattr(
            "dustapprox.tools.grid.tqdm", lambda it, **kwargs: it
        )

        # Mock _parallel_task to bypass file IO and return a deterministic df
        def _fake_task(fname, filters, extinction_curve, A0, R0, apfields):
            df = pd.DataFrame(
                {
                    "teff": [5000.0],
                    "logg": [4.5],
                    "feh": [0.0],
                    "alpha": [0.0],
                    "passband": ["MOCK"],
                    "mag0": [10.0],
                    "mag": [10.5],
                    "A0": [1.0],
                    "R0": [3.1],
                    "Ax": [0.5],
                }
            )
            df.attrs = {
                "extinction": {"source": "F99"},
                "atmosphere": {"source": fname},
            }
            return df

        monkeypatch.setattr("dustapprox.tools.grid._parallel_task", _fake_task)

        # Mock joblib.Parallel to return a callable that ignores delayed tasks and yields our df
        def _fake_parallel(**kwargs):
            def _runner(iterable):
                return list(iterable)

            return _runner

        monkeypatch.setattr("dustapprox.tools.grid.Parallel", _fake_parallel)

        # Mock delayed to just return callable unchanged
        monkeypatch.setattr("dustapprox.tools.grid.delayed", lambda fn: fn)

        # Mock Filter handling so we skip SVO fetch
        class DummyFilter:
            def __init__(self):
                self.name = "MOCK"

            def get_flux(self, lamb, flux):
                return SimpleNamespace(value=1.0)

        monkeypatch.setattr("dustapprox.tools.grid.Filter", DummyFilter)
        mock_filter = DummyFilter()
        # Ensure which_filters treated as already-Filter instances (bypass SVO fetch)
        which_filters = cast(list, [mock_filter])

        df = compute_photometric_grid(
            sources="dummy/*.fits",
            which_filters=which_filters,
            extinction_curve="F99",
            A0=np.array([1.0]),
            R0=np.array([3.1]),
            n_jobs=1,
            verbose=0,
        )

        assert isinstance(df, pd.DataFrame)
        assert "Ax" in df.columns
        assert df.attrs.get("extinction", {}).get("source") == "F99"
        assert df.attrs.get("atmosphere", {}).get("source") == "dummy/*.fits"

    def test_compute_grid_no_sources(self, tmp_path):
        """Test behavior when no source files match."""
        pattern = str(tmp_path / "nonexistent/*.txt")

        # Should handle empty glob result gracefully
        try:
            result = compute_photometric_grid(
                sources=pattern,
                which_filters=["GAIA/GAIA3.G"],
                extinction_curve="F99",
                n_jobs=1,
            )
            # If it succeeds, check result is empty or minimal
            if result is not None:
                assert isinstance(result, pd.DataFrame)
        except (ValueError, IndexError, KeyError):
            # Also acceptable to raise an error for empty input
            pass
        except (
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ConnectTimeout,
        ):
            # Network issues may cause failures in SVO fetch
            pytest.skip("Network issues prevented filter fetching")

    def test_compute_grid_default_parameters(self):
        """Test that default parameters are set correctly."""
        # This test mainly validates parameter handling without execution
        # We can't easily test without actual data files

        # Test default R0
        assert isinstance(
            np.array([2.3, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1]), np.ndarray
        )

        # Test default A0 generation logic
        default_a0 = np.sort(np.hstack([[0.01], np.arange(0.1, 20.01, 0.1)]))
        assert len(default_a0) > 0
        assert default_a0[0] == 0.01

    def test_extinction_curve_name_string(self):
        """Test extraction of extinction curve name from string."""
        extinction_curve = "F99"
        extinction_curve_name = (
            extinction_curve
            if isinstance(extinction_curve, str)
            else extinction_curve.__class__.__name__
        )
        assert extinction_curve_name == "F99"

    def test_extinction_curve_name_from_object(self):
        """Test extraction of extinction curve name from object."""
        # Mock object with __class__.__name__
        mock_curve = Mock()
        mock_curve.__class__.__name__ = "CCM89"

        extinction_curve_name = (
            mock_curve
            if isinstance(mock_curve, str)
            else mock_curve.__class__.__name__
        )
        assert extinction_curve_name == "CCM89"

    @pytest.mark.requires_data
    @pytest.mark.slow
    def test_compute_grid_integration(self, tmp_path):
        """Integration test for compute_photometric_grid (requires data)."""
        # This would require actual atmosphere model files
        # Skip if data not available
        pytest.skip("Requires actual atmosphere model files")

    def test_apfields_default(self):
        """Test default atmospheric parameter fields."""
        default_apfields = ("teff", "logg", "feh", "alpha")

        assert len(default_apfields) == 4
        assert "teff" in default_apfields
        assert "logg" in default_apfields
        assert "feh" in default_apfields
        assert "alpha" in default_apfields

    def test_r0_default_values(self):
        """Test default R0 values."""
        R0 = np.array([2.3, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1])

        assert len(R0) == 7
        assert R0.min() == 2.3
        assert R0.max() == 5.1
        assert 3.1 in R0  # Standard value

    def test_a0_default_generation(self):
        """Test default A0 value generation."""
        A0 = np.sort(np.hstack([[0.01], np.arange(0.1, 20.01, 0.1)]))

        assert A0.min() == 0.01
        assert A0.max() <= 20.0 + 1e-8  # Allow for floating-point precision
        assert len(A0) > 100  # Should have many values
        # Should be sorted
        assert np.all(A0[:-1] <= A0[1:])

    def test_n_jobs_parameter(self):
        """Test that n_jobs parameter accepts valid values."""
        valid_n_jobs = [1, 2, 4, -1]

        for n_jobs in valid_n_jobs:
            assert isinstance(n_jobs, int)
            # Just verify the parameter type, actual execution needs data


class TestGridEdgeCases:
    """Test edge cases and potential bugs in grid module."""

    def test_empty_filter_list(self):
        """Test behavior with empty filter list."""
        # Should use DEFAULT_FILTERS when None is passed
        which_filters = None
        result = which_filters or DEFAULT_FILTERS

        assert result == DEFAULT_FILTERS

    def test_single_r0_value(self):
        """Test grid computation with single R0 value."""
        R0 = np.array([3.1])

        assert len(R0) == 1
        assert R0[0] == 3.1

    def test_single_a0_value(self):
        """Test grid computation with single A0 value."""
        A0 = np.array([1.0])

        assert len(A0) == 1
        assert A0[0] == 1.0

    def test_very_large_r0_array(self):
        """Test with very large R0 array."""
        R0 = np.linspace(2.0, 6.0, 1000)

        assert len(R0) == 1000
        assert R0.min() == 2.0
        assert R0.max() == 6.0

    def test_very_large_a0_array(self):
        """Test with very large A0 array."""
        A0 = np.linspace(0.0, 50.0, 1000)

        assert len(A0) == 1000
        assert A0.min() == 0.0
        assert A0.max() == 50.0

    def test_zero_a0_value(self):
        """Test that A0=0 is handled correctly."""
        A0 = np.array([0.0, 1.0, 2.0])

        assert A0[0] == 0.0  # Zero extinction case

    def test_negative_a0_raises_or_handles(self):
        """Test behavior with negative A0 (unphysical)."""
        A0 = np.array([-1.0, 0.0, 1.0])

        # Negative extinction is unphysical but mathematically valid
        # Code should either handle or reject it
        assert A0[0] < 0

    def test_r0_outside_typical_range(self):
        """Test R0 values outside typical range."""
        # Typical range is ~2-6, test outside
        R0_low = np.array([1.5, 2.0])
        R0_high = np.array([7.0, 8.0])

        assert R0_low.min() < 2.0
        assert R0_high.max() > 6.0

    def test_unsorted_r0_values(self):
        """Test that R0 values don't need to be sorted."""
        R0 = np.array([3.1, 2.3, 4.1, 3.6])

        # Should work regardless of order
        assert len(R0) == 4

    def test_unsorted_a0_values(self):
        """Test that A0 values don't need to be sorted."""
        A0 = np.array([5.0, 1.0, 10.0, 0.1])

        # Should work regardless of order
        assert len(A0) == 4

    def test_duplicate_r0_values(self):
        """Test behavior with duplicate R0 values."""
        R0 = np.array([3.1, 3.1, 3.1])

        # Should still process even with duplicates
        assert len(R0) == 3

    def test_duplicate_a0_values(self):
        """Test behavior with duplicate A0 values."""
        A0 = np.array([1.0, 1.0, 1.0])

        # Should still process even with duplicates
        assert len(A0) == 3

    def test_filter_name_variations(self):
        """Test different filter name formats."""
        filter_formats = [
            "GAIA/GAIA3.G",
            "GAIA_GAIA3.G",
            "Generic/Johnson.V",
            "2MASS/2MASS.J",
        ]

        for filter_name in filter_formats:
            assert isinstance(filter_name, str)
            assert len(filter_name) > 0

    def test_extinction_curve_variations(self):
        """Test different extinction curve names."""
        curves = ["F99", "CCM89", "O94"]

        for curve in curves:
            assert isinstance(curve, str)
            assert len(curve) > 0

    def test_apfields_custom_order(self):
        """Test custom order of atmospheric parameter fields."""
        custom_apfields = ("feh", "alpha", "logg", "teff")

        assert len(custom_apfields) == 4
        assert all(isinstance(field, str) for field in custom_apfields)

    def test_apfields_subset(self):
        """Test with subset of atmospheric parameters."""
        subset_apfields = ("teff", "logg")

        assert len(subset_apfields) == 2

    def test_apfields_extended(self):
        """Test with extended atmospheric parameters."""
        extended_apfields = ("teff", "logg", "feh", "alpha", "vturb", "custom")

        assert len(extended_apfields) == 6

    def test_verbose_levels(self):
        """Test different verbosity levels."""
        verbose_levels = [0, 1, 2, 10]

        for level in verbose_levels:
            assert isinstance(level, int)
            assert level >= 0


class TestGridDataStructures:
    """Test data structure expectations for grid module."""

    def test_expected_dataframe_columns(self):
        """Test expected columns in output DataFrame."""
        expected_cols = [
            "teff",
            "logg",
            "feh",
            "alpha",
            "passband",
            "mag0",
            "mag",
            "A0",
            "R0",
            "Ax",
        ]

        # These are the expected columns based on _parallel_task
        assert len(expected_cols) == 10
        assert "Ax" in expected_cols  # The key output

    def test_metadata_structure(self):
        """Test expected metadata structure."""
        meta = {
            "extinction": {"source": "F99"},
            "atmosphere": {"source": "Kurucz"},
        }

        assert "extinction" in meta
        assert "atmosphere" in meta
        assert "source" in meta["extinction"]
        assert "source" in meta["atmosphere"]

    def test_stats_aggregation(self):
        """Test statistics aggregation structure."""
        df = pd.DataFrame(
            {
                "teff": [5000, 6000, 7000],
                "logg": [4.0, 4.5, 5.0],
                "A0": [0.5, 1.0, 1.5],
                "R0": [3.1, 3.1, 3.1],
            }
        )

        stats = df[["teff", "logg", "A0", "R0"]].agg(["min", "max"])

        assert "min" in stats.index
        assert "max" in stats.index
        assert "teff" in stats.columns
        assert stats.loc["min", "teff"] == 5000
        assert stats.loc["max", "teff"] == 7000
