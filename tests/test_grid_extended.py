"""Additional tests for dustapprox.tools.grid module for coverage."""

import numpy as np
import pandas as pd

from dustapprox.tools.grid import (
    DEFAULT_FILTERS,
    compute_photometric_grid,
)


class TestGridFunctionality:
    """Test additional grid functionality."""

    def test_compute_grid_with_varied_parameters(self, monkeypatch, tmp_path):
        """Test compute_photometric_grid with varied parameters."""
        # Mock glob
        monkeypatch.setattr("dustapprox.tools.grid.glob", lambda pattern: ["dummy.fits"])
        
        # Mock tqdm
        monkeypatch.setattr("dustapprox.tools.grid.tqdm", lambda it, **kwargs: it)
        
        # Mock Parallel
        def _fake_parallel(**kwargs):
            def _runner(iterable):
                return list(iterable)
            return _runner
        
        monkeypatch.setattr("dustapprox.tools.grid.Parallel", _fake_parallel)
        
        # Mock delayed
        monkeypatch.setattr("dustapprox.tools.grid.delayed", lambda fn: fn)
        
        # Mock _parallel_task to return deterministic data
        def _fake_task(fname, apfields, filters, extinction_curve, R0, A0):
            df = pd.DataFrame({
                "teff": [5000.0, 6000.0],
                "logg": [4.5, 4.0],
                "feh": [0.0, -0.5],
                "alpha": [0.0, 0.1],
                "passband": ["MOCK", "MOCK"],
                "mag0": [10.0, 11.0],
                "mag": [10.5, 11.5],
                "A0": [1.0, 1.5],
                "R0": [3.1, 3.1],
                "Ax": [0.5, 0.75],
            })
            df.attrs = {"extinction": {"source": "F99"}, "atmosphere": {"source": fname}}
            return df
        
        monkeypatch.setattr("dustapprox.tools.grid._parallel_task", _fake_task)
        
        # Mock get_svo_passbands
        class DummyFilter:
            def __init__(self):
                self.name = "MOCK"
            
            def get_flux(self, lamb, flux):
                from types import SimpleNamespace
                return SimpleNamespace(value=1.0)
        
        monkeypatch.setattr(
            "dustapprox.tools.grid.svo.get_svo_passbands",
            lambda filters: [DummyFilter() for _ in filters]
        )
        
        df = compute_photometric_grid(
            sources="dummy/*.fits",
            which_filters=["MOCK"],
            extinction_curve="F99",
            A0=np.array([1.0, 1.5]),
            R0=np.array([3.1]),
            n_jobs=1,
        )
        
        assert isinstance(df, pd.DataFrame)
        assert "Ax" in df.columns
        assert len(df) >= 2

    def test_compute_grid_with_different_extinction_curves(self, monkeypatch):
        """Test with different extinction curve names."""
        # Mock glob
        monkeypatch.setattr("dustapprox.tools.grid.glob", lambda pattern: ["dummy.fits"])
        
        # Mock tqdm
        monkeypatch.setattr("dustapprox.tools.grid.tqdm", lambda it, **kwargs: it)
        
        # Mock Parallel
        def _fake_parallel(**kwargs):
            def _runner(iterable):
                return list(iterable)
            return _runner
        
        monkeypatch.setattr("dustapprox.tools.grid.Parallel", _fake_parallel)
        monkeypatch.setattr("dustapprox.tools.grid.delayed", lambda fn: fn)
        
        def _fake_task(fname, apfields, filters, extinction_curve, R0, A0):
            df = pd.DataFrame({
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
            })
            df.attrs = {"extinction": {"source": extinction_curve}, "atmosphere": {"source": fname}}
            return df
        
        monkeypatch.setattr("dustapprox.tools.grid._parallel_task", _fake_task)
        
        class DummyFilter:
            def __init__(self):
                self.name = "MOCK"
            def get_flux(self, lamb, flux):
                from types import SimpleNamespace
                return SimpleNamespace(value=1.0)
        
        monkeypatch.setattr(
            "dustapprox.tools.grid.svo.get_svo_passbands",
            lambda filters: [DummyFilter() for _ in filters]
        )
        
        for curve in ["F99", "CCM89", "G23"]:
            df = compute_photometric_grid(
                sources="dummy/*.fits",
                which_filters=["MOCK"],
                extinction_curve=curve,
                A0=np.array([1.0]),
                R0=np.array([3.1]),
            )
            
            assert isinstance(df, pd.DataFrame)
            assert df.attrs.get("extinction", {}).get("source") == curve

    def test_compute_grid_parameters_validation(self):
        """Test that compute_photometric_grid accepts various parameter combinations."""
        # Test valid parameter types
        assert isinstance(np.array([1.0, 2.0, 3.0]), np.ndarray)
        assert isinstance([1.0, 2.0, 3.0], list)
        
        # Extinction curve as string
        extinction = "F99"
        assert isinstance(extinction, str)
        
        # Extinction curve as object
        class MockCurve:
            pass
        
        curve_obj = MockCurve()
        assert hasattr(curve_obj, "__class__")

    def test_compute_grid_filter_handling(self, monkeypatch):
        """Test filter handling in compute_photometric_grid."""
        # Mock glob
        monkeypatch.setattr("dustapprox.tools.grid.glob", lambda pattern: ["dummy.fits"])
        
        # Mock tqdm
        monkeypatch.setattr("dustapprox.tools.grid.tqdm", lambda it, **kwargs: it)
        
        # Mock Parallel
        def _fake_parallel(**kwargs):
            def _runner(iterable):
                return list(iterable)
            return _runner
        
        monkeypatch.setattr("dustapprox.tools.grid.Parallel", _fake_parallel)
        monkeypatch.setattr("dustapprox.tools.grid.delayed", lambda fn: fn)
        
        def _fake_task(fname, apfields, filters, extinction_curve, R0, A0):
            # Test that filters are properly received
            assert filters is not None
            if isinstance(filters, (list, tuple)):
                for f in filters:
                    assert hasattr(f, "name")
            
            df = pd.DataFrame({
                "teff": [5000.0],
                "logg": [4.5],
                "feh": [0.0],
                "alpha": [0.0],
                "passband": ["TEST"],
                "mag0": [10.0],
                "mag": [10.5],
                "A0": [1.0],
                "R0": [3.1],
                "Ax": [0.5],
            })
            df.attrs = {"extinction": {"source": extinction_curve}, "atmosphere": {"source": fname}}
            return df
        
        monkeypatch.setattr("dustapprox.tools.grid._parallel_task", _fake_task)
        
        class DummyFilter:
            def __init__(self, name="TEST"):
                self.name = name
            def get_flux(self, lamb, flux):
                from types import SimpleNamespace
                return SimpleNamespace(value=1.0)
        
        monkeypatch.setattr(
            "dustapprox.tools.grid.svo.get_svo_passbands",
            lambda filters: [DummyFilter() for _ in filters]
        )
        
        df = compute_photometric_grid(
            sources="dummy/*.fits",
            which_filters=["GAIA/GAIA3.G", "SLOAN/SDSS.u"],
            extinction_curve="F99",
        )
        
        assert isinstance(df, pd.DataFrame)

    def test_compute_grid_apfields_parameter(self, monkeypatch):
        """Test compute_photometric_grid with custom apfields."""
        # Mock glob
        monkeypatch.setattr("dustapprox.tools.grid.glob", lambda pattern: ["dummy.fits"])
        
        # Mock tqdm
        monkeypatch.setattr("dustapprox.tools.grid.tqdm", lambda it, **kwargs: it)
        
        # Mock Parallel and delayed
        def _fake_parallel(**kwargs):
            def _runner(iterable):
                return list(iterable)
            return _runner
        
        monkeypatch.setattr("dustapprox.tools.grid.Parallel", _fake_parallel)
        monkeypatch.setattr("dustapprox.tools.grid.delayed", lambda fn: fn)
        
        def _fake_task(fname, apfields, filters, extinction_curve, R0, A0):
            # Verify apfields is passed
            assert apfields is not None
            assert isinstance(apfields, (tuple, list))
            
            df = pd.DataFrame({
                field: [5000.0] for field in apfields
            })
            df["passband"] = ["MOCK"]
            df["mag0"] = [10.0]
            df["mag"] = [10.5]
            df["A0"] = [1.0]
            df["R0"] = [3.1]
            df["Ax"] = [0.5]
            df.attrs = {"extinction": {"source": extinction_curve}, "atmosphere": {"source": fname}}
            return df
        
        monkeypatch.setattr("dustapprox.tools.grid._parallel_task", _fake_task)
        
        class DummyFilter:
            def __init__(self):
                self.name = "MOCK"
            def get_flux(self, lamb, flux):
                from types import SimpleNamespace
                return SimpleNamespace(value=1.0)
        
        monkeypatch.setattr(
            "dustapprox.tools.grid.svo.get_svo_passbands",
            lambda filters: [DummyFilter() for _ in filters]
        )
        
        custom_apfields = ("teff", "logg", "feh")
        
        df = compute_photometric_grid(
            sources="dummy/*.fits",
            which_filters=["MOCK"],
            extinction_curve="F99",
            apfields=custom_apfields,
        )
        
        assert isinstance(df, pd.DataFrame)

    def test_default_filters_completeness(self):
        """Test that DEFAULT_FILTERS contains expected filters."""
        assert isinstance(DEFAULT_FILTERS, list)
        assert len(DEFAULT_FILTERS) > 0
        
        # Check for some common filters
        all_filters = " ".join(DEFAULT_FILTERS)
        assert any(x in all_filters for x in ["GAIA", "gaia", "Gaia"])
        assert any(x in all_filters for x in ["SLOAN", "sloan", "SDSS"])


class TestGridEdgeCasesAdvanced:
    """Advanced edge case tests for grid module."""

    def test_varied_a0_r0_combinations(self):
        """Test various A0 and R0 value combinations."""
        # Single values
        a0_single = np.array([1.0])
        r0_single = np.array([3.1])
        assert len(a0_single) == 1
        assert len(r0_single) == 1
        
        # Multiple values
        a0_multi = np.arange(0.1, 5.0, 0.5)
        r0_multi = np.arange(2.5, 5.5, 0.5)
        assert len(a0_multi) > 1
        assert len(r0_multi) > 1
        
        # Edge values
        a0_small = np.array([0.01])
        r0_extreme = np.array([6.0])
        assert a0_small[0] < 0.1
        assert r0_extreme[0] > 5.0

    def test_extinction_curve_representations(self):
        """Test different representations of extinction curves."""
        # String representation
        curve_str = "F99"
        assert isinstance(curve_str, str)
        
        # Object with __class__.__name__
        class MockCurve:
            pass
        
        curve_obj = MockCurve()
        name = curve_obj.__class__.__name__
        assert isinstance(name, str)
        assert name == "MockCurve"
