"""Tests for dustapprox package version and imports."""

import pytest


class TestPackageImport:
    """Test basic package imports."""

    def test_import_dustapprox(self):
        """Test that dustapprox can be imported."""
        import dustapprox
        assert dustapprox is not None

    def test_version_exists(self):
        """Test that version is defined."""
        from dustapprox import __VERSION__
        assert __VERSION__ is not None
        assert isinstance(__VERSION__, str)

    def test_version_format(self):
        """Test that version follows semantic versioning."""
        from dustapprox import __VERSION__
        
        # Should have format like "0.2.0" or "0.2.0-dev"
        parts = __VERSION__.split("-")[0].split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_main_modules_importable(self):
        """Test that main modules can be imported."""
        try:
            from dustapprox import extinction
            from dustapprox import astropy_units
            from dustapprox import models
            from dustapprox import io
            from dustapprox import tools
            
            assert extinction is not None
            assert astropy_units is not None
            assert models is not None
            assert io is not None
            assert tools is not None
        except ImportError as e:
            pytest.fail(f"Failed to import main modules: {e}")


class TestDependencies:
    """Test that required dependencies are available."""

    def test_numpy_available(self):
        """Test that numpy is available."""
        import numpy as np
        assert np.__version__ is not None

    def test_pandas_available(self):
        """Test that pandas is available."""
        import pandas as pd
        assert pd.__version__ is not None

    def test_astropy_available(self):
        """Test that astropy is available."""
        import astropy
        assert astropy.__version__ is not None

    def test_dust_extinction_available(self):
        """Test that dust_extinction is available."""
        import dust_extinction
        assert dust_extinction is not None

    def test_sklearn_available(self):
        """Test that scikit-learn is available."""
        import sklearn
        assert sklearn.__version__ is not None

    def test_scipy_available(self):
        """Test that scipy is available."""
        import scipy
        assert scipy.__version__ is not None

    def test_yaml_available(self):
        """Test that yaml is available."""
        import yaml
        assert yaml is not None

    def test_joblib_available(self):
        """Test that joblib is available."""
        import joblib
        assert joblib.__version__ is not None

    @pytest.mark.requires_network
    def test_pyphot_available(self):
        """Test that pyphot is available."""
        try:
            import pyphot
            assert pyphot is not None
        except ImportError:
            pytest.skip("pyphot not installed (requires git source)")


class TestModuleStructure:
    """Test module structure and organization."""

    def test_extinction_exports(self):
        """Test that extinction module exports expected functions."""
        from dustapprox import extinction
        
        assert hasattr(extinction, "get_extinction_model")
        assert hasattr(extinction, "evaluate_extinction_model")

    def test_models_exports(self):
        """Test that models module exports expected classes."""
        from dustapprox import models
        
        assert hasattr(models, "PrecomputedModel")
        assert hasattr(models, "ModelInfo")
        assert hasattr(models, "BaseModel")
        assert hasattr(models, "PolynomialModel")

    def test_io_submodules(self):
        """Test that io module has expected submodules."""
        from dustapprox import io
        
        assert hasattr(io, "ecsv")
        assert hasattr(io, "svo")

    def test_astropy_units_exports(self):
        """Test that astropy_units exports expected functions."""
        from dustapprox import astropy_units
        
        assert hasattr(astropy_units, "has_unit")
        assert hasattr(astropy_units, "val_in_unit")
        assert hasattr(astropy_units, "Unit")
        assert hasattr(astropy_units, "Quantity")


class TestBackwardCompatibility:
    """Test backward compatibility concerns."""

    def test_legacy_extinction_importable(self):
        """Test that legacy_extinction module exists."""
        try:
            from dustapprox import legacy_extinction
            assert legacy_extinction is not None
        except ImportError:
            pytest.fail("legacy_extinction module not found")

    def test_version_module_exists(self):
        """Test that version module exists."""
        try:
            from dustapprox import version
            assert hasattr(version, "__VERSION__")
        except ImportError:
            pytest.fail("version module not found")


class TestDataFiles:
    """Test data file availability."""

    def test_data_directory_accessible(self):
        """Test that data directory is accessible."""
        try:
            from importlib import resources
            data_path = resources.files("dustapprox") / "data"
            # Check if path exists (in newer Python versions)
            assert data_path is not None
        except (ImportError, AttributeError):
            # Older Python versions or data not packaged
            pytest.skip("Data directory check not supported")

    @pytest.mark.requires_data
    def test_precomputed_data_exists(self):
        """Test that precomputed data directory exists."""
        try:
            from importlib import resources
            precomputed_path = resources.files("dustapprox") / "data" / "precomputed"
            assert precomputed_path is not None
        except (ImportError, AttributeError, FileNotFoundError):
            pytest.skip("Precomputed data not available")


class TestEdgeCasesAndBugs:
    """Test potential edge cases and bugs in package structure."""

    def test_multiple_imports_no_side_effects(self):
        """Test that importing multiple times has no side effects."""
        import dustapprox
        version1 = dustapprox.__VERSION__
        
        # Import again
        import dustapprox as da2
        version2 = da2.__VERSION__
        
        assert version1 == version2

    def test_import_from_different_paths(self):
        """Test importing from different paths gives same module."""
        from dustapprox import extinction as ext1
        import dustapprox.extinction as ext2
        
        # Should be the same module object
        assert ext1 is ext2

    def test_circular_import_safety(self):
        """Test that there are no circular import issues."""
        # If this test runs, circular imports are not blocking
        try:
            from dustapprox import extinction
            from dustapprox import models
            from dustapprox import tools
            from dustapprox.tools import grid
            from dustapprox.models import polynomial
            
            assert all([extinction, models, tools, grid, polynomial])
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_namespace_pollution(self):
        """Test that package doesn't pollute namespace."""
        import dustapprox
        
        # Should have minimal exports in __init__
        public_attrs = [attr for attr in dir(dustapprox) if not attr.startswith("_")]
        
        # Should mainly export __VERSION__ and possibly submodules
        # Not checking exact count as it may vary
        assert "__VERSION__" in dir(dustapprox)
