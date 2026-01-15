"""Tests for dustapprox.models module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, cast

from dustapprox.models import (
    PrecomputedModel,
    ModelInfo,
    BaseModel,
    PolynomialModel,
    kinds,
)
from dustapprox.io import ecsv


class TestBaseModel:
    """Test BaseModel class."""

    def test_base_model_instantiation(self):
        """Test that BaseModel can be instantiated."""
        model = BaseModel()
        assert hasattr(model, "meta")
        assert hasattr(model, "name_")

    def test_base_model_with_meta(self):
        """Test BaseModel with metadata."""
        meta = {"source": "test", "version": "1.0"}
        model = BaseModel(meta=meta)
        assert model.meta == meta

    def test_base_model_with_name(self):
        """Test BaseModel with name."""
        model = BaseModel(name="test_model")
        assert model.name_ == "test_model"

    def test_fit_not_implemented(self):
        """Test that fit raises NotImplementedError."""
        model = BaseModel()
        with pytest.raises(NotImplementedError):
            model.fit()

    def test_predict_not_implemented(self):
        """Test that predict raises NotImplementedError."""
        model = BaseModel()
        with pytest.raises(NotImplementedError):
            model.predict()

    def test_to_pandas_not_implemented(self):
        """Test that to_pandas raises NotImplementedError."""
        model = BaseModel()
        with pytest.raises(NotImplementedError):
            model.to_pandas()


class TestModelInfo:
    """Test ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo instance."""
        info = ModelInfo(
            atmosphere={"source": "Kurucz", "teff": [3500, 50000]},
            extinction={"source": "F99", "R0": 3.1},
            comment=["test comment"],
            model={"kind": "polynomial", "degree": 3},
            passbands=["GAIA_GAIA3.G"],
            filename="test.ecsv",
        )
        
        assert info.atmosphere["source"] == "Kurucz"
        assert info.extinction["source"] == "F99"
        assert len(info.passbands) == 1
        assert info.filename == "test.ecsv"

    def test_model_info_repr(self):
        """Test ModelInfo string representation."""
        info = ModelInfo(
            atmosphere={"source": "Kurucz"},
            extinction={"source": "F99"},
            comment=["test"],
            model={"kind": "polynomial"},
            passbands=["GAIA_GAIA3.G"],
            filename="test.ecsv",
        )
        
        repr_str = repr(info)
        assert "Precomputed Model Information" in repr_str
        assert "Kurucz" in repr_str
        assert "F99" in repr_str

    def test_model_info_copy(self):
        """Test copying ModelInfo."""
        info = ModelInfo(
            atmosphere={"source": "Kurucz"},
            extinction={"source": "F99"},
            comment=["test"],
            model={"kind": "polynomial"},
            passbands=["GAIA_GAIA3.G"],
            filename="test.ecsv",
        )
        
        info_copy = info.copy()
        assert info_copy.atmosphere == info.atmosphere
        assert info_copy.extinction == info.extinction
        assert info_copy is not info

    def test_load_model_without_library_raises(self):
        """Test that load_model raises error without source library."""
        info = ModelInfo(
            atmosphere={},
            extinction={},
            comment=[],
            model={},
            passbands=[],
            filename="",
        )
        
        with pytest.raises(ValueError, match="source library is not set"):
            info.load_model()


class TestKindsRegistry:
    """Test the kinds registry."""

    def test_kinds_contains_polynomial(self):
        """Test that polynomial kind is registered."""
        assert "polynomial" in kinds
        assert kinds["polynomial"] == PolynomialModel

    def test_kinds_is_dict(self):
        """Test that kinds is a dictionary."""
        assert isinstance(kinds, dict)

    def test_polynomial_model_in_kinds(self):
        """Test that we can access PolynomialModel through kinds."""
        poly_class = kinds.get("polynomial")
        assert poly_class is not None
        assert poly_class == PolynomialModel


class TestPrecomputedModel:
    """Test PrecomputedModel class."""

    def test_precomputed_model_init_default(self):
        """Test PrecomputedModel initialization with default location."""
        lib = PrecomputedModel()
        assert lib.location is not None
        assert lib._info is None

    def test_precomputed_model_init_custom_location(self, tmp_path):
        """Test PrecomputedModel initialization with custom location."""
        lib = PrecomputedModel(location=str(tmp_path))
        assert lib.location == str(tmp_path)

    def test_get_models_info_returns_list(self):
        """Test that get_models_info returns a list."""
        lib = PrecomputedModel()
        try:
            info = lib.get_models_info()
            assert isinstance(info, (list, tuple))
        except (FileNotFoundError, OSError):
            # If no models are available, that's okay for this test
            pytest.skip("No precomputed models available")

    def test_get_models_info_caching(self):
        """Test that get_models_info caches results."""
        lib = PrecomputedModel()
        try:
            info1 = lib.get_models_info()
            info2 = lib.get_models_info()
            # Should return the same cached object
            assert info1 is info2
        except (FileNotFoundError, OSError):
            pytest.skip("No precomputed models available")

    @pytest.mark.requires_data
    def test_find_by_passband(self):
        """Test finding models by passband."""
        lib = PrecomputedModel()
        try:
            # Try to find GAIA models
            results = lib.find(passband="GAIA")
            if len(results) > 0:
                assert all(
                    "passband" in str(r.passbands).lower()
                    or "gaia" in str(r.passbands).lower()
                    for r in results
                )
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip(
                "No precomputed models available or passband not found"
            )

    @pytest.mark.requires_data
    def test_find_by_extinction(self):
        """Test finding models by extinction curve."""
        lib = PrecomputedModel()
        try:
            results = lib.find(extinction="F99")
            if len(results) > 0:
                assert all(
                    "f99" in r.extinction["source"].lower()
                    or "fitzpatrick" in r.extinction["source"].lower()
                    for r in results
                )
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("No precomputed models available")

    @pytest.mark.requires_data
    def test_find_by_atmosphere(self):
        """Test finding models by atmosphere."""
        lib = PrecomputedModel()
        try:
            results = lib.find(atmosphere="kurucz")
            if len(results) > 0:
                assert all(
                    "kurucz" in r.atmosphere["source"].lower() for r in results
                )
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("No precomputed models available")

    @pytest.mark.requires_data
    def test_find_by_kind(self):
        """Test finding models by kind."""
        lib = PrecomputedModel()
        try:
            results = lib.find(kind="polynomial")
            if len(results) > 0:
                assert all(r.model["kind"] == "polynomial" for r in results)
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("No precomputed models available")

    @pytest.mark.requires_data
    def test_find_case_insensitive(self):
        """Test that find is case-insensitive."""
        lib = PrecomputedModel()
        try:
            results_lower = lib.find(passband="gaia")
            results_upper = lib.find(passband="GAIA")
            results_mixed = lib.find(passband="Gaia")

            # All should return the same results
            assert (
                len(results_lower) == len(results_upper) == len(results_mixed)
            )
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("No precomputed models available")

    @pytest.mark.requires_data
    def test_find_multiple_criteria(self):
        """Test finding with multiple search criteria."""
        lib = PrecomputedModel()
        try:
            results = lib.find(
                passband="GAIA", extinction="F99", kind="polynomial"
            )
            # Should filter by all criteria
            if len(results) > 0:
                for r in results:
                    assert "gaia" in str(r.passbands).lower()
                    assert r.model["kind"] == "polynomial"
        except (FileNotFoundError, OSError, AttributeError):
            pytest.skip("No precomputed models available")

    def test_find_no_matches(self):
        """Test find with criteria that match nothing."""
        lib = PrecomputedModel()
        try:
            results = lib.find(passband="NONEXISTENT_PASSBAND_XYZ123")
            assert len(results) == 0
        except (FileNotFoundError, OSError):
            pytest.skip("No precomputed models available")

    @pytest.mark.requires_data
    def test_load_model_not_implemented_kind(self, tmp_path):
        """Test loading model with unimplemented kind."""
        lib = PrecomputedModel()
        
        # Create a mock ModelInfo with unsupported kind
        info = ModelInfo(
            atmosphere={},
            extinction={},
            comment=[],
            model={"kind": "unsupported_model_type"},
            passbands=["TEST"],
            filename=str(tmp_path / "test.ecsv"),
        )
        
        with pytest.raises(NotImplementedError, match="not implemented"):
            lib.load_model(info, passband="TEST")


class TestPolynomialModel:
    """Test PolynomialModel class (basic tests without data)."""

    def test_polynomial_model_is_base_model(self):
        """Test that PolynomialModel inherits from BaseModel."""
        assert issubclass(PolynomialModel, BaseModel)

    def test_polynomial_model_has_predict(self):
        """Test that PolynomialModel has predict method."""
        assert hasattr(PolynomialModel, "predict")

    def test_polynomial_model_has_from_file(self):
        """Test that PolynomialModel has from_file class method."""
        assert hasattr(PolynomialModel, "from_file")


class TestPolynomialModelFitting:
    """Exercise PolynomialModel fit/predict/IO for coverage."""

    def _build_simple_df(self):
        teff = np.array([5000.0, 5500.0, 6000.0, 6500.0, 7000.0])
        A0 = np.array([1.0, 2.0, 1.5, 0.5, 1.2])
        # simple linear relation on teffnorm to keep the regression stable
        kx = 0.2 + 0.01 * (teff / 5040.0)
        Ax = A0 * kx
        df = pd.DataFrame({
            "teff": teff,
            "A0": A0,
            "Ax": Ax,
        })
        # mimic metadata from grid generation
        df.attrs = {
            "atmosphere": {"source": "test"},
            "extinction": {"source": "CCM89"},
            "model": {"kind": "polynomial", "feature_names": ["teff", "A0"]},
        }
        return df

    def test_fit_and_predict_roundtrip(self):
        """Fit on synthetic data and predict back."""
        df = self._build_simple_df()
        model = PolynomialModel(name="GAIA.TEST")
        model.fit(df, features=["teff", "A0"], degree=1)

        preds = model.predict(df[["teff", "A0"]])
        expected_kx = df["Ax"] / df["A0"]
        assert preds.shape == expected_kx.shape
        # allow small tolerance from lasso fit
        assert np.allclose(preds, expected_kx, rtol=1e-2, atol=1e-2)

    def test_predict_adds_teffnorm(self):
        """Ensure teffnorm is auto-added when missing."""
        df = self._build_simple_df()
        model = PolynomialModel(name="GAIA.TEST")
        model.fit(df, features=["teff", "A0"], degree=1)

        # omit teffnorm; model should compute it
        inputs = df[["teff", "A0"]].copy()
        preds = model.predict(inputs)
        assert preds.shape[0] == len(df)

    def test_to_pandas_and_from_file_roundtrip(self, tmp_path):
        """Write to ECSV then restore and compare predictions."""
        df = self._build_simple_df()
        model = PolynomialModel(name="GAIA.TEST")
        model.fit(df, features=["teff", "A0"], degree=1)

        fname = tmp_path / "poly_model.ecsv"
        # to_pandas does not store passband column; add it before persisting
        df_model = model.to_pandas()
        passband = model.name or "GAIA.TEST"
        df_model.insert(0, "passband", passband)
        ecsv.write(
            df_model,
            str(fname),
            **cast(dict[str, Any], dict(df_model.attrs)),
        )

        loaded = PolynomialModel.from_file(str(fname), passband)
        preds_orig = model.predict(df[["teff", "A0"]])
        preds_loaded = loaded.predict(df[["teff", "A0"]])
        assert np.allclose(preds_loaded, preds_orig, rtol=1e-2, atol=1e-2)

    def test_predict_requires_fit(self):
        """predict should raise if model not fitted."""
        model = PolynomialModel(name="GAIA.TEST")
        with pytest.raises(ValueError):
            model.predict(pd.DataFrame({"teff": [5000], "A0": [1.0]}))

    def test_set_transformer_invalid_kind(self):
        """_set_transformer should reject unknown kinds."""
        model = PolynomialModel(name="GAIA.TEST", meta={
            "model": {"feature_names": ["teff", "A0"]}
        })
        with pytest.raises(NotImplementedError):
            model._set_transformer(kind="spline", degree=2)


class TestModelsEdgeCases:
    """Test edge cases and potential bugs in models module."""

    def test_model_info_with_empty_passbands(self):
        """Test ModelInfo with empty passband list."""
        info = ModelInfo(
            atmosphere={},
            extinction={},
            comment=[],
            model={},
            passbands=[],
            filename="",
        )
        
        assert info.passbands == []
        assert len(info.passbands) == 0

    def test_model_info_with_multiple_comments(self):
        """Test ModelInfo with multiple comments."""
        comments = ["comment1", "comment2", "comment3"]
        info = ModelInfo(
            atmosphere={},
            extinction={},
            comment=comments,
            model={},
            passbands=[],
            filename="",
        )
        
        assert len(info.comment) == 3
        assert info.comment == comments

    def test_precomputed_model_empty_location(self, tmp_path):
        """Test PrecomputedModel with empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        lib = PrecomputedModel(location=str(empty_dir))
        info = lib.get_models_info()
        
        # Should return empty list for empty directory
        assert isinstance(info, (list, tuple))
        assert len(info) == 0

    def test_find_filters_passbands_correctly(self):
        """Test that find filters passbands in results."""
        lib = PrecomputedModel()
        
        # Create mock info
        mock_info = ModelInfo(
            atmosphere={},
            extinction={"source": "test"},
            comment=[],
            model={"kind": "polynomial"},
            passbands=["GAIA_GAIA3.G", "GAIA_GAIA3.Gbp", "SLOAN_SDSS.g"],
            filename="test.ecsv",
            _source_library=lib,
        )
        
        lib._info = [mock_info]
        
        # Find GAIA passbands
        results = lib.find(passband="GAIA")
        
        if len(results) > 0:
            # Should only include GAIA passbands
            for passband in results[0].passbands:
                assert "gaia" in passband.lower()

    def test_base_model_kwargs_handling(self):
        """Test that BaseModel handles arbitrary kwargs."""
        model = BaseModel(
            meta={"test": "value"},
            name="test_name",
            extra_param="should_be_ignored"
        )
        
        assert model.meta == {"test": "value"}
        assert model.name_ == "test_name"
        # extra_param should not cause an error
