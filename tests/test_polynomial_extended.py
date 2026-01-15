"""Tests for dustapprox.models.polynomial module."""

import pytest
import numpy as np
import pandas as pd

from dustapprox.models.polynomial import (
    approx_model,
    PolynomialModel,
)


class TestApproxModel:
    """Test the approx_model function."""

    @pytest.fixture
    def sample_grid_data(self):
        """Create sample grid data for testing."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "passband": ["GAIA_GAIA3.G"] * n_samples,
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        return pd.DataFrame(data)

    def test_approx_model_default_parameters(self, sample_grid_data):
        """Test approx_model with default parameters."""
        result = approx_model(sample_grid_data)

        assert isinstance(result, dict)
        assert "features" in result
        assert "coefficients" in result
        assert "mae" in result
        assert "rmse" in result
        assert len(result["features"]) == len(result["coefficients"])

    def test_approx_model_custom_passband(self, sample_grid_data):
        """Test approx_model with custom passband."""
        sample_grid_data["passband"] = ["SLOAN_SDSS.u"] * len(sample_grid_data)

        result = approx_model(sample_grid_data, passband="SLOAN_SDSS.u")

        assert isinstance(result, dict)
        assert "coefficients" in result

    def test_approx_model_different_degree(self, sample_grid_data):
        """Test approx_model with different polynomial degrees."""
        for degree in [1, 2, 3]:
            result = approx_model(sample_grid_data, degree=degree)

            assert isinstance(result, dict)
            assert "features" in result

    def test_approx_model_statistics_validity(self, sample_grid_data):
        """Test that returned statistics are valid."""
        result = approx_model(sample_grid_data)

        assert result["mae"] >= 0
        assert result["rmse"] >= 0

    def test_approx_model_different_input_params(self, sample_grid_data):
        """Test approx_model with different input parameters."""
        result = approx_model(
            sample_grid_data, input_parameters=["teff", "A0", "logg"]
        )

        assert isinstance(result, dict)
        assert "coefficients" in result


class TestPolynomialModel:
    """Test the PolynomialModel class."""

    @pytest.fixture
    def sample_polynomial_model(self):
        """Create a sample PolynomialModel for testing."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df)
        return model

    def test_polynomial_model_fit(self):
        """Test PolynomialModel fit method."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df)

        assert model is not None
        assert model.transformer_ is not None
        assert model.coeffs_ is not None

    def test_polynomial_model_predict(self, sample_polynomial_model):
        """Test PolynomialModel predict method."""
        model = sample_polynomial_model

        test_data = pd.DataFrame(
            {
                "teff": [5000, 6000, 7000],
                "logg": [4.5, 4.0, 3.5],
                "feh": [0.0, -0.5, 0.5],
                "alpha": [0.0, 0.1, 0.2],
                "A0": [1.0, 1.5, 2.0],
            }
        )

        predictions = model.predict(test_data)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3
        assert all(np.isfinite(predictions))

    def test_polynomial_model_predict_requires_fit(self):
        """Test that predict raises error before fit."""
        model = PolynomialModel()

        test_data = pd.DataFrame(
            {
                "teff": [5000],
                "logg": [4.5],
                "feh": [0.0],
                "alpha": [0.0],
                "A0": [1.0],
            }
        )

        with pytest.raises((ValueError, AttributeError)):
            model.predict(test_data)

    def test_polynomial_model_to_pandas(self, sample_polynomial_model):
        """Test PolynomialModel to_pandas method."""
        model = sample_polynomial_model

        result = model.to_pandas()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_polynomial_model_degree(self, sample_polynomial_model):
        """Test PolynomialModel degree_ property."""
        model = sample_polynomial_model

        degree = model.degree_

        assert degree is not None
        assert isinstance(degree, (int, np.integer))
        assert degree > 0

    def test_polynomial_model_fit_custom_degree(self):
        """Test PolynomialModel fit with custom degree."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df, degree=2)

        assert model.degree_ == 2

    def test_polynomial_model_fit_custom_features(self):
        """Test PolynomialModel fit with custom features."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df, features=["teff", "logg", "A0"])

        assert model is not None

    def test_polynomial_model_predict_single_row(
        self, sample_polynomial_model
    ):
        """Test predict with single row."""
        model = sample_polynomial_model

        test_data = pd.DataFrame(
            {
                "teff": [5000],
                "logg": [4.5],
                "feh": [0.0],
                "alpha": [0.0],
                "A0": [1.0],
            }
        )

        predictions = model.predict(test_data)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1
        assert np.isfinite(predictions[0])

    def test_polynomial_model_predict_many_rows(self, sample_polynomial_model):
        """Test predict with many rows."""
        model = sample_polynomial_model

        n_test = 100
        test_data = pd.DataFrame(
            {
                "teff": np.random.uniform(3500, 10000, n_test),
                "logg": np.random.uniform(0, 5, n_test),
                "feh": np.random.uniform(-2, 1, n_test),
                "alpha": np.random.uniform(0, 0.5, n_test),
                "A0": np.random.uniform(0.1, 5.0, n_test),
            }
        )

        predictions = model.predict(test_data)

        assert len(predictions) == n_test
        assert all(np.isfinite(predictions))

    def test_polynomial_model_predict_teffnorm(self, sample_polynomial_model):
        """Test that predict correctly handles teff normalization."""
        model = sample_polynomial_model

        test_data = pd.DataFrame(
            {
                "teff": [5040],
                "logg": [4.5],
                "feh": [0.0],
                "alpha": [0.0],
                "A0": [1.0],
            }
        )

        predictions = model.predict(test_data)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 1


class TestPolynomialModelEdgeCases:
    """Test edge cases for PolynomialModel."""

    def test_polynomial_model_fit_interaction_only(self):
        """Test fit with interaction_only flag."""
        np.random.seed(42)
        n_samples = 100

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df, interaction_only=True)

        assert model is not None

    def test_polynomial_model_consolidate_named_data(self):
        """Test _consolidate_named_data method."""
        np.random.seed(42)
        n_samples = 200

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df)

        X_df = df[["teff", "logg", "feh", "alpha", "A0"]]
        consolidated = model._consolidate_named_data(X_df)

        assert isinstance(consolidated, pd.DataFrame)

    def test_polynomial_model_get_transformed_feature_names(self):
        """Test get_transformed_feature_names method."""
        np.random.seed(42)
        n_samples = 200

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df)

        feature_names = model.get_transformed_feature_names()

        assert isinstance(feature_names, (list, np.ndarray))
        assert len(feature_names) > 0


class TestPolynomialModelIntegration:
    """Integration tests for PolynomialModel."""

    def test_fit_predict_roundtrip(self):
        """Test full fit-predict roundtrip."""
        np.random.seed(42)
        n_samples = 200

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        train = df.iloc[:150]
        test = df.iloc[150:]

        model = PolynomialModel()
        model.fit(train)

        test_features = test[["teff", "logg", "feh", "alpha", "A0"]]
        predictions = model.predict(test_features)

        assert len(predictions) == len(test)
        assert all(np.isfinite(predictions))

    def test_repr_method(self):
        """Test __repr__ method."""
        np.random.seed(42)
        n_samples = 200

        data = {
            "teff": np.linspace(3500, 10000, n_samples),
            "logg": np.random.uniform(0, 5, n_samples),
            "feh": np.random.uniform(-2, 1, n_samples),
            "alpha": np.random.uniform(0, 0.5, n_samples),
            "A0": np.random.uniform(0.1, 5.0, n_samples),
            "Ax": np.random.uniform(0.05, 2.5, n_samples),
        }
        df = pd.DataFrame(data)

        model = PolynomialModel()
        model.fit(df)

        repr_str = repr(model)

        assert isinstance(repr_str, str)
        assert "PolynomialModel" in repr_str
