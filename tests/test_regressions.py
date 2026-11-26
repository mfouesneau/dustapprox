"""Regression tests for bugs found during the November 2026 code review.

Each test pins a behaviour that was previously broken, so the bug cannot
come back silently.
"""

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import astropy.units as u
import matplotlib
import numpy as np
import pandas as pd
import pytest
import requests

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from dust_extinction.parameter_averages import F99  # noqa: E402

from dustapprox.extinction import evaluate_extinction_model  # noqa: E402
from dustapprox.io import ecsv  # noqa: E402
from dustapprox.models import PrecomputedModel  # noqa: E402
from dustapprox.models.polynomial import (  # noqa: E402
    PolynomialModel,
    approx_model,
    quick_plot_models,
)
from dustapprox.tools import downloader, generate_model  # noqa: E402
from dustapprox.tools import model_characteristics as mc  # noqa: E402


@pytest.fixture
def training_grid():
    """Small synthetic grid, including A0 = 0 rows like real grids."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "teff": rng.uniform(4000, 9000, n),
            "A0": np.tile([0.0, 0.5, 1.0, 2.0, 4.0, 8.0], n // 6),
            "R0": rng.choice([2.6, 3.1, 4.1], n),
        }
    )
    kx = 0.9 - 0.01 * df["A0"] + 0.02 * (df["teff"] / 5040.0) - 0.03 * df["R0"]
    df["Ax"] = kx * df["A0"]
    df["passband"] = "TEST_BAND"
    return df


# --- extinction --------------------------------------------------------------


@pytest.mark.parametrize("extrapolate", [True, False])
def test_extinction_instance_uses_requested_R0(extrapolate):
    """A curve instance used to ignore R0 (grids got mislabelled R0 rows)."""
    lam = np.array([0.44, 0.55, 0.8]) * u.micron
    with pytest.warns(UserWarning, match="R0=5.1"):
        got = evaluate_extinction_model(
            F99(Rv=3.1), lam, A0=1.0, R0=5.1, extrapolate=extrapolate
        )
    expected = evaluate_extinction_model(
        "F99", lam, A0=1.0, R0=5.1, extrapolate=extrapolate
    )
    np.testing.assert_allclose(got, expected)


def test_extinction_instance_matching_R0_does_not_warn():
    lam = np.array([0.44, 0.55]) * u.micron
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        evaluate_extinction_model(F99(Rv=3.1), lam, A0=1.0, R0=3.1)


# --- polynomial model --------------------------------------------------------


def test_fit_rmse_is_root_mean_squared(training_grid):
    """rmse used to hold the mean squared error."""
    model = PolynomialModel(name="TEST_BAND").fit(
        training_grid, ["teff", "A0", "R0"], degree=2
    )
    used = training_grid[training_grid["A0"] > 0]
    residuals = model.predict(used) - used["Ax"] / used["A0"]
    assert model.meta["rmse"] == pytest.approx(
        np.sqrt(np.mean(residuals**2)), rel=1e-6
    )


def test_approx_model_rmse_matches_fit(training_grid):
    res = approx_model(
        training_grid,
        passband="TEST_BAND",
        degree=2,
        input_parameters=["teff", "A0", "R0"],
    )
    model = PolynomialModel().fit(
        training_grid, ["teff", "A0", "R0"], degree=2
    )
    assert res["rmse"] == pytest.approx(model.meta["rmse"])


def test_fit_drops_A0_zero_rows(training_grid):
    """A0 = 0 rows used to become NaN and crash PolynomialFeatures."""
    with_zeros = PolynomialModel().fit(training_grid, ["teff", "A0"], degree=2)
    without = PolynomialModel().fit(
        training_grid[training_grid["A0"] > 0], ["teff", "A0"], degree=2
    )
    pd.testing.assert_series_equal(with_zeros.coeffs_, without.coeffs_)


def test_fit_without_teff(training_grid):
    """teff used to be required even when not in the features."""
    model = PolynomialModel().fit(training_grid, ["A0", "R0"], degree=2)
    assert model.feature_names == ["A0", "R0"]
    pred = model.predict(training_grid[training_grid["A0"] > 0])
    assert np.all(np.isfinite(pred))


def test_repr_of_unfitted_model():
    text = repr(PolynomialModel(name="x"))
    assert "not fitted" in text


def test_stats_survive_ecsv_roundtrip(training_grid, tmp_path):
    """from_file looked for mean/stddev while files store *_residuals."""
    model = PolynomialModel(name="TEST_BAND").fit(
        training_grid, ["teff", "A0"], degree=2
    )
    fname = tmp_path / "models.ecsv"
    generate_model.export_trained_model_to_ecsv(fname, [model])
    loaded = PolynomialModel.from_file(str(fname), "TEST_BAND")
    for key in ("mae", "rmse", "mean_residuals", "std_residuals"):
        assert loaded.meta[key] == pytest.approx(model.meta[key]), key


def test_quick_plot_models_returns_dataframe(training_grid):
    """It was a generator (never ran in the docs) and passed approx_model
    kwargs to imshow in the tall-image branch."""
    try:
        res = quick_plot_models(
            training_grid, input_parameters=["teff", "A0"], degree=2
        )
        assert isinstance(res, pd.DataFrame)
        assert res["passband"].tolist() == ["TEST_BAND"]
    finally:
        plt.close("all")


# --- precomputed library -----------------------------------------------------


def test_load_model_unknown_passband_raises_keyerror():
    """A KeyError from from_file was reported as an unimplemented kind."""
    lib = PrecomputedModel()
    fname = lib.find(passband="GAIA3.G")[0].filename
    with pytest.raises(KeyError, match="NOPE"):
        lib.load_model(fname, passband="NOPE")


def test_load_model_unknown_kind_still_not_implemented():
    info = {
        "model": {"kind": "spline"},
        "filename": "unused.ecsv",
        "passbands": ["X"],
    }
    with pytest.raises(NotImplementedError, match="spline"):
        PrecomputedModel().load_model(info, passband="X")


def test_get_models_info_respects_glob_pattern():
    """The cache used to ignore a different glob_pattern."""
    lib = PrecomputedModel()
    everything = lib.get_models_info()
    gaia_only = lib.get_models_info(glob_pattern="/**/*gaia_dr3*.ecsv")
    assert 0 < len(gaia_only) < len(everything)
    assert all("gaia_dr3" in m.filename for m in gaia_only)
    assert len(lib.get_models_info()) == len(everything)


# --- ECSV --------------------------------------------------------------------


def test_ecsv_vector_columns_keep_their_own_subtype(tmp_path):
    """Converters were late-bound to the last column's subtype."""
    fname = tmp_path / "vectors.ecsv"
    fname.write_text(
        "# %ECSV 1.0\n# ---\n# delimiter: ','\n# datatype:\n"
        "# - {name: a, datatype: string, subtype: 'float64[2]'}\n"
        "# - {name: b, datatype: string, subtype: 'int64[2]'}\n"
        'a,b\n"[1.5, 2.5]","[1, 2]"\n'
    )
    df = ecsv.read(str(fname))
    assert df["a"][0].dtype == np.float64
    np.testing.assert_array_equal(df["a"][0], [1.5, 2.5])
    assert df["b"][0].dtype == np.int64


# --- downloader --------------------------------------------------------------


def _response(headers, chunks):
    resp = Mock(headers=headers)
    resp.iter_content = Mock(return_value=iter(chunks))
    resp.content = b"".join(chunks)
    return resp


def test_download_zero_content_length_streaming(tmp_path):
    """total=0 used to divide by zero once the progress bar updated."""
    resp = _response({"content-length": "0"}, [b"x" * 10] * 3)
    target = tmp_path / "f.bin"
    with (
        patch("requests.get", return_value=resp),
        patch.object(downloader, "time") as fake_time,
    ):
        fake_time.time.side_effect = [0, 10, 20, 30, 40]
        downloader.download_file("http://example.invalid/f", str(target))
    assert target.read_bytes() == b"x" * 30


def test_download_http_error_is_raised_not_saved(tmp_path):
    resp = _response({"content-length": "9"}, [b"not found"])
    resp.raise_for_status.side_effect = requests.HTTPError("404")
    target = tmp_path / "f.bin"
    with patch("requests.get", return_value=resp):
        with pytest.raises(requests.HTTPError):
            downloader.download_file("http://example.invalid/f", str(target))
    assert not target.exists()


def test_download_uses_timeout(tmp_path):
    resp = _response({"content-length": "4"}, [b"data"])
    with patch("requests.get", return_value=resp) as get:
        downloader.download_file(
            "http://example.invalid/f", str(tmp_path / "f.bin"), timeout=5
        )
    assert get.call_args.kwargs["timeout"] == 5


# --- grid generation ---------------------------------------------------------


def _stored_grid(extinction="F99"):
    df = pd.DataFrame(
        {
            "teff": 5000.0,
            "logg": 4.5,
            "feh": 0.0,
            "alpha": 0.0,
            "passband": "GAIA_GAIA3.G",
            "mag0": 1.0,
            "mag": 1.5,
            "A0": np.array([0.01, 1.0], dtype="float32"),
            "R0": np.float32(3.1),
            "Ax": 0.5,
        }
    )
    df.attrs["extinction"] = {"source": extinction}
    return df


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("models/Kurucz2003all/*.fl.dat.txt", "Kurucz2003all"),
        ("/abs/data/Kurucz_2003/*.txt", "Kurucz"),
        ("*.txt", "*.txt"),
    ],
)
def test_generate_grid_default_atmosphere_name(tmp_path, pattern, expected):
    """The name was taken from split('/')[1]: IndexError without a '/'."""
    with patch.object(
        generate_model.grid,
        "compute_photometric_grid",
        return_value=_stored_grid(),
    ) as compute:
        generate_model.generate_grid(
            pattern, tmp_path / "grid.ecsv", ["GAIA/GAIA3.G"]
        )
    assert compute.call_args.kwargs["atmosphere_name"] == expected


@pytest.fixture
def stored_grid_file(tmp_path):
    fname = tmp_path / "grid.ecsv"
    ecsv.write(_stored_grid(), str(fname))
    return fname


def _reuse(fname, **overrides):
    params = dict(
        pbset=["GAIA/GAIA3.G"],
        extinction_curve="F99",
        A0=np.array([0.01, 1.0]),
        R0=np.array([3.1]),
    )
    params.update(overrides)
    with patch.object(
        generate_model.grid,
        "compute_photometric_grid",
        side_effect=AssertionError("must reuse the stored grid"),
    ):
        return generate_model.generate_grid("models/x/*.txt", fname, **params)


def test_generate_grid_reuses_matching_grid(stored_grid_file):
    assert len(_reuse(stored_grid_file)) == 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"extinction_curve": "G23"}, "extinction"),
        ({"pbset": ["GAIA/GAIA3.Gbp"]}, "passbands"),
        ({"A0": np.array([0.01, 2.0])}, "A0"),
        ({"R0": np.array([2.6, 3.1])}, "R0"),
        ({"apfields": ["teff", "vtur"]}, "vtur"),
    ],
)
def test_generate_grid_refuses_mismatching_grid(
    stored_grid_file, overrides, message
):
    """An existing grid used to be reused whatever the parameters were."""
    with pytest.raises(ValueError, match=message):
        _reuse(stored_grid_file, **overrides)


# --- model characteristics ---------------------------------------------------


def test_run_model_characteristics_selects_extinction(tmp_path, monkeypatch):
    """All curves were tested against a grid built with one curve."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "gaia_kurucz_G23_small_a0r0_grid.ecsv").touch()
    with (
        patch.object(mc, "ModelCharacteristicsRunner") as runner,
        patch.object(mc.svo, "SVOSpectrum", MagicMock()),
    ):
        mc.run_model_characteristics(passbands="Gaia", extinction_curve="G23")
    selection = runner.call_args.kwargs["library_selection"]
    assert selection == {"passband": "Gaia", "extinction": "G23"}


def test_figure_names_differ_per_extinction_curve():
    """F99 and G23 figures for one passband used to overwrite each other."""
    names = {
        mc._figure_name(
            1,
            SimpleNamespace(
                meta={"extinction": {"source": src}}, name="GAIA_GAIA3.G"
            ),
        )
        for src in ("F99", "G23")
    }
    assert len(names) == 2


# --- grid attenuation --------------------------------------------------------


def _flat_svo_spectrum(path):
    """Minimal SVO-format spectrum with a flat flux."""
    lines = [
        "# teff = 5000 K (Effective temperature)",
        "# logg = 4.5 log(cm/s2) (Surface gravity)",
        "# meta = 0  (Metallicity)",
        "# alpha = 0  (Alpha)",
        "# column 1: WAVELENGTH (ANGSTROM), Wavelength in Angstrom",
        "# column 2: FLUX (ERG/CM2/S/A), Flux in erg/cm2/s/A",
    ]
    lines += [f"{w:.1f} 1.0" for w in np.arange(3000.0, 10000.0, 5.0)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_grid_attenuation_uses_magnitudes(tmp_path):
    """Extinction is in magnitudes: flux * 10**(-0.4 A), not exp(-A).

    For a flat spectrum seen through a narrow box at 550 nm, Ax / A0 must
    equal the curve's A(550 nm) / A0 (exp(-A) made it 2.5 / ln(10) ~ 1.086
    times larger).
    """
    from pyphot import Filter
    from pyphot import config as pyphot_config

    from dustapprox.tools.grid import _parallel_task

    wave = np.arange(5440.0, 5561.0, 1.0)
    trans = ((wave >= 5450) & (wave <= 5550)).astype(float)
    box = Filter(
        wave * pyphot_config.units.U("angstrom"),
        trans,
        name="BOX550",
        dtype="photon",
    )
    fname = _flat_svo_spectrum(tmp_path / "flat.txt")
    df = _parallel_task(fname, ["teff"], [box], "F99", [3.1], [0.01, 1.0, 5.0])
    kx = (df["Ax"] / df["A0"]).to_numpy()
    expected = evaluate_extinction_model(
        "F99", np.array([5500.0]) * u.angstrom, A0=1.0, R0=3.1
    )[0]
    np.testing.assert_allclose(kx, expected, atol=0.005)


# --- extinction: x extrapolation only where the curve allows it -------------


@pytest.mark.parametrize(
    "name, inside, outside",
    [
        ("D22", [0.5, 1.0], [3.0, 8.0]),  # near-IR only: blows up in the UV
        (
            "GCC09",
            [4.0, 8.0],
            [1.0, 2.0],
        ),  # UV only: returned 0 in the optical
        ("M14", [1.0, 3.0], [5.0, 8.0]),  # optical only: returned 0 in the UV
    ],
)
def test_partial_range_curves_are_nan_outside(name, inside, outside):
    x = np.array(inside + outside)
    lam = (1.0 / x) * u.micron
    vals = evaluate_extinction_model(
        name, lam, A0=1.0, R0=3.1, extrapolate=True
    )
    assert np.all(np.isfinite(vals[: len(inside)]))
    assert np.all(vals[: len(inside)] > 0)
    assert np.all(np.isnan(vals[len(inside) :]))


@pytest.mark.parametrize("name", ["CCM89", "F99", "F04", "F19", "G23", "O94"])
def test_full_range_curves_still_extrapolate(name):
    lam = (1.0 / np.array([0.2, 1.0, 9.5])) * u.micron
    vals = evaluate_extinction_model(
        name, lam, A0=1.0, R0=3.1, extrapolate=True
    )
    assert np.all(np.isfinite(vals))


@pytest.mark.parametrize("name", ["M14", "F99", "G23"])
def test_extrapolate_does_not_leak_class_ranges(name):
    import dust_extinction.parameter_averages as pa

    cls = getattr(pa, name)
    before = (list(cls.x_range), list(cls.Rv_range))
    evaluate_extinction_model(name, [5500.0] * u.AA, A0=1.0, R0=3.1)
    assert (list(cls.x_range), list(cls.Rv_range)) == before
    # outside the native range, extrapolate=False must still refuse
    with pytest.raises(ValueError):
        evaluate_extinction_model(
            name, [1.0e6] * u.AA, A0=1.0, R0=3.1, extrapolate=False
        )

