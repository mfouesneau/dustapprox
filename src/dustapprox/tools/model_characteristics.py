"""Generates some characteristic model plots."""

import pathlib
from dataclasses import dataclass, field
from typing import Any, Generator, Sequence, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d

from .. import models
from ..io import ecsv, svo
from .generate_model import GridParameters

__all__ = ["ModelCharacteristicsRunner", "run_model_characteristics"]


@dataclass
class ModelCharacteristicsRunner:
    """Class to run model characteristics plots for a set of precomputed models."""

    library_path: str
    """which precomputed model library path to use"""
    library_selection: dict[str, Any]
    """Selection criteria to select the precomputed model library"""
    testdata_path: str
    """Path to the test data photometric grid"""
    _what: list[models.ModelInfo] = field(default_factory=list)
    """Which models to run"""
    _vega_zpt: dict[str, float] = field(default_factory=dict[str, float])
    """Vega zeropoints for each passband"""
    vega_spec_approx: svo.SVOSpectrum | None = None
    """An approximate Vega spectrum for zeropoint calculations"""
    _data: pd.DataFrame | None = None
    """Data grid with photometric values and stellar parameters to test on"""

    @property
    def data(self) -> pd.DataFrame:
        """Data grid with photometric values and stellar parameters to test on
        Load the test data grid if not already done.
        """
        if self._data is None:
            if self.testdata_path is None:
                raise ValueError(
                    "No testdata_path provided to load test data grid."
                )
            if not isinstance(self.testdata_path, str):
                raise ValueError(
                    "testdata_path should be a string path to the data file."
                )
            print(f"Loading test data grid from {self.testdata_path}...")
            if self.testdata_path.strip().endswith(".ecsv"):
                self._data = cast(pd.DataFrame, ecsv.read(self.testdata_path))
            elif self.testdata_path.strip().endswith(".csv"):
                self._data = pd.read_csv(self.testdata_path)
            else:
                raise ValueError(
                    "Unsupported testdata_path file format. Use .ecsv or .csv"
                )
        return self._data

    @property
    def what(self) -> Sequence[models.ModelInfo]:
        """Which models to run."""
        if not self._what:
            print(
                f"Selecting precomputed model library from {self.library_path}..."
            )
            lib = models.PrecomputedModel(self.library_path)
            print("Selection criteria:", self.library_selection)
            self._what = list(lib.find(**self.library_selection))
            print(f"Selected {len(self._what)} models.")
        return self._what

    def set_vega_zeropoints(self, vega_spec: svo.SVOSpectrum) -> None:
        """Compute Vega zeropoints for each passband in the selected models.

        Parameters
        ----------
        vega_spec: svo.SVOSpectrum
            An approximate Vega spectrum for zeropoint calculations
        """
        self.vega_spec_approx = vega_spec
        self._vega_zpt = {}
        print("Computing Vega zeropoints...")
        for pb in self.passbands():
            # no need to recompute if overlapping passbands
            if pb in self._vega_zpt:
                continue
            passband_ = svo.get_pyphot_filter(pb.replace("_", "/", 1))
            zpt = -2.5 * np.log10(
                passband_.get_flux(vega_spec.λ, vega_spec.flux.value).value
            )
            self._vega_zpt[pb] = zpt
            print(f"  {pb}: {zpt:.4f} mag")

    def passbands(self) -> Generator[str, None, None]:
        """Return a generator on the passbands of the selected models."""
        for what_ in self.what:
            yield from what_.passbands

    def models(self) -> Generator[models.BaseModel, None, None]:
        """Return a generator on the selected models."""
        lib = models.PrecomputedModel(self.library_path)
        for what_ in self.what:
            models_ = lib.load_model(what_)
            if not isinstance(models_, Sequence):
                models_ = [models_]
            yield from models_

    def plot_2d_residuals(
        self, /, close_figure: bool = True, **kwargs
    ) -> None:
        """Plot 2D residuals for each selected model.

        Parameters
        ----------
        close_figure: bool, optional
            Whether to close the figure after saving (default: True)

        :see also: imshow_binned_statistic_2d
        """
        print("Plotting 2D residuals...")
        # plot 2D residuals per model
        labels = {
            "log10(teff)": r"log$_{10}$T$_{eff}$ [K]",
            "A0": r"A$_0$ [mag]",
            "logg": r"log g [dex]",
            "R0": r"R$_0$ [mag]",
            "feh": r"[Fe/H] [dex]",
            "median(delta_kg)": r"Median Predicted - True $A_x / A_0$ [mag]",
            "mean(delta_kg)": r"Mean Predicted - True $A_x / A_0$ [mag]",
        }

        kwargs_ = {
            "statistic": "mean",
            "cmap": "RdBu",
            "labels": labels,
        }

        kwargs_.update(kwargs)

        # prep data
        query = 'passband == "{pb}" & A0 > 0'

        for model in self.models():
            pb = str(model.name)  # pyright: ignore
            # zpt = self._vega_zpt.get(pb, 0.0)
            df = self.data.query(query.format(pb=pb))
            kg_pred = model.predict(df)
            df = df.assign(
                kg_pred=kg_pred,
                kg_true=df["Ax"] / df["A0"],
                delta_kg=kg_pred - df["Ax"] / df["A0"],
            )
            _, axes = plt.subplots(1, 4, figsize=(14, 3))
            quantiles = df["delta_kg"].quantile([0.01, 0.99])
            if "vmin" not in kwargs:
                kwargs_["vmin"] = -np.max(np.abs(quantiles))
            if "vmax" not in kwargs:
                kwargs_["vmax"] = np.max(np.abs(quantiles))
            imshow_binned_statistic_2d(
                df, "log10(teff)", "A0", "delta_kg", ax=axes[0], **kwargs_
            )
            axes[0].invert_xaxis()

            imshow_binned_statistic_2d(
                df, "log10(teff)", "logg", "delta_kg", ax=axes[1], **kwargs_
            )
            axes[1].invert_xaxis()
            axes[1].invert_yaxis()

            imshow_binned_statistic_2d(
                df, "A0", "R0", "delta_kg", ax=axes[2], **kwargs_
            )

            imshow_binned_statistic_2d(
                df, "A0", "feh", "delta_kg", ax=axes[3], **kwargs_
            )

            plt.setp(plt.gcf().get_axes()[-4:-1], visible=False)

            # fig.suptitle(title)
            ax = plt.gcf().get_axes()[0]
            ax.text(
                -0.3,
                0.5,
                pb,
                transform=ax.transAxes,
                rotation=90,
                va="center",
                fontweight="bold",
            )

            plt.tight_layout()
            plt.savefig(f"model_characteristics_2_{pb.replace('/', '_')}.png")
            print(
                f"   - Saved model_characteristics_2_{pb.replace('/', '_')}.png"
            )
            if close_figure:
                plt.close()

    def plot_1d_residuals(
        self, /, close_figure: bool = True, **kwargs
    ) -> None:
        """Plot 1D residuals for each selected model.

        Parameters
        ----------
        close_figure: bool, optional
            Whether to close the figure after saving (default: True)

        :see also: plot_quick_diagnostic_1d
        """
        print("Plotting 1D residuals...")
        query = 'passband == "{pb}" & A0 > 0'
        for model in self.models():
            pb = str(model.name)  # pyright: ignore
            df = self.data.query(query.format(pb=pb))
            zpt = self._vega_zpt.get(pb, 0.0)
            plot_quick_diagnostic_1d(model, df, mag_zpt=zpt)

            ax = plt.gcf().get_axes()[0]
            ax.text(
                -0.4,
                0.5,
                model.name,  # pyright: ignore
                transform=ax.transAxes,
                rotation=90,
                va="center",
                fontweight="bold",
            )
            plt.savefig(f"model_characteristics_1_{pb.replace('/', '_')}.png")
            print(
                f"   - Saved model_characteristics_1_{pb.replace('/', '_')}.png"
            )
            if close_figure:
                plt.close()


def generate_small_kurucz_testgrid(
    name: str,
    passbands: list[str],
    model_pattern: str = "models/Kurucz2003all/*.fl.dat.txt",
    extinction_curve: str = "F99",
) -> pathlib.Path:
    """Generate a small grid of photometric data for testing purposes.

    This grid is based on Kurucz (2003) atmospheric models and uses the
    specified passbands. The grid covers a limited range of A0 and R0 values.
    It should take ~3 min to generate on a good machine.
    """
    # helps to robustly define grid parameters
    grid_parameters = GridParameters(
        model_pattern=model_pattern,
        pbset=passbands,
        atmosphere_name="Kurucz (ODFNEW/NOVER 2003)",
        atmosphere_shortname="kurucz",
        extinction_curve=extinction_curve,
        apfields=[
            "teff",
            "logg",
            "feh",
            "alpha",
        ],  # no additional output parameters
        n_jobs=10,
        A0=np.array([0.01, 0.1, 1, 2, 4, 6, 8, 10]),
        R0=np.array([2.3, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1]),
    )
    grid_output_path = pathlib.Path(
        f"{name}_{grid_parameters.atmosphere_shortname}_{grid_parameters.extinction_curve}_small_a0r0_grid.ecsv"
    )

    # run grid generation
    grid = grid_parameters.generate_grid(grid_output_path)

    print(
        f"Generated small grid at {grid_output_path} with {len(grid)} entries."
    )
    return grid_output_path


def plot_quick_diagnostic_1d(
    model: models.BaseModel,
    df: pd.DataFrame,
    mag_zpt: float = 0.0,
    ylim: tuple[float, float] | None = None,
):
    """Plot Δk(X) = A(X) / A0  predicted - true as a function of A0, mag0, and Teff
    Plots are computed for the small precomputed model grid provided with Dustapprox.
    """
    # values Ax / A0
    kg_pred = model.predict(df)
    df = df.assign(
        kg_pred=kg_pred,
        kg_true=df["Ax"] / df["A0"],
        delta_kg=kg_pred - df["Ax"] / df["A0"],
    )

    cmap = mpl.colormaps["inferno_r"]
    kwargs_all = dict(
        rasterized=True,
        edgecolor="None",
        cmap=cmap,
        c=df["A0"],
        s=6,
        alpha=0.6,
    )
    noise = np.random.uniform(-0.5, 0.5, size=len(df))

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    ax = axes[0]
    sc = ax.scatter(
        df["A0"] + 0.3 * noise, df["kg_pred"] - df["kg_true"], **kwargs_all
    )
    plt.colorbar(sc, label="A0 [mag]")
    grps = df.groupby("A0")[["delta_kg"]]
    grps.mean().plot(
        y="delta_kg", ax=ax, color="k", lw=2, label="Mean prediction error"
    )
    a, b = grps.quantile(0.16), grps.quantile(0.84)
    ax.fill_between(
        a.index, a["delta_kg"], b["delta_kg"], color="gray", alpha=0.2
    )
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel("A0 [mag]")
    ax.set_ylabel("Predicted - True $A_x / A_0$ [mag]")
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    sc = ax.scatter(
        df["mag0"] - mag_zpt, df["kg_pred"] - df["kg_true"], **kwargs_all
    )
    plt.colorbar(sc, label="A0 [mag]")
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel(r"M$_G$ [mag]")
    ax.set_ylabel("Predicted - True $A_x / A_0$ [mag]")

    ax = axes[2]
    sc = ax.scatter(
        np.log10(df["teff"]), df["kg_pred"] - df["kg_true"], **kwargs_all
    )
    plt.colorbar(sc, label="A0 [mag]")
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel(r"log$_{10}$T$_{eff}$ [K]")
    ax.set_ylabel("Predicted - True $A_x / A_0$ [mag]")

    ax = axes[3]
    sc = ax.scatter(
        df["R0"] + 0.2 * noise, df["kg_pred"] - df["kg_true"], **kwargs_all
    )
    cbar = plt.colorbar(sc, label="A0 [mag]")
    if cbar.solids:
        cbar.solids.set_alpha(1)
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel(r"R0")
    ax.set_ylabel("Predicted - True $A_x / A_0$ [mag]")

    plt.setp(fig.get_axes()[-axes.size : -1], visible=False)
    if ylim is None:
        quantiles = df["delta_kg"].quantile([0.01, 0.99])
        # ylim = -np.max(np.abs(quantiles)), np.max(np.abs(quantiles))
        ylim = quantiles[0.01], quantiles[0.99]
    plt.setp(axes, ylim=ylim)

    plt.tight_layout()


def imshow_binned_statistic_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    what: str,
    bins_x: np.ndarray | None = None,
    bins_y: np.ndarray | None = None,
    statistic: str = "median",
    ax=None,
    labels: dict[str, str] = {},
    **imshow_kwargs,
):
    """Plot binned statistic 2d"""
    x_ = df.eval(x)
    y_ = df.eval(y)
    w_ = df.eval(what)
    # get bins
    if bins_x is None:
        x_vals = np.sort(x_.unique())  # pyright: ignore
        x_vals[-1] += 1e-3
        x_vals[0] -= 1e-3
        bins_x = (x_vals[1:] + x_vals[:-1]) / 2
        bins_x = np.concatenate(
            [
                [x_vals[0] - 0.5 * (bins_x[0] - x_vals[0])],
                bins_x,
                [x_vals[-1] + 0.5 * (x_vals[-1] - bins_x[-1])],
            ]
        )
    if bins_y is None:
        y_vals = np.sort(y_.unique())  # pyright: ignore
        y_vals[-1] += 1e-3
        y_vals[0] -= 1e-3
        bins_y = (y_vals[1:] + y_vals[:-1]) / 2
        bins_y = np.concatenate(
            [
                [y_vals[0] - 0.5 * (bins_y[0] - y_vals[0])],
                bins_y,
                [y_vals[-1] + 0.5 * (y_vals[-1] - bins_y[-1])],
            ]
        )

    # get statistics
    bin_means, xedges, yedges, _ = binned_statistic_2d(
        x_,
        y_,
        w_,
        statistic=statistic,
        bins=[bins_x, bins_y],  # pyright: ignore
    )

    # plot
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(
        bin_means.T,
        origin="lower",
        extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
        aspect="auto",
        interpolation="nearest",
        **imshow_kwargs,
    )

    plt.colorbar(im).set_label(
        labels.get(f"{statistic}({what})", f"{statistic}({what})")
    )
    ax.set_xlabel(labels.get(x, x))
    ax.set_ylabel(labels.get(y, y))


def run_model_characteristics(
    *,
    passbands: str = "Gaia",
    extinction_curve: str = "F99",
    plot_1d: bool = True,
    plot_2d: bool = True,
    close_figure: bool = True,
):
    """Run the model characteristics plots for a given set of passbands and extinction curve.

    This function generates a small photometric test grid if not already
    existing, and then produces 1D and/or 2D residual plots for the selected
    passbands.

    Parameters
    ----------
    passbands: str
        Name of the passband set to use (e.g., "Gaia", "Generic"), this
        corresponds to :func:`dustapprox.models.PrecomputedModel.find` passband
        argument.
    extinction_curve: str
        Name of the extinction curve to use (e.g., "F99", "G23"), this corresponds to
        :func:`dustapprox.models.PrecomputedModel.find` extinction argument and
        :class:`dustapprox.tools.generate_model.GridParameters` extinction_curve
        field.
    plot_1d: bool
        Whether to generate 1D residual plots (default: True).
    plot_2d: bool
        Whether to generate 2D residual plots (default: True).
    close_figure: bool
        Whether to close figures after saving (default: True).
    """
    # prepare testdata path
    testdata_path = pathlib.Path(
        f"{passbands.lower()}_kurucz_{extinction_curve}_small_a0r0_grid.ecsv"
    )
    r = ModelCharacteristicsRunner(
        library_path="precomputed",
        library_selection={"passband": passbands},
        testdata_path=str(testdata_path),
    )

    # generate small grid if not existing
    if not testdata_path.exists():
        print(f"Generating small test grid at {testdata_path}...")
        generate_small_kurucz_testgrid(
            name=passbands.lower(),
            passbands=[name.replace("_", "/", 1) for name in r.passbands()],
            extinction_curve=extinction_curve,
        )

    # set vega for zeropoints
    # use an approximate Kurucz model for Vega
    # alpha_lyr_mod_003 is using Kurucz 9550K Vega spectrum T/g=9550/3.95
    # we use the nearest Kurucz model available: feh=-0.5, logg=3.5, teff=9500K
    # This is an approximation for the plots
    vega_spec_approx = svo.SVOSpectrum(
        "models/Kurucz2003all/fm05at9500g35k2odfnew.fl.dat.txt"
    )
    r.set_vega_zeropoints(vega_spec_approx)

    # plot 2d residuals
    if plot_2d:
        r.plot_2d_residuals(close_figure=close_figure)

    # plot 1d residuals
    if plot_1d:
        r.plot_1d_residuals(close_figure=close_figure)

    print("Done.")
