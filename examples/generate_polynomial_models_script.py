"""Tools to generate pre-computed models from atmosphere grids
This module provides functions to generate photometric grids from atmosphere models,
train polynomial models on these grids, and export the trained models to ECSV files.
"""

import pathlib
from importlib import resources

import numpy as np
from pyphot import Filter
from pyphot.libraries import Ascii_Library

from dustapprox.models import PrecomputedModel
from dustapprox.tools.generate_model import (
    GridParameters,
    train_polynomial_model,
)


def get_gaia_c1_filters() -> list[Filter]:
    """Load Gaia DR3 C1 passbands from pyphot's Ascii_Library"""
    DATA_PATH = str(resources.files("dustapprox").joinpath("data", "Gaia2"))
    lib = Ascii_Library(f"{DATA_PATH}", glob_pattern="*csv")
    fnames = [f for f in lib.content if f.endswith(".csv")]
    filters = lib.load_filters(fnames)
    return filters


gaia_dr3_filters = [
    "GAIA/GAIA3.G",
    "GAIA/GAIA3.Gbp",
    "GAIA/GAIA3.Grp",
]
sloan_filters = [
    "SLOAN/SDSS.u",
    "SLOAN/SDSS.g",
    "SLOAN/SDSS.r",
    "SLOAN/SDSS.i",
    "SLOAN/SDSS.z",
]
twomass_filters = [
    "2MASS/2MASS.J",
    "2MASS/2MASS.H",
    "2MASS/2MASS.Ks",
]
wise_filters = [
    "WISE/WISE.W1",
    "WISE/WISE.W2",
    "WISE/WISE.W3",
    "WISE/WISE.W4",
]
galex_filters = [
    "GALEX/GALEX.FUV",
    "GALEX/GALEX.NUV",
]
generic_filters = [
    "Generic/Johnson.U",
    "Generic/Johnson.B",
    "Generic/Johnson.V",
    "Generic/Cousins.R",
    "Generic/Cousins.I",
    "Generic/Bessell_JHKLM.J",
    "Generic/Bessell_JHKLM.H",
    "Generic/Bessell_JHKLM.K",
]

filter_set = dict(
    (
        ("gaia_dr3", gaia_dr3_filters),
        ("sloan", sloan_filters),
        ("twomass", twomass_filters),
        ("wise", wise_filters),
        ("galex", galex_filters),
        ("generic", generic_filters),
        ("gaiac1", get_gaia_c1_filters()),
    )
)


def main_example() -> PrecomputedModel:
    """Generate a pre-computed polynomial model library end-to-end"""

    name = "gaia"
    features = ["teff", "A0", "R0"]

    # helps to robustly define grid parameters
    grid_parameters = GridParameters(
        model_pattern="models/Kurucz2003all/*.fl.dat.txt",
        pbset=gaia_dr3_filters,
        atmosphere_name="Kurucz (ODFNEW/NOVER 2003)",
        atmosphere_shortname="kurucz",
        extinction_curve="F99",
        apfields=[
            "teff",
            "logg",
            "feh",
            "alpha",
        ],  # no additional output parameters
        n_jobs=-1,
        A0=np.sort(np.hstack([[0.01], np.arange(0.1, 20.01, 0.1)])),
        R0=np.array([2.3, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1]),
    )

    grid_output_path = pathlib.Path(
        f"precomputed/grids/{name}_{grid_parameters.atmosphere_shortname}_{grid_parameters.extinction_curve}_grid.ecsv"
    )

    # run grid generation
    grid = grid_parameters.generate_grid(grid_output_path)

    # Train models on the generated grid
    model_output_path = pathlib.Path(
        f"precomputed/polynomial/{grid_parameters.extinction_curve}/{grid_parameters.atmosphere_shortname}/{name}_{grid_parameters.atmosphere_shortname}_{grid_parameters.extinction_curve}_{'_'.join(features)}.ecsv"
    )
    model_kwargs = {"degree": 3}

    train_polynomial_model(grid, model_output_path, features, **model_kwargs)

    lib = PrecomputedModel(str(model_output_path.parent))
    return lib
