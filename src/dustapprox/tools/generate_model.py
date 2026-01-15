"""Tools to generate pre-computed models from atmosphere grids

This module provides functions to generate photometric grids from atmosphere models,
train polynomial models on these grids, and export the trained models to ECSV files.
"""

import pathlib
from dataclasses import dataclass
from typing import Dict, List, LiteralString, Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from pyphot import Filter

from ..extinction import BaseExtRvModel
from ..io import ecsv
from ..models import PrecomputedModel, polynomial
from ..models.basemodel import BaseModel
from . import grid

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


@dataclass
class GridParameters:
    """Parameters for generating a photometric grid
    Attributes
    ----------
    model_pattern : str
        The pattern to identify the atmosphere models.
    pbset : List[str] | List[Filter]
        The list of passbands to compute.
    atmosphere_name : Optional[str]
        The name of the atmosphere set.
    extinction_curve : Union[str, BaseExtRvModel]
        The extinction curve to use (from dust_extinction).
    A0 : Optional[npt.NDArray]
        Array of A0 values to use.
    R0 : Optional[npt.NDArray]
        Array of R0 values to use.
    apfields : Optional[List[str]]
        Additional atmosphere parameters to include in the output.
    n_jobs : int
        Number of parallel jobs to use (-1 uses all available cores).
    """

    model_pattern: str
    pbset: List[str] | List[Filter]
    atmosphere_name: Optional[str] = None
    atmosphere_shortname: Optional[str] = None
    extinction_curve: Union[str, BaseExtRvModel] = "F99"
    A0: Optional[npt.NDArray] = None
    R0: Optional[npt.NDArray] = None
    apfields: Optional[List[str]] = None
    n_jobs: int = -1

    def generate_grid(
        self, output_path: Union[str, pathlib.Path]
    ) -> pd.DataFrame:
        """Shortcut to generate the photometric grid based on the parameters

        seealso:: :func:`generate_grid`

        Returns
        -------
        pd.DataFrame
            DataFrame with the computed grid
        """
        return generate_grid(
            model_pattern=self.model_pattern,
            grid_fname=output_path,
            pbset=self.pbset,
            atmosphere_name=self.atmosphere_name,
            extinction_curve=self.extinction_curve,
            A0=self.A0,
            R0=self.R0,
            apfields=self.apfields,
            n_jobs=self.n_jobs,
        )

    def copy(self) -> "GridParameters":
        """Create a copy of the current GridParameters instance

        Returns
        -------
        GridParameters
            A new instance of GridParameters with the same attributes.
        """
        return GridParameters(
            model_pattern=self.model_pattern,
            pbset=self.pbset.copy(),
            atmosphere_name=self.atmosphere_name,
            atmosphere_shortname=self.atmosphere_shortname,
            extinction_curve=self.extinction_curve,
            A0=self.A0.copy() if self.A0 is not None else None,
            R0=self.R0.copy() if self.R0 is not None else None,
            apfields=self.apfields.copy()
            if self.apfields is not None
            else None,
            n_jobs=self.n_jobs,
        )


@dataclass
class ModelParameters:
    """Parameters for training a model

    Attributes
    ----------
    output_fname : Union[str, pathlib.Path]
        The output filename for the trained models.
    features : Union[List[str], List[LiteralString]]
        The list of features to use for training.
    model_kwargs: dict
        additional parameters to the model
    """

    output_fname: Union[str, pathlib.Path]
    features: Union[List[str], List[LiteralString]]
    kwargs: Optional[Dict] = None


def generate_grid(
    model_pattern: str,
    grid_fname: Union[str, pathlib.Path],
    pbset: List[str] | List[Filter],
    atmosphere_name: Optional[str] = None,
    extinction_curve: Union[str, BaseExtRvModel] = "F99",
    A0: Optional[npt.NDArray] = None,
    R0: Optional[npt.NDArray] = None,
    apfields: Optional[Sequence[str]] = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Generate the photometric grid from models
    Parameters
    ----------
    model_pattern: str
        Pattern to identify the atmosphere models
    grid_fname: Union[str, pathlib.Path]
        Output filename for the grid
    pbset: List[str]
        List of passbands to compute
    atmosphere_name: Optional[str]
        Name of the atmosphere set
    extinction_curve: Union[str, BaseExtRvModel]
        Extinction curve to use (from dust_extinction)
    A0: Optional[npt.NDArray]
        Array of A0 values to use
    R0: Optional[npt.NDArray]
        Array of R0 values to use
    apfields: Optional[Sequence[str]]
        Additional atmosphere parameters to include in the output
    n_jobs: int
        Number of parallel jobs to use (-1 uses all available cores)

    Returns
    -------
    pd.DataFrame
        DataFrame with the computed grid
    """

    atmosphere_name = (
        atmosphere_name or model_pattern.split("/")[1].split("_")[0]
    )

    # show some info
    print("Computing photometric grid")
    # ensure directory exists
    grid_fname = pathlib.Path(grid_fname)
    grid_fname.parent.mkdir(parents=True, exist_ok=True)
    pbset_names = [f.name if isinstance(f, Filter) else f for f in pbset]
    if not grid_fname.exists():
        print(f"   - Using passbands: {', '.join(pbset_names)}")
        print(f"   - Using model pattern: {model_pattern}")
        print(f"   - Using atmospheres: {atmosphere_name}")
        print(f"   - Using extinction curve: {extinction_curve}")
        if A0 is None:
            print("   - Using default A0 values: [0.01, 0.1, 0.2 ... 20.0]")
        else:
            print(f"   - Using custom A0 values: {A0}")
        if R0 is None:
            print(
                "   - Using default R0 values: [2.3, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1]"
            )
        else:
            print(f"   - Using custom R0 values: {R0}")
        if apfields is not None:
            print(f"   - Using additional parameters: {', '.join(apfields)}")
        else:
            print(
                "   - Using default parameters: ('teff', 'logg', 'feh', 'alpha')"
            )
        r = grid.compute_photometric_grid(
            model_pattern,
            pbset,
            extinction_curve=extinction_curve,
            A0=A0,
            R0=R0,
            apfields=apfields,
            n_jobs=n_jobs,
            atmosphere_name=atmosphere_name,
        )
        print(f"Exporting grid to {grid_fname}.", end="")
        ecsv.write(r, str(grid_fname))
        print(" Done.")
    else:
        print(
            f"   - Existing grid from {grid_fname}. Skipping computation and reloading from file."
        )
        r = cast(pd.DataFrame, ecsv.read(str(grid_fname)))
    return r


def export_trained_model_to_ecsv(
    fname: Union[str, pathlib.Path], models: Sequence[BaseModel], **kwargs
):
    """Export a collection of models
    Parameters
    ----------
    fname: str
        The name of the file to write.
    models: Sequence[_BaseModel]
        The list of models to export.
    **kwargs: keyword arguments
        Additional metadata to attach to the resulting ECSV file.
    Returns
    -------
    None
    """
    data = pd.concat([model.to_pandas() for model in models]).reset_index(
        names=["passband"]
    )
    meta = models[0].to_pandas().attrs
    meta.update(**kwargs)
    pathlib.Path(fname).parent.mkdir(parents=True, exist_ok=True)
    ecsv.write(data, str(fname), **meta)


def train_polynomial_model(
    r: pd.DataFrame,
    output_fname: Union[str, pathlib.Path],
    features: Union[List[str], List[LiteralString]],
    *,
    degree: int = 3,
) -> List[polynomial.PolynomialModel]:
    """Train polynomial models for the given features from the photometric grid
    Parameters
    ----------
    r: pd.DataFrame
        The photometric grid DataFrame.
    output_fname: Union[str, pathlib.Path]
        The output filename for the trained models.
    features: Union[List[str], List[LiteralString]]
        The list of features to use for training.
    degree: int
        The degree of the polynomial to fit.
    Returns
    -------
    List[polynomial.PolynomialModel]
        The list of trained polynomial models.
    """
    print(f"Training polynomial models for features: {', '.join(features)}")
    models = []
    for passband in tqdm(r.passband.unique(), desc="fit"):
        df = r[(r.passband == passband) & (r["A0"] > 0)].copy()
        model = polynomial.PolynomialModel(name=passband).fit(
            df, features, degree=degree
        )
        models.append(model)
    print(f"Exporting grid to {output_fname}.", end="")
    export_trained_model_to_ecsv(output_fname, models)
    print(" done.")
    return models


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
