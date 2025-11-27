"""Generate a grid of models with extinction from an atmosphere library

Example of script that produces a grid of dust attenuated stellar models from an
atmosphere library.

This example can run in parallel on multiple processes or cores.

.. seealso::

    :func:`compute_photometric_grid`
"""

from glob import glob
from typing import List, Optional, Sequence, Union, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from pyphot import Filter
from tqdm import tqdm

from ..extinction import BaseExtRvModel
from ..io import svo

DEFAULT_FILTERS = [
    "GAIA/GAIA3.G",
    "GAIA/GAIA3.Gbp",
    "GAIA/GAIA3.Grp",
    "SLOAN/SDSS.u",
    "SLOAN/SDSS.g",
    "SLOAN/SDSS.r",
    "SLOAN/SDSS.i",
    "SLOAN/SDSS.z",
    "2MASS/2MASS.J",
    "2MASS/2MASS.H",
    "2MASS/2MASS.Ks",
    "WISE/WISE.W1",
    "WISE/WISE.W2",
    "WISE/WISE.W3",
    "WISE/WISE.W4",
    "GALEX/GALEX.FUV",
    "GALEX/GALEX.NUV",
    "Generic/Johnson.U",
    "Generic/Johnson.B",
    "Generic/Johnson.V",
    "Generic/Cousins.R",
    "Generic/Cousins.I",
    "Generic/Bessell_JHKLM.J",
    "Generic/Bessell_JHKLM.H",
    "Generic/Bessell_JHKLM.K",
]


def _parallel_task(
    fname: str,
    apfields: Sequence[str],
    passbands: Sequence[Filter],
    extinction_curve: Union[str, BaseExtRvModel],
    R0: Sequence[float],
    A0: Sequence[float],
) -> pd.DataFrame:
    """Task per spectrum

    This task reads in a spectrum from `fname`
    applies the extinction dimension to it and extract the relevant photometric values.

    Parameters
    ----------
    fname: str
        Filename of the spectrum to process (assuming an svo formatted file)
    apfields: Sequence[str]
        Which fields to extract from the model atmospheres
    passbands: Sequence[pyphot.astropy.UnitFilter]
        List of passbands to extract photometry from
    extinction_curve: Union[str, BaseExtRvModel]
        Extinction curve to apply
    Rv: Sequence[float]
        List of Rv values to apply
    Av: Sequence[float]
        List of Av values to apply

    Returns
    -------
    pd.DataFrame
        Dataframe with the photometric values for each passband
        this includes the stellar parameters (Teff, logg, [Fe/H]), A0, R0,
        intrinsic and reddened photometry, and the relative difference (Ax)
    """
    # imports retricted to subprocesses
    import numpy as np
    import pandas as pd

    from dustapprox.extinction import evaluate_extinction_model
    from dustapprox.io import svo

    data = svo.spectra_file_reader(fname)

    # extract model relevant information
    λ_unit, flux_unit = svo.get_svo_sprectum_units(data)
    λ = data["data"]["WAVELENGTH"].values * λ_unit
    flux = data["data"]["FLUX"].values * flux_unit
    apvalues = [data[k]["value"] for k in apfields]

    columns = list(apfields) + ["passband", "mag0", "mag", "A0", "R0", "Ax"]
    logs = []

    for r0_val in R0:
        # wavelength definition varies between models
        Aλ_per_A0 = evaluate_extinction_model(
            extinction_curve, λ, A0=1.0, R0=r0_val, extrapolate=True
        )

        # Dust magnitudes
        for pk in passbands:
            # Dust free values
            mag0 = -2.5 * np.log10(pk.get_flux(λ, flux).value)  # pyright: ignore / dumb
            # possibly we redo a0[0] = 0, but it's cheap for consistency gain
            for a0_val in A0:
                new_flux = flux * np.exp(-Aλ_per_A0 * a0_val)
                mag = -2.5 * np.log10(pk.get_flux(λ, new_flux).value)  # pyright: ignore / dumb
                delta = mag - mag0
                logs.append(apvalues + [str(pk.name), mag0, mag, a0_val, r0_val, delta])
    logs = pd.DataFrame.from_records(logs, columns=columns).astype(
        {
            "passband": "string",
            "mag0": "float32",
            "mag": "float32",
            "A0": "float32",
            "R0": "float32",
            "Ax": "float32",
        }
    )
    return logs


def compute_photometric_grid(
    sources: str = "models/Kurucz2003all/*.fl.dat.txt",
    which_filters: Optional[Union[Sequence[str], Sequence[Filter]]] = None,
    extinction_curve: Optional[Union[str, BaseExtRvModel]] = "F99",
    A0: Optional[npt.NDArray] = None,
    R0: Optional[npt.NDArray] = None,
    apfields: Optional[Sequence[str]] = None,
    n_jobs: int = 1,
    verbose: int = 0,
    atmosphere_name: Optional[str] = None,
    **kwargs,
):
    """Run the computations of the photometric grid in parallel

    Parameters
    ----------
    sources: str
        pattern of atmospehric models to process
        (using glob syntax)
    which_filters: Sequence[Union[str, UnitFilter]], optional
        list of filter names or UnitFilter instances to compute photometry for.
        If None, use a default set of filters.
    extinction_curve: Union[str, BaseExtRvModel], optional, default="F99"
        Name of the extinction curve to use from dust_extinction
        or an instance of BaseExtRvModel.
    A0: npt.NDArray, optional
        array of A0 values to compute (default: from 0.01 to 20.0 in steps of 0.1)
    R0: npt.NDArray, optional
        array of R0 values to compute (default: [2.3, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1])
    apfields: Sequence[str], optional
        list of atmospheric parameter field names to extract from the atmosphere files.
        If None, use ('teff', 'logg', 'feh', 'alpha')
    n_jobs: int
        number of parallel processes to run (default: 1, -1 for as many as CPUs)
    verbose: int
        verbosity level (default: 0)
    atmosphere_name: str, optional
        Name of the atmosphere model set used. If None, use the sources pattern.
    **kwargs: keyword arguments
        Additional metadata to attach to the resulting DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with the photometric values for each passband
    """

    # Load relevant passbands
    which_filters = which_filters or DEFAULT_FILTERS
    if not isinstance(which_filters[0], Filter):
        passbands = svo.get_svo_passbands(cast(List[str], which_filters))
    else:
        passbands = which_filters

    # Extinction
    extinction_curve_name = (
        extinction_curve
        if isinstance(extinction_curve, str)
        else extinction_curve.__class__.__name__
    )

    R0 = R0 or np.array([2.3, 2.6, 3.1, 3.6, 4.1, 4.6, 5.1])
    A0 = A0 or np.sort(np.hstack([[0.01], np.arange(0.1, 20.01, 0.1)]))
    apfields = apfields or ("teff", "logg", "feh", "alpha")

    # get model file list
    models = glob(sources)

    # trick to get progress bar with joblib+tqdm
    res = [
        r
        for r in tqdm(
            Parallel(
                return_as="generator",
                n_jobs=n_jobs,
                verbose=verbose,
                prefer="processes",
            )(
                delayed(_parallel_task)(
                    fname, apfields, passbands, extinction_curve, R0, A0
                )
                for fname in models
            ),
            total=len(models),
            desc="Grid",
        )
    ]
    df = pd.concat(res)
    stats = df[list(apfields) + ["R0", "A0"]].agg(["min", "max"])

    meta = {
        "extinction": {"source": extinction_curve_name},
        "atmosphere": {"source": atmosphere_name or sources},
    }
    meta.update(kwargs)

    for key in ("R0", "A0"):
        meta["extinction"][key] = [float(stats[key]["min"]), float(stats[key]["max"])]
    for key in apfields:
        meta["atmosphere"][key] = [float(stats[key]["min"]), float(stats[key]["max"])]
    df.attrs.update(**meta)
    return df
