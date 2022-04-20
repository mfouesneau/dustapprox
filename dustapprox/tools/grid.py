""" Generate a grid of models with extinction from an atmosphere library

Example of script that produces a grid of dust attenuated stellar models from an
atmosphere library.

This example can run in parallel on multiple processes or cores.

.. seealso::

    :func:`compute_photometric_grid`
"""
import numpy as np
import pandas as pd
from typing import Sequence
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
from pyphot.astropy import UnitFilter
from dustapprox.io import svo
from .parallel import tqdm_joblib
from ..extinction import ExtinctionLaw, F99


def _parallel_task(fname: str, apfields: Sequence[str],
                   passbands: Sequence[UnitFilter],
                   extc: ExtinctionLaw,
                   Rv: Sequence[float],
                   Av: Sequence[float]) -> pd.DataFrame:
    """ Task per spectrum

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
    extc: ExtinctionLaw
        Extinction law to apply
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
    from dustapprox.io import svo

    data = svo.spectra_file_reader(fname)

    # extract model relevant information
    lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
    lamb = data['data']['WAVELENGTH'].values * lamb_unit
    flux = data['data']['FLUX'].values * flux_unit
    apvalues = [data[k]['value'] for k in apfields]

    # columns = ['teff', 'logg', 'feh', 'alpha',
    columns = (list(apfields) +
               ['passband', 'mag0', 'mag',
                'A0', 'R0', 'Ax'])
    logs = []

    for rv_val in Rv:
        # wavelength definition varies between models
        alambda_per_av = extc(lamb, 1.0, Rv=rv_val)

        # Dust magnitudes
        for pk in passbands:
            # Dust free values
            mag0 = -2.5 * np.log10(pk.get_flux(lamb, flux).value)
            # possibly we redo av[0] = 0, but it's cheap for consistency gain
            for av_val in Av:
                new_flux = flux * np.exp(- alambda_per_av * av_val)
                mag = -2.5 * np.log10(pk.get_flux(lamb, new_flux).value)
                delta = (mag - mag0)
                logs.append(apvalues + [pk.name, mag0, mag, av_val, rv_val, delta])
                pass
    logs = pd.DataFrame.from_records(logs, columns=columns)
    return logs


def compute_photometric_grid(sources='models/Kurucz2003all/*.fl.dat.txt',
                             n_jobs=1, verbose=0):
    """ Run the computations of the photometric grid in parallel

    Parameters
    ----------
    sources: str
        pattern of atmospehric models to process
        (using glob syntax)
    n_jobs: int
        number of parallel processes to run (default: 1, -1 for as many as CPUs)
    verbose: int
        verbosity level (default: 0)

    Returns
    -------
    pd.DataFrame
        Dataframe with the photometric values for each passband
    """
    # Load relevant passbands
    which_filters = ['GAIA/GAIA3.G', 'GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp',
                     'SLOAN/SDSS.u', 'SLOAN/SDSS.g', 'SLOAN/SDSS.r', 'SLOAN/SDSS.i', 'SLOAN/SDSS.z',
                     '2MASS/2MASS.J', '2MASS/2MASS.H', '2MASS/2MASS.Ks',
                     'WISE/WISE.W1', 'WISE/WISE.W2', 'WISE/WISE.W3', 'WISE/WISE.W4',
                     'GALEX/GALEX.FUV', 'GALEX/GALEX.NUV',
                     'Generic/Johnson.U', 'Generic/Johnson.B', 'Generic/Johnson.V',
                     'Generic/Cousins.R', 'Generic/Cousins.I',
                     'Generic/Bessell_JHKLM.J', 'Generic/Bessell_JHKLM.H', 'Generic/Bessell_JHKLM.K',]
    passbands = svo.get_svo_passbands(which_filters)


    # Extinction
    extc = F99()
    Rv = np.array([3.1,])
    Av = np.sort(np.hstack([[0.01], np.arange(0.1, 10.01, 0.1)]))

    models = glob(sources)
    apfields = 'teff', 'logg', 'feh', 'alpha'

    with tqdm_joblib(tqdm(desc="Grid", total=len(models))):
        res = Parallel(n_jobs=n_jobs, verbose=verbose, prefer='processes')(
            delayed(_parallel_task)(fname, apfields, passbands, extc, Rv, Av) for fname in models
            )
    return pd.concat(res)