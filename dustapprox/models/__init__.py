"""We provide various modeling schemes for extinction in a given photometric band.

.. todo::

    * add script to generate grid of models
    * add polynomial training.
    * move literature models to here.
    * compare literature values to ours.
"""

from pkg_resources import resource_filename
from glob import glob
from typing import Union

from tables import UnImplemented
from ..io import ecsv
from .polynomial import PolynomialModel

_DATA_PATH_ = resource_filename('dustapprox', 'data/precomputed')

kinds = {'polynomial':  PolynomialModel,
         }


class PrecomputedModel:

    def __init__(self, location=None):
        """ Constructor """
        if location is None:
            location = _DATA_PATH_
        self._info = None
        self.location = location

    def get_models_info(self, glob_pattern='/**/*.ecsv') -> dict:
        """ Retrieve the information for all models available and files """
        if self._info is not None:
            return self._info
        location = self.location
        lst = glob(f'{location:s}{glob_pattern:s}', recursive=True)

        info = {}
        for fname in lst:
            info.update(self._get_file_info(fname))
        self._info = info
        return info

    def _get_file_info(self, fname:str) -> dict:
        """ Extract information from a file """
        info = {}
        where = fname.replace(_DATA_PATH_, '')
        df = ecsv.read(fname)
        info[where] = df.attrs.copy()
        info[where]['passbands'] = list(df['passband'].values)
        info[where]['filename'] = fname
        return info

    def find(self, passband=None, extinction=None, atmosphere=None, kind=None) -> dict:
        """ Find all the computed models that match the given parameters.

        The search is case insentive and returns all matches.

        Parameters
        ----------
        passband : str
            The passband to be used.
        extinction : str
            The extinction model to be used. (e.g., 'Fitzpatrick')
        atmosphere : str
            The atmosphere model to be used. (e.g., 'kurucz')
        kind : str
            The kind of model to be used (e.g., polynomial).
        """
        info = self.get_models_info()

        results = {}
        for key, value in info.items():
            if passband is not None and passband.lower() not in ' '.join(value['passbands']).lower():
                continue
            if extinction is not None and extinction.lower() not in value['extinction']['source'].lower():
                continue
            if atmosphere is not None and atmosphere.lower() not in value['atmosphere']['source'].lower():
                continue
            if kind is not None and kind.lower() not in value['model']['kind'].lower():
                continue
            results[key] = value.copy()
            if passband is not None:
                results[key]['passbands'] = [pk for pk in results[key]['passbands'] if passband.lower() in pk.lower()]
        return results

    def load_model(self, fname: Union[str, dict], passband: str = None):

        if isinstance(fname, dict):
            key = list(fname.keys())[0]
            fname_ = fname[key]['filename']
            info = fname
        else:
            fname_ = fname
            info = self._get_file_info(fname_)

        try:
            info = info[fname_.replace(_DATA_PATH_, '')]
        except KeyError:
            pass

        model_kind = info['model']['kind']

        if passband is None:
            print("Available passbands:")
            print(info['passbands'])
            raise ValueError("You need to specify a passband to load the model.")

        try:
            return kinds[model_kind].from_file(fname_, passband=passband)
        except KeyError:
            raise NotImplementedError(f'Model kind {model_kind} not implemented.')