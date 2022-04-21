"""We provide various modeling schemes for extinction in a given photometric band.

.. todo::

    * add script to generate grid of models
    * add polynomial training.
    * compare literature values to ours.
"""

from pkg_resources import resource_filename
from glob import glob
from typing import Union, Sequence

from ..io import ecsv
from .polynomial import PolynomialModel
from .basemodel import _BaseModel


_DATA_PATH_ = resource_filename('dustapprox', 'data/precomputed')

kinds = {'polynomial':  PolynomialModel,
         }

class PrecomputedModel:
    """ Access to precomputed models

    .. code-block:: python

        from dustapprox.models import PrecomputedModel
        lib = PrecomputedModel()
        # search for GALEX passbands if present
        r = lib.find(passband='galex')
        print(r)
        # load both available models
        models = []
        for source in r.values():
            models.extend([lib.load_model(r, passband=pbname) for pbname in source['passbands']])

    .. code-block:: text
        :caption: result from :func:`PrecomputedModel.find`

        [{'atmosphere': {'source': 'Kurucz (ODFNEW/NOVER 2003)',
            'teff': [3500.0, 50000.0],
            'logg': [0.0, 5.0],
            'feh': [-4, 0.5],
            'alpha': [0, 0.4]},
          'extinction': {'source': 'Fitzpatrick (1999)', 'R0': 3.1, 'A0': [0, 10]},
          'comment': ['teffnorm = teff / 5040', 'predicts kx = Ax / A0'],
          'model': {'kind': 'polynomial',
          'degree': 3,
          'interaction_only': False,
          'include_bias': True,
          'feature_names': ['A0', 'teffnorm']},
          'passbands': ['GALEX_GALEX.FUV', 'GALEX_GALEX.NUV'],
          'filename': 'dustapprox/data/precomputed/polynomial/f99/kurucz/kurucz_f99_a0_teff.ecsv'}]

    .. code-block:: text
        :caption: result when loading models with from :func:`PrecomputedModel.load_model`

        [PolynomialModel: GALEX_GALEX.FUV
        <dustapprox.models.polynomial.PolynomialModel object at 0x12917b6a0>
            from: A0, teffnorm   polynomial degree: 3,
        PolynomialModel: GALEX_GALEX.NUV
        <dustapprox.models.polynomial.PolynomialModel object at 0x129170820>
            from: A0, teffnorm   polynomial degree: 3]
    """

    def __init__(self, location=None):
        """ Constructor """
        if location is None:
            location = _DATA_PATH_
        self._info = None
        self.location = location

    def get_models_info(self, glob_pattern='/**/*.ecsv') -> Sequence[dict]:
        """ Retrieve the information for all models available and files """
        if self._info is not None:
            return self._info
        location = self.location
        lst = glob(f'{location:s}{glob_pattern:s}', recursive=True)

        info = []
        for fname in lst:
            info.append(self._get_file_info(fname))
        self._info = info
        return info

    def _get_file_info(self, fname:str) -> dict:
        """ Extract information from a file """
        info = {}
        where = fname.replace(_DATA_PATH_, '')
        df = ecsv.read(fname)
        info = df.attrs.copy()
        info['passbands'] = list(df['passband'].values)
        info['filename'] = fname
        return info

    def find(self, passband=None, extinction=None, atmosphere=None, kind=None) -> Sequence[dict]:
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

        Returns
        -------

        """
        info = self.get_models_info()

        results = []
        for value in info:
            if passband is not None and passband.lower() not in ' '.join(value['passbands']).lower():
                continue
            if extinction is not None and extinction.lower() not in value['extinction']['source'].lower():
                continue
            if atmosphere is not None and atmosphere.lower() not in value['atmosphere']['source'].lower():
                continue
            if kind is not None and kind.lower() not in value['model']['kind'].lower():
                continue
            content = value.copy()
            if passband is not None:
                content['passbands'] = [pk for pk in content['passbands'] if passband.lower() in pk.lower()]
            results.append(content)
        return results

    def load_model(self, fname: Union[str, dict],
                   passband: str = None):
        """ Load a model from a file or description (:func:`PrecomputedModel.find`)

        Parameters
        ----------
        fname : str or dict
            The filename of the model to be loaded or a description of the model
            returned by :func:`PrecomputedModel.find`

        passband : str
            The passband to be loaded. If `None`, loads all available passband models.

        Returns
        -------
        model : :class:`dustapprox.models.polynomial.PolynomialModel`
        """

        if isinstance(fname, dict):
            fname_ = fname['filename']
            info = fname
        else:
            fname_ = fname
            info = self._get_file_info(fname_)

        model_kind = info['model']['kind']

        if passband is None:
            return [self.load_model(fname, pbname) for pbname in info['passbands']]

        try:
            return kinds[model_kind].from_file(fname_, passband=passband)
        except KeyError:
            raise NotImplementedError(f'Model kind {model_kind} not implemented.')