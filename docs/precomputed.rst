Extinction approximation models
--------------------------------

Extinction coefficients per passbands depend on both the source spectral energy distribution
and on the extinction itself (e.g., Gordon et al., 2016, Jordi et al., 2010).
To first order, the shape of the SED through a given passband determine the mean
photon wavelength and therefore the mean extinction through that passband.  Of
course in practice this also depends on the source spectral features in the
passband and the dust properties.

Please have a look to the following pages for the ingredients we used in our precomputed models

* :doc:`/atmospheres` - The source stellar atmosphere models
* :doc:`/photometry` - The photometric computations
* :doc:`/extinction` - The extinction curves


Precomputed models
------------------

We provide some already pre-computed model approximations for the extinction in various passbands.

:class:`dustapprox.models.PrecomputedModel` provides convenient search and load functions.

* use :class:`dustapprox.models.PrecomputedModel.find` to find available models and associated passbands
    * The search can be on passband, extinction, atmosphere, and model kind. It is caseless and does not need to contain the complete name.

.. code-block:: python
    :caption: examples of searching for models

    from dustapprox.models import PrecomputedModel
    lib = PrecomputedModel()
    lib.find(passband='Gaia')   # returns all models with Gaia passband
    lib.find(passband='galex', atmosphere='Atlas')   # returns nothing (we did not provide Atlas9 atmosphere)
    lib.find(passband='galex', atmosphere='kurucz')  # return only kurucz based models

.. code-block:: text
    :caption: result from :func:`dustapprox.models.PrecomputedModel.find` for `passband="galex"`

    {'/polynomial/f99/kurucz/kurucz_f99_a0_teff.ecsv': {'atmosphere': {'source': 'Kurucz (ODFNEW/NOVER 2003)',
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
        'filename': 'dustapprox/data/precomputed/polynomial/f99/kurucz/kurucz_f99_a0_teff.ecsv'}}


* use :class:`dustapprox.models.PrecomputedModel.load_model` to load any model (for a given passband)


.. code-block:: python
    :caption: example of loading Gaia passband approximations

    from dustapprox.models import PrecomputedModel
    lib = PrecomputedModel()
    r = lib.find(passband='Gaia')
    models = []
    for source in r.values():
        models.extend([lib.load_model(r, passband=pbname) for pbname in source['passbands']])

.. important::
    We currently provide only a limited set of models and approximation methods.
    We plan to expand in the future releases.

    Please contact us if you would like a particular passband (or set of passbands) to be included by default.


.. seealso::

    * list of provided models: :doc:`/precomputed_content`

Generating models
-----------------

Generating a photometric extinction model or approximation requires first that
we have some atmosphere spectral model. We provide some tools associated with the
`SVO Theoretical <spectra: http://svo2.cab.inta-csic.es/theory/newov2/index.php>`_
in :doc:`/atmospheres` (:mod:`dustapprox.io.svo`) but you can also use your own atmosphere models.

Second, we need an extinction presscription. We provide some mean extinction
curves in :doc:`/extinction` (:mod:`dustapprox.extinction`).

Finally, we need passband definitions and functions to do the photometric
calculations.  For the photometry, we use the external package `pyphot
<https://mfouesneau.github.io/pyphot/index.html>`_ a suite to compute synthetic
photometry in flexible ways.  In addition,
:func:`dustapprox.io.svo.get_svo_passbands` interfaces the `SVO Filter Profile
Service <http://svo2.cab.inta-csic.es/theory/fps/index.php>`_, which provides us
with a large collection of passbands. (wrapper from `pyphot`_).

Once we have the above ingredients, we can bring them together to generate a
large collection of photometric extinction values in various bands.


Creating a photometric grid of dust attenuated stars
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To compute an extinction approximation model, we need to first compute the exact
effects of extinction on well known stars when assuming a given extinction curve.

We detail below the steps to do this.

* We first need to get the set of transmission curves that we find relevant for the :doc:`/photometry`.

.. code-block:: python3
   :caption: Get transmission curves from the `SVO Filter Profile Service`_.

   from dustapprox.io import svo
   which_filters = ['GAIA/GAIA3.G', 'GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp']
   passbands = svo.get_svo_passbands(which_filters)


.. code-block:: python3
   :caption: Get the Gaia C1 transmission curves provided with `dustapprox` (see :mod:`dustapprox.literature.c1`)

   from pkg_resources import resource_filename
   from pyphot.astropy import UnitAscii_Library

   where = resource_filename('dustapprox', 'data/Gaia2')
   lib = UnitAscii_Library([where])
   passbands = lib.load_all_filters()

* We set which atmosphere library files we use (note that we do not provide these internally; :doc:`/atmospheres`).

.. code-block:: python3
   :caption: Set the atmosphere models and parameter fields to report

   from glob import glob

   models = glob('models/Kurucz2003all/*.fl.dat.txt')
   apfields = ['teff', 'logg', 'feh', 'alpha']

* We then need to get the set of extinction curves that we find relevant.

.. code-block:: python3
   :caption: Extinction curve and parameter sets

   import numpy as np
   from dustapprox.extinction import F99

   # Extinction
   extc = F99()
   Rv = np.array([3.1,])
   Av = np.arange(0, 10.01, 0.2)


* Finally we loop through the elements and store relevant information (e.g., `apfields`, `Rv`, `Av`, `mag0`, `mag`).


.. code-block:: python3
   :caption: An example of **not optimized** script to generate an extinction grid over all the atmosphere models

   import numpy as np
   import pandas as pd
   from tqdm import tqdm
   from dustapprox.io import svo

   logs = []
   for fname in tqdm(models):
       data = svo.spectra_file_reader(fname)
       # extract model relevant information
       lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
       lamb = data['data']['WAVELENGTH'].values * lamb_unit
       flux = data['data']['FLUX'].values * flux_unit
       apvalues = [data[k]['value'] for k in apfields]

       # wavelength definition varies between models
       alambda_per_av = extc(lamb, 1.0, Rv=Rv)

       # Dust magnitudes
       columns = apfields + ['passband', 'mag0', 'mag', 'A0', 'Ax']
       for pk in passbands:
           mag0 = -2.5 * np.log10(pk.get_flux(lamb, flux).value)
           # we redo av = 0, but it's cheap, allows us to use the same code
           for av_val in Av:
               new_flux = flux * np.exp(- alambda_per_av * av_val)
               mag = -2.5 * np.log10(pk.get_flux(lamb, new_flux).value)
               delta = (mag - mag0)
               logs.append(apvalues + [pk.name, mag0, mag, av_val, delta])

   logs = pd.DataFrame.from_records(logs, columns=columns)

The above script works, but it could be very time consuming if you have many
passbands and many extinction parameters to grid. However, every piece of
information are independent of one another: atmosphere spectra, passbands,
extinction grid points. Hence this is a massively parallel problem.

As the first rule of optimization is to start by the most outer loop, we
provide a script that parallelizes the the procedure with respect to the atmosphere files in
:mod:`dustapprox.tools.grid` (using
`joblib <https://joblib.readthedocs.io/en/latest/>`_) in particular
:func:`dustapprox.tools.grid.compute_photometric_grid`.