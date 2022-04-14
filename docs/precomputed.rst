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


Generating models
-----------------

Generating a photometric extinction model or approximation requires first that
we have some atmosphere spectral model. We provide some tools associated with the
`SVO Theoretical <spectra: http://svo2.cab.inta-csic.es/theory/newov2/index.php>`_
in :doc:`/atmospheres` (:mod:`dustapprox.io.svo`) but you can also use your own atmosphere models.

Second, we need an extinction presscription. We provide some mean extinction
curves in :doc:`/extinction` (:mod:`dustapprox.io.extinction`).

Finally, we need passband definitions and functions to do the photometric
calculations.  For the photometry, we use the external package `pyphot
<https://mfouesneau.github.io/pyphot/index.html>`_ a suite to compute synthetic
photometry in flexible ways.  In addition,
:func:`dustapprox.io.svo.get_svo_passbands` interfaces the `SVO Filter Profile
Service <http://svo2.cab.inta-csic.es/theory/fps/index.php>`_, which provides us
with a large collection of passbands. (wrapper from `pyphot`_).

Once we have the above ingredients, we can bring them together to generate a
large collection of photometric extinction values in various bands.


Creating a grid of models
--------------------------

.. code-block:: python3
   :caption: An example of **not optimized** script to generate an extinction grid over all the atmosphere models

   import numpy as np
   import pandas as pd
   from glob import glob
   from tqdm import tqdm
   from dustapprox.io import svo
   from dustapprox.extinction import F99
   from pyphot.astropy.sandbox import Unit as U


   which_filters = ['GAIA/GAIA3.G', 'GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp']
   passbands = svo.get_svo_passbands(which_filters)
   # Technically it does not matter what zeropoint we use since we'll do relative values to get the dust effect

   models = glob('models/Kurucz2003all/*.fl.dat.txt')

   # Extinction
   extc = F99()
   Rv = 3.1
   Av = np.arange(0, 20.01, 0.2)

   logs = []
   for fname in tqdm(models):
       data = svo.spectra_file_reader(fname)
       # extract model relevant information
       lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
       lamb = data['data']['WAVELENGTH'].values * lamb_unit
       flux = data['data']['FLUX'].values * flux_unit
       teff = data['teff']['value']
       logg = data['logg']['value']
       feh = data['feh']['value']
       print(fname, teff, logg, feh)

       # wavelength definition varies between models
       alambda_per_av = extc(lamb, 1.0, Rv=Rv)

       # Dust magnitudes
       columns = ['teff', 'logg', 'feh', 'passband', 'mag0', 'mag', 'A0', 'Ax']
       for pk in passbands:
           mag0 = -2.5 * np.log10(pk.get_flux(lamb, flux).value)
           # we redo av = 0, but it's cheap, allows us to use the same code
           for av_val in Av:
               new_flux = flux * np.exp(- alambda_per_av * av_val)
               mag = -2.5 * np.log10(pk.get_flux(lamb, new_flux).value)
               delta = (mag - mag0)
               logs.append([teff, logg, feh, pk.name, mag0, mag, av_val, delta])

   logs = pd.DataFrame.from_records(logs, columns=columns)
