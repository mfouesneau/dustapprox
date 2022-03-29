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

Mathematical details
^^^^^^^^^^^^^^^^^^^^


If we assume :math:`F_\lambda^0` is the intrinsic atmosphere energy distribution of a star
as a function of wavelength :math:`\lambda` and the extinction curve :math:`\tau_\lambda`, the apparent
wavelength dependent light observed from a star is given by:

.. math::

    \begin{equation}
    f_\lambda = F_\lambda^0 \exp(-\tau_\lambda).
    \end{equation}

If we consider a filter throughput (a.k.a, transmission curve, or response
function) defined in wavelength by the dimensionless function :math:`T(\lambda)`,
this function tells you what fraction of the arriving photons at wavelength
:math:`\lambda` actually get through the instrument.

Consequently, the statistical mean of the flux density through :math:`T`, :math:`\overline{f_T}` is

.. math::

        \begin{equation}
        \overline{f_T} = \frac{\int_\lambda \lambda f_\lambda T(\lambda) d\lambda}{\int_\lambda \lambda T(\lambda) d\lambda}.
        \end{equation}

and the magnitude in :math:`T` is given by

.. math::

        \begin{equation}
        m_T = -2.5 \times \log_{10} \left(\overline{f_T}/\overline{f_{zero}}\right),
        \end{equation}

where :math:`\overline{f_{zero}` is the zero-point flux density of the filter
depending on the photometric systems and detector count types (energy or photons).

However, the magnitude effect :math:`A(T)` or :math:`A_T` of the extinction in
:math:`T` does not require an explicit zero-point as it is a relative effect:

.. math::

        \begin{eqnarray}
        A_T &=& m_T - M_T \\
            &=& -2.5 \times \log_{10} \frac{
                            \int_\lambda \lambda F_\lambda^0 \exp(-\tau_\lambda) T(\lambda) d\lambda}{
                            \int_\lambda \lambda F_\lambda^0 T(\lambda) d\lambda},
        \end{eqnarray}

From the equation above, it should be clear that :math:`A(T)` is not a constant,
nor a simple expression form. However, one can approximate :math:`A(T)` by a function of various parameters.
From the integral's perspective, we see that the shape of the SED matters, which
often leads to approximations as functions of stellar temperatures, :math:`T_{eff}`.


**Example showing the effect of extinction on a given star**

The example below illustrates the effect of dust extinction using the tools we provide in this package.

.. plot::
   :caption: **Figure 5.** Effect of extinction on a given star. The reference star parameters
             are indicated at the top. We gridded :math:`A_0` from 0 to 5 mag (by 0.1 mag step).
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   from dustapprox.io import svo
   from dustapprox.extinction import F99

   modelfile = 'models/Kurucz2003all/fm05at10500g40k2odfnew.fl.dat.txt'
   data = svo.spectra_file_reader(modelfile)

   # extract model relevant information
   lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
   lamb = data['data']['WAVELENGTH'].values * lamb_unit
   flux = data['data']['FLUX'].values * flux_unit

   # Extinction
   extc = F99()
   Rv = 3.1
   Av = np.arange(0, 5.01, 0.1)
   alambda_per_av = extc(lamb, 1.0, Rv=Rv)

   # Dust magnitudes
   cmap = plt.cm.inferno_r
   sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Av.min(), vmax=Av.max()))
   for av_val in Av:
      new_flux = flux * np.exp(- alambda_per_av * av_val)
      plt.loglog(lamb, new_flux, label=f'A0={av_val:.2f}', color=cmap(av_val / Av.max()))
   plt.loglog(lamb, flux, color='k')
   plt.ylim(1e-6, 1e9)
   plt.xlim(750, 5e4)
   plt.xlabel('Wavelength [{}]'.format(lamb_unit))
   plt.ylabel('Flux [{}]'.format(flux_unit))
   label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'
   plt.title(label.format(teff=data['teff']['value'],
                          logg=data['logg']['value'],
                          feh=data['feh']['value']))
   plt.colorbar(sm).set_label(r'A$_0$ [mag]')
   plt.tight_layout()

   plt.show()


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