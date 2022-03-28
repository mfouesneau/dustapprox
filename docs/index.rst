Welcome to dustapprox's documentation!
===========================================

This package is a set of tools to compute photometric extinction coefficients in a *quick and dirty* way.

.. todo::

   * we need to warn about bad practices.

   * we need to give the various references.

Literature Extinction approximations
------------------------------------

We provide multiple literature approximations with this package

* :class:`dustapprox.edr3.edr3_ext` provides the Riello et al. (2020) approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_0` for Gaia eDR3 passbands (G, BP, RP).

  .. warning::

     Their calibration only accounted for solar metallicity.

* :class:`dustapprox.c1m.C1_extinction` provides the Bellazzini et al. (2022) approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_G` for Gaia :math:`C1` passbands.

  .. warning::

     Their relations use :math:`A_G`, not :math:`A_0` as input.


Our extinction approximation models
-----------------------------------

Extinction coefficients per passbands depend on both the source spectral energy distribution
and on the extinction itself (e.g., Gordon et al., 2016, Jordi et al., 2010).
To first order, the shape of the SED through a given passband determine the mean
photon wavelength and therefore the mean extinction through that passband.  Of
course in practice this also depends on the source spectral features in the
passband and the dust properties.

Atmosphere models
^^^^^^^^^^^^^^^^^

We took the the Kurucz (ODFNEW/NOVER 2003) atmosphere library from the `SVO
Theoretical spectra <http://svo2.cab.inta-csic.es/theory/newov2/index.php>`_
:ref:`Figure 1 <fig-Kurucz-coverage>` shows the coverage of this library.

.. _fig-Kurucz-coverage:

.. figure:: http://svo2.cab.inta-csic.es/theory/newov2/temp/models/plots/range_1648123268.6215.png

   **Figure 1.** Kurucz (ODFNEW/NOVER 2003) atmosphere library coverage. These models span
   ranges of :math:`T_{eff}` from 3500 K to 50000 K, :math:`\log g` from 0 to 5
   dex, and metallicity :math:`[Fe/H]` from -4 to 0 dex. Each square symbol
   indicates the availability of a synthetic spectrum.

SVO provides many other libraries. Our approach is agnostics to the exact library itself.
All files from SVO have the same format, but the spectra are not on the same wavelength scale (even for a single atmosphere source).
The parameters of the spectra may vary from source to source. For instance, they may not provide microturbulence velocity or alpha/Fe etc.
We technically require only :math:`T_{eff}` and :math:`[Fe/H]` to be provided.

* :class:`dustapprox.io.svo` provides means to read SVO atmosphere models.

  see :func:`dustapprox.io.svo.spectra_file_reader`, and :func:`dustapprox.io.svo.get_svo_sprectum_units`


**example usage**
To convert the flux to the observed flux at Earth, we need to multiply by a
factor of :math:`(R/D)^2` where :math:`R`` is the stellar radius, and :math:`D``
is the distance to Earth in consistent units.

.. plot::
   :caption: **Figure 2.** Examples of Kurucz models.
   :include-source:

   import matplotlib.pyplot as plt
   from dustapprox.io import svo

   models = ['models/Kurucz2003all/fm05at10500g40k2odfnew.fl.dat.txt',
             'models/Kurucz2003all/fm05at5000g25k2odfnew.fl.dat.txt',
             'models/Kurucz2003all/fm40at6000g00k2odfnew.fl.dat.txt']

   label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'

   for fname in models:
      data = svo.spectra_file_reader(fname)
      lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
      lamb = data['data']['WAVELENGTH'].values * lamb_unit
      flux = data['data']['FLUX'].values * flux_unit

      plt.loglog(lamb, flux,
                 label=label.format(teff=data['teff']['value'],
                                    logg=data['logg']['value'],
                                    feh=data['feh']['value']))

   plt.legend(loc='upper right', frameon=False)
   plt.xlabel('Wavelength [{}]'.format(lamb_unit))
   plt.ylabel('Flux [{}]'.format(flux_unit))
   plt.ylim(1e2, 5e9)
   plt.xlim(800, 1e5)


.. note::

   For the Kurucz ODFNEW /NOVER (2003) library, refer to

   * `Castelli and Kurucz 2003, IAUS 210, A20 <http://adsabs.harvard.edu/abs/2003IAUS..210P.A20C>`_
   * `Castelli and Kurucz Atlas <https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas>`_
   * `Castelli ATLAS9 grids web page. <https://wwwuser.oats.inaf.it/castelli/grids.html>`_


.. warning::

   We do not provide an interface to download the atmospheres from SVO. (in this version at least)


Photometric bands
^^^^^^^^^^^^^^^^^

For the photometry, we use `pyphot
<https://mfouesneau.github.io/pyphot/index.html>`_ a suite to compute synthetic
photometry in flexible ways.

* :func:`dustapprox.io.svo.get_svo_passbands` to interface the `SVO Filter Profile Service
  <http://svo2.cab.inta-csic.es/theory/fps/index.php>`_, which provides us with
  a large collection of passbands. This a wrapper around `pyphot <https://mfouesneau.github.io/pyphot/index.html>`_.


**example usage**

.. plot::
   :caption: **Figure 3.** This figure shows the Gaia eDR3 passbands retrieved from the SVO service.
   :include-source:

   import matplotlib.pyplot as plt

   from dustapprox.io import svo
   which_filters = ['GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp', 'GAIA/GAIA3.G']
   passbands = svo.get_svo_passbands(which_filters)

   for pb in passbands:
      plt.plot(pb.wavelength.to('nm'), pb.transmit, label=pb.name)

   plt.legend(loc='upper right', frameon=False)

   plt.xlabel('wavelength [nm]')
   plt.ylabel('transmission')
   plt.tight_layout()
   plt.show()


Extinction curves
^^^^^^^^^^^^^^^^^
In our application, we need the models that predict spectra or SEDs of
star extinguished by dust. Interstellar dust extinguishes stellar light as it
travels from the star's surface to the observer. The wavelength dependence of
the extinction from the ultraviolet to the near-infrared has been measured along
many sightlines in the Milky Way (Cardelli et al. 1989; Fitzpatrick 1999;
Valencic et al. 2004; Gordon et al. 2009) and for a handful of sightlines in the
Magellanic Clouds (Gordon & Clayton 1998; Misselt et al. 1999; Maiz Apellaniz &
Rubio 2012) as well as in M31 (Bianchi et al. 1996, Clayton et al. 2015).


* :class:`dustapprox.extinction` provides a common interface to many
  commonly used extinction curves.

.. plot::
   :caption: **Figure 4.** Differences between extinction curves. This figure compares a few different presscriptions at fixed :math:`R_0`.
   :include-source:

   import numpy as np
   import matplotlib.pyplot as plt
   import astropy.units as u

   from dustapprox.extinction import CCM89, F99

   #define the wave numbers
   x = np.arange(0.1, 10, 0.1)    # in microns^{-1}
   lamb = 1. / x * u.micron

   curves = [CCM89(), F99()]
   Rv = 3.1

   for c in curves:
       name = c.name
       plt.plot(x, c(lamb, Rv=Rv), label=f'{name:s}, R(V) = {Rv:0.1f}', lw=2)
   plt.xlabel(r'Wave number [$\mu$m$^{-1}$]')
   plt.ylabel(r'$A(x)/A(V)$')
   plt.legend(loc='upper left', frameon=False, title='Ext. Curve')

   plt.tight_layout()
   plt.show()


Generating models
^^^^^^^^^^^^^^^^^

Once we have the above ingredients, we can bring them together to generate a large collection of photometric extinction values in various bands.

**Example showing the effect of extinction on a given star**

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
^^^^^^^^^^^^^^^^^^^^^^^^^

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


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
