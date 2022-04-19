Welcome to dustapprox's documentation!
===========================================

This package is a set of tools to compute photometric extinction coefficients in a *quick and dirty* way.

Extinction coefficients per passbands depend on both the source spectral energy distribution
and on the extinction itself (e.g., `Gordon et al., 2016 <https://ui.adsabs.harvard.edu/abs/2016ApJ...826..104G/abstract>`_,
`Jordi et al., 2010 <https://ui.adsabs.harvard.edu/abs/2010A%26A...523A..48J/abstract>`_ ).
To first order, the shape of the SED through a given passband determine the mean
photon wavelength and therefore the mean extinction through that passband.  Of
course in practice this also depends on the source spectral features in the
passband and the dust properties.

We provide the methodology to compute approximation models of the extinction for a given
passband as well as some precomputed models that are ready to use or integrate
with larger projects.

We also detailed the various ingredients of the models in subsequent pages listed below

.. toctree::
   :maxdepth: 1
   :caption: Details

   atmospheres
   extinction
   photometry
   precomputed

.. todo::

   * we need to warn about bad practices.

   * we need to give the various references.

Quick Start
-----------

.. todo::

    * add some quick examples of usage from precomputed models


Why an approximation?
---------------------

*very light mathematical details*

If we assume :math:`F_\lambda^0` is the intrinsic atmosphere energy distribution of a star
as a function of wavelength :math:`\lambda` and the extinction curve :math:`\tau_\lambda`, the apparent
wavelength dependent light observed from a star is given by:

.. math::

    \begin{equation}
    f_\lambda = F_\lambda^0 \exp(-\tau_\lambda).
    \end{equation}

.. plot::
   :caption: **Figure 1.** Effect of extinction on a given star. The reference star parameters
             are indicated at the top. We gridded :math:`A_0` from 0 to 5 mag (by 0.1 mag step).
             The code also illustrates adding the effect of dust extinction with the tools we provide.

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

If we consider a filter photon throughput (a.k.a, transmission curve, or response
function) defined in wavelength by the dimensionless function :math:`T(\lambda)`,
this function tells you what fraction of the arriving photons at wavelength
:math:`\lambda` actually get through the instrument.

Consequently, the statistical mean of the flux density through :math:`T`, :math:`\overline{f_T}` is

.. math::

        \begin{equation}
        \overline{f_T} = \frac{\int_\lambda \lambda f_\lambda T(\lambda) d\lambda}{\int_\lambda \lambda T(\lambda) d\lambda}.
        \end{equation}

The flux equation above slightly change if we consider energy detector types, but the general idea remains the same.
The magnitude in :math:`T` is given by

.. math::

        \begin{equation}
        m_T = -2.5 \times \log_{10} \left(\overline{f_T}/\overline{f_{zero}}\right),
        \end{equation}

where :math:`\overline{f_{zero}}` is the zero-point flux density of the filter
depending on the photometric systems.

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

.. plot::
   :caption: **Figure 2.** Integrated extinction effect in the Gaia G, BP and RP bands.
             The left panel illustrates that :math:`A(T)` is not a simple rescaling of :math:`A_0`
             but a distribution of values (the black line indicates the identity line).
             The right panel illustates that the distribution of
             :math:`A(T)/A_0` depends strongly on the temperature
             :math:`T_{eff}`, i.e. the shape of the SED throught the passband :math:`T`

   import pylab as plt
   import pandas as pd

   r = pd.read_csv('./models/precomputed/kurucs_gaiaedr3_small_a0_grid.csv')
   r['kx'] = np.where(r['A0'] > 0, r['Ax'] / r['A0'], float('NaN'))

   plt.figure(figsize=(9, 4))
   ax1 = plt.subplot(121)
   ax2 = plt.subplot(122, sharey=ax1)
   for key, grp in r.groupby('passband'):
      ax1.scatter(grp['A0'], grp['kx'], label=key, rasterized=True)
      ax2.scatter(grp['teff'], grp['kx'], label=key, rasterized=True)
   ax1.legend(loc='best', frameon=False)
   ax1.set_ylabel(r'$A(T)\ /\ A_0$ [mag]')
   ax1.set_xlabel(r'$A_0$ [mag]')
   ax2.set_xlabel(r'$T_{eff}$ [K]')
   ax2.set_xscale('log')
   plt.tight_layout()
   plt.show()



However, calculating :math:`A(T)` correctly is not a trivial task. It first requires having an atmosphere model at the exact
stellar parameter set (:math:`T_{eff}, \log g, [Fe/H], \ldots`). This may be also a long computation or an interpolation.
Then applying the dust extinction curve, and integrating through the passband, twice, to obtain :math:`A(T)`.
It may become computationally a very expensive task.

On another hand, it is often useful to convert extinction from :math:`A(T)` to :math:`A(T^\prime)`: for instance from
:math:`A_0` to :math:`A(V)`, :math:`A(G_{BP})`, or :math:`A(Ks)`. This could also become a difficult task.


Precomputed models
------------------

This package allows one to generate new models from spectral libraries and extinction curves.
However, we also provide some pre-computed models that can be used directly.

* :class:`dustapprox.models.PrecomputedModel` provides convenient search and load functions.

.. seealso::

   * model training details :doc:`/precomputed`


Literature Extinction approximations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also provide multiple literature approximations with this package (:mod:`dustapprox.literature`).

.. warning::

   Their relations may not use the same approach or parametrizations.


.. todo::

   Add some comparison to the literature in the submodule :mod:`dustapprox.literature`.


How to contribute?
-------------------

We love contributions! This project is open source, built on open source
libraries.

Please open a new issue or new pull request for bugs, feedback, or new features
you would like to see. If there is an issue you would like to work on, please
leave a comment and we will be happy to assist. New contributions and
contributors are very welcome!

Being a contributor doesn't just mean writing code. You can
help out by writing documentation, tests, or even giving feedback about the
project (yes - that includes giving feedback about the contribution
process).

We are committed to providing a strong and enforced code of conduct and expect
everyone in our community to follow these guidelines when interacting with
others in all forums. Our goal is to keep ours a positive, inclusive, thriving,
and growing community. The community of participants in open source Astronomy
projects such as this present work includes members from around the globe with
diverse skills, personalities, and experiences. It is through these differences
that our community experiences success and continued growth.
Please have a look at our `code of conduct <https://github.com/mfouesneau/dustapprox/main/CODE_OF_CONDUCT.md>`_.


This project work follows a `BSD 3-Clause license <https://github.com/mfouesneau/dustapprox/blob/main/LICENSE>`_.


How to cite this work?
----------------------

If you use this software, please cite it using the metadata below.

**APA citation**

.. code-block:: text

   Fouesneau, M., Andrae, R., Sordo, R., & Dharmawardena, T. (2022).
      dustapprox (Version 0.1) [Computer software].
      https://github.com/mfouesneau/dustapprox

**Bibtex citation**

.. code-block:: text

   @software{Fouesneau_dustapprox_2022,
      author = {Fouesneau, Morgan
                and Andrae, René
                and Sordo, Rosanna
                and Dharmawardena, Thavisha},
      month = {3},
      title = {{dustapprox}},
      url = {https://github.com/mfouesneau/dustapprox},
      version = {0.1},
      year = {2022}
      }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :maxdepth: 3
   :caption: Module API

   modules