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
   precomputed_content


.. warning::

   This package provides **approximations** to the extinction effects in photometric bands.
   It is not meant to be a full implementation of the extinction curves but a shortcut.

   In this current version, we only provide global uncertainties (e.g., rms,
   biases). To obtain complete uncertainties, one needs to use models and
   compute the relevant statistics. We show how to compute a model grid in :doc:`/precomputed/`.


Quick Start
-----------

The following example shows how to use the predictions from a precomputed model.


.. code-block:: python
   :caption: Example of using an approximation model.

   import pandas as pd
   from dustapprox import models
   from dustapprox.literature import edr3
   import pylab as plt

   # get Gaia models
   lib = models.PrecomputedModel()
   r = lib.find(passband='Gaia')[0]  # taking the first one
   model = lib.load_model(r, passband='GAIA_GAIA3.G')

   # get some data
   data = pd.read_csv('models/precomputed/kurucs_gaiaedr3_small_a0_grid.csv')
   df = data[(data['passband'] == 'GAIA_GAIA3.G') & (data['A0'] > 0)]

   # values
   kg_pred = model.predict(df)


Why an approximation?
---------------------

*very light mathematical details*

If we assume :math:`F_\lambda^0` is the intrinsic atmosphere energy distribution of a star
as a function of wavelength :math:`\lambda` and the extinction curve :math:`\tau_\lambda`, the apparent
wavelength-dependent light observed from a star is given by:

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

The flux equation above changes slightly if we consider energy detector types, but the general idea remains the same.
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
   import numpy as np

   r = pd.read_csv('./models/precomputed/kurucs_gaiaedr3_small_a0_grid.csv')
   r['kx'] = np.where(r['A0'] > 0, r['Ax'] / r['A0'], float('NaN'))

   plt.figure(figsize=(9, 4))
   ax1 = plt.subplot(121)
   ax2 = plt.subplot(122, sharey=ax1)
   colors = {'Gbp': 'C0', 'G': 'C2', 'Grp': 'C3'}
   for key, grp in r.groupby('passband'):
      color = colors[key.split('.')[-1]]
      ax1.scatter(grp['A0'], grp['kx'], label=key, rasterized=True, color=color)
      ax2.scatter(grp['teff'], grp['kx'], label=key, rasterized=True, color=color)
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
   * list of provided models: :doc:`/precomputed_content`


Literature Extinction approximations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We also provide multiple literature approximations with this package (:mod:`dustapprox.literature`).

.. warning::

   Their relations may not use the same approach or parametrizations.

The following figure (code provided) compares the predictions from a :class:`dustapprox.models.polynomial.PolynomialModel` to those of :class:`dustapprox.literature.edr3.edr3_ext`. The latter is a model described `Riello et al. (2020) <https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...3R/abstract>`_.

However, one must note that the literature models are often valid on a restricted stellar parameter space, here in particular, the temperature :math:`T_{eff}` is limited to the range :math:`[3500, 10 000]` K. Our model uses the same polynomial degree as the literature version.
Note that these are evidently approximations to the explicit calculations used as reference on all the y-axis.
Our package also allows one to update the model parameters, such as the polynomial degree, or the range of validity of the model.

.. plot::
   :caption:  **Figure 3.** Comparing a Gaia G approximation model to the EDR3 one (:mod:`dustapprox.literature.edr3`).
              We plot the residuals of :math:`k_G = A_G/A_0` versus the intrinsic G-magnitude (left) and temperature (right). One
              can see the polynomial oscillations.  Note that the performance degrades with
              stellar temperature. As the EDR3 model is only valid for some limited range
              of temperature, we indicate extrapolation data with smaller dots.

   import pandas as pd
   from dustapprox import models
   from dustapprox.literature import edr3
   import pylab as plt

   # get Gaia models
   lib = models.PrecomputedModel()
   r = lib.find(passband='Gaia')[0]  # taking the first one
   model = lib.load_model(r, passband='GAIA_GAIA3.G')

   # get some data
   data = pd.read_csv('./models/precomputed/kurucs_gaiaedr3_small_a0_grid.csv')
   df = data[(data['passband'] == 'GAIA_GAIA3.G') & (data['A0'] > 0)]

   # values
   ydata = (df['mag'] - df['mag0']) / df['A0']
   kg_edr3 = edr3.edr3_ext().from_teff('kG', df['teff'], df['A0'])
   delta_edr3 = ydata - kg_edr3
   delta = ydata - model.predict(df)
   # note: edr3 model valid for 3500 < teff < 10_000
   selection = '3500 < teff < 10_000'
   select = df.eval(selection)

   plt.figure(figsize=(10, 8))
   cmap = plt.cm.inferno_r

   title = "{name:s}: kind={kind:s}\n using: {features}".format(
         name=model.name,
         kind=model.__class__.__name__,
         features=', '.join(model.feature_names))

   kwargs_all = dict(rasterized=True, edgecolor='w', cmap=cmap, c=df['A0'])
   kwargs_select = dict(rasterized=True, cmap=cmap, c=df['A0'][select])

   ax0 = plt.subplot(221)
   plt.scatter(df['mag0'], delta, **kwargs_all)
   plt.scatter(df['mag0'][select], delta[select], **kwargs_select)
   plt.colorbar().set_label(r'A$_0$ [mag]')
   plt.ylabel(r'$\Delta$k$_G$')
   plt.text(0.01, 0.99, title, fontsize='medium',
         transform=plt.gca().transAxes, va='top', ha='left')

   ax1 = plt.subplot(222, sharey=ax0)
   plt.scatter(df['teff'], delta, **kwargs_all)
   plt.scatter(df['teff'][select], delta[select], **kwargs_select)
   plt.colorbar().set_label(r'A$_0$ [mag]')
   plt.text(0.01, 0.99, title, fontsize='medium',
         transform=plt.gca().transAxes, va='top', ha='left')

   title = "{name:s}: kind={kind:s}\n using: {features}".format(
            name="EDR3",
            kind=edr3.edr3_ext().__class__.__name__,
            features=', '.join(('teffnorm', 'A0')))

   ax = plt.subplot(223, sharex=ax0, sharey=ax0)
   plt.scatter(df['mag0'], delta_edr3, **kwargs_all)
   plt.scatter(df['mag0'][select], delta_edr3[select], **kwargs_select)
   plt.colorbar().set_label(r'A$_0$ [mag]')
   plt.xlabel(r'M$_G$ [mag] + constant')
   plt.ylabel(r'$\Delta$k$_G$')
   plt.ylim(-0.1, 0.1)
   plt.text(0.01, 0.99, title, fontsize='medium',
         transform=plt.gca().transAxes, va='top', ha='left')


   plt.subplot(224, sharex=ax1, sharey=ax1)
   plt.scatter(df['teff'], delta_edr3, **kwargs_all)
   plt.scatter(df['teff'][select], delta_edr3[select], **kwargs_select)
   plt.xlabel(r'T$_{\rm eff}$ [K]')
   plt.colorbar().set_label(r'A$_0$ [mag]')
   plt.ylim(-0.1, 0.1)

   plt.setp(plt.gcf().get_axes()[1:-1:2], visible=False)

   plt.text(0.01, 0.99, title, fontsize='medium',
         transform=plt.gca().transAxes, va='top', ha='left')

   plt.tight_layout()
   plt.show()


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
Please have a look at our `code of conduct <https://github.com/mfouesneau/dustapprox/blob/main/CODE_OF_CONDUCT.md>`_.


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