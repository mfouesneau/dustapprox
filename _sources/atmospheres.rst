Atmosphere models
=================

For the precomputed models, we took the the Kurucz (ODFNEW/NOVER 2003)
atmosphere library from the `SVO Theoretical spectra
<http://svo2.cab.inta-csic.es/theory/newov2/index.php>`_ :ref:`Figure 1
<fig-Kurucz-coverage>` shows the coverage of this library.

.. _fig-Kurucz-coverage:

.. figure:: https://keeper.mpdl.mpg.de/f/5c751ff156d443b38692/?dl=1

   **Figure 1.** Kurucz (ODFNEW/NOVER 2003) atmosphere library coverage. These models span
   ranges of :math:`T_{eff}` from 3500 K to 50000 K, :math:`\log g` from 0 to 5
   dex, and metallicity :math:`[Fe/H]` from -4 to 0 dex. Each square symbol
   indicates the availability of a synthetic spectrum.

However, `SVO Theoretical spectra
<http://svo2.cab.inta-csic.es/theory/newov2/index.php>`_  provides many other libraries. Our approach is agnostics to the exact library itself.
All files from SVO have the same format, but the spectra are not on the same wavelength scale (even for a single atmosphere source).
The parameters of the spectra may vary from source to source. For instance, they may not provide microturbulence velocity or alpha/Fe etc.
We technically require only :math:`T_{eff}` and :math:`[Fe/H]` to be provided.

* :mod:`dustapprox.io.svo` provides means to read SVO atmosphere models.

  see :class:`~dustapprox.io.svo.SVOSpectrum`, :func:`~dustapprox.io.svo.spectra_file_reader`, and :func:`~dustapprox.io.svo.get_svo_spectrum_units`


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
      spec = svo.SVOSpectrum(fname)

      plt.loglog(spec.λ, spec.flux,
                 label=label.format(teff=spec.meta['teff'],
                                    logg=spec.meta['logg'],
                                    feh=spec.meta['feh']))

   plt.legend(loc='upper right', frameon=False)
   plt.xlabel('Wavelength [{}]'.format(spec.units[0]))
   plt.ylabel('Flux [{}]'.format(spec.units[1]))
   plt.ylim(1e2, 5e9)
   plt.xlim(800, 1e5)


Pre-compiled atmosphere libraries
---------------------------------

`SVO Theoretical spectra`_ provides many atmosphere libraries. Our approach is agnostics to the exact library itself.
All files from SVO have the same format, but the spectra are not on the same wavelength scale (even for a single atmosphere source).

.. warning::

   We do not provide an interface to download the atmospheres from SVO. (in this version at least)


We compiled tarballs of some atmosphere libraries we use in our models (and the associated references).

.. important::

   Please cite the appropriate references we provided to the model atmospheres you use in your applications.

* `Kurucz (ODFNEW/NOVER 2003) <https://keeper.mpdl.mpg.de/f/a80ede0816674d729f4e/>`_ (tar.gz; 78 Mo)
   * from `SVO Theoretical spectra`_
   * `Castelli and Kurucz 2003, IAUS 210, A20 <http://adsabs.harvard.edu/abs/2003IAUS..210P.A20C>`_
   * `Castelli and Kurucz Atlas <https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/castelli-and-kurucz-atlas>`_
   * `Castelli ATLAS9 grids web page. <https://wwwuser.oats.inaf.it/castelli/grids.html>`_

.. todo::

   * add CU8 atmospheres with proper references. (MARCS, PHOENIX, OB, A, BTSettl, libraries)
