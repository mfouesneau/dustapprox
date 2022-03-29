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