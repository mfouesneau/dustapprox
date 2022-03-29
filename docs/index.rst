Welcome to dustapprox's documentation!
===========================================

This package is a set of tools to compute photometric extinction coefficients in a *quick and dirty* way.

Extinction coefficients per passbands depend on both the source spectral energy distribution
and on the extinction itself (e.g., Gordon et al., 2016, Jordi et al., 2010).
To first order, the shape of the SED through a given passband determine the mean
photon wavelength and therefore the mean extinction through that passband.  Of
course in practice this also depends on the source spectral features in the
passband and the dust properties.

We provide the methodology to compute the extinction coefficients for a given
passband as well as some precomputed models that are ready to use or integrate
with larger projects.


.. todo::

   * we need to warn about bad practices.

   * we need to give the various references.

Quick Start
-----------

.. todo::

    * add some quick examples of usage from precomputed models


Precomputed models
------------------

This package allows one to generate new models from spectral libraries and extinction curves.
However, we also provide some pre-computed models that can be used directly.

* :mod:`dustapprox.models.polynomial`: polynomial models.
   * :class:`dustapprox.models.polynomial.precomputed`: get available models.


.. seealso::

   * model training details :doc:`/precomputed`


Literature Extinction approximations
------------------------------------

We also provide multiple literature approximations with this package

* :class:`dustapprox.edr3.edr3_ext` provides the Riello et al. (2020) approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_0` for Gaia eDR3 passbands (G, BP, RP).

  .. warning::

     Their calibration only accounted for solar metallicity.

* :class:`dustapprox.c1m.C1_extinction` provides the Bellazzini et al. (2022) approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_G` for Gaia :math:`C1` passbands.

  .. warning::

     Their relations use :math:`A_G`, not :math:`A_0` as input.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   atmospheres
   extinction
   photometry
   precomputed
   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
