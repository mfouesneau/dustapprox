.. my_package documentation master file, created by
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gdr3_dustapprox's documentation!
===========================================

This package is a set of tools to compute photometric extinction coefficients in a *quick and dirty* way.

.. todo::

   * we need to warn about bad practices.

   * we need to give the various references.


* :class:`gdr3_dustapprox.edr3.edr3_ext` provides the Riello et al. (2020) approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_0` for Gaia eDR3 passbands (G, BP, RP).

* :class:`gdr3_dustapprox.c1m.C1_extinction` provides the Bellazzini et al. (2022) approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_G` for Gaia :math:`C1` passbands.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
