"""
Literature Extinction approximations
====================================

We provide multiple literature approximations with this package.

* :class:`dustapprox.literature.edr3.edr3_ext` provides the `Riello et al. (2020) <https://ui.adsabs.harvard.edu/abs/2021A%26A...649A...3R/abstract>`_ approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_0` for Gaia eDR3 passbands (G, BP, RP).

  .. warning::

     Their calibration only accounted for solar metallicity.

* :class:`dustapprox.literature.c1.dr3_ext` provides the `Bellazzini et al. (2022) <https://www.cosmos.esa.int/web/gaia/dr3-papers>`_ approximation, i.e.,
  extinction coefficient :math:`k_x = A_x / A_G` for Gaia :math:`C1` passbands (defined in `Jordi et al (2006) <https://academic.oup.com/mnras/article/367/1/290/1018790>`_).

  .. warning::

     Their relations use :math:`A_G`, not :math:`A_0` as input.
"""

from . import edr3
from . import c1