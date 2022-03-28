r"""  Coefficients of the reddening laws to correct magnitudes in the :math:`C1` system.

The procedure is detailed in the appendix E of Bellazzini et al. (2022).

In brief, they fit polynomial relations to predicted SED values.
The simulated predictions originate from the BTSettl atmosphere library combined
with the :math:`C1` passbands definitions and the Fitzpatrick extinction curve.

They derive the absorption in any :math:`X` band as a function of the absorption in :math:`G`, :math:`A_G`,
the Gaia :math:`G_{BP}-G_{RP}` color:

.. math::

        \frac{A_X}{A_G}=\alpha+\sum_{i=1}^{4}\beta_i \cdot ({G_{BP}-G_{RP}})^i+\sum_{j=1}^3 \gamma_j \cdot A_G^j + \delta \cdot ({G_{BP}-G_{RP}})\cdot A_G

.. warning::

    Those relations depend on :math:`A_G`, not :math:`A_0`.

The :math:`C1` Gaia photometric system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original design for Gaia included a set of photometric filters (see Jordi et
al., 2006), the C1B and C1M systems for the broad and medium band passbands,
respectively.  The C1 system was specifically desgined to maximize the
scientific return in terms of stellar astrophysical parameters.

Eventually, construction and budget constraints led ESA to adopt prisms in the
final design of Gaia, which cover for those passbands.

For example :math:`C1M467-C1M515` color is sensitive to
surface gravity (:math:`\log g`) and :math:`C1B556-C1B996` is sensitive to the effective
temperature :math:`T_{eff}`.

.. todo::

    Make a table with the C1 indices

"""
from typing import Union
import numpy as np
from .io import ecsv


class C1_extinction:
    """Function to get the extinction coefficients A_X / A_G in C1M system

    The procedure is detailed in the appendix E of Bellazzini et al. (2022).

    Parameters
    ----------
    name : str
        The name of the X band
    bprp : float or array
        The Gaia G_BP - G_RP color
    ag : float or array
        The Gaia A_G value

    Returns
    -------
    A_X / A_G : float or array
        The extinction coefficients
    """
    def __init__(self, data='gaia_C1_extinction.ecsv'):
        """Constructor that loads the external data table."""
        self.data = ecsv.read('gaia_C1_extinction.ecsv').set_index('X')

    def __call__(self,
                 name: str,
                 bprp: Union[float, np.array],
                 ag:float) -> Union[float, np.array]:
        """
        Returns A_X / A_G values

        Parameters
        ----------
        name : str
            The name of the X band
        bprp : float or array
            The Gaia G_BP - G_RP color
        ag : float or array
            The Gaia A_G value

        Returns
        -------
        A_X / A_G : float or array
            The extinction coefficients
        """
        bprp_ = np.atleast_1d(bprp)
        ag_ = np.atleast_1d(ag)
        c = self.data.loc[name]
        ax = np.atleast_1d(c["alpha"])
        for i in range(1, 5):
            ax = ax + c["beta_{0:d}".format(i)] * bprp_ ** i
        for j in range(1, 4):
            ax = ax + (c["gamma_{0:d}".format(j)] * ag_ ** j)
        ax += c["delta"] * bprp_ * ag_
        return np.squeeze(ax)