r"""
Provide extinction coefficient :math:`k_x = A_x / A_0` for Gaia eDR3 passbands (G, BP, RP).

Data and equations taken from: https://www.cosmos.esa.int/web/gaia/edr3-extinction-law

Extinction coefficients depend on the source spectral energy distribution
and on the extinction itself (e.g., Gordon et al., 2016; Jordi et al., 2010).

Following the method presented in Danielski et al. (2018), DPAC computed the
extinction coefficient in the :math:`x` band

.. math::

    k_x = A_x / A_0,

with :math:`A_0` the extinction at :math:`550 nm`, as a function of the star's
intrinsic color or effective temperature (both denoted by :math:`X`):

.. math::

    k_x = a_1 &+ a_2 \times X + a_3 \times X^2 + a_4 \times X^3 \\
            &+ a_5 \times A0 + a_6 \times A_0^2 + a_7 \times A_0^3 \\
            &+ a_8 \times A_0 \times X + a_9 \times A_0 \times X^2 \\
            &+ a_10 \times X \times A_0^2

Riello et al. (2020) fit the above formula on a grid of extinctions obtained by convolving
the Gaia eDR3 passbands with Kurucz spectra (Castelli & Kurucz, 2003) and the
Fitzpatrick et al. (2019) extinction curve. They constructed a grid with :math:`3500 K < T_{eff}
< 10000 K` in steps of :math:`250 K`, and :math:`0.01 < A_0 < 20 mag` with a step linearly
increasing with :math:`0.01 mag`.

.. warning::

    Their work only include solar metallicity.

.. note::

    * The work cannot be reproduced from the information contained on the webpage only.

    * Internally, they did a fit for main-sequence stars (:math:`log g > 4.5`)
      and for the upper HR diagram (giants, upper MS, etc.) up to :math:`M_G
      \sim 5 mag`.

"""
from typing import Union, Sequence
from pkg_resources import resource_filename
import pandas as pd
import numpy as np
import os

_DATA_PATH_ = resource_filename('dustapprox', 'data')

class edr3_ext:
    """ provide extinction coefficient :math:`k_x = A_x / A_0` for Gaia eDR3 passbands (G, BP, RP)

    This class provides a simple access to their expressions
    for :math:`X=(G_{BP}-G_{RP})_0, (G-K)_0`, and :math:`T_{eff}`,
    and for the bands :math:`m = G, GBP, GRP`,  and :math:`J, H`, and
    :math:`Ks`, corresponding to the 2MASS passbands.  The fit itself has a
    maximum uncertainty of 3.5%, 1.5%, and 1% in the :math:`G, G_{BP}`, and
    :math:`G_{RP}` bands, respectively.

    .. note::

        The relations with temperature use internally :math:`T_{eff}^{Norm} = T_{eff} / 5040 K`, but
        the function argument is :math:`T_{eff}` for simplicity.

    Data taken from: https://www.cosmos.esa.int/web/gaia/edr3-extinction-law
    """
    def __init__(self):
        datafiles = (
            "Fitz19_EDR3_extinctionlawcoefficients/Fitz19_EDR3_HRDTop.csv",
            "Fitz19_EDR3_extinctionlawcoefficients/Fitz19_EDR3_MainSequence.csv")
        self.Ay_top = pd.read_csv(os.path.join(_DATA_PATH_, datafiles[0])).set_index(['Kname', 'Xname'])
        self.Ay_ms = pd.read_csv(os.path.join(_DATA_PATH_, datafiles[1])).set_index(['Kname', 'Xname'])

    def _from(self, name: str, Xname: str,
              Xval: Union[float, Sequence[float], np.array],
              a0: Union[float, Sequence[float], np.array],
              flavor: str ='top'
             ) -> Union[float, Sequence[float], np.array]:
        """Internal access to the equations

        Parameters
        ----------
        name : str
            The name of the band ('G', 'BP', or 'RP')
        Xname : str
            The name of the variable to be used as X (e.g., TeffNorm, BPRP, GK)
        Xval : float or array_like
            The value of the variable to be used as X
        a0 : float or array_like
            The value of the extinction at 550 nm
        flavor : str
            The type of the data to be used ('top' or 'ms')

        Returns
        -------
        float or array_like
            The extinction coefficient A_x / A_0
        """
        if flavor == 'top':
            data = self.Ay_top
        else:
            data = self.Ay_ms
        coeffs = data.loc[name, Xname]
        X_ = np.atleast_1d(Xval)
        X_2 = X_ ** 2
        X_3 = X_ ** 3
        a0_ = np.atleast_1d(a0)
        a0_2 = a0_ ** 2
        a0_3 = a0_ ** 3

        ay = coeffs['Intercept']
        for key, Xk in zip(['X', 'X2', 'X3'], [X_, X_2, X_3]):
            ay = ay + coeffs[key] * Xk
        for key, Ak in zip(['A', 'A2', 'A3'], [a0_, a0_2, a0_3]):
            ay = ay + coeffs[key] * Ak

        ay += (coeffs['XA'] * X_ * a0_ +
               coeffs['XA2'] * X_ * a0_2 +
               coeffs['AX2'] * X_2 * a0_)
        return ay

    def from_teff(self, name: str,
              teff: Union[float, Sequence[float], np.array],
              a0: Union[float, Sequence[float], np.array],
              flavor: str ='top'
             ) -> Union[float, Sequence[float], np.array]:
        """Relation based on temperature of the source

        Parameters
        ----------
        name : str
            The name of the band ('G', 'BP', or 'RP')
        teff : float or array_like
            The values of :math:`T_{eff}`
        a0 : float or array_like
            The value of the extinction at 550 nm
        flavor : str
            The type of the data to be used ('top' or 'ms')

        Returns
        -------
        float or array_like
            The extinction coefficient A_x / A_0
        """
        teffnorm = teff / 5040.
        return self._from(name, 'TeffNorm', teffnorm, a0, flavor)

    def from_bprp(self, name: str,
              bprp: Union[float, Sequence[float], np.array],
              a0: Union[float, Sequence[float], np.array],
              flavor: str ='top'
             ) -> Union[float, Sequence[float], np.array]:
        """Relation based on BP-RP of the source

        Parameters
        ----------
        name : str
            The name of the band ('G', 'BP', or 'RP')
        bprp : float or array_like
            The :math:`G_{BP} - G_{RP}` color values
        a0 : float or array_like
            The value of the extinction at 550 nm
        flavor : str
            The type of the data to be used ('top' or 'ms')

        Returns
        -------
        float or array_like
            The extinction coefficient A_x / A_0
        """
        return self._from(name, 'BPRP', bprp, a0, flavor)

    def from_GmK(self, name: str,
              gmk: Union[float, Sequence[float], np.array],
              a0: Union[float, Sequence[float], np.array],
              flavor: str ='top'
             ) -> Union[float, Sequence[float], np.array]:
        """Relation based on G-Ks of the source

        Parameters
        ----------
        name : str
            The name of the band ('G', 'BP', or 'RP')
        gmk : float or array_like
            The G-Ks color values
        a0 : float or array_like
            The value of the extinction at 550 nm
        flavor : str
            The type of the data to be used ('top' or 'ms')

        Returns
        -------
        float or array_like
            The extinction coefficient A_x / A_0
        """
        return self._from(name, 'GK', gmk, a0, flavor)