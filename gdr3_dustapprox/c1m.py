from typing import Union
import numpy as np


class C1M_extinction:
    """Function to get the extinction coefficients in C1M system

    obtained when fitting D.1 from Bellazzini et al. (2022) to the passbands in the C1 system
    using BTSettl SED library.
    """
    def __init__(self, data='gaia_C1_extinction.ecsv'):
        self.data = IO_ECSV.read('gaia_C1_extinction.ecsv').set_index('X')

    def __call__(self,
                 name: str,
                 bprp: Union[float, np.array],
                 ag:float) -> Union[float, np.array]:
        """
        Returns the Ax/Ag values
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