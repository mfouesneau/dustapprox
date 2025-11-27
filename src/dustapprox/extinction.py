r"""
Dust Extinction curves
----------------------

The observations show a wide range of dust column normalized extinction curves,
:math:`A(\lambda) / A(V)`.  This package provides a common interface to the `dust_extinction`_ package
which includes commonly used extinction curves.

.. _dust_extinction: http://dust-extinction.readthedocs.io/

.. note::

    This module is able to handle values with units from `pyphot.astropy` (intefaced to this package) and `astropy`.
    We recommend the users to provide units in their inputs.


**Example of comparing extinction curves**

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u

    from dustapprox.extinction import evaluate_extinction_model

    #define the wave numbers
    x = np.arange(0.1, 10, 0.1)    # in microns^{-1}
    λ = 1. / x * u.micron

    curves = ["CCM89", "F99", "O94", "G23"]
    R0 = 3.1

    for name in curves:
        values = evaluate_extinction_model(name, λ, A0=1.0, R0=R0)
        plt.plot(x, values, label=f'{name:s}, R(V) = {R0:0.1f}', lw=2)
    plt.xlabel(r'Wave number [$\mu$m$^{-1}$]')
    plt.ylabel(r'$A(x)/A(V)$')
    plt.legend(loc='upper left', frameon=False, title='Ext. Curve')
    plt.tight_layout()
    plt.show()

"""

from typing import Union

import astropy.units as u
import dust_extinction.parameter_averages
import numpy as np
import numpy.typing as npt
from dust_extinction.parameter_averages import BaseExtRvModel

from .astropy_units import val_in_unit

__all__ = ["evaluate_extinction_model", "get_extinction_model", "BaseExtRvModel"]


def get_extinction_model(name: Union[str, BaseExtRvModel]) -> BaseExtRvModel:
    """Get an extinction model from dust_extinction by name

    Parameters
    ----------
    name: str or BaseExtRvModel
        Name of the extinction model from `dust_extinction.parameter_averages`
        or an instance of `dust_extinction.parameter_averages.BaseExtRvModel`

    Returns
    -------
    BaseExtRvModel
        Instance of the requested extinction model
    """
    if isinstance(name, BaseExtRvModel):
        return name
    elif name not in dust_extinction.parameter_averages.__all__:
        raise ValueError(
            f"Extinction model '{name}' not found in dust_extinction.parameter_averages"
        )
    else:
        return getattr(dust_extinction.parameter_averages, name)


def evaluate_extinction_model(
    name: Union[str, BaseExtRvModel],
    λ: Union[npt.ArrayLike, u.Quantity],
    A0: float,
    R0: float,
    *,
    extrapolate: bool = True,
) -> np.ndarray:
    """Evaluate an extinction model from dust_extinction by name at given wavelengths

    This function is a quick interface that makes sure the wavelength input is
    properly converted to the expected units and that the extinction model is
    properly instantiated.

    Note: dust_extinction assumes evaluation at wavenumbers, here in wavelengths.

    Parameters
    ----------
    name: str or BaseExtRvModel
        Name of the extinction model from `dust_extinction.parameter_averages`
        or an instance of `dust_extinction.parameter_averages.BaseExtRvModel`
    λ: array-like or Quantity
        Wavelength(s) at which to evaluate the extinction model
    A0: float
        Initial amplitude of the extinction curve
    R0: float
        Initial slope of the extinction curve
    extrapolate: bool
        Whether to allow extrapolation beyond the model's valid range in R0

    Returns
    -------
    np.ndarray
        Evaluated extinction curve values at the given wavelengths
    """
    # make sure we have an instance of extinction curve
    model_cls: BaseExtRvModel = get_extinction_model(name)
    λ_ = val_in_unit("extinction wavelength", λ, "angstrom").to("micron")
    # convert wavelength to wavenumber (inverse in microns)
    x = 1.0 / λ_
    if extrapolate:
        # prepare output array of A(λ)/A0
        τ = np.zeros(λ_.shape, dtype=float) * float("nan")
        # allow extrapolation!
        #  Above Rv=6.3, G23 FUV extinction becomes lower at shorter wavelengths
        #  Below Rv=2.2, G23 MIR before 10um silicates goes unrealistically small
        model_cls.Rv_range = [2.0, 6.51]
        model = model_cls(Rv=R0)
        valid = (x.value < model.x_range[1]) & (x.value > model.x_range[0])
        τ_ = model(x[valid])
        τ[valid] = τ_
    else:
        τ = model_cls(Rv=R0)(x)
    # return extinction curve evaluated at given x, A0, R0
    return τ * A0
