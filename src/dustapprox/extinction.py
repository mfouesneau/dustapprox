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

import warnings
from contextlib import contextmanager
from typing import Iterator, Sequence, Union

import astropy.units as u
import dust_extinction.parameter_averages
import numpy as np
import numpy.typing as npt
from dust_extinction.parameter_averages import BaseExtRvModel

from .astropy_units import val_in_unit

__all__ = [
    "evaluate_extinction_model",
    "get_extinction_model",
    "BaseExtRvModel",
]


def _allow_F9x_extrapolation() -> None:
    """Patch to dust_extinction to allow F99 wavelength extrapolation

    Bug reported to dust_extinction [issue #262](https://github.com/karllark/dust_extinction/issues/262).
    """
    from dust_extinction.shapes import FM90

    FM90.x_range = [0.001, 100]


# Curves whose native range spans the optical to the far-UV can be evaluated
# beyond their edges (smooth low-order forms). Curves defined on one region only
# (e.g., D22 near-IR, GCC09/VCG04 UV, M14 optical) blow up or return 0 outside
# it, so they are left NaN there instead.
_EXTRAPOLABLE_X_SPAN = (0.3, 8.0)  # [micron^-1] native x_range must cover this
_EXTRAPOLATION_X_RANGE = [0.001, 100.0]
#  Above Rv=6.3, G23 FUV extinction becomes lower at shorter wavelengths
#  Below Rv=2.2, G23 MIR before 10um silicates goes unrealistically small
_EXTRAPOLATION_RV_RANGE = [2.0, 6.51]


def _can_extrapolate_x(model_cls: type) -> bool:
    """Whether the curve's native x range covers the optical to the far-UV"""
    xmin, xmax = model_cls.x_range  # pyright: ignore[reportAttributeAccessIssue]
    return (xmin <= _EXTRAPOLABLE_X_SPAN[0]) and (
        xmax >= _EXTRAPOLABLE_X_SPAN[1]
    )


@contextmanager
def _class_ranges(
    model_cls: type, **ranges: Sequence[float]
) -> Iterator[None]:
    """Temporarily override class-level validity ranges (e.g., x_range, Rv_range)

    dust_extinction checks these class attributes; restoring them afterwards keeps
    later calls (e.g., with ``extrapolate=False``) validating against the native
    ranges.
    """
    missing = object()
    saved = {k: model_cls.__dict__.get(k, missing) for k in ranges}
    try:
        for k, v in ranges.items():
            setattr(model_cls, k, list(v))
        yield
    finally:
        for k, v in saved.items():
            if v is missing:
                delattr(model_cls, k)
            else:
                setattr(model_cls, k, v)


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
    model_cls = get_extinction_model(name)
    if not isinstance(model_cls, type):
        # An instance carries its own Rv: evaluate the same curve at R0 instead
        # of silently ignoring R0 (e.g., when looping over an R0 grid).
        instance_rv = float(model_cls.Rv.value)
        if not np.isclose(instance_rv, R0):
            warnings.warn(
                f"{type(model_cls).__name__} instance has Rv={instance_rv:g} "
                f"but R0={R0:g} was requested; using R0={R0:g}.",
                stacklevel=2,
            )
        model_cls = type(model_cls)
    λ_ = val_in_unit("extinction wavelength", λ, "angstrom").to("micron")
    # convert wavelength to wavenumber (inverse in microns)
    x = 1.0 / λ_
    if extrapolate:
        # prepare output array of A(λ)/A0: NaN where the curve is not defined
        τ = np.full(λ_.shape, np.nan, dtype=float)
        # Rv extrapolation for all curves; x extrapolation only where it is safe.
        # Because models are low order polynomials this should be ok, but no
        # guarantees.
        x_range = (
            _EXTRAPOLATION_X_RANGE
            if _can_extrapolate_x(model_cls)
            else list(model_cls.x_range)  # pyright: ignore[reportAttributeAccessIssue]
        )
        with _class_ranges(
            model_cls, x_range=x_range, Rv_range=_EXTRAPOLATION_RV_RANGE
        ):
            if model_cls.__name__.startswith("F9"):
                _allow_F9x_extrapolation()
            model = model_cls(Rv=R0)
            valid = (x.value < x_range[1]) & (x.value > x_range[0])
            if np.any(valid):
                τ[valid] = model(x[valid])
    else:
        τ = model_cls(Rv=R0)(x)
    # return extinction curve evaluated at given x, A0, R0
    return τ * A0
