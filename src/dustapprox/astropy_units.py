"""
Declare missing photometric and spectral units for use with astropy.
"""

from typing import Any, Union
import warnings

from astropy.units import Unit, def_unit, add_enabled_units
from astropy.units import Quantity
from astropy.units.core import Unit as U

__all__ = ["Unit", "Quantity", "has_unit", "val_in_unit", "U"]

new_units = dict(
    flam="erg * s ** (-1) * AA ** (-1) * cm **(-2)",
    fnu="erg * s ** (-1) * Hz ** (-1) * cm **(-2)",
    photflam="photon * s ** (-1) * AA ** (-1) * cm **(-2)",
    photfnu="photon * s ** (-1) * Hz ** (-1) * cm **(-2)",
    angstroms="angstrom",
    lsun="Lsun",
    ergs="erg",
)

add_enabled_units([def_unit([k], Unit(v)) for k, v in new_units.items()]).__enter__()


def _warning_on_one_line(
    message: str, category: Any, filename: str, lineno: int, file=None, line=None
) -> str:
    """Prints a complete warning that includes exactly the code line triggering it from the stack trace."""
    return " {:s}:{:d} {:s}:{:s}".format(
        filename, lineno, category.__name__, str(message)
    )


def has_unit(val: Any) -> bool:
    """Check if a unit is defined in astropy."""
    return hasattr(val, "units") or hasattr(val, "unit")


def val_in_unit(
    varname: str,
    value: Union[Any, Quantity],
    defaultunit: str,
) -> Quantity:
    """check units and convert to defaultunit or create the unit information

    Parameters
    ----------
    varname: str
        name of the variable
    value: object
        value of the variable, which may be unitless
    defaultunit: str
        default units is unitless

    Returns
    -------
    quantity: Quantity
        value with units

    Example
    -------
    >>> r = 0.5
    >>> print(val_in_unit('r', r, 'degree'))
    # UserWarning: Variable r does not have explicit units. Assuming `degree`
    <Quantity(0.5, 'degree')>
    >>> r = 0.5 * unit['degree']
    >>> print(val_in_unit('r', r, 'degree'))
    <Quantity(0.5, 'degree')>
    """

    if not has_unit(value):
        warnings.formatwarning = _warning_on_one_line
        msg = "Variable {0:s} does not have explicit units. Assuming `{1:s}`\n"
        # stacklevel makes the correct code reference
        warnings.warn(msg.format(varname, defaultunit), stacklevel=4)
        return value * U(defaultunit)
    else:
        return value.to(defaultunit)
