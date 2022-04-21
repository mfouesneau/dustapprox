"""
Dust Extinction curves
----------------------

The observations show a wide range of dust column normalized extinction curves,
:math:`A(\lambda) / A(V)`.  This package provides a common interface to many
commonly used extinction curves.

.. note::

    This module is able to handle values with units from `pyphot.astropy` (intefaced to this package) and `astropy`.
    We recommend the users to provide units in their inputs.


**Example of comparing extinction curves**

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u

    from dustapprox.extinction import CCM89, F99

    #define the wave numbers
    x = np.arange(0.1, 10, 0.1)    # in microns^{-1}
    lamb = 1. / x * u.micron

    curves = [CCM89(), F99()]
    Rv = 3.1

    for c in curves:
        name = c.name
        plt.plot(x, c(lamb, Rv=Rv), label=f'{name:s}, R(V) = {Rv:0.1f}', lw=2)
    plt.xlabel(r'Wave number [$\mu$m$^{-1}$]')
    plt.ylabel(r'$A(x)/A(V)$')
    plt.legend(loc='upper left', frameon=False, title='Ext. Curve')
    plt.tight_layout()
    plt.show()


"""
import warnings
import numpy as np
from scipy import interpolate
from pyphot.astropy.sandbox import Unit as U
from typing import Union, Sequence
Quantity = type(U())


__all__ = ['ExtinctionLaw', 'CCM89', 'F99']

def _warning_on_one_line(message, category, filename, lineno, file=None, line=None) -> str:
    """ Prints a complete warning that includes exactly the code line triggering it from the stack trace. """
    return " {0:s}:{1:d} {2:s}:{3:s}".format(filename, lineno,
                                            category.__name__, str(message))


def _val_in_unit(varname: str, value: object, defaultunit: str) -> Quantity:
    """ check units and convert to defaultunit or create the unit information

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

    if not (hasattr(value, 'unit') or hasattr(value, 'units')):
        warnings.formatwarning = _warning_on_one_line
        msg = 'Variable {0:s} does not have explicit units. Assuming `{1:s}`\n'
        # stacklevel makes the correct code reference
        warnings.warn(msg.format(varname, defaultunit), stacklevel=4)
        return value * U(defaultunit)
    else:
        return value.to(defaultunit)


class ExtinctionLaw(object):
    """ Template function class

    Attributes
    ----------
    name: str
        name of the curve

    Parameters
    ----------
    lamb: float, np.array, Quantity
        wavelength. User should prefer a Quantity which provides units.

    Returns
    -------
    val: ndarray
        expected values of the law evaluated at lamb

    """
    def __init__(self):
        self.name = 'None'

    def __repr__(self):
        return '{0:s}\n{1:s}'.format(self.name, object.__repr__(self))

    def __call__(self, lamb: Union[float, np.array, Quantity], *args, **kwargs) -> np.array:
        """ Make the extinction law callable object using :func:`self.function`"""
        raise NotImplementedError

    def isvalid(self, *args, **kwargs):
        """ Check if the current arguments are in the validity domain of the law
        Must be redefined if any restriction applies to the law
        """
        return True


class CCM89(ExtinctionLaw):
    """ Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model.

    from Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245)

    **Example showing CCM89 curves for a range of R(V) values.**

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dustapprox.extinction import CCM89

        #define the wave numbers
        x = np.arange(0.1, 10, 0.1)    # in microns^{-1}
        lamb = 1. / x * u.micron

        c = CCM89()
        Rvs = np.arange(2, 6.01, 1.)

        for Rv in Rvs:
            plt.plot(x, c(lamb, Rv=Rv), label=f'R(V) = {Rv:0.1f}', lw=2)
        plt.xlabel(r'Wave number [$\mu$m$^{-1}$]')
        plt.ylabel(r'$A(x)/A(V)$')
        plt.legend(loc='upper left', frameon=False, title='CCM (1989)')
        plt.tight_layout()
        plt.show()


    """
    def __init__(self):
        self.name = 'CCM89'
        self.long_name = 'Cardelli, Clayton, & Mathis (1989)'

    def __call__(self, lamb, Av=1., Rv=3.1, Alambda=True, **kwargs):
        """ Cardelli extinction curve

        Parameters
        ----------
        lamb: float or ndarray(dtype=float)
            wavelength [in Angstroms] at which evaluate the law.

        Av: float
            desired A(V) (default: 1.0)

        Rv: float
            desired R(V) (default: 3.1)

        Alambda: bool
            if set returns +2.5*1./log(10.)*tau, tau otherwise

        Returns
        -------
        r: float or ndarray(dtype=float)
            attenuation as a function of wavelength
            depending on Alambda option +2.5*1./log(10.)*tau,  or tau

        """
        _lamb = _val_in_unit('lamb', lamb, 'angstrom').value

        if isinstance(_lamb, float) or isinstance(_lamb, np.float_):
            _lamb = np.asarray([_lamb])
        else:
            _lamb = _lamb[:]

        # init variables
        x = 1.e4 / _lamb  # wavenumber in um^-1
        a = np.zeros(np.size(x))
        b = np.zeros(np.size(x))
        # Infrared (Eq 2a,2b)
        ind = np.where((x >= 0.3) & (x < 1.1))
        a[ind] =  0.574 * x[ind] ** 1.61
        b[ind] = -0.527 * x[ind] ** 1.61
        # Optical & Near IR
        # Eq 3a, 3b
        ind = np.where((x >= 1.1) & (x <= 3.3))
        y = x[ind] - 1.82
        a[ind] = 1. + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4 + 0.01979 * y ** 5 - 0.77530 * y ** 6 + 0.32999 * y ** 7
        b[ind] =      1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4 - 0.62251 * y ** 5 + 5.30260 * y ** 6 - 2.09002 * y ** 7
        # UV
        # Eq 4a, 4b
        ind = np.where((x >= 3.3) & (x <= 8.0))
        a[ind] =  1.752 - 0.316 * x[ind] - 0.104 / ((x[ind] - 4.67) ** 2 + 0.341)
        b[ind] = -3.090 + 1.825 * x[ind] + 1.206 / ((x[ind] - 4.62) ** 2 + 0.263)

        ind = np.where((x >= 5.9) & (x <= 8.0))
        Fa     = -0.04473 * (x[ind] - 5.9) ** 2 - 0.009779 * (x[ind] - 5.9) ** 3
        Fb     =  0.21300 * (x[ind] - 5.9) ** 2 + 0.120700 * (x[ind] - 5.9) ** 3
        a[ind] = a[ind] + Fa
        b[ind] = b[ind] + Fb
        # Far UV
        # Eq 5a, 5b
        ind = np.where((x >= 8.0) & (x <= 10.0))
        # Fa = Fb = 0
        a[ind] = -1.073 - 0.628 * (x[ind] - 8.) + 0.137 * ((x[ind] - 8.) ** 2) - 0.070 * (x[ind] - 8.) ** 3
        b[ind] = 13.670 + 4.257 * (x[ind] - 8.) + 0.420 * ((x[ind] - 8.) ** 2) + 0.374 * (x[ind] - 8.) ** 3

        # Case of -values x out of range [0.3,10.0]
        ind = np.where((x > 10.0) | (x < 0.3))
        a[ind] = 0.0
        b[ind] = 0.0

        # Return Extinction vector
        # Eq 1
        if (Alambda):
            return( ( a + b / Rv ) * Av)
        else:
            # return( 1./(2.5 * 1. / np.log(10.)) * ( a + b / Rv ) * Av)
            return( 0.4 * np.log(10.) * ( a + b / Rv ) * Av)


class F99(ExtinctionLaw):
    """ Fitzpatrick (1999, PASP, 111, 63) [1999PASP..111...63F]_

    R(V) dependent extinction curve that explicitly deals with optical/NIR
    extinction being measured from broad/medium band photometry.
    Based on fm_unred.pro from the IDL astronomy library


    Parameters
    ----------
    lamb: float or ndarray(dtype=float) or Quantity
        wavelength at which evaluate the law.
        units are assumed to be Angstroms if not provided

    Av: float, optional
        desired A(V) (default 1.0)

    Rv: float, optional
        desired R(V) (default 3.1)

    Alambda: bool, optional
        if set returns +2.5*1./log(10.)*tau, tau otherwise

    Returns
    -------
    r: float or ndarray(dtype=float)
        attenuation as a function of wavelength
        depending on Alambda option +2.5*1./log(10.)*tau,  or tau


    **Example showing F99 curves for a range of R(V) values.**

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dustapprox.extinction import F99

        #define the wave numbers
        x = np.arange(0.1, 10, 0.1)    # in microns^{-1}
        lamb = 1. / x * u.micron

        c = F99()
        Rvs = np.arange(2, 6.01, 1.)

        for Rv in Rvs:
            plt.plot(x, c(lamb, Rv=Rv), label=f'R(V) = {Rv:0.1f}', lw=2)
        plt.xlabel(r'Wave number [$\mu$m$^{-1}$]')
        plt.ylabel(r'$A(x)/A(V)$')
        plt.legend(loc='upper left', frameon=False, title='Fitzpatrick (1999)')
        plt.tight_layout()
        plt.show()


    .. [1999PASP..111...63F] http://adsabs.harvard.edu/abs/1999PASP..111...63F

    .. note::

        this function assumed the wavelength in Anstroms if `lamb` is not a
        `Quantity`.

    """
    def __init__(self):
        self.name = 'F99'
        self.long_name = 'Fitzpatrick (1999)'

    def __call__(self,
                 lamb: Union[float, np.array, Quantity],
                 Av: float = 1, Rv: float = 3.1,
                 Alambda: bool = True,
                 **kwargs):
        """
        Fitzpatrick99 extinction curve

        .. note::

            this function assumed the wavelength in Anstroms if `lamb` is not a
            `Quantity`.

        Parameters
        ----------
        lamb: float or ndarray(dtype=float) or Quantity
            wavelength at which evaluate the law.
            units are assumed to be Angstroms if not provided

        Av: float, optional
            desired A(V) (default 1.0)

        Rv: float, optional
            desired R(V) (default 3.1)

        Alambda: bool, optional
            if set returns +2.5*1./log(10.)*tau, tau otherwise

        Returns
        -------
        r: float or ndarray(dtype=float)
            attenuation as a function of wavelength
            depending on Alambda option +2.5*1./log(10.)*tau,  or tau
        """
        _lamb = _val_in_unit('lamb', lamb, 'angstrom').value

        if isinstance(_lamb, float) or isinstance(_lamb, np.float_):
            _lamb = np.asarray([_lamb])
        else:
            _lamb = _lamb[:]

        c2 = -0.824 + 4.717 / Rv
        c1 = 2.030 - 3.007 * c2
        c3 = 3.23
        c4 = 0.41
        x0 = 4.596
        gamma = 0.99

        x = 1.e4 / _lamb
        k = np.zeros(np.size(x))

        # compute the UV portion of A(lambda)/E(B-V)
        xcutuv = 10000.0 / 2700.
        xspluv = 10000.0 / np.array([2700., 2600.])
        yspluv = c1 + (c2 * xspluv) + c3 * ((xspluv) ** 2) / ( ((xspluv) ** 2 - (x0 ** 2)) ** 2 + (gamma ** 2) * ((xspluv) ** 2 ))
        ind = (x >= xcutuv)

        if True in ind:
            k[ind] = c1 + (c2 * x[ind]) + c3 * ((x[ind]) ** 2) / ( ((x[ind]) ** 2 - (x0 ** 2)) ** 2 + (gamma ** 2) * ((x[ind]) ** 2 ))

            # FUV portion
            fuvind = np.where(x >= 5.9)
            k[fuvind] += c4 * (0.5392 * ((x[fuvind] - 5.9) ** 2) + 0.05644 * ((x[fuvind] - 5.9) ** 3))

            k[ind] += Rv
            yspluv += Rv

        # Optical/NIR portion

        ind = x < xcutuv
        if True in ind:
            xsplopir = np.zeros(7)
            xsplopir[0] = 0.0
            xsplopir[1: 7] = 10000.0 / np.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0])

            ysplopir = np.zeros(7)
            ysplopir[0: 3] = np.array([0.0, 0.26469, 0.82925]) * Rv / 3.1

            ysplopir[3: 7] = np.array([np.poly1d([2.13572e-04, 1.00270, -4.22809e-01])(Rv),
                                       np.poly1d([-7.35778e-05, 1.00216, -5.13540e-02])(Rv),
                                       np.poly1d([-3.32598e-05, 1.00184,  7.00127e-01])(Rv),
                                       np.poly1d([ 1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1])(Rv)])

            tck = interpolate.splrep(np.hstack([xsplopir, xspluv]), np.hstack([ysplopir, yspluv]), k=3)
            k[ind] = interpolate.splev(x[ind], tck)

        # convert from A(lambda)/E(B-V) to A(lambda)/A(V)
        k /= Rv

        if (Alambda):
            return(k * Av)
        else:
            return(k * Av * (np.log(10.) * 0.4))