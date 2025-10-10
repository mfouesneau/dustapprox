r"""  Coefficients of the reddening laws to correct magnitudes in the :math:`C1` system.

The :math:`C1` Gaia photometric system
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original design for Gaia included a set of photometric filters (see `Jordi et
al., 2006 <https://academic.oup.com/mnras/article/367/1/290/1018790>`_), the C1B and C1M systems for the broad and medium band passbands,
respectively.  The C1 system was specifically desgined to maximize the
scientific return in terms of stellar astrophysical parameters.

The C1B component has five broad passbands covering the wavelength range of the
unfiltered light from the blue to the far-red (i.e.  400-1000 nm). The basic
response curve of the filters versus wavelength is a symmetric quasi-trapezoidal
shape. The filter designs represent a compromise between the astrophysical needs and the
specific requirements for chromaticity calibration of Gaia.

The C1M component consists of 14 passbands. Their basic response curves are
symmetric quasi-trapezoidal shapes as well.

Eventually, construction and budget constraints led ESA to adopt prisms in the
final design of Gaia, which cover for those passbands.

For example :math:`C1M467-C1M515` color is sensitive to
surface gravity (:math:`\log g`) and :math:`C1B556-C1B996` is sensitive to the effective
temperature :math:`T_{eff}`.

**Comments on the Passbands and colors** (see details in Sect. 5.1 of `Jordi et al., 2006`_)

* C1B431: similar to that of a Johnson's B passband.
* C1B556: similar to that of a Johnson's V passband.
* C1B768: similar to that of a Cousin's I passband, SDSS i prime, or HST F814W
* C1M326: is affected by strong metallicity dependent absorption lines.
* C1M379: corresponding to the higher energy levels of the Balmer series.
* C1M395: mainly to measure the Ca II H line
* C1M467: similar to Strömgren b
* C1M549: similar to Strömgren y
* C1M515: dominated by Mg I triplet and the MgH band.
* C1M656: is a direct Halpha line measurement.
* C1M716: dominated by TiO (713 nm)
* C1M825: mostly the strong Carbon-Nitrogen (CN), but also weak TiO bands


* C1B431-C1B556: is similar to Johnson's B-V.
* C1B556-C1B768: is similar to a V-I color, HST F555W-F814W.
* C1B655-C1M656: forms an Halpha index
* C1B768-C1B916: strength of the Paschen jump.
* C1M326-C1B431: is a Balmer jump index, function of Teff and log g
* C1M326-C1M410: is a Balmer jump index, function of Teff and log g
* C1M379-C1M467: is a metallicity index.
* C1M395-C1M410: correlates with W(CaT*), i.e. the equivalent width of the Calcium triplet (in RVS).
* C1M395-C1M515: (and similar indices) allows to disentangle Fe and alpha-process element abundances.
* C1M716-C1M747: is an index for the presence and intensity of TiO.
* C1M861-C1M965: is an index of gravity-sensitive absorption of the high member lines of the Paschen series.

* C1B556-C1B768, C1B556-C1B916, C1B768-C1B916 indices can allow for the
  separation of cool oxygen-rich (M) and carbon-rich (N) stars (for dust-free stars).
* C1M395-C1M410 vs. W(CaT*) may be used as a log g estimator.
* C1M825-C1M861 vs. C1M861-C1M965 helps separating M-, R- and N-type stars


C1 passbands transmission curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The C1 passbands are available from `Jordi et al., 2006`_.

For convenience, we also provide them with this package in
`dustapprox/data/Gaia2` directory in the `pyphot
<https://mfouesneau.github.io/pyphot/>`_ ascii format.


.. code-block:: python
    :caption: Use the C1 passbands transmission curves with `pyphot`_

    from pkg_resources import resource_filename
    from pyphot.astropy import UnitAscii_Library

    where = resource_filename('dustapprox', 'data/Gaia2')
    lib = UnitAscii_Library([where])
    lib['C1B556'].info()

The last line of the above example shows the information of the C1B556 passband.

.. code-block:: text


    Filter object information:
        name:                 C1B556
        detector type:        photon
        wavelength units:     nm
        central wavelength:   556.000128 nm
        pivot wavelength:     554.743690 nm
        effective wavelength: 548.702778 nm
        photon wavelength:    551.180067 nm
        minimum wavelength:   481.000000 nm
        maximum wavelength:   631.000000 nm
        norm:                 115.200850
        effective width:      128.000944 nm
        fullwidth half-max:   0.500000 nm
        definition contains 337 points

        Zeropoints
            Vega: 21.142893 mag,
                3.490138982485388e-09 erg / (Angstrom cm2 s),
                3582.669614072932 Jy
                879.1910941155257 ph / (Angstrom cm2 s)
            AB: 21.128410 mag,
                3.5370073412867404e-09 erg / (Angstrom cm2 s),
                3630.780547700996 Jy
            ST: 21.100000 mag,
                3.6307805477010028e-09 erg / (Angstrom cm2 s),
                3727.0398711607527 Jy

.. plot::
    :include-source:
    :caption: C1 passband transmission curves. Black and blue lines indicate the C1B and C1M passbands, respectively.

    import pylab as plt
    from pkg_resources import resource_filename
    from pyphot.astropy import UnitAscii_Library

    where = resource_filename('dustapprox', 'data/Gaia2')
    lib = UnitAscii_Library([where])
    pbs = lib.load_all_filters()

    plt.figure(figsize=(8, 4))
    for pb in pbs:
        if pb.name.startswith('C1B'):
            kwargs = dict(color='k', lw=2)
        else:
            kwargs = dict(color='C0', lw=1)
        plt.plot(pb.wavelength.to('nm'), pb.transmit, **kwargs)
    plt.xlabel('wavelength [nm]')
    plt.ylabel('throughput')
    plt.tight_layout()
    plt.show()


Model from Bellazzini et al. (2022; Gaia DR3)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The procedure is detailed in the appendix E of `Bellazzini et al. (2022) <https://www.cosmos.esa.int/web/gaia/dr3-papers>`_.

In brief, they fit polynomial relations to predicted SED values.
The simulated predictions originate from the BTSettl atmosphere library combined
with the :math:`C1` passbands definitions and the Fitzpatrick extinction curve.

They derive the absorption in any :math:`X` band as a function of the absorption in :math:`G`, :math:`A_G`,
the Gaia :math:`G_{BP}-G_{RP}` color:

.. math::

        \frac{A_X}{A_G}=\alpha+\sum_{i=1}^{4}\beta_i \cdot ({G_{BP}-G_{RP}})^i+\sum_{j=1}^3 \gamma_j \cdot A_G^j + \delta \cdot ({G_{BP}-G_{RP}})\cdot A_G

.. warning::

    Those relations depend on :math:`A_G`, not :math:`A_0`.

"""
from typing import Union
import numpy as np
from ..io import ecsv


class dr3_ext:
    """Function to get the extinction coefficients A_X / A_G in C1 system

    The procedure is detailed in the appendix E of `Bellazzini et al. (2022)`_.

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