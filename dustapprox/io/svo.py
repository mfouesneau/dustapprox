"""
Interface to the SVO Theoretical spectra web server and Filter Profile Service.

The SVO Theory Server provides data for many sources of theoretical spectra
and observational templates.

* SVO Theoretical spectra: http://svo2.cab.inta-csic.es/theory/newov2/index.php
* SVO Filter Profile Service: http://svo2.cab.inta-csic.es/theory/fps/index.php

.. todo::

    * Add support for the SVO Theoretical spectra service (create one similarly to pyphot, though it may be harder).
        * currently manual download of the data. It's not that hard, but we could put some guidance in our documentation.
"""
from typing import Sequence, Union
import pandas as pd
from pyphot.astropy.sandbox import Unit as U
from pyphot.astropy.sandbox import UnitFilter
from pyphot.svo import get_pyphot_astropy_filter
Quantity = type(U())


def spectra_file_reader(fname: str) -> dict:
    """Read the model file from the SVO Theoretical spectra service.

    They all have the same format, but the spectra are not on the same
    wavelength scale (even for a single atmosphere source).

    .. note::

        The parameters of the spectra may vary from source to source. For instance, they may not provide
        microturbulence velocity or alpha/Fe etc.


    example of the model file::

        # Kurucz ODFNEW /NOVER (2003)
        # teff = 3500 K (value for the effective temperature for the model. Temperatures are given in K)
        # logg = 0 log(cm/s2) (value for Log(G) for the model.)
        # meta = -0.5  (value for the Metallicity for the model.)
        # lh = 1.25  (l/Hp where l is the  mixing length of the convective element and Hp is the pressure scale height)
        # vtur = 2 km/s (Microturbulence velocity)
        # alpha = 0  (Alpha)
        #
        # column 1: WAVELENGTH (ANGSTROM), Wavelength in Angstrom
        # column 2: FLUX (ERG/CM2/S/A), Flux in erg/cm2/s/A
                147.2    5.04158e-191
                151      5.95223e-186
                155.2    1.22889e-180
                158.8    2.62453e-176
                162      1.2726e-172
                166      3.23807e-168
                170.3    1.03157e-163
                ...      ...

    Parameters
    ----------
    fname:  str
        The name of the file to read.

    Returns
    -------
    data: dict
        The data read from the file.
        it contains the various parameters (e.g., teff, logg, metallicity, alpha, lh, vtur)
    """
    data = {'columns': {}, 'data': None}

    rename_parameters = {'meta': 'feh'}

    n_header_lines = 0
    with open(fname, 'r') as fin:
        for line in fin:
            # if not a comment line stop.
            if line[0] != '#':
                break
            n_header_lines += 1

            line = line[1:].strip()
            if not line:
                continue
            if '=' in line:
                parameter, valdesc = line.split('=')
                parameter = rename_parameters.get(parameter.strip(), parameter.strip())
                val = float(valdesc.split()[0])
                candidate_unit = valdesc.split()
                if '(' not in candidate_unit[1]:
                    val_unit = candidate_unit[1]
                    val_desc = ' '.join(candidate_unit[2:])[1:-1]  # remove ()
                else:
                    val_unit = None
                    val_desc = ' '.join(candidate_unit[1:])[1:-1]  # remove ()
                data[parameter] = {'value': val, 'unit': val_unit, 'description': val_desc}
            elif line.startswith('column'):
                comment = ' '.join(line.split(':')[1:]).split()
                name, unit = comment[:2]
                desc = ' '.join(comment[2:])
                unit = unit.strip()[1:-2] # remove (),
                if 'ERG/CM2/S/A' in unit:
                    # 'A' unit is too vague and can be misinterpreted.
                    unit = 'erg/cm2/s/Angstrom'
                data['columns'].update({name.strip(): {'unit': unit, 'description': desc}})

    data['data'] = pd.read_csv(fname, skiprows=n_header_lines,
                               comment='#',
                               delim_whitespace=True,
                               names=list(data['columns'].keys()))

    return data


def get_svo_sprectum_units(data: dict) -> Sequence[Quantity]:
    """ Get the units objects of the wavelength and flux from an SVO spectrum.

    Parameters
    ----------
    data: dict
        The data read from the file using :func:`spectra_file_reader`

    Returns
    -------
    lamb_unit: Quantity
        The wavelength unit.
    flux_unit: Quantity
        The flux unit.
    """
    # remove the UnitWarning ("contains multiple slashes, which is discouraged" warning)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            lamb_unit = U(data['columns']['WAVELENGTH']['unit'])
        except ValueError:
            lamb_unit = U(data['columns']['WAVELENGTH']['unit'].lower())
        try:
            flux_unit = U(data['columns']['FLUX']['unit'])
        except ValueError:
            flux_unit = U(data['columns']['FLUX']['unit'].lower())
    return lamb_unit, flux_unit

def get_svo_passbands(identifiers: Union[str, Sequence[str]]) -> Sequence[UnitFilter]:
    """ Query the SVO filter profile service and return the pyphot filter objects.

    Parameters
    ----------
    identifier : str, Sequence[str]
        SVO identifier(s) of the filter profile
        e.g., 2MASS/2MASS.Ks HST/ACS_WFC.F475W
        The identifier is the first column on the webpage of the facilities.

    Returns
    -------
    filter : UnitFilter, Sequence[UnitFilter]
        list of Filter objects


    **example of the Gaia filter profile**

    .. plot::
        :include-source:

        import matplotlib.pyplot as plt

        from dustapprox.io import svo
        which_filters = ['GAIA/GAIA3.Gbp', 'GAIA/GAIA3.Grp', 'GAIA/GAIA3.G']
        passbands = svo.get_svo_passbands(which_filters)

        for pb in passbands:
            plt.plot(pb.wavelength.to('nm'), pb.transmit, label=pb.name)

        plt.legend(loc='upper right', frameon=False)

        plt.xlabel('wavelength [nm]')
        plt.ylabel('transmission')
        plt.tight_layout()
        plt.show()


    .. seealso:: :class:`pyphot.astropy.sandbox.UnitFilter`

        `pyphot <https://mfouesneau.github.io/pyphot/index.html>`_ is a set of
        tools to compute synthetic photometry in a simple way, ideal to
        integrate in larger projects.

        We internally use their astropy backend for the spectral units.

    """
    if isinstance(identifiers, str):
        identifiers = [identifiers]
        return get_pyphot_astropy_filter(identifiers)

    return [get_pyphot_astropy_filter(k) for k in identifiers]