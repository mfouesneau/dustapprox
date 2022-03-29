Extinction curves
=================

In our application, we need the models that predict spectra or SEDs of
star extinguished by dust. Interstellar dust extinguishes stellar light as it
travels from the star's surface to the observer. The wavelength dependence of
the extinction from the ultraviolet to the near-infrared has been measured along
many sightlines in the Milky Way (Cardelli et al. 1989; Fitzpatrick 1999;
Valencic et al. 2004; Gordon et al. 2009) and for a handful of sightlines in the
Magellanic Clouds (Gordon & Clayton 1998; Misselt et al. 1999; Maiz Apellaniz &
Rubio 2012) as well as in M31 (Bianchi et al. 1996, Clayton et al. 2015).


* :class:`dustapprox.extinction` provides a common interface to many
  commonly used extinction curves.

.. plot::
   :caption: **Figure 4.** Differences between extinction curves. This figure compares a few different presscriptions at fixed :math:`R_0`.
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