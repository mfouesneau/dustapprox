import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from dustapprox.legacy_extinction import CCM89, F99

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