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