import pylab as plt
import pandas as pd
import numpy as np

r = pd.read_csv('models/precomputed/kurucz_gaiaedr3_small_a0_grid.csv')
r['kx'] = np.where(r['A0'] > 0, r['Ax'] / r['A0'], float('NaN'))

plt.figure(figsize=(9, 4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122, sharey=ax1)
colors = {'Gbp': 'C0', 'G': 'C2', 'Grp': 'C3'}
for key, grp in r.groupby('passband'):
   color = colors[key.split('.')[-1]]
   ax1.scatter(grp['A0'], grp['kx'], label=key, rasterized=True, color=color)
   ax2.scatter(grp['teff'], grp['kx'], label=key, rasterized=True, color=color)
ax1.legend(loc='best', frameon=False)
ax1.set_ylabel(r'$A(T)\ /\ A_0$ [mag]')
ax1.set_xlabel(r'$A_0$ [mag]')
ax2.set_xlabel(r'$T_{eff}$ [K]')
ax2.set_xscale('log')
plt.tight_layout()
plt.show()