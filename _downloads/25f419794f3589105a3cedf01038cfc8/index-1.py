import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from dustapprox.io import svo
from dustapprox.extinction import evaluate_extinction_model

modelfile = 'models/Kurucz2003all/fm05at10500g40k2odfnew.fl.dat.txt'
spec = svo.SVOSpectrum(modelfile)

# extract model relevant information
lamb = spec.λ
flux = spec.flux

# Extinction
R0 = 3.1
alambda_per_av = evaluate_extinction_model('F99', lamb, 1., R0)
A0 = np.arange(0, 5.01, 0.1)

# Dust magnitudes
cmap = mpl.colormaps["inferno_r"]
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=A0.min(), vmax=A0.max()))
for a0_val in A0:
   new_flux = flux * np.exp(- alambda_per_av * a0_val)
   plt.loglog(lamb, new_flux, label=f'A0={a0_val:.2f}', color=cmap(a0_val / A0.max()))
plt.loglog(lamb, flux, color='k')
plt.ylim(1e-6, 1e9)
plt.xlim(750, 5e4)
plt.xlabel('Wavelength [{}]'.format(spec.units[0]))
plt.ylabel('Flux [{}]'.format(spec.units[1]))
label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'
plt.title(label.format(teff=spec.meta['teff'],
                        logg=spec.meta['logg'],
                        feh=spec.meta['feh']))
plt.colorbar(sm, ax=plt.gca()).set_label(r'A$_0$ [mag]')
plt.tight_layout()

plt.show()