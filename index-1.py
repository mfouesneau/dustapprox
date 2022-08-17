import numpy as np
import matplotlib.pyplot as plt
from dustapprox.io import svo
from dustapprox.extinction import F99

modelfile = 'models/Kurucz2003all/fm05at10500g40k2odfnew.fl.dat.txt'
data = svo.spectra_file_reader(modelfile)

# extract model relevant information
lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
lamb = data['data']['WAVELENGTH'].values * lamb_unit
flux = data['data']['FLUX'].values * flux_unit

# Extinction
extc = F99()
Rv = 3.1
Av = np.arange(0, 5.01, 0.1)
alambda_per_av = extc(lamb, 1.0, Rv=Rv)

# Dust magnitudes
cmap = plt.cm.inferno_r
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=Av.min(), vmax=Av.max()))
for av_val in Av:
   new_flux = flux * np.exp(- alambda_per_av * av_val)
   plt.loglog(lamb, new_flux, label=f'A0={av_val:.2f}', color=cmap(av_val / Av.max()))
plt.loglog(lamb, flux, color='k')
plt.ylim(1e-6, 1e9)
plt.xlim(750, 5e4)
plt.xlabel('Wavelength [{}]'.format(lamb_unit))
plt.ylabel('Flux [{}]'.format(flux_unit))
label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'
plt.title(label.format(teff=data['teff']['value'],
                       logg=data['logg']['value'],
                       feh=data['feh']['value']))
plt.colorbar(sm).set_label(r'A$_0$ [mag]')
plt.tight_layout()

plt.show()