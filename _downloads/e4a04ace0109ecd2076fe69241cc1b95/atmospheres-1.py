import matplotlib.pyplot as plt
from dustapprox.io import svo

models = ['models/Kurucz2003all/fm05at10500g40k2odfnew.fl.dat.txt',
          'models/Kurucz2003all/fm05at5000g25k2odfnew.fl.dat.txt',
          'models/Kurucz2003all/fm40at6000g00k2odfnew.fl.dat.txt']

label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'

for fname in models:
   spec = svo.SVOSpectrum(fname)

   plt.loglog(spec.λ, spec.flux,
              label=label.format(teff=spec.meta['teff'],
                                 logg=spec.meta['logg'],
                                 feh=spec.meta['feh']))

plt.legend(loc='upper right', frameon=False)
plt.xlabel('Wavelength [{}]'.format(spec.units[0]))
plt.ylabel('Flux [{}]'.format(spec.units[1]))
plt.ylim(1e2, 5e9)
plt.xlim(800, 1e5)