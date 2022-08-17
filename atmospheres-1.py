import matplotlib.pyplot as plt
from dustapprox.io import svo

models = ['models/Kurucz2003all/fm05at10500g40k2odfnew.fl.dat.txt',
          'models/Kurucz2003all/fm05at5000g25k2odfnew.fl.dat.txt',
          'models/Kurucz2003all/fm40at6000g00k2odfnew.fl.dat.txt']

label = 'teff={teff:4g} K, logg={logg:0.1g} dex, [Fe/H]={feh:0.1g} dex'

for fname in models:
   data = svo.spectra_file_reader(fname)
   lamb_unit, flux_unit = svo.get_svo_sprectum_units(data)
   lamb = data['data']['WAVELENGTH'].values * lamb_unit
   flux = data['data']['FLUX'].values * flux_unit

   plt.loglog(lamb, flux,
              label=label.format(teff=data['teff']['value'],
                                 logg=data['logg']['value'],
                                 feh=data['feh']['value']))

plt.legend(loc='upper right', frameon=False)
plt.xlabel('Wavelength [{}]'.format(lamb_unit))
plt.ylabel('Flux [{}]'.format(flux_unit))
plt.ylim(1e2, 5e9)
plt.xlim(800, 1e5)