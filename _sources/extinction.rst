Extinction curves
=================

In our application, we need the models that predict spectra or SEDs of star extinguished by dust. Interstellar dust extinguishes stellar light as it travels from the star's surface to the observer. The wavelength dependence of the extinction from the ultraviolet to the near-infrared has been measured along many sightlines in the Milky Way (`Cardelli et al. 1989 <https://ui.adsabs.harvard.edu/abs/1989ApJ...345..245C/abstract>`_; `Fitzpatrick 1999 <https://doi.org/10.1086/316293>`_; `Valencic et al. 2004 <https://doi.org/10.1086/424922>`_; `Gordon et al. 2009 <https://doi.org/10.1088/0004-637X/705/2/1320>`_) and for a handful of sightlines in the Magellanic Clouds (`Gordon & Clayton 1998 <https://ui.adsabs.harvard.edu/abs/1998ApJ...500..816G/abstract>`_; `Misselt et al. 1999 <https://ui.adsabs.harvard.edu/abs/1999ApJ...515..128M/abstract>`_; `Maiz Apellaniz & Rubio 2012 <https://ui.adsabs.harvard.edu/abs/2012A%26A...541A..54M/abstract>`_) as well as in M31 (`Bianchi et al. 1996 <https://ui.adsabs.harvard.edu/abs/1996ApJ...471..203B/abstract>`_, `Clayton et al. 2015 <https://ui.adsabs.harvard.edu/abs/2015ApJ...815...14C/abstract>`_).

Since v.0.20, Dusapprox uses `dust_extinction <https://dust-extinction.readthedocs.io/en/latest/>`_ as provider of dust extinction curves.  The following figure shows some of the available extinction curves.  Not all extinction curves are suitable for all applications, wavelength ranges, parameters, etc vary between curves. We primarily use the `parametric average average curves <https://dust-extinction.readthedocs.io/en/latest/dust_extinction/choose_model.html#parameter-dependent-average-curves>`_ here, but you can always manually use a different one. Please refer to the `dust_extinction <https://dust-extinction.readthedocs.io/en/latest/>`_ documentation for more details.


.. plot::
  :caption: **Figure 4.** Differences between extinction curves. This figure compares a few different presscriptions at fixed :math:`R_0`.
  :include-source:

  import numpy as np
  import matplotlib.pyplot as plt
  import astropy.units as u

  from dustapprox.extinction import evaluate_extinction_model

  #define the wave numbers
  x = np.arange(0.1, 10, 0.1)    # in microns^{-1}
  λ = 1. / x * u.micron

  curves = 'CCM89', 'O94', 'F99', 'F04', 'F19', 'G23'
  R0 = 3.1

  curve_data = {}
  _, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
  for name in curves:
      values = evaluate_extinction_model(name, λ, A0=1.0, R0=R0)
      curve_data[name] = values
      axes[0].plot(x, values, label=f'{name:s}', lw=2)

  mean_curve = np.nanmean(list(curve_data.values()), axis=0)
  for name, values in curve_data.items():
      diff = values - mean_curve
      axes[1].plot(x, diff, label=f'{name:s}', lw=2)

  axes[0].set_ylabel(r'$A(x)/A_V$')
  axes[0].legend(loc='upper left', frameon=False, title=rf'Ext. Curve', bbox_to_anchor=(1.0, 1.0))
  axes[0].text(0.05, 0.9, rf'$R_V={R0:.1f}$', transform=axes[0].transAxes)
  axes[1].set_xlabel(r'Wave number [$\mu$m$^{-1}$]')
  axes[1].set_ylabel(r'$A(x)/A_V - <A(x)/A_V>$')
  axes[1].axhline(0.0, color='0.5', ls='-', lw=1)
  axes[0].set_ylim(-0.2, 5.2)
  axes[1].set_ylim(-0.42, 0.19)
  plt.tight_layout()
  plt.show()
