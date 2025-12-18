import pandas as pd
from dustapprox import models
from dustapprox.literature import edr3
import pylab as plt

# get Gaia G band models
# let's take the first one with teffnorm and A0 features only
lib = models.PrecomputedModel()
options = lib.find(passband='Gaia')
selected = None
for model in options:
   if model.model['feature_names'] == ['A0', 'teffnorm']:
      selected = model
      break
model = lib.load_model(selected, passband='GAIA_GAIA3.G')

# get some data
data = pd.read_csv('models/precomputed/kurucz_gaiaedr3_small_a0_grid.csv')
df = data[(data['passband'] == 'GAIA_GAIA3.G') & (data['A0'] > 0)]

# values
ydata = (df['mag'] - df['mag0']) / df['A0']
kg_edr3 = edr3.edr3_ext().from_teff('kG', df['teff'], df['A0'])
delta_edr3 = ydata - kg_edr3
delta = ydata - model.predict(df)
# note: edr3 model valid for 3500 < teff < 10_000
selection = '3500 < teff < 10_000'
select = df.eval(selection)

plt.figure(figsize=(10, 8))
cmap = plt.cm.inferno_r

title = "{name:s}: kind={kind:s}\n using: {features}".format(
      name=model.name,
      kind=model.__class__.__name__,
      features=', '.join(model.feature_names))

kwargs_all = dict(rasterized=True, edgecolor='w', cmap=cmap, c=df['A0'])
kwargs_select = dict(rasterized=True, cmap=cmap, c=df['A0'][select])

ax0 = plt.subplot(221)
plt.scatter(df['mag0'], delta, **kwargs_all)
plt.scatter(df['mag0'][select], delta[select], **kwargs_select)
plt.colorbar().set_label(r'A$_0$ [mag]')
plt.ylabel(r'$\Delta$k$_G$')
plt.text(0.01, 0.99, title, fontsize='medium',
      transform=plt.gca().transAxes, va='top', ha='left')

ax1 = plt.subplot(222, sharey=ax0)
plt.scatter(df['teff'], delta, **kwargs_all)
plt.scatter(df['teff'][select], delta[select], **kwargs_select)
plt.colorbar().set_label(r'A$_0$ [mag]')
plt.text(0.01, 0.99, title, fontsize='medium',
      transform=plt.gca().transAxes, va='top', ha='left')

title = "{name:s}: kind={kind:s}\n using: {features}".format(
         name="EDR3",
         kind=edr3.edr3_ext().__class__.__name__,
         features=', '.join(('teffnorm', 'A0')))

ax = plt.subplot(223, sharex=ax0, sharey=ax0)
plt.scatter(df['mag0'], delta_edr3, **kwargs_all)
plt.scatter(df['mag0'][select], delta_edr3[select], **kwargs_select)
plt.colorbar().set_label(r'A$_0$ [mag]')
plt.xlabel(r'M$_G$ [mag] + constant')
plt.ylabel(r'$\Delta$k$_G$')
plt.ylim(-0.1, 0.1)
plt.text(0.01, 0.99, title, fontsize='medium',
      transform=plt.gca().transAxes, va='top', ha='left')


plt.subplot(224, sharex=ax1, sharey=ax1)
plt.scatter(df['teff'], delta_edr3, **kwargs_all)
plt.scatter(df['teff'][select], delta_edr3[select], **kwargs_select)
plt.xlabel(r'T$_{\rm eff}$ [K]')
plt.colorbar().set_label(r'A$_0$ [mag]')
plt.ylim(-0.1, 0.1)

plt.setp(plt.gcf().get_axes()[1:-1:2], visible=False)

plt.text(0.01, 0.99, title, fontsize='medium',
      transform=plt.gca().transAxes, va='top', ha='left')

plt.tight_layout()
plt.show()