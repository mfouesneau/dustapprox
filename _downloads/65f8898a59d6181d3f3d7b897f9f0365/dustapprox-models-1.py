import pylab as plt
from dustapprox.models import polynomial
import pandas as pd

r = pd.read_csv('models/precomputed/kurucz_gaiaedr3_small_a0_grid.csv')
# polynomial.quick_plot_models(r, input_parameters='teff A0'.split())
polynomial.quick_plot_models(r, input_parameters='teff A0 feh logg'.split())
plt.show()