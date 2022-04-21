""" Creates the list table for the precomputed models """
import sys
sys.path.insert(0, '../')

import pandas as pd
from dustapprox.models import PrecomputedModel

lib = PrecomputedModel()

data_ = []
for info in lib.get_models_info():

    kind = info['model']['kind']
    features = ' '.join(info['model']['feature_names'])
    extc = info['extinction']['source']
    atm = info['atmosphere']['source']
    teffmin = info['atmosphere']['teff'][0]
    teffmax = info['atmosphere']['teff'][1]
    loggmin = info['atmosphere']['logg'][0]
    loggmax = info['atmosphere']['logg'][1]
    fehmin = info['atmosphere']['feh'][0]
    fehmax = info['atmosphere']['feh'][1]
    a0min = info['extinction']['A0'][0]
    a0max = info['extinction']['A0'][1]

    if kind == 'polynomial':
        kind = "{kind:s}(degree={degree:d})".format(kind=kind, 
                                                    degree=info['model']['degree'])

    for pb in info['passbands']:
        data_.append([pb, kind, features, 
                     extc, a0min, a0max, 
                     atm, teffmin, teffmax, loggmin, loggmax, fehmin, fehmax])
columns=['passband', 'kind', 'features', 
         'extinction', 'A0 min', 'A0 max',
         'atmosphere', 'teff min', 'teff max', 'logg min', 'logg max',
         '[Fe/H] min', '[Fe/H] max']

df = pd.DataFrame\
       .from_records(data_, columns=columns)\
       .set_index('passband')\
       .sort_index()\
       .to_csv('precomputed_table.csv', float_format='%0.8g')
