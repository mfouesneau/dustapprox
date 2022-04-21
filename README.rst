dustapprox -- A tool for computing approximative extinction coefficients
=============================================================================

This is a set of tools to compute photometric extinction coefficients in a *quick and dirty* way.

full documentation at: http://mfouesneau.github.io/dustapprox/


Quick Start
-----------

.. code-block:: python

  import pandas as pd
  from dustapprox import models
  from dustapprox.literature import edr3
  import pylab as plt

  # get Gaia models
  lib = models.PrecomputedModel()
  r = lib.find(passband='Gaia')[0]  # taking the first one
  model = lib.load_model(r, passband='GAIA_GAIA3.G')

  # get some data
  data = pd.read_csv('models/precomputed/kurucs_gaiaedr3_small_a0_grid.csv')
  df = data[(data['passband'] == 'GAIA_GAIA3.G') & (data['A0'] > 0)]

  # values
  kg_pred = model.predict(df)

Installation
------------
* Installation from PyPI

.. code::

  pip install git+https://github.com/mfouesneau/gdr3_extinction

* Manual installation

download the repository and run the setup

.. code::

  git clone https://github.com/mfouesneau/gdr3_extinction
  python setup.py install

Contributors
------------

Main contributors:

- M. Fouesneau (@mfouesneau),
- R. Andrae,
- R. Sordo,
- R. Drimmel,
- T. E. Dharmawardena


Contributing
------------

Please open a new issue or new pull request for bugs, feedback, or new features
you would like to see. If there is an issue you would like to work on, please
leave a comment, and we will be happy to assist. New contributions and
contributors are very welcome!