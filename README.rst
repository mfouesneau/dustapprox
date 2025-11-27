dustapprox -- A tool for computing approximative extinction coefficients
=============================================================================

.. image:: https://img.shields.io/pypi/v/dustapprox.svg
    :target: https://pypi.org/project/dustapprox/

.. image:: https://img.shields.io/badge/python-3.10,_3.11,_3.12,_3.13-blue.svg

This is a set of tools to compute photometric extinction coefficients in a *quick and dirty* way.

full documentation at: http://mfouesneau.github.io/dustapprox/


TODO
----
- [x] update python supported versions
- [x] upgrade packaging to pyproject.toml only
- [x] typing annotations
- [x] update internal extinction curves to use dust_extinction instead
- [x] update for pyphot 2.0
- [ ] generate workflow to train model
- [ ] check how we handle R0
  - [ ] add (A0, R0) models
- [ ] add additional models:
  - [ ] G23 curve
- [ ] additional unit tests
- [ ] debug warnings from doc workflow
- [ ] additional examples
- [ ] add notes about A0 vs A(V) etc.
- [ ] add Gaia DR4 passbands
- [ ] add Euclid passbands? others?
- [ ] check python 3.14 compatibility


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

  pip install dustapprox

* Installation from pip+github

.. code::

  pip install git+https://github.com/mfouesneau/gdr3_extinction

* Manual installation

download the repository and run the setup

.. code::

  git clone https://github.com/mfouesneau/gdr3_extinction
  python setup.py install

Contributors
------------

- Morgan Fouesneau (@mfouesneau)
- René Andrae
- Rosanna Sordo
- Thavisha Dharmawardena


Contributing
------------

Please open a new issue or new pull request for bugs, feedback, or new features
you would like to see. If there is an issue you would like to work on, please
leave a comment, and we will be happy to assist. New contributions and
contributors are very welcome!
