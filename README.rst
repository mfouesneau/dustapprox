dustapprox -- A tool for computing approximative extinction coefficients
========================================================================

.. image:: https://img.shields.io/pypi/v/dustapprox.svg
    :target: https://pypi.org/project/dustapprox/

.. image:: https://img.shields.io/badge/python-3.10,_3.11,_3.12,_3.13-blue.svg

This is a set of tools to compute photometric extinction coefficients in a *quick and dirty* way.

full documentation at: http://mfouesneau.github.io/dustapprox/

see recent changes at: `What's new <http://mfouesneau.github.io/dustapprox/whats_new.html>`_


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

  pip install git+https://github.com/mfouesneau/dustapprox

* Manual installation

download the repository and run the setup

.. code::

  git clone https://github.com/mfouesneau/dustapprox
  python -m pip install .

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

see `contributing guide <http://mfouesneau.github.io/dustapprox/contributing.html>`_ for more information.

TODO
----
Upcoming tasks and features planned for future releases

- [ ] add additional models:
  - [ ] add Gaia DR4 passbands
  - [ ] add Euclid passbands? 
  - [ ] others?
- [ ] improve documentation
  - [ ] additional examples
  - [ ] add notes about A0 vs A(V) etc.
  - [ ] add notes about A0, R0 models
- [ ] additional unit tests (current coverage 63%)