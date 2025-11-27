List of provided precomputed models
====================================

* use :class:`dustapprox.models.PrecomputedModel.find` to find available models and associated passbands
    * The search can be on passband, extinction, atmosphere, and model kind.  It is caseless and does not need to contain the complete name.


.. code-block:: python
    :caption: examples of searching for models

    from dustapprox.models import PrecomputedModel
    lib = PrecomputedModel()
    lib.find(passband='Gaia')   # returns all models with Gaia passband
    lib.find(passband='galex', atmosphere='Atlas')   # returns nothing (we did not provide Atlas9 atmosphere)
    lib.find(passband='galex', atmosphere='kurucz')  # return only kurucz based models



This table provides the list of precomputed models provided with this package.


.. csv-table:: List of provided precomputed models
   :file: precomputed_table.csv
