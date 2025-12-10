Version History
===============

This page lists the recent changes in versions of dustapprox.

v0.2.0
------
[December 10, 2025]

Breaking changes
~~~~~~~~~~~~~~~~
- moved :mod:`dustapprox.extinction` classes to :mod:`dustapprox.legacy_extinction` module
    - replaced by direct use of `dust_extinction <https://dust-extinction.readthedocs.io/en/latest/>`_ models
    - added :func:`dustapprox.extinction.evaluate_extinction_model` helper function
    - users need to adapt their code accordingly

Non-breaking changes
~~~~~~~~~~~~~~~~~~~~

- new precomputed models:
    - F99 with A0, R0, Teff dependency for Generic, Galex, Gaia DR3, Gaia C1, Sloan, TwoMASS, WISE passbands
    - G23 with A0, R0, Teff dependency for Generic, Galex, Gaia DR3, Gaia C1, Sloan, TwoMASS, WISE passbands

- added :mod:`dustapprox.tools.generate_model` module to simplify the generation of pre-computed models
    - includes end-to-end example function

- added :mod:`dustapprox.tools.model_characteristics` module to help assess model systematics
    - can compute a sparse model grid
    - evaluates systematics and makes some plots (see :doc:`systematics`)

- adopted `dust_extinction <https://dust-extinction.readthedocs.io/en/latest/>`_ as provider of dust extinction curves
    - outsourced definitions of extinction curves to experts
    - allows dustapprox to include other curves (e.g. G23)

- updated `pyphot <https://mfouesneau.github.io/pyphot/>`_ dependency to v2.0.0
    - see `pyphot v2.0.0 release notes <https://mfouesneau.github.io/pyphot/whats_new.html#id1>`_ for details

- unit tests improvements
    - increased coverage to 61% (from 0!)

- updated documentation
    - added :doc:`whats_new`, :doc:`contributing`, :doc:`systematics` pages
    - updated :doc:`extinction`, :doc:`precomputed_content`
