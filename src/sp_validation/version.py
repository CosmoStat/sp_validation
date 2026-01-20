# -*- coding: utf-8 -*-

"""sp_validation PACKAGE.

Validation of weak-lensing catalogues (galaxy and star shapes and other
parameters) produced by ShapePipe

References
----------
This package makes use of the following third-party packages:

- `Matplotlib <https://matplotlib.org/>`_ :cite:`Hunter:2007`
- `Numpy <https://numpy.org/>`_ :cite:`Harris:2020`

.. warning::

    `WPS410 <https://wemake-python-stylegui.de/en/latest/pages/usage/violation
    s/best_practices.html#wemake_python_styleguide.violations.best_practices.W
    rongModuleMetadataViolation>`_ and `WPS412 <https://wemake-python-stylegui.
    de/en/latest/pages/usage/violations/best_practices.html#wemake_python_style
    guide.violations.best_practices.InitModuleHasLogicViolation>`_ errors are
    supressed in this module to allow the defintion of a package
    ``__version__``, which is standard for most Python packages.

"""

from warnings import warn

try:
    from importlib_metadata import version

    __version__ = version("sp_validation")
except Exception:  # pragma: no cover
    __version__ = "Unkown"
    warn(
        "Could not extract package metadata. Make sure the package is "
        + "correctly installed.",
    )
