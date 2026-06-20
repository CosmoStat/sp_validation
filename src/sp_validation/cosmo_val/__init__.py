"""Cosmology validation package.

``CosmologyValidation`` is the public entry point. ``cs_plots`` is re-exported
so the ``cosmo_val.cs_plots`` attribute path (the cs_util plotting alias used at
each call site) keeps resolving for callers that import the package as a module
handle.
"""

from cs_util import plots as cs_plots

from .core import CosmologyValidation

__all__ = ["CosmologyValidation", "cs_plots"]
