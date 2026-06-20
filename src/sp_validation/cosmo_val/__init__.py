"""Cosmology validation package.

``CosmologyValidation`` is the public entry point. ``cs_plots`` is re-exported
so the ``cosmo_val.cs_plots`` attribute path (the cs_util plotting alias used at
each call site) keeps resolving for callers that import the package as a module
handle.
"""

from .core import CosmologyValidation, cs_plots

__all__ = ["CosmologyValidation", "cs_plots"]
