"""COSEBI root finding with maxsteps = 200, cosmo_numba's upstream setting (03e4f4a).

The image's cosmo_numba calls `sympy.nroots` with its default maxsteps = 50, which
fails to converge on some catalogues' θ ranges. The blinded all-tiles data run
measured its COSEBIs through the same shim.
"""
import sympy

_nroots = sympy.nroots


def nroots(f, n=15, maxsteps=200, cleanup=True):
    return _nroots(f, n=n, maxsteps=maxsteps, cleanup=cleanup)


sympy.nroots = nroots
