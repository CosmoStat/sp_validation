"""Run make_glass_sim.py unchanged, with the seed-independent Gaussian-spectrum solve cached.

`glass.solve_gaussian_spectra(fields, cls)` depends only on the shells and matter
spectra, not the seed, and takes ~75 single-threaded minutes at nside 4096. The
first call saves its result beside the generator's own cls cache; later seeds load
it. Everything else is the generator's own code path.

    python sky.py <cache.npy> <make_glass_sim.py> [make_glass_sim args...]
"""
import os
import runpy
import sys

import glass
import numpy as np

cache, script, *argv = sys.argv[1:]
solve = glass.solve_gaussian_spectra


def cached_solve(fields, cls):
    if os.path.exists(cache):
        print(f"[sky.py] loading Gaussian spectra from {cache}")
        return list(np.load(cache, allow_pickle=True))
    gls = solve(fields, cls)
    tmp = f"{cache}.{os.getpid()}.npy"
    arr = np.empty(len(gls), dtype=object)
    arr[:] = list(gls)
    np.save(tmp, arr, allow_pickle=True)
    os.replace(tmp, cache)
    print(f"[sky.py] saved Gaussian spectra to {cache}")
    return gls


glass.solve_gaussian_spectra = cached_solve
sys.argv = [script, *argv]
runpy.run_path(script, run_name="__main__")
