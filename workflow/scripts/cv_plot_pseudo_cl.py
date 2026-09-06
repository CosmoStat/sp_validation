"""Rule cv_plot_pseudo_cl: the EE/EB/BB pseudo-Cl figures.

Plot-only, and an ingest like the other B-mode rules: the spectra come from the
analysis pseudo-Cl parts and their NaMaster covariances, the same pair the
summary and the terminal file are built from, so the figures cannot show
something the data products do not.
"""

import numpy as np
from astropy.io import fits
from cv_runner import _unbuffer_streams, verify_outputs
from snakemake.script import snakemake

from sp_validation import sacc_io
from sp_validation.cosmo_val.pseudo_cl import plot_pseudo_cl_spectrum

_unbuffer_streams()
p = snakemake.params

spectra = {}
for i, version in enumerate(p["versions"]):
    part = sacc_io.load(snakemake.input["pseudo_cl"][i])
    ell, ee, bb, eb, _window = sacc_io.get_pseudo_cl(part, (0, 0))
    with fits.open(snakemake.input["pseudo_cl_cov"][i]) as hdul:
        covs = {
            name: np.asarray(hdul[f"COVAR_{name}_{name}"].data, float)
            for name in ("EE", "EB", "BB")
        }
    for name, cl in (("EE", ee), ("EB", eb), ("BB", bb)):
        spectra.setdefault(name, {})[version] = {
            "ell": ell,
            "cl": cl,
            "cov": covs[name],
            "style": {"marker": p["markers"][i], "colour": p["colours"][i]},
        }

for name, datasets in spectra.items():
    plot_pseudo_cl_spectrum(datasets, name, snakemake.output[f"figure_{name.lower()}"])

verify_outputs(snakemake)
