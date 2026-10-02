"""Rule cv_ratio_xi_sys_xi: ratio of PSF systematics to cosmic-shear signal.

Joins two upstream chains: each version's reporting ξ± part and xi_psf_sys
(recomputed in memory from the rho/tau FITS via the lazy cv.xi_psf_sys property
-- the PSF-error fit is not persisted by cosmo_val.py). Both are declared as
inputs so the DAG shows the join. Writes ratio_xi_sys_xi.png at a fixed path
(declared output).
"""

from cv_runner import _unbuffer_streams, make_cv, take_reporting_parts, verify_outputs

_unbuffer_streams()
cv = make_cv(snakemake)
take_reporting_parts(cv, snakemake.input["xi"])
cv.plot_ratio_xi_sys_xi(offset=snakemake.params.get("offset", 0.1))
verify_outputs(snakemake)
