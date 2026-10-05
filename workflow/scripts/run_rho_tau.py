"""Rule rho_tau_stats: ρ/τ PSF statistics for one version.

Writes the ρ/τ FITS tables and the version's ρ/τ SACC part.
"""

from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation.cosmo_val import CosmologyValidation

_unbuffer_streams()
params = snakemake.params

cv = CosmologyValidation(
    versions=[params["ver"]],
    theta_min=float(params["min_sep"]),
    theta_max=float(params["max_sep"]),
    nbins=int(params["nbins"]),
    npatch=int(params["npatch"]),
    catalog_config=params["cat_config"],
    output_dir=params["output_dir"],
)
cv.calculate_rho_tau_stats(tomography=False)
verify_outputs(snakemake)
