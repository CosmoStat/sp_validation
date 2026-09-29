"""Rule rho_tau_stats: ρ/τ PSF statistics for one version.

Writes the ρ/τ FITS tables and the version's ρ/τ SACC part, sealed under the
catalogue's custody.
"""

from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation.cosmo_val import CosmologyValidation

_unbuffer_streams()
params = snakemake.params  # noqa: F821 — injected by Snakemake's script: directive

cv = CosmologyValidation(
    versions=[params["ver"]],
    theta_min=float(params["min_sep"]),
    theta_max=float(params["max_sep"]),
    nbins=int(params["nbins"]),
    npatch=int(params["npatch"]),
    catalog_config=params["cat_config"],
    output_dir=params["output_dir"],
    custody={params["ver"]: params["custody"]},
)
cv.calculate_rho_tau_stats()
verify_outputs(snakemake)  # noqa: F821
