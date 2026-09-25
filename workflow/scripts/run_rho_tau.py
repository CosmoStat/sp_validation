"""Rule rho_tau_stats: ρ/τ PSF statistics for one version.

Writes the ρ/τ FITS tables and the version's ρ/τ SACC part, sealed under the
catalogue's custody.
"""

from pathlib import Path

from cv_runner import _unbuffer_streams

from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.custody import confirm

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
)
confirm(cv.custody(params["ver"]), params["custody"])
cv.calculate_rho_tau_stats()

outputs = snakemake.output  # noqa: F821
for label in ("rho_stats", "tau_stats", "rho_tau"):
    target = Path(outputs[label])
    if not target.exists():
        raise FileNotFoundError(
            f"Expected {label} file not found after CosmologyValidation run: {target}"
        )
