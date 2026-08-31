# %%
# if interactive
import os
import sys
from pathlib import Path

from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")
else:
    # Force unbuffered stdout and stderr
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)  # line-buffered
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

from sp_validation.cosmo_val import CosmologyValidation  # noqa: E402

print("Finished imports")

if ipython is not None:
    from snakemake_helpers import snakemake_interactive

    snakemake = snakemake_interactive(
        "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_val/output/rho_tau_stats/rho_stats_SP_v1.4.5.fits",
        "/home/cdaley/n17data/unions/pure_eb",
    )
else:
    from snakemake.script import snakemake

params = snakemake.params  # type: ignore

# %%
os.chdir("/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_val")
print("Starting CosmologyValidation")

# Use parameters passed from Snakemake rule
cv = CosmologyValidation(
    versions=[params["ver"]],
    theta_min=float(params["min_sep"]),
    theta_max=float(params["max_sep"]),
    nbins=int(params["nbins"]),
    npatch=int(params["npatch"]),
)

cv.calculate_rho_tau_stats()

# Confirm CosmologyValidation produced the requested outputs: the rho/tau FITS
# and the born-as-SACC rho_tau part.
outputs = snakemake.output  # type: ignore
for label in ("rho_stats", "tau_stats", "rho_tau"):
    target = Path(outputs[label])
    if not target.exists():
        raise FileNotFoundError(
            f"Expected {label} file not found after CosmologyValidation run: {target}"
        )
