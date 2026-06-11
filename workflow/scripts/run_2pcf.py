# %%
# if interactive
import os
import sys

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

if hasattr(sys, "ps1"):
    from snakemake_helpers import snakemake_interactive

    snakemake = snakemake_interactive(
        "/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output/SP_v1.4.5_xi_minsep=1_maxsep=250_nbins=20_npatch=1.txt",
        "/home/cdaley/n17data/unions/pure_eb",
    )
else:
    from snakemake.script import snakemake

params = snakemake.params  # type: ignore
params.nbins = int(params["nbins"])

# %%
os.chdir("/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val")
print("Starting CosmologyValidation")
cv = CosmologyValidation(
    versions=[params["ver"]],
)

gg = cv.calculate_2pcf(**params, save_fits=True)

# %%
