# %%
# Plot binned quantities, which are the outputs of leakage_minimal.py

# %%
from IPython import get_ipython

ipython = get_ipython()

# enable autoreload for interactive sessions
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%
import numpy as np

from sp_validation import io, plots

## %%
## enable inline plotting for interactive sessions
## (must be done *after* importing package that sets agg backend)
# if ipython is not None:
#    print("matplotlib inline")
#    ipython.run_line_magic("matplotlib", "inline")

# %%
bin_edges = {}
quantities = {}

# %%
keys = [
    "number",
    "response",
    "leakage",
    "w_iv",
    "mag",
    "NGMIX_Tpsf_NOSHEAR",
    "N_EPOCH",
    "e1_PSF",
    "e2_PSF",
    "fwhm_PSF",
]


for key in keys:
    fname = f"{key}_binned.npz"
    result = io.read_binned_quantity(fname)
    for xy in result:
        if xy != "quantity":
            bin_edges[xy] = result[xy]
    quantities[key] = result["quantity"]

xlabel = "SNR"
ylabel = r"$r / r_{\rm psf}$"

lines = {
    "x": [10, 500],
    "y": [0.5, 3],
}


# %%
vmin = {"diag": -0.2, "offdiag": -0.1}
vmax = {"diag": 1.2, "offdiag": 0.1}

plots.plot_binned(
    quantities,
    "response",
    bin_edges["snr"],
    bin_edges["size_ratio"],
    title="R",
    vmin=vmin,
    vmax=vmax,
    xlabel=xlabel,
    ylabel=ylabel,
    lines=lines,
    close_fig=False,
)

# %%
plots.plot_binned(
    quantities,
    "number",
    bin_edges["snr"],
    bin_edges["size_ratio"],
    title="n",
    vmin=1,
    vmax=np.nanmax(quantities["number"]),
    xlabel=xlabel,
    ylabel=ylabel,
    lines=lines,
    close_fig=False,
)

# %%
vmin = {"diag": -0.2, "offdiag": -0.2}
vmax = {"diag": 0.2, "offdiag": 0.2}

plots.plot_binned(
    quantities,
    "leakage",
    bin_edges["snr"],
    bin_edges["size_ratio"],
    title=r"\alpha",
    vmin=vmin,
    vmax=vmax,
    xlabel=xlabel,
    ylabel=ylabel,
    lines=lines,
    close_fig=False,
)

# %%
bin_edges
# %%
