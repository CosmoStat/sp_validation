# %%
import os
import sys

# Append any useful folder in the path
sys.path.append("/home/guerrini/sp_validation/cosmo_inference/scripts/")
sys.path.append(
    "/home/guerrini/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_unblinding/"
)

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import seaborn as sns
from getdist import plots

from sp_validation.rho_tau import SquareRootScale

mscale.register_scale(SquareRootScale)

import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

import chain_postprocessing as cp

plt.style.use(
    "/home/guerrini/sp_validation/papers/harmonic/matplotlib_config/paper.mplstyle"
)

plt.rcParams["text.usetex"] = True

sns.set_palette("husl")

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 30
g.settings.axes_labelsize = 30
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 40

# Directory where the chains are located
root_dir = "/n09data/guerrini/output_chains"

# Path to the ini files used
path_ini_files = "/home/guerrini/sp_validation/cosmo_inference/cosmosis_config"
path_datavectors = "/home/guerrini/sp_validation/cosmo_inference/data/"
path_output_chains = "/n09data/guerrini/output_chains/"

best_fit_method = "2Dkde"
assert best_fit_method in ["2Dkde", "1Dkde", "weighted_mean"], (
    "Invalid best fit method. Choose one of: '2Dkde', '1Dkde', 'weighted_mean'"
)

# %%
# Prepare settings for the plot
blind = "B"

colour_blind = {"A": "royalblue", "B": "crimson", "C": "forestgreen"}

root_to_plot = [
    f"SP_v1.4.6.3_leak_corr_{blind}",
    f"SP_v1.4.6.3_leak_corr_halofit_{blind}",
    f"SP_v1.4.6.3_{blind}_fiducial_config",
]

labels = [
    r"UNIONS $C_\ell$",
    r"UNIONS $C_\ell$, \texttt{Halofit}",
    r"UNIONS $\xi_\pm(\vartheta)$, (Goh et al., 2026)",
]

line_args = [
    {"color": colour_blind[blind], "linestyle": "-"},
    {"color": colour_blind[blind], "linestyle": "--"},
    {"color": "orange", "linestyle": "-"},
]

analysis_type = ["harmonic", "harmonic", "configuration"]

ini_file_root = os.path.join(
    path_ini_files,
    f"config_space_v1.4.6.3_fiducial/pipeline/blind_{blind}/fiducial.ini",
)

properties = {}
for i, root in enumerate(root_to_plot):
    properties = cp.update_properties_w_roots(
        properties,
        root,
        path_ini_files,
        with_configuration=analysis_type[i] == "configuration",
        path_to_this_ini=ini_file_root if analysis_type[i] == "configuration" else None,
    )

# %%
# Perform the computation of the best fit for each chain
for i, root in enumerate(root_to_plot):
    analysis_type_i = analysis_type[i]
    if analysis_type_i == "harmonic":
        path_samples = os.path.join(root_dir, root, root, f"samples_{root}_cell.txt")
        path_gd = os.path.join(root_dir, root, root, f"getdist_{root}_cell")
    elif analysis_type_i == "configuration":
        path_samples = os.path.join(root_dir, root, f"samples_{root}.txt")
        path_gd = os.path.join(root_dir, root, f"getdist_{root}")
    else:
        raise ValueError(f"Invalid analysis type: {analysis_type_i}")

    # Read files, format and load chains
    cp.load_samples_and_write_paramnames(path_samples, path_gd + ".paramnames")
    cp.write_samples_getdist_format(path_samples, path_gd + ".txt")
    chain = cp.load_chain(path_gd, smoothing_scale=0.5)

    # Get the bestfit parameters
    best_fit = cp.extract_best_fit_params(chain, best_fit_method=best_fit_method)
    print(best_fit["S_8"])
    # Get the datavector for the best fit
    if analysis_type_i == "harmonic":
        cp.compute_best_fit(
            path_ini_files, best_fit, root, is_harmonic=True, blind=blind
        )
    elif analysis_type_i == "configuration":
        cp.compute_best_fit(
            path_ini_files,
            best_fit,
            root,
            is_harmonic=False,
            blind=blind,
            ini_file_root=ini_file_root,
        )
    else:
        raise ValueError(f"Invalid analysis type: {analysis_type_i}")


# %%
# Plot hyperparameter
os.chdir("/home/guerrini/sp_validation/papers/harmonic/")
loc_legend = "lower center"
bbox_to_anchor = (0.685, 0.69)

# Perform the plot PNG format
cp.plot_best_fit(
    f"SP_v1.4.6.3_leak_corr_{blind}",
    root_to_plot,
    path_output_chains,
    line_args,
    savefile="./plots/paperplot_Cell_EE_and_best_fit.png",
    labels=labels,
    loc_legend=loc_legend,
    bbox_to_anchor=bbox_to_anchor,
    properties=properties,
)

# Perform the plot PDF format
cp.plot_best_fit(
    f"SP_v1.4.6.3_leak_corr_{blind}",
    root_to_plot,
    path_output_chains,
    line_args,
    savefile="./plots/paperplot_Cell_EE_and_best_fit.pdf",
    labels=labels,
    loc_legend=loc_legend,
    bbox_to_anchor=bbox_to_anchor,
    properties=properties,
)
# %%
