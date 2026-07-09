#
# This notebook plots the whisker plot of $S_8$, $\Omega_m$ and $\sigma_8$


import os
import sys

# Trick to plot with tex
os.environ["LD_LIBRARY_PATH"] = ""
os.environ["CONDA_PREFIX"] = "/home/guerrini/.conda/envs/sp_validation_3.11"

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from getdist import plots

sys.path.append("/home/guerrini/sp_validation/cosmo_inference/scripts")

import chain_postprocessing as cp

plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

plt.rc("text", usetex=True)

sns.set_palette("husl")

g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 60
g.settings.axes_labelsize = 60
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 60


# SPECIFY DATA DIRECTORY AND DESIRED CHAINS TO ANALYSE
root_dir = "/n09data/guerrini/output_chains/"
root_external = f"{root_dir}/ext_data/"
blind = "B"

roots = [
    f"SP_v1.4.6.3_{blind}_fiducial_config",
    f"SP_v1.4.6.3_leak_corr_{blind}",
    "Planck18",
    "DES Y6",
    "KiDS-Legacy_bandpowers",
    "KiDS-Legacy_cosebis",
    "KiDS-Legacy_xipm",
    "HSC_Y3",
    "HSC_Y3_cell",
    f"SP_v1.4.6.3_{blind}_small_scales_config",
    f"SP_v1.4.6.3_{blind}_flat_alpha_beta_config",
    f"SP_v1.4.6.3_{blind}_no_xi_sys_config",
    f"SP_v1.4.6.3_{blind}_no_leak_corr_config",
    f"SP_v1.4.6.3_{blind}_flat_delta_z_config",
    f"SP_v1.4.6.3_{blind}_no_delta_z_config",
    f"SP_v1.4.6.3_{blind}_flat_ia_config",
    f"SP_v1.4.6.3_{blind}_no_ia_config",
    f"SP_v1.4.6.3_{blind}_no_m_bias_config",
    f"SP_v1.4.6.3_{blind}_unmasked_covmat_config",
    f"SP_v1.4.6.3_{blind}_halofit_config",
    f"SP_v1.4.6.3_{blind}_no_baryons_config",
    f"SP_v1.4.6.3_{blind}_nautilus_config",
    f"SP_v1.4.6.3_{blind}_planck_config",
    f"SP_v1.4.6.3_{blind}_planck_desi_config",
]

legend_labels = [
    r"UNIONS-3500 $\xi_{\pm}(\theta)$ (This work)",
    r"UNIONS-3500 $C_\ell$ (Guerrini et al. 2026)",
    r"$\textit{Planck}$ 2018",
    r"DES Y6 $\xi_{\pm}$, NLA",
    r"KiDS-Legacy Bandpowers ($C_{\rm E}$)",
    r"KiDS-Legacy COSEBIs ($E_n$)",
    r"KiDS-Legacy $\xi_{\pm}(\theta)$",
    r"HSC-Y3 $\xi_{\pm}(\theta)$",
    r"HSC-Y3 $C_\ell$",
    r"$\xi_+$ small scales, $\theta$=[5,83] arcmin",
    r"Flat $\alpha_{\rm{PSF}}$ and $\beta_{\rm{PSF}}$ priors",
    r"No $\xi^{\rm sys}_{\pm}$",
    r"No leakage correction",
    r"Flat $\Delta z$ priors",
    r"No $\Delta z$",
    r"Flat $A_{\rm IA}$ prior",
    r"No $A_{\rm IA}$",
    r"No $m$ bias",
    r"Unmasked covmat",
    r"$\texttt{Halofit}$",
    r"$\texttt{HMCode}$ no baryons",
    r"Nautilus sampler",
    r"UNIONS-3500 + $\textit{Planck}$",
    r"UNIONS-3500 + $\textit{Planck}$ + DESI BAO",
]

categories = [
    "configuration",
    "harmonic",
    "external",
    "external",
    "external",
    "external",
    "external",
    "external",
    "external",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
    "configuration",
]
colours = [
    "darkorange",
    "royalblue",
    "violet",
    "black",
    "black",
    "black",
    "black",
    "black",
    "black",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
    "forestgreen",
]


chains = []
for i, root in enumerate(roots):
    category = categories[i]
    if root == "DES Y6":
        continue
    if category != "external":
        if category == "configuration":
            path_samples = os.path.join(root_dir, f"{root}/samples_{root}.txt")
            path_getdist = os.path.join(root_dir, f"{root}/getdist_{root}")
        elif category == "harmonic":
            path_samples = os.path.join(
                root_dir, f"{root}/{root}/samples_{root}_cell.txt"
            )
            path_getdist = os.path.join(root_dir, f"{root}/{root}/getdist_{root}")
        elif category == "external_compute_sample":
            path_samples = os.path.join(root_dir, f"ext_data/{root}/samples_{root}.txt")
            path_getdist = os.path.join(root_dir, f"ext_data/{root}/getdist_{root}")
        else:
            raise ValueError(f"The category, {category}, of {root} is not correct")
        if "nautilus" not in root:
            cp.load_samples_and_write_paramnames(
                path_samples, path_getdist + ".paramnames"
            )
            cp.write_samples_getdist_format(path_samples, path_getdist + ".txt")
        else:
            cp.load_samples_and_write_paramnames(
                path_samples, path_getdist + ".paramnames", chain_type="nautilus"
            )
            cp.write_samples_getdist_format(
                path_samples, path_getdist + ".txt", chain_type="nautilus"
            )
        chains.append(cp.load_chain(path_getdist, smoothing_scale=0.5))
    else:
        path_getdist = os.path.join(root_dir, f"ext_data/{root}/getdist_{root}")
        chains.append(cp.load_chain(path_getdist))


name_list = [
    "OMEGA_M",
    "ombh2",
    "h0",
    "n_s",
    "SIGMA_8",
    "S_8",
    "s_8_input",
    "logt_agn",
    "a",
    "m1",
    "bias_1",
]
label_list = [
    r"\Omega_{\rm m}",
    r"\omega_b h^2",
    r"h_0",
    r"n_s",
    r"\sigma_8",
    r"S_8",
    r"S_8",
    r"\log T_{\rm AGN}",
    r"A_{\rm IA}",
    r"m_1",
    r"\Delta z_1",
]

for i, chain in enumerate(chains):
    print(legend_labels[i])
    param_names = chain.getParamNames()
    for name, label in zip(name_list, label_list):
        try:
            param_names.parWithName(name).label = label
        except Exception:
            warnings.warn(f"Parameter {name} not found in chain {roots[i]}.")


# Micro management of external chains

# Account for the missing parameter conventions

idx = roots.index("KiDS-Legacy_xipm")
cp.derive_parameter_S8(chains[idx])

idx = roots.index("KiDS-Legacy_bandpowers")
cp.derive_parameter_S8(chains[idx])

idx = roots.index("KiDS-Legacy_cosebis")
cp.derive_parameter_S8(chains[idx])

# OMEGA_M not in HSC_Y3_cell
idx = roots.index("HSC_Y3_cell")
cp.adjust_paramname_chain(chains[idx], "omega_m", "OMEGA_M", r"\Omega_{\rm m}")


param_values = np.array(
    [
        "# Expt",
        "Colour",
        "S8_Mean",
        "S8_low",
        "S8_high",
        "sigma_8_Mean",
        "sigma_8_low",
        "sigma_8_high",
        "Omega_m_Mean",
        "Omega_m_low",
        "Omega_m_high",
    ]
)
escaped = np.char.replace(legend_labels, "\\", "\\\\")

for i, root in enumerate(roots):
    chain = chains[i]
    if root == "DES Y6":
        param_values = np.vstack(
            (
                param_values,
                [
                    escaped[i],
                    colours[i],
                    0.798,
                    0.015,
                    0.014,
                    0.763,
                    0.057,
                    0.050,
                    0.332,
                    0.040,
                    0.035,
                ],
            )
        )
    else:
        best_fit_params = cp.extract_best_fit_params(chain, best_fit_method="2Dkde")
        margestats = chain.getMargeStats()

        s8_stats = margestats.parWithName("S_8")
        sigma8_stats = margestats.parWithName("SIGMA_8")
        omegam_stats = margestats.parWithName("OMEGA_M")

        param_values = np.vstack(
            (
                param_values,
                [
                    escaped[i],
                    colours[i],
                    best_fit_params["S_8"],
                    best_fit_params["S_8"] - s8_stats.limits[0].lower,
                    s8_stats.limits[0].upper - best_fit_params["S_8"],
                    best_fit_params["SIGMA_8"],
                    best_fit_params["SIGMA_8"] - sigma8_stats.limits[0].lower,
                    sigma8_stats.limits[0].upper - best_fit_params["SIGMA_8"],
                    best_fit_params["OMEGA_M"],
                    best_fit_params["OMEGA_M"] - omegam_stats.limits[0].lower,
                    omegam_stats.limits[0].upper - best_fit_params["OMEGA_M"],
                ],
            )
        )
print(param_values)
np.savetxt(
    f"{root_dir}/param_values.txt",
    param_values,
    fmt=["%s" for i in range(11)],
    delimiter=";",
)


# Load the value of the parameters
cosmo = np.loadtxt(
    f"{root_dir}/param_values.txt",
    dtype={
        "names": (
            "Expt",
            "colour",
            "s8_mean",
            "s8_low",
            "s8_high",
            "sigma8_mean",
            "sigma8_low",
            "sigma8_high",
            "omegam_mean",
            "omegam_low",
            "omegam_high",
        ),
        "formats": (
            "U250",
            "U20",
            "U20",
            "U20",
            "U20",
            "U20",
            "U20",
            "U20",
            "U20",
            "U20",
            "U20",
        ),
    },
    skiprows=1,
    delimiter=";",
)
expt = np.char.replace(cosmo["Expt"], "\\\\", "\\")
colours = cosmo["colour"]
s8_mean = cosmo["s8_mean"].astype(np.float64)
s8_low = cosmo["s8_low"].astype(np.float64)
s8_high = cosmo["s8_high"].astype(np.float64)
sigma8_mean = cosmo["sigma8_mean"].astype(np.float64)
sigma8_low = cosmo["sigma8_low"].astype(np.float64)
sigma8_high = cosmo["sigma8_high"].astype(np.float64)
omegam_mean = cosmo["omegam_mean"].astype(np.float64)
omegam_low = cosmo["omegam_low"].astype(np.float64)
omegam_high = cosmo["omegam_high"].astype(np.float64)


from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(13, 8))
gs = GridSpec(1, 3, width_ratios=[1, 0.5, 0.5])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharey=ax1)
ax3 = fig.add_subplot(gs[2], sharey=ax1)

axs = [ax1, ax2, ax3]

params = [
    (s8_mean, s8_low, s8_high, r"$S_8$"),
    (sigma8_mean, sigma8_low, sigma8_high, r"$\sigma_8$"),
    (omegam_mean, omegam_low, omegam_high, r"$\Omega_{\rm m}$"),
]
reference = r"UNIONS-3500 $\xi_{\pm}(\theta)$ (This work)"

separation_after = [
    r"UNIONS-3500 $C_\ell$ (Guerrini et al. 2026)",
    r"HSC-Y3 $C_\ell$",
    r"$\xi_+$ small scales, $\theta$=[5,83] arcmin",
    r"Unmasked covmat",
    r"$\texttt{HMCode}$ no baryons",
    r"Nautilus sampler",
]
list_section_index = [r"(ii)", r"(iii)", r"(iv)", r"(v)", r"(vi)", r"(vii)"]

preliminary_watermark = False
blind_axes = False
row_spacing = 0.2

index_ref = np.where(expt == reference)[0][0]

y = np.arange(len(expt))
for ax, param in zip(axs, params):
    means, lows, highs, label = param
    for i, mean, low, high, color in zip(y, means, lows, highs, colours):
        ax.errorbar(
            mean,
            0.05 + i * row_spacing,
            xerr=np.array([low, high])[:, None],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=3,
        )
    ax.set_xlabel(label, fontsize=14)

    ax.grid(False)
    ax.tick_params(axis="y", left=False, labelleft=False)
    if label == r"$S_8$":
        ax.axvspan(
            s8_mean[index_ref] - s8_low[index_ref],
            s8_mean[index_ref] + s8_high[index_ref],
            color=colours[index_ref],
            alpha=0.2,
        )
        ax.set_xlim(0.6, 1.35)
        if blind_axes:
            ref_tick = np.mean(s8_mean[:4])
            ax.set_xticks([ref_tick + i * 0.1 for i in range(-5, 5)], labels=[])
    elif label == r"$\sigma_8$":
        ax.axvspan(
            sigma8_mean[index_ref] - sigma8_low[index_ref],
            sigma8_mean[index_ref] + sigma8_high[index_ref],
            color=colours[index_ref],
            alpha=0.2,
        )
        ax.set_xlim(0.5, 1.35)
        if blind_axes:
            ref_tick = np.mean(sigma8_mean[:4])
            ax.set_xticks([ref_tick + i * 0.2 for i in range(-2, 2)], labels=[])
    elif label == r"$\Omega_{\rm m}$":
        ax.axvspan(
            omegam_mean[index_ref] - omegam_low[index_ref],
            omegam_mean[index_ref] + omegam_high[index_ref],
            color=colours[index_ref],
            alpha=0.2,
        )
        ax.set_xlim(0.1, 0.5)
        if blind_axes:
            ref_tick = np.mean(omegam_mean[:4])
            ax.set_xticks([ref_tick + i * 0.1 for i in range(-2, 3)], labels=[])


ax1.set_yticks(0.01 + y * row_spacing)
ax1.set_yticklabels([])
for label, color in zip(expt, colours):
    if "This work" in label:
        label_bold = (
            r"$\bf{UNIONS}$-$\bf{3500}$ $\xi_{\pm}(\theta)$ $\bf{(This\ work)}$"
        )
        ax1.text(
            -0.6,
            0.05 + row_spacing * np.where(expt == label)[0][0],
            label_bold,
            fontsize=12,
            ha="left",
            va="center",
            color=color,
        )
    else:
        ax1.text(
            -0.6,
            0.05 + row_spacing * np.where(expt == label)[0][0],
            label,
            fontsize=12,
            ha="left",
            va="center",
            color=color,
        )
    if label != reference:
        index = np.where(expt == label)[0][0]
        s8_tension = cp.get_sigma_tension(
            s8_mean[index],
            s8_low[index],
            s8_high[index],
            s8_mean[index_ref],
            s8_low[index_ref],
            s8_high[index_ref],
        )
        sign_str = "+" if s8_tension > 0 else "-"
        ax1.text(
            1.32,
            0.05 + row_spacing * index,
            rf"${sign_str}{np.abs(s8_tension):.2f}" + r"\, \sigma$",
            fontsize=10,
            ha="right",
            va="center",
            color=color,
        )
# Add separation lines
for i, sep in enumerate(separation_after):
    print(sep)
    index_sep = np.where(expt == sep)[0][0]
    ax2.axhline(
        row_spacing * (index_sep + 1) - 0.07,
        color="black",
        linestyle="dotted",
        linewidth=1,
    )
    ax3.axhline(
        row_spacing * (index_sep + 1) - 0.07,
        color="black",
        linestyle="dotted",
        linewidth=1,
    )
    ax1.axhline(
        row_spacing * (index_sep + 1) - 0.07,
        xmin=-1.8,
        color="black",
        linestyle="dotted",
        linewidth=1,
        clip_on=False,
    )
    ax1.text(
        -0.61,
        row_spacing * (index_sep + 1) + 0.05,
        list_section_index[i],
        fontsize=12,
        fontweight="bold",
        va="center",
        ha="right",
    )


# --- Add section label (i)) ---
ax1.text(-0.61, 0.05, r"(i)", fontsize=12, fontweight="bold", va="center", ha="right")

if preliminary_watermark:
    plt.figtext(
        0.5,
        0.5,
        "PRELIMINARY",
        fontsize=50,
        color="gray",
        ha="center",
        va="center",
        alpha=0.3,
        rotation=330,
    )

plt.gca().invert_yaxis()

plt.tight_layout()

# #Save pdf
plt.savefig("./../../results/S8_whisker_plot.pdf", bbox_inches="tight")
