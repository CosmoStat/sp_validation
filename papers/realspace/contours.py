# # 2D contour plots
#
# This notebook produces the plots for all the 2D contours in the results section.


import os.path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from getdist import plots

plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

plt.rcParams["text.usetex"] = True

sns.set_palette("husl")
g = plots.get_subplot_plotter(width_inch=30)
g.settings.axes_fontsize = 70
g.settings.axes_labelsize = 80
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 70


# SPECIFY DATA DIRECTORY AND DESIRED CHAINS TO ANALYSE

root_dir = "/n09data/guerrini/output_chains/"
path_datavectors = "/home/guerrini/sp_validation/cosmo_inference/data/"
path_output_chains = "/n09data/guerrini/output_chains/"

data = fits.open(
    os.path.join(
        path_datavectors,
        "SP_v1.4.6.3_config/SP_v1.4.6.3_B/cosmosis_SP_v1.4.6.3_leak_corr_B_masked.fits",
    )
)

roots_fid = {
    "SP_v1.4.6.3_leak_corr_B": r"UNIONS-3500 $C_\ell$",
    "SP_v1.4.6.3_B_fiducial_config": r"UNIONS-3500 $\xi_\pm$ (This work) ",
    "KiDS-Legacy_xipm": r"KiDS-Legacy $\xi_\pm$",
    "HSC_Y3": r"HSC-Y3 $\xi_\pm$",
    "Planck18": r"$\textit{Planck}$ 2018",
}

roots_full = {
    "SP_v1.4.6.3_B_fiducial_config": r"UNIONS-3500 $\xi_\pm$ (This work) ",
}

roots_ia = {
    "SP_v1.4.6.3_B_fiducial_config": r"Gaussian $A_{\rm{IA}}$ prior",
    "SP_v1.4.6.3_B_flat_ia_config": r"Flat $A_{\rm{IA}}$ prior",
    "SP_v1.4.6.3_B_no_ia_config": r"No IA",
}

roots_ext = {
    "SP_v1.4.6.3_B_fiducial_config": r"UNIONS-3500 $\xi_\pm$",
    "SP_v1.4.6.3_B_planck_config": r"UNIONS-3500 $\xi_\pm$ + CMB",
    "SP_v1.4.6.3_B_planck_desi_config": r"UNIONS-3500 $\xi_\pm$ + CMB + BAO",
    "Planck18": r"$\textit{Planck}$ 2018",
}

roots_dz = {
    "SP_v1.4.6.3_B_fiducial_config": r"Gaussian $\Delta z$ prior",
    "SP_v1.4.6.3_B_flat_delta_z_config": r"Flat $\Delta z$ prior",
    "SP_v1.4.6.3_B_no_delta_z_config": r"No $\Delta z$ modelling",
}

roots_psf = {
    "SP_v1.4.6.3_B_flat_alpha_beta_config": r"Flat $\alpha$ and $\beta$ priors",
    "SP_v1.4.6.3_B_fiducial_config": r"Gaussian $\alpha$ and $\beta$ priors",
    "SP_v1.4.6.3_B_no_xi_sys_config": r"No $\xi^{\rm sys}$ included",
    "SP_v1.4.6.3_B_no_leak_corr_config": r"No object-wise leakage correction",
}

roots_scale = {
    "SP_v1.4.6.3_B_fiducial_config": r"$\xi_+$: $\theta=[12,83]$",
    "SP_v1.4.6.3_B_small_scales_config": r"$\xi_+$: $\theta=[5,83]$",
}

roots_nonlin = {
    "SP_v1.4.6.3_B_fiducial_config": r"Fiducial (\texttt{HMCode2020}, $\log(T_{\rm AGN})$)",
    "SP_v1.4.6.3_B_no_baryons_config": r"\texttt{HMCode2020} no baryons",
    "SP_v1.4.6.3_B_halofit_config": r"\texttt{Halofit}",
}
roots = roots_ext


# ## Retrieve the chains


# READ CHAIN

chains = []

for i, root in enumerate(list(roots.keys())):
    burnin = 0
    if "SP" not in root:
        chain = g.samples_for_root(
            root_dir + "ext_data/{}/getdist_{}".format(root, root),
            cache=False,
            settings={
                "ignore_rows": burnin,
                #  'smooth_scale_2D':0.2,
                #  'smooth_scale_1D':0.2
            },
        )
        p = chain.getParams()
        if hasattr(p, "S_8") == False:
            omega_m = chain.getParams().OMEGA_M
            sigma_8 = chain.getParams().SIGMA_8

            s_8 = sigma_8 * (omega_m / 0.3) ** 0.5

            chain.addDerived(s_8, name="S_8", label=r"S_8")

            p = chain.paramNames.parWithName("S_8")

    elif "config" in root:
        if os.path.isfile(root_dir + "{}/getdist_{}.txt".format(root, root)) == False:
            samples = np.loadtxt(root_dir + "{}/samples_{}.txt".format(root, root))

            if "nautilus" in root:
                weights = np.exp(samples[:, -3])
                neglogL = samples[:, -2] - samples[:, -1]

                samples = np.column_stack((weights, neglogL, samples[:, 0:-3]))
            elif "mh" in root:
                samples = np.column_stack(
                    (
                        np.ones_like(samples[:, -1]),
                        np.log(samples[:, -1]) - np.log(samples[:, -2]),
                        samples[:, 0:-2],
                    )
                )
                burnin = 0.3
            else:
                samples = np.column_stack(
                    (samples[:, -1], samples[:, -3], samples[:, 0:-4])
                )

            np.savetxt(root_dir + "{}/getdist_{}.txt".format(root, root), samples)

        chain = g.samples_for_root(
            root_dir + "{}/getdist_{}".format(root, root),
            cache=False,
            settings={
                "ignore_rows": burnin,
                #  'smooth_scale_2D':0.2,
                #  'smooth_scale_1D':0.2
            },
        )
    else:
        if (
            os.path.isfile(
                root_dir + "{}/{}/getdist_{}_cell.txt".format(root, root, root)
            )
            == False
        ):
            samples = np.loadtxt(
                root_dir + "{}/{}/samples_{}_cell.txt".format(root, root, root)
            )

            if "nautilus" in root:
                weights = np.exp(samples[:, -3])
                neglogL = samples[:, -2] - samples[:, -1]

                samples = np.column_stack((weights, neglogL, samples[:, 0:-3]))
            elif "mh" in root:
                samples = np.column_stack(
                    (
                        np.ones_like(samples[:, -1]),
                        np.log(samples[:, -1]) - np.log(samples[:, -2]),
                        samples[:, 0:-2],
                    )
                )
                burnin = 0.3
            else:
                samples = np.column_stack(
                    (samples[:, -1], samples[:, -3], samples[:, 0:-4])
                )

            np.savetxt(
                root_dir + "{}/{}/getdist_{}_cell.txt".format(root, root, root), samples
            )

        chain = g.samples_for_root(
            root_dir + "{}/{}/getdist_{}_cell".format(root, root, root),
            cache=False,
            settings={
                "ignore_rows": burnin,
                #  'smooth_scale_2D':0.2,
                #  'smooth_scale_1D':0.2
            },
        )
    p = chain.getParams()

    chains.append(chain)


name_list = [
    "OMEGA_M",
    "ombh2",
    "h0",
    "n_s",
    "SIGMA_8",
    "S_8",
    "logt_agn",
    "a",
    "m1",
    "bias_1",
    "alpha",
    "beta",
    "omch2",
]
label_list = [
    r"\Omega_{\rm m}",
    r"\omega_{\rm b}",
    r"h",
    r"n_{\rm s}",
    r"\sigma_8",
    r"S_8",
    r"\log T_{\rm AGN}",
    r"A_{\rm IA}",
    r"m_1",
    r"\Delta z",
    r"\alpha_{\rm PSF}",
    r"\beta_{\rm PSF}",
    r"\omega_{\rm c}",
]

for chain in chains:
    param_names = chain.getParamNames()
    p = chain.getParams()
    for name, label in zip(name_list, label_list):
        if hasattr(p, name):
            param_names.parWithName(name).label = label

legend_labels = list(roots.values())


# ## Plot the chains


# ### FIDUCIAL PLOT


colours = [
    "royalblue",
    "orange",
    "crimson",
    "forestgreen",
    "indigo",
]

linestyle = ["solid", "solid", "solid", "solid", "solid"]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]

# FIDUCIAL PLOT
g.triangle_plot(
    chains,
    ["SIGMA_8", "S_8", "OMEGA_M"],  #
    legend_labels=legend_labels,
    line_args=line_args,
    contour_colors=colours,
    label_order=[1, 0, 2, 3, 4],
    filled=[True, True, False, False, True],
)

g.export("./../../results/SP_v1.4.6.3_B_fiducial_config_contour_plot.pdf")


# ### FULL PLOT


g.settings.axes_fontsize = 40
g.settings.axes_labelsize = 50

colours = [
    "orange",
]

linestyle = [
    "solid",
]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]

# FIDUCIAL PLOT
g.triangle_plot(
    chains,
    [
        "OMEGA_M",
        "ombh2",
        "h0",
        "n_s",
        "SIGMA_8",
        "S_8",
        "logt_agn",
        "a",
        "m1",
        "bias_1",
    ],
    legend_labels=legend_labels,
    line_args=line_args,
    contour_colors=colours,
    filled=True,
)

g.export("./../../results/SP_v1.4.6.3_B_fiducial_config_contour_plot_full.pdf")


# ### IA PLOT


colours = [
    "orange",
    "royalblue",
    "forestgreen",
]

linestyle = [
    "solid",
    "solid",
    "solid",
]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]

g.triangle_plot(
    chains,
    ["S_8", "OMEGA_M", "a"],  #
    legend_labels=legend_labels,
    line_args=line_args,
    contour_args={"alpha": 0.6},
    contour_colors=colours,
    filled=[True, False, True],
)

g.export("./../../results/SP_v1.4.6.3_B_fiducial_config_contour_plot_ia.pdf")


# ### PSF PLOT


colours = [
    "royalblue",
    "orange",
    "hotpink",
    "slategray",
]

linestyle = [
    "solid",
    "solid",
    "solid",
    "solid",
]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]

g.triangle_plot(
    chains,
    ["S_8", "OMEGA_M", "alpha", "beta"],  #
    legend_labels=legend_labels,
    line_args=line_args,
    contour_args=[{"alpha": 1}, {"alpha": 0.6}, {"alpha": 0.8}, {"alpha": 0.8}],
    contour_colors=colours,
    legend_loc="upper right",
    label_order=[1, 0, 2, 3],
    filled=[False, True, True, True],
)

g.subplots[3, 2].scatter(
    0.005, 0.81, color="k", marker="X", s=400, label="Fiducial config best-fit"
)
g.subplots[3, 2].scatter(
    0.022, 0.798, color="k", marker="P", s=400, label="Fiducial config best-fit"
)

g.export("./../../results/SP_v1.4.6.3_B_fiducial_config_contour_plot_psf.pdf")


# ### DELTA Z PLOT


colours = [
    "orange",
    "royalblue",
    "indigo",
]

linestyle = [
    "solid",
    "solid",
    "solid",
]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]
g.triangle_plot(
    chains,
    ["S_8", "OMEGA_M", "bias_1"],  #
    legend_labels=legend_labels,
    line_args=line_args,
    contour_args=[{"alpha": 1.0}, {"alpha": 0.9}, {"alpha": 0.5}],
    contour_colors=colours,
    filled=[True, False, True],
)

g.export("./../../results/SP_v1.4.6.3_B_fiducial_config_contour_plot_dz.pdf")


# ### EXTERNAL DATA


colours = [
    "orange",
    "royalblue",
    "crimson",
    "forestgreen",
]

linestyle = [
    "solid",
    "solid",
    "solid",
    "solid",
    "solid",
]

line_args = [dict(color=col, ls=ls) for col, ls in zip(colours, linestyle)]

g = plots.get_subplot_plotter(width_inch=10)
g.settings.axes_fontsize = 25
g.settings.axes_labelsize = 25
g.settings.legend_fontsize = 22

g.plot_2d(
    chains,
    ["S_8", "OMEGA_M", "SIGMA_8"],  #
    line_args=line_args,
    contour_colors=colours,
    legend_labels=legend_labels,
    alphas=[0.7, 1.0, 1.0, 1.0],
    filled=[True, True, True, False],
)

g.add_y_bands(0.2975, 0.0086, alpha2=0, color="k", label="BAO")
g.add_legend(legend_labels, legend_loc="upper right")

g.export("./../../results/SP_v1.4.6.3_B_fiducial_config_contour_plot_ext.pdf")


# ### Small scales


colours = [
    "orange",
    "dodgerblue",
]

linestyle = [
    "solid",
    "solid",
]

line_args = [dict(color=col, ls=ls) for col, ls in zip(colours, linestyle)]

g = plots.get_subplot_plotter(width_inch=9)
g.settings.axes_fontsize = 25
g.settings.axes_labelsize = 25
g.settings.alpha_filled_add = 0.7
g.settings.legend_fontsize = 30

g.plot_2d(
    chains,
    ["S_8", "OMEGA_M"],  #
    line_args=line_args,
    contour_args=[{"alpha": 0.7}, {"alpha": 1.0}],
    contour_colors=colours,
    filled=[True, True],
)
g.add_legend(legend_labels, legend_loc="upper right")

g.export("./../../results/SP_v1.4.6.3_B_fiducial_config_contour_plot_scales.pdf")


# ### BBN Prior


from getdist.gaussian_mixtures import Gaussian1D

colours = [
    "orange",
    "royalblue",
]

linestyle = [
    "solid",
    "solid",
]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]

# BBN PRIOR
bbn_prior = Gaussian1D(
    mean=0.02218,
    sigma=0.00055,
    name="ombh2",
    labels=[r"\omega_{\rm b}"],
    label="BBN prior",
)
bbn_chain = bbn_prior.MCSamples(3000, label="BBN prior")

g.triangle_plot(
    chains + [bbn_chain],
    name_list,
    legend_labels=legend_labels,
    line_args=line_args,
    contour_colors=colours,
    filled=[True, False],
)


# ## Plot the best-fit $\xi_\pm$


xi_p_data = data["XI_PLUS"].data
xi_m_data = data["XI_MINUS"].data
cov_mat = data["COVMAT"].data

labels = roots_scale.values()

bbox_to_anchor_xip = (0.685, 0.09)
bbox_to_anchor_xim = (0.3, 0.65)
theta_min = 1.0
theta_max = 250.0
loc_legend = "lower center"


colours = [
    "orange",
    "dodgerblue",
]

linestyle = [
    "solid",
    "solid",
]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]

labels = roots_scale.values()

fig, ax = plt.subplots(1, 1, figsize=(11, 7))

theta, xi_p, xi_m = xi_p_data["ANG"], xi_p_data["VALUE"], xi_m_data["VALUE"]
ax.errorbar(
    theta,
    theta * xi_p,
    yerr=theta * np.sqrt(np.diag(cov_mat[: len(theta), : len(theta)])),
    fmt="o",
    color="black",
    capsize=2,
)

for idx, (label, root) in enumerate(zip(labels, roots_scale)):
    # Read the results
    theta = (
        (
            np.loadtxt(
                path_output_chains + "{}/best_fit/shear_xi_plus/theta.txt".format(root)
            )
        )
        * 180
        / np.pi
        * 60
    )
    xi_plus = np.loadtxt(
        path_output_chains + "{}/best_fit/shear_xi_plus/bin_1_1.txt".format(root)
    )
    xi_minus = np.loadtxt(
        path_output_chains + "{}/best_fit/shear_xi_minus/bin_1_1.txt".format(root)
    )
    xi_sys_plus = np.loadtxt(
        path_output_chains + "{}/best_fit/xi_sys/shear_xi_plus.txt".format(root)
    )
    xi_sys_minus = np.loadtxt(
        path_output_chains + "{}/best_fit/xi_sys/shear_xi_minus.txt".format(root)
    )
    theta_xi_sys = (
        np.loadtxt(path_output_chains + "{}/best_fit/xi_sys/theta.txt".format(root))
        * 180
        / np.pi
        * 60
    )

    xi_sys_plus = np.interp(theta, theta_xi_sys, xi_sys_plus)
    xi_sys_minus = np.interp(theta, theta_xi_sys, xi_sys_minus)
    xi_plus += xi_sys_plus
    xi_minus += xi_sys_minus

    mask = (theta > theta_min) & (theta < theta_max)
    theta = theta[mask]
    ax.plot(theta, theta * xi_plus[mask], label=label, **line_args[idx])

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]

ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color="gray", alpha=0.2)
ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=5, color="gray", alpha=0.7)
ax.fill_betweenx(y=[ymin, ymax], x1=83, x2=300, color="gray", alpha=0.2)

ax.set_ylim(ymin, ymax)

ax.set_ylabel(r"$\theta \xi_\pm$", fontsize=26)
ax.set_xlabel(r"$\theta$ (arcmin)", fontsize=26)
ax.set_xlim([theta.min() - 0.1, theta.max() + 20])
ax.set_title(r"$\xi_+(\theta)$", fontsize=26)
ax.set_xscale("log")
ax.set_xticks(np.array([1, 10, 100]))
ax.tick_params(axis="x", which="minor", length=2, width=0.8)
ax.tick_params(axis="both", which="major", labelsize=24)
ax.tick_params(axis="both", which="minor", labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(24)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xip, fontsize=20)


plt.savefig("./../../results/scale_cut_xipm_SP_v1.4.6.3_B.pdf", bbox_inches="tight")


labels = roots_nonlin.values()

colours = ["orange", "hotpink", "teal"]

linestyle = ["solid", "solid", "dashed"]

line_args = [dict(color=col, ls=ls, lw=2) for col, ls in zip(colours, linestyle)]

fig, [ax, ax2] = plt.subplots(2, 1, figsize=(11, 14))

theta, xi_p, xi_m = xi_p_data["ANG"], xi_p_data["VALUE"], xi_m_data["VALUE"]
ax.errorbar(
    theta,
    theta * xi_p,
    yerr=theta * np.sqrt(np.diag(cov_mat[: len(theta), : len(theta)])),
    fmt="o",
    color="black",
    capsize=2,
)
ax2.errorbar(
    theta,
    theta * xi_m,
    yerr=theta
    * np.sqrt(
        np.diag(cov_mat[len(theta) : 2 * len(theta), len(theta) : 2 * len(theta)])
    ),
    fmt="o",
    color="black",
    capsize=2,
)

for idx, (label, root) in enumerate(zip(labels, roots_nonlin)):
    # Read the results
    theta = (
        (
            np.loadtxt(
                path_output_chains + "{}/best_fit/shear_xi_plus/theta.txt".format(root)
            )
        )
        * 180
        / np.pi
        * 60
    )
    xi_plus = np.loadtxt(
        path_output_chains + "{}/best_fit/shear_xi_plus/bin_1_1.txt".format(root)
    )
    xi_minus = np.loadtxt(
        path_output_chains + "{}/best_fit/shear_xi_minus/bin_1_1.txt".format(root)
    )
    xi_sys_plus = np.loadtxt(
        path_output_chains + "{}/best_fit/xi_sys/shear_xi_plus.txt".format(root)
    )
    xi_sys_minus = np.loadtxt(
        path_output_chains + "{}/best_fit/xi_sys/shear_xi_minus.txt".format(root)
    )
    theta_xi_sys = (
        np.loadtxt(path_output_chains + "{}/best_fit/xi_sys/theta.txt".format(root))
        * 180
        / np.pi
        * 60
    )

    xi_sys_plus = np.interp(theta, theta_xi_sys, xi_sys_plus)
    xi_sys_minus = np.interp(theta, theta_xi_sys, xi_sys_minus)
    xi_plus += xi_sys_plus
    xi_minus += xi_sys_minus

    mask = (theta > theta_min) & (theta < theta_max)
    theta = theta[mask]
    ax.plot(theta, theta * xi_plus[mask], label=label, **line_args[idx])
    ax2.plot(theta, theta * xi_minus[mask], label=label, **line_args[idx])

ymin = ax.get_ylim()[0]
ymax = ax.get_ylim()[1]
ax.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color="gray", alpha=0.2)
ax.fill_betweenx(y=[ymin, ymax], x1=83, x2=300, color="gray", alpha=0.2)

ax.set_ylim(ymin, ymax)

ax.set_ylabel(r"$\theta \xi_\pm$", fontsize=26)
ax.set_xlabel(r"$\theta$ (arcmin)", fontsize=26)
ax.set_xlim([theta.min() - 0.1, theta.max() + 20])
ax.set_title(r"$\xi_+(\theta)$", fontsize=26)
ax.set_xscale("log")
ax.set_xticks(np.array([1, 10, 100]))
ax.tick_params(axis="x", which="minor", length=2, width=0.8)
ax.tick_params(axis="both", which="major", labelsize=24)
ax.tick_params(axis="both", which="minor", labelsize=20)
ax.yaxis.get_offset_text().set_fontsize(24)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


ymin = ax2.get_ylim()[0]
ymax = ax2.get_ylim()[1]
ax2.fill_betweenx(y=[ymin, ymax], x1=0, x2=12, color="gray", alpha=0.2)
ax2.fill_betweenx(y=[ymin, ymax], x1=83, x2=3000, color="gray", alpha=0.2)

ax2.set_ylim(ymin, ymax)
ax2.set_xlabel(r"$\theta$ (arcmin)", fontsize=26)
ax2.set_xlim([theta.min() - 0.1, theta.max()])
ax2.set_xscale("log")
ax2.set_title(r"$\xi_-(\vartheta)$", fontsize=26)
ax2.set_xticks(np.array([1, 10, 100]))
ax2.tick_params(axis="x", which="minor", length=2, width=0.8)
ax2.tick_params(axis="both", which="major", labelsize=24)
ax2.tick_params(axis="both", which="minor", labelsize=20)
ax2.yaxis.get_offset_text().set_fontsize(24)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax2.legend(loc=loc_legend, bbox_to_anchor=bbox_to_anchor_xim, fontsize=20)

plt.savefig("./../../results/nonlin_xipm_SP_v1.4.6.3_B.pdf", bbox_inches="tight")
