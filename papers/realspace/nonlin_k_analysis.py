# # Nonlinear $k$ contributions
#
# This notebook plots the 2D heatmap of ratio of scale contributions to the $\xi_\pm$ 2PCF given angular scale $\theta$ and wavenumber $k$.



import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

plt.style.use("/home/guerrini/matplotlib_config/paper.mplstyle")

plt.rcParams["text.usetex"] = True

plt.rcParams.update(
    {
        "font.size": 20,
        "axes.titlesize": 21,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 21,
    }
)
sns.set_palette("husl")

blind = "B"
ver = "v1.4.6.3"


data_dir = "/n23data1/n06data/lgoh/scratch/UNIONS/cosmo_inference/data/"

# Read the 2D array from the text file

file_headers = ["xip_%s_%s" % (ver, blind), "xim_%s_%s" % (ver, blind)]

for f in file_headers:
    xis = np.loadtxt(data_dir + f"theta_k_{f}.txt")
    xis_reshaped = xis.reshape(-1, 201)
    sorted_xis = xis_reshaped[np.argsort(xis_reshaped[:, 0])]

    np.savetxt(data_dir + f"theta_k_{f}_sorted.txt", sorted_xis)


fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# --- k grid ---
h = 0.6766
k_plot = np.logspace(-4, 2, 200)

file_header = "%s_%s" % (ver, blind)

xi_thetas = np.loadtxt(data_dir + f"theta_k_xip_{file_header}_sorted.txt")
thetas = xi_thetas[:, 0]
xis = xi_thetas[:, 1:]

# normalise
xi_plot = xis / np.max(xis, axis=1, keepdims=True)

T, K = np.meshgrid(thetas, k_plot)

axs[0].contour(T, K, xi_plot.T, levels=[0.9], colors="red", linewidths=1.7)
pcm = axs[0].pcolormesh(T, K, xi_plot.T, shading="auto", cmap="viridis")
pcm.set_rasterized(True)

axs[0].axvline(5, color="k", ls="dashed", lw=1.2)
axs[0].axvline(12, color="white", ls="dashed", lw=1.6)
axs[0].axhline(1, color="k", ls="dashed", lw=1.2)  # converted to h/Mpc space if needed
axs[0].axhline(0.425, color="white", ls="dashed", lw=1.6)

axs[0].set_yscale("log")
axs[0].set_xlabel(r"$\theta\ \mathrm{(arcmin)}$")
axs[0].set_ylabel(r"$k\ (h$ Mpc$^{-1})$")

axs[0].set_title(r"$\xi_+$")

xi_thetas = np.loadtxt(data_dir + f"theta_k_xim_{file_header}_sorted.txt")
thetas = xi_thetas[:, 0]
xis = xi_thetas[:, 1:]

xi_plot = xis / np.max(xis, axis=1, keepdims=True)

T, K = np.meshgrid(thetas, k_plot)

axs[1].contour(T, K, xi_plot.T, levels=[0.9], colors="red", linewidths=1.7)
pcm = axs[1].pcolormesh(T, K, xi_plot.T, shading="nearest", cmap="viridis")
pcm.set_rasterized(True)

axs[1].axvline(12, color="white", ls="dashed", lw=1.6)
axs[1].axhline(2.85, color="white", ls="dashed", lw=1.6)


axs[1].set_yscale("log")
axs[1].set_xlabel(r"$\theta\ \mathrm{(arcmin)}$")
axs[1].set_ylabel(r"$k\ (h$ Mpc$^{-1})$")
axs[1].set_title(r"$\xi_-$")


fig.tight_layout()

cbar_ax = fig.add_axes([0.99, 0.15, 0.02, 0.7])
cbar = fig.colorbar(pcm, cax=cbar_ax)

fig.savefig("./../../results/theta_k_xip_xim_{ver}_{blind}.pdf", bbox_inches="tight"
)
