# %%
import IPython

ipython = IPython.get_ipython()

if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(".//matplotlib_config/paper.mplstyle")
plt.rcParams["text.usetex"] = True

sns.set_palette("husl")

if ipython is not None:
    ipython.run_line_magic("matplotlib", "inline")

# %%
blind = "B"
path_redshift_distribution = (
    f"/n17data/sguerrini/UNIONS/WL/nz/v1.4.6.3/nz_SP_v1.4.6.3_{blind}.txt"
)
redshift_distribution = np.loadtxt(path_redshift_distribution)


# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    redshift_distribution[:, 0],
    redshift_distribution[:, 1],
    label="SP v1.4.6 A",
)
ax.fill_between(
    redshift_distribution[:, 0],
    redshift_distribution[:, 1],
    alpha=0.3,
)
ax.set_xlim(0, 3)
ax.set_ylim(0, None)
ax.set_xlabel(r"$z$", fontsize=14)
ax.set_ylabel(r"$n(z)$", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)

plt.savefig("./plots/redshift_distribution_SP_v1.4.6.png", dpi=300)
# Save PDF
plt.savefig("./plots/redshift_distribution_SP_v1.4.6.pdf")
plt.show()
# %%
