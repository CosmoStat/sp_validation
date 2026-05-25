"""Plot the three blinded n(z) curves used in the Moriond slides."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plotting_utils import PAPER_MPLSTYLE


def load_nz(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


plt.style.use(PAPER_MPLSTYLE)
sns.set_palette("husl", 3)

fig, ax = plt.subplots(figsize=(4.5, 3.2))

for label, path in [
    ("Blind A", snakemake.input.nz_A),
    ("Blind B", snakemake.input.nz_B),
    ("Blind C", snakemake.input.nz_C),
]:
    z, nz = load_nz(path)
    ax.plot(z, nz, linewidth=1.6, label=label)

ax.set_xlabel(r"$z$", fontsize=18)
ax.set_ylabel(r"$n(z)$", fontsize=18)
ax.set_xlim(0, 2)
ax.set_ylim(bottom=0)
ax.legend(frameon=False, loc="upper right", fontsize=15)
ax.tick_params(labelsize=15)
ax.grid(False)

fig.tight_layout()
Path(snakemake.output[0]).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(snakemake.output[0], dpi=220, bbox_inches="tight")
