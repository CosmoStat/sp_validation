"""Render calibration diagnostics and saved-delivery z2 coverage."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from sp_validation.nz_diagnostics import COLORS, render

matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("recipe")
parser.add_argument("output")
parser.add_argument("figures")
args = parser.parse_args()
recipe = yaml.safe_load(Path(args.recipe).read_text())
output = Path(args.output)
figures = Path(args.figures)
render(
    dict(
        recipe["diagnostics"],
        calibration_arrays=output / "calibration_arrays.npz",
        model=output / "som_assignment_model.npz",
        summary=output / "summary.json",
        source_catalogue=recipe["inputs"]["gold_reference"],
        side=recipe["som"]["side"],
        workers=16,
    ),
    figures,
)
# This comparison uses saved delivery counts; production runs only z+z2.
delivery = Path("/n17data/cdaley/unions/analyses/shear_tomographic/nz_v2_full/products")
retention = json.loads((delivery / "summary.json").read_text())["retention"]
stages = ["all", "complete_photometry", "bpz_bounds", "gold"]
x = np.arange(len(stages))
fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
width = 0.34
for j, (key, label, color) in enumerate(
    (("zonly", "Canonical z only", COLORS[2]), ("both", "z + z2", COLORS[0]))
):
    bars = ax.bar(
        x + (j - 0.5) * width,
        [100 * retention[key][stage] for stage in stages],
        width,
        label=label,
        color=color,
    )
    ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
ax.set(
    xticks=x,
    xticklabels=["All", "Complete\nphotometry", "BPZ\nbounds", "Gold"],
    ylabel="Retained bound-9 shear weight (%)",
    ylim=(0, 110),
    title="Coverage with and without z2 (saved delivery)",
)
ax.legend()
fig.savefig(figures / "coverage_z2.png", dpi=100, bbox_inches="tight")
plt.close(fig)
