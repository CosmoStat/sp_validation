"""Run the pipeline's object-wise regression on the clean and PSF mocks."""

import json
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sp_validation.cosmo_val import CosmologyValidation

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/cvpaint")
config = yaml.safe_load((ROOT / "config.yaml").read_text())
versions = ["GLASS1_clean", "GLASS1_psfleak"]
cv = CosmologyValidation(
    versions=versions,
    catalog_config=str(ROOT / "cat_config.yaml"),
    output_dir=str(ROOT / "output"),
    npatch=100,
    theta_min=1.0,
    theta_max=250.0,
    nbins=20,
    cosmo_params=config["cosmo_val"]["cosmo_params"],
)
cv.calculate_objectwise_leakage(tomography=False)
coefficients = {
    v: {
        bin_id: {key: [item.nominal_value, item.std_dev] for key, item in coeff.items()}
        for bin_id, coeff in bins.items()
    }
    for v, bins in cv.leakage_coeff.items()
}
(ROOT / "objectwise_early.json").write_text(json.dumps(coefficients, indent=2))
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for j, version in enumerate(versions):
    for ax, key in zip(axes, ("a11", "a22", "aii_mean")):
        value, error = coefficients[version]["tomo_bin_all"][key]
        truth = 0.1 if version.endswith("psfleak") else 0.0
        ax.errorbar(j, value, yerr=error, fmt="o")
        ax.plot(j, truth, "kx", ms=7)
        ax.set(
            title=key,
            xticks=[0, 1],
            xticklabels=["clean", "α=0.1 injection"],
            ylabel="Object-wise leakage slope",
        )
fig.tight_layout()
fig.savefig(ROOT / "figures/objectwise_psf.png", dpi=170)
plt.close(fig)
print(
    "Saved objectwise_early.json and figures/objectwise_psf.png; inspect the figure.",
    flush=True,
)
