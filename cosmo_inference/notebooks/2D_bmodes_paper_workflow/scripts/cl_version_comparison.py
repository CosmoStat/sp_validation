"""Harmonic-space C_ell^BB version comparison.

Claim: B-mode power spectra consistent with zero for v1.4.6 and v1.4.8,
while v1.4.5 shows marginal tension.

Visualizes C_ell^BB and C_ell^EB normalized by error bars (C_ell / sigma).
Statistical PTEs reported separately in evidence.json.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy import stats


plt.style.use(
    "/n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/config/paper.mplstyle"
)

# Scale cuts from harmonic space paper (Guerrini et al.)
ELL_MIN_CUT = 300
ELL_MAX_CUT = 1600


# SquareRootScale for x-axis
class SquareRootScale(mscale.ScaleBase):
    name = "squareroot"

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self, axis, **kwargs)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mticker.AutoLocator())
        axis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        axis.set_minor_locator(mticker.NullLocator())
        axis.set_minor_formatter(mticker.NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0.0, vmin), vmax

    class SquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 0.5

        def inverted(self):
            return SquareRootScale.InvertedSquareRootTransform()

    class InvertedSquareRootTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.array(a) ** 2

        def inverted(self):
            return SquareRootScale.SquareRootTransform()

    def get_transform(self):
        return self.SquareRootTransform()


mscale.register_scale(SquareRootScale)


DEFAULT_VERSION_ALPHA = {
    "SP_v1.4.5_leak_corr": 0.4,
    "SP_v1.4.6_leak_corr": 1.0,
    "SP_v1.4.8_leak_corr": 0.4,
}


VERSION_LABELS = {
    "SP_v1.4.5_leak_corr": "Initial",
    "SP_v1.4.6_leak_corr": "Fiducial",
    "SP_v1.4.8_leak_corr": "Masked",
}


def _version_label(version):
    return VERSION_LABELS.get(version, version.replace("SP_", "").replace("_leak_corr", ""))


def _compute_pte(data, covariance):
    chi2 = float(data @ np.linalg.solve(covariance, data))
    dof = len(data)
    pte = stats.chi2.sf(chi2, dof)
    return pte, chi2, dof


def main():
    from snakemake.script import snakemake

    # Read config
    import yaml
    with open(snakemake.input["config"]) as f:
        config = yaml.safe_load(f)

    versions = config["versions"]
    version_alpha = DEFAULT_VERSION_ALPHA

    # Load data for all versions
    datasets = []
    colors = sns.color_palette("colorblind", len(versions))

    for i, version in enumerate(versions):
        hdu = fits.open(snakemake.input["pseudo_cl"][i])
        data = hdu["PSEUDO_CELL"].data
        hdu.close()

        ell = data["ELL"]
        cl_bb = data["BB"]
        cl_eb = data["EB"]

        hdu_cov = fits.open(snakemake.input["pseudo_cl_cov"][i])
        cov_bb = hdu_cov["COVAR_BB_BB"].data
        cov_eb = hdu_cov["COVAR_EB_EB"].data
        hdu_cov.close()

        sigma_bb = np.sqrt(np.diag(cov_bb))
        sigma_eb = np.sqrt(np.diag(cov_eb))

        pte_bb, chi2_bb, dof_bb = _compute_pte(cl_bb, cov_bb)
        pte_eb, chi2_eb, dof_eb = _compute_pte(cl_eb, cov_eb)

        datasets.append({
            "version": version,
            "label": _version_label(version),
            "color": colors[i],
            "alpha": version_alpha.get(version, 1.0),
            "ell": ell,
            "cl_bb": cl_bb,
            "cl_eb": cl_eb,
            "sigma_bb": sigma_bb,
            "sigma_eb": sigma_eb,
            "pte_bb": pte_bb,
            "chi2_bb": chi2_bb,
            "dof_bb": dof_bb,
            "pte_eb": pte_eb,
            "chi2_eb": chi2_eb,
            "dof_eb": dof_eb,
        })

    # Two-panel figure: BB (top) and EB (bottom)
    # Full-width figure (7.24 inches for two-column A&A format)
    fig_width = 7.24
    fig, (ax_bb, ax_eb) = plt.subplots(2, 1, figsize=(fig_width, fig_width * 0.6), sharex=True)

    ell_ref = datasets[0]["ell"]
    ell_widths = np.diff(ell_ref)
    ell_widths = np.append(ell_widths, ell_widths[-1])
    jitter_fraction = 0.15

    # Marker styles matching cosebis_version_comparison
    marker_styles = ["o", "s", "D"]

    legend_handles = []
    legend_labels = []

    for i, data in enumerate(datasets):
        ell = data["ell"]
        jitter_offset = (i - (len(datasets) - 1) / 2) * jitter_fraction
        ell_jittered = ell + jitter_offset * ell_widths
        color = data["color"]
        label = data["label"]
        alpha = data["alpha"]
        marker = marker_styles[i] if i < len(marker_styles) else "o"

        # Normalize by error: C_ell / sigma
        cl_bb_normalized = data["cl_bb"] / data["sigma_bb"]
        cl_eb_normalized = data["cl_eb"] / data["sigma_eb"]

        line_bb = ax_bb.errorbar(
            ell_jittered, cl_bb_normalized, yerr=np.ones_like(cl_bb_normalized),
            fmt=marker, color=color, alpha=alpha,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5,
            markersize=4, capsize=2, capthick=0.8, linewidth=0.8, elinewidth=0.8,
        )

        ax_eb.errorbar(
            ell_jittered, cl_eb_normalized, yerr=np.ones_like(cl_eb_normalized),
            fmt=marker, color=color, alpha=alpha,
            markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.5,
            markersize=4, capsize=2, capthick=0.8, linewidth=0.8, elinewidth=0.8,
        )

        # Collect legend handles from BB panel only
        legend_handles.append(line_bb)
        legend_labels.append(label)

    all_ell = np.concatenate([d["ell"] for d in datasets])
    ell_min, ell_max = all_ell.min() * 0.9, all_ell.max() * 1.05

    major_ticks = np.array([100, 400, 900, 1600])
    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]

    for ax, ylabel in [(ax_bb, r"$C_\ell^{BB} / \sigma$"), (ax_eb, r"$C_\ell^{EB} / \sigma$")]:
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_xscale("squareroot")
        ax.set_xlim(ell_min, ell_max)

        # Shade excluded regions (matching cl_fiducial)
        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], ELL_MIN_CUT, alpha=0.1, color="gray", zorder=0)
        ax.axvspan(ELL_MAX_CUT, xlim[1], alpha=0.1, color="gray", zorder=0)
        ax.set_xlim(xlim)  # Restore limits after shading

        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", length=2, width=0.8)
        ax.set_ylabel(ylabel)

    # Legend only on upper panel, single row with three columns, at bottom
    ax_bb.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.9,
    )

    ax_eb.set_xlabel(r"$\ell$")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)

    # Save outputs
    output_dir = Path(snakemake.output["evidence"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_path = Path(snakemake.output["figure"])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved {fig_path}")

    paper_path = Path(snakemake.output["paper_figure"])
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fig_path, paper_path)
    print(f"Copied to {paper_path}")

    plt.close(fig)

    # Build evidence
    evidence_versions = {}
    for data in datasets:
        evidence_versions[data["version"]] = {
            "pte_bb": float(data["pte_bb"]),
            "chi2_bb": float(data["chi2_bb"]),
            "dof_bb": int(data["dof_bb"]),
            "pte_eb": float(data["pte_eb"]),
            "chi2_eb": float(data["chi2_eb"]),
            "dof_eb": int(data["dof_eb"]),
        }

    spec_paths = snakemake.input["specs"]

    evidence_data = {
        "spec_id": "cl_version_comparison",
        "spec_path": spec_paths[0],
        "generated": datetime.now().isoformat(),
        "evidence": {
            "versions": evidence_versions,
            "ell_min": float(datasets[0]["ell"].min()),
            "ell_max": float(datasets[0]["ell"].max()),
            "n_ell_bins": int(len(datasets[0]["ell"])),
        },
        "artifacts": {
            "figure": fig_path.name,
        },
    }

    evidence_path = Path(snakemake.output["evidence"])
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


if __name__ == "__main__":
    main()
