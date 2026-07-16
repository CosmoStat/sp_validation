"""Harmonic-space fiducial B-mode power spectrum.

Claim: Each catalog version shows C_ell^BB and C_ell^EB consistent with zero.

Produces 9 figures:
- figure.png: fiducial version, leak-corrected, no title (paper)
- figure_v{X.Y.Z}.png: version X.Y.Z, leak-corrected, with title
- figure_v{X.Y.Z}_uncorrected.png: version X.Y.Z, uncorrected, with title
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

# Import shared utilities (also registers SquareRootScale)
from plotting_utils import (
    FIG_WIDTH_SINGLE,
    PAPER_MPLSTYLE,
    compute_chi2_pte,
    ell_bin_mask,
    get_powspace_bin_edges,
    iter_version_figures,
)

plt.style.use(PAPER_MPLSTYLE)


def _compute_pte_with_cuts(data, covariance, ell, ell_min, ell_max):
    """Compute PTE for B-mode null test with scale cuts.

    A bin is included only if its full multipole range falls within
    [ell_min, ell_max] (lower edge >= ell_min AND upper edge <= ell_max).
    """
    mask = ell_bin_mask(ell, ell_min, ell_max)
    data_cut = data[mask]
    cov_cut = covariance[np.ix_(mask, mask)]
    chi2, pte, dof = compute_chi2_pte(data_cut, cov_cut)
    return pte, chi2, dof


def _load_pseudo_cl_data(pseudo_cl_path, pseudo_cl_cov_path):
    """Load pseudo-Cl data and covariance from FITS files."""
    hdu = fits.open(pseudo_cl_path)
    data = hdu["PSEUDO_CELL"].data
    hdu.close()

    ell = data["ELL"]
    cl_eb = data["EB"]
    cl_bb = data["BB"]

    hdu_cov = fits.open(pseudo_cl_cov_path)
    cov_eb = hdu_cov["COVAR_EB_EB"].data
    cov_bb = hdu_cov["COVAR_BB_BB"].data
    hdu_cov.close()

    sigma_eb = np.sqrt(np.diag(cov_eb))
    sigma_bb = np.sqrt(np.diag(cov_bb))

    return ell, cl_bb, cl_eb, cov_bb, cov_eb, sigma_bb, sigma_eb


def _create_cl_figure(
    ell, cl_bb, cl_eb, sigma_bb, sigma_eb, ell_min_cut, ell_max_cut, title=None
):
    """Create two-panel Cl figure.

    Args:
        ell_min_cut: Lower scale cut for shading excluded region
        ell_max_cut: Upper scale cut for shading excluded region
        title: Optional title for the figure (None for paper figure)
    """
    fig, (ax_bb, ax_eb) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH_SINGLE, FIG_WIDTH_SINGLE * 0.75), sharex=True
    )

    color_bb = "#2c5f8a"  # dark blue (distinct harmonic-space scheme)
    color_eb = "#c45a2c"  # burnt orange

    minor_ticks = [i * 10 for i in range(1, 10)] + [i * 100 for i in range(1, 21)]

    # BB panel
    bb_label = r"$C_\ell^{BB}$"
    if title:
        bb_label = rf"$C_\ell^{{BB}}$ ({title})"
    ax_bb.errorbar(
        ell,
        cl_bb / sigma_bb,
        yerr=np.ones_like(cl_bb),
        fmt="o",
        mfc=color_bb,
        mec=color_bb,
        color=color_bb,
        capsize=2,
        markersize=4,
        linewidth=1.0,
        label=bb_label,
    )
    ax_bb.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax_bb.set_xscale("squareroot")
    ax_bb.set_ylabel(r"$C_\ell / \sigma$")
    ax_bb.legend(loc="upper left", framealpha=0.9)

    # EB panel
    eb_label = r"$C_\ell^{EB}$"
    if title:
        eb_label = rf"$C_\ell^{{EB}}$ ({title})"
    ax_eb.errorbar(
        ell,
        cl_eb / sigma_eb,
        yerr=np.ones_like(cl_eb),
        fmt="s",
        mfc="none",
        mec=color_eb,
        color=color_eb,
        capsize=2,
        markersize=4,
        linewidth=1.0,
        label=eb_label,
    )
    ax_eb.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax_eb.set_xscale("squareroot")
    ax_eb.set_xlabel(r"$\ell$")
    ax_eb.set_ylabel(r"$C_\ell / \sigma$")
    ax_eb.legend(loc="upper left", framealpha=0.9)

    # Match y-axis range between panels (use BB range)
    ax_eb.set_ylim(ax_bb.get_ylim())

    # Shade excluded bins using actual bin edges
    ell_low, ell_high = get_powspace_bin_edges(ell)
    mask = ell_bin_mask(ell, ell_min_cut, ell_max_cut)
    included = np.where(mask)[0]
    shade_low = ell_low[included[0]]  # lower edge of first included bin
    shade_high = ell_high[included[-1]]  # upper edge of last included bin

    for ax in [ax_bb, ax_eb]:
        ell_min_data = ell.min()
        ell_max_data = ell.max()
        ax.set_xlim(ell_min_data * 0.95, ell_max_data * 1.05)

        xlim = ax.get_xlim()
        ax.axvspan(xlim[0], shade_low, alpha=0.1, color="gray", zorder=0)
        ax.axvspan(shade_high, xlim[1], alpha=0.1, color="gray", zorder=0)
        ax.set_xlim(xlim)

        ax.set_xticks(np.array([100, 400, 900, 1600]))
        ax.minorticks_on()
        ax.tick_params(axis="x", which="minor", length=2, width=0.8)
        ax.set_xticks(minor_ticks, minor=True)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Canonical per-version file paths (COSMO_VAL results tree). The DAG read these
# via _pseudo_cl_path() in claims.smk; the CLI reconstructs them from --results-dir
# so the version sweep is self-contained (lc produces only the fiducial version).
# ---------------------------------------------------------------------------
def _pseudo_cl(results_dir, ver, blind="A", nbins=32):
    return f"{results_dir}/pseudo_cl_{ver}_blind={blind}_powspace_nbins={nbins}.fits"


def _pseudo_cl_cov(results_dir, ver, blind="A", nbins=32):
    return (
        f"{results_dir}/pseudo_cl_cov_{ver}_blind={blind}_powspace_nbins={nbins}.fits"
    )


def _resolve_pseudo_cl_paths(
    results_dir,
    ver,
    fiducial_version=None,
    fiducial_pseudo_cl_path=None,
    fiducial_pseudo_cl_cov_path=None,
    blind="A",
    nbins=32,
):
    """Resolve the pseudo-Cl and covariance paths for one version in the sweep.

    Every version is read from the reconstructed COSMO_VAL pattern, except the
    fiducial version when explicit override paths are supplied: the lc-produced
    fiducial FITS files are named `pseudo_cl_{version}.fits` /
    `pseudo_cl_cov_{version}.fits` (no blind=/nbins= tokens), so pattern
    reconstruction can't find them and an explicit path is required instead.
    """
    if (
        ver == fiducial_version
        and fiducial_pseudo_cl_path
        and fiducial_pseudo_cl_cov_path
    ):
        return fiducial_pseudo_cl_path, fiducial_pseudo_cl_cov_path
    return (
        _pseudo_cl(results_dir, ver, blind, nbins),
        _pseudo_cl_cov(results_dir, ver, blind, nbins),
    )


def main(
    config,
    results_dir,
    out_dir,
    fiducial_version=None,
    fiducial_pseudo_cl_path=None,
    fiducial_pseudo_cl_cov_path=None,
):
    version = config["fiducial"]["version"]
    version_labels = config["plotting"]["version_labels"]
    ell_min_cut = int(config["cl"]["fiducial_ell_min"])
    ell_max_cut = int(config["cl"]["fiducial_ell_max"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Track generated artifacts
    output = {}

    # Generate all 9 figures
    for fig_spec in iter_version_figures(version_labels, version):
        ver = (
            fig_spec["version_leak_corr"]
            if fig_spec["leak_corrected"]
            else fig_spec["version_uncorr"]
        )

        # Load data
        pseudo_cl_path, pseudo_cl_cov_path = _resolve_pseudo_cl_paths(
            results_dir,
            ver,
            fiducial_version,
            fiducial_pseudo_cl_path,
            fiducial_pseudo_cl_cov_path,
        )
        ell, cl_bb, cl_eb, cov_bb, cov_eb, sigma_bb, sigma_eb = _load_pseudo_cl_data(
            pseudo_cl_path, pseudo_cl_cov_path
        )

        # Create figure with appropriate title
        fig = _create_cl_figure(
            ell,
            cl_bb,
            cl_eb,
            sigma_bb,
            sigma_eb,
            ell_min_cut,
            ell_max_cut,
            title=fig_spec["title"],
        )

        # Save figure
        fig_path = out_dir / fig_spec["filename"]
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        print(f"Saved {fig_path}")
        plt.close(fig)

        # Track artifact
        output[fig_spec["filename"].replace(".png", "")] = fig_spec["filename"]

        # Paper figure into the lc output directory
        if fig_spec["is_paper_figure"]:
            paper_path = out_dir / "cl_data_vector.pdf"
            fig.savefig(paper_path, bbox_inches="tight")
            print(f"Saved {paper_path}")

    # Compute PTEs for evidence (fiducial version, leak-corrected only)
    pseudo_cl_path, pseudo_cl_cov_path = _resolve_pseudo_cl_paths(
        results_dir,
        version,
        fiducial_version,
        fiducial_pseudo_cl_path,
        fiducial_pseudo_cl_cov_path,
    )
    ell, cl_bb, cl_eb, cov_bb, cov_eb, sigma_bb, sigma_eb = _load_pseudo_cl_data(
        pseudo_cl_path, pseudo_cl_cov_path
    )

    # Compute PTEs (null tests) using full ell range
    chi2_eb_full, pte_eb_full, dof_eb_full = compute_chi2_pte(cl_eb, cov_eb)
    chi2_bb_full, pte_bb_full, dof_bb_full = compute_chi2_pte(cl_bb, cov_bb)

    # Compute PTEs with scale cuts
    pte_eb_cut, chi2_eb_cut, dof_eb_cut = _compute_pte_with_cuts(
        cl_eb, cov_eb, ell, ell_min_cut, ell_max_cut
    )
    pte_bb_cut, chi2_bb_cut, dof_bb_cut = _compute_pte_with_cuts(
        cl_bb, cov_bb, ell, ell_min_cut, ell_max_cut
    )

    evidence_data = {
        "spec_id": "cl_data_vector",
        "generated": datetime.now().isoformat(),
        "evidence": {
            # Full range PTEs
            "pte_eb_full": float(pte_eb_full),
            "chi2_eb_full": float(chi2_eb_full),
            "dof_eb_full": int(dof_eb_full),
            "pte_bb_full": float(pte_bb_full),
            "chi2_bb_full": float(chi2_bb_full),
            "dof_bb_full": int(dof_bb_full),
            # Scale cut PTEs
            "pte_eb_cut": float(pte_eb_cut),
            "chi2_eb_cut": float(chi2_eb_cut),
            "dof_eb_cut": int(dof_eb_cut),
            "pte_bb_cut": float(pte_bb_cut),
            "chi2_bb_cut": float(chi2_bb_cut),
            "dof_bb_cut": int(dof_bb_cut),
            # Version
            "version": version,
        },
        "output": output,
    }

    evidence_path = out_dir / "evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(evidence_data, f, indent=2)
    print(f"Saved evidence to {evidence_path}")


def _from_cli(argv=None):
    import yaml

    ap = argparse.ArgumentParser(
        description="Harmonic-space Cl^BB/Cl^EB data vector figures (paper Fig 4 + 9-figure set)."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    ap.add_argument(
        "--results-dir",
        required=True,
        help="COSMO_VAL output dir with per-version pseudo_cl_* / pseudo_cl_cov_* FITS",
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    ap.add_argument(
        "--fiducial-version",
        default=None,
        help=(
            "Version string whose pseudo-Cl inputs should be read from explicit "
            "override paths instead of the reconstructed COSMO_VAL pattern "
            "(e.g. SP_v1.4.6.3_leak_corr)."
        ),
    )
    ap.add_argument(
        "--fiducial-pseudo-cl-path",
        default=None,
        help=(
            "Explicit path to the fiducial version's pseudo-Cl FITS (lc output). "
            "Requires --fiducial-version and --fiducial-pseudo-cl-cov-path."
        ),
    )
    ap.add_argument(
        "--fiducial-pseudo-cl-cov-path",
        default=None,
        help=(
            "Explicit path to the fiducial version's pseudo-Cl covariance FITS "
            "(lc output). Requires --fiducial-version and --fiducial-pseudo-cl-path."
        ),
    )
    a = ap.parse_args(argv)
    with open(a.config) as f:
        config = yaml.safe_load(f)
    main(
        config,
        a.results_dir,
        a.out,
        fiducial_version=a.fiducial_version,
        fiducial_pseudo_cl_path=a.fiducial_pseudo_cl_path,
        fiducial_pseudo_cl_cov_path=a.fiducial_pseudo_cl_cov_path,
    )


if __name__ == "__main__":
    _from_cli()
