"""Test whether the v1.4.8 star-mask footprint leaks pure B modes geometrically.

Referee comment 4 asks whether the masked catalogue's ξ± pure-mode B failures
are contamination or geometry.  GLASS lognormal mocks are systematics-free and
carry no B power by construction, so any pure-B signal they develop under a
footprint is purely an estimator/geometry effect (ambiguous-mode leakage from
the finite, more complex mask).

The same realizations are transformed under two footprints -- the fiducial one
and the v1.4.8 masked one -- and compared four ways:

1. Mean B vector against the empirical covariance of the mean (Hartlap on the
   inverse), per footprint.
2. Per-realization χ² against each footprint's own empirical covariance, with a
   KS test between the two χ² samples.
3. The paired per-realization difference (masked − fiducial), where cosmic
   variance cancels; this is the sharpest geometry test.
4. Per-realization χ² against the paper's MC-propagated analytic pure-EB
   covariance, on the full 20 bins and on the paper's [12, 83] arcmin cut.

Gather-only and standalone: it reads already-produced per-mock ``pure_eb``
npz files from two globs and writes ``figure.png`` + ``evidence.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    from plotting_utils import PAPER_MPLSTYLE, compute_chi2_pte
except ImportError:  # pragma: no cover - only used in unusually small images
    PAPER_MPLSTYLE = None

    def compute_chi2_pte(data, covariance, n_samples=None):
        chi2 = float(data @ np.linalg.solve(covariance, data))
        dof = len(data)
        if n_samples is not None:
            chi2 *= (n_samples - dof - 2) / (n_samples - 1)
        return chi2, float(stats.chi2.sf(chi2, dof)), dof


try:
    if PAPER_MPLSTYLE:
        plt.style.use(PAPER_MPLSTYLE)
except OSError as exc:  # pragma: no cover - depends on checkout relocation
    print(f"Could not load paper matplotlib style ({exc}); using matplotlib defaults")


# Block order of the stored 120x120 pure-EB covariance, matching the six arrays
# written by mock_pure_eb_scatter.py: E, B, then ambiguous, each (+, -).
BLOCK_ORDER = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")
PRIMARY_STATS = ("xip_B", "xim_B")
SECONDARY_STATS = ("xip_amb", "xim_amb")
STAT_TEX = {
    "xip_B": r"$\xi_+^{B}$",
    "xim_B": r"$\xi_-^{B}$",
    "xip_amb": r"$\xi_+^{\rm amb}$",
    "xim_amb": r"$\xi_-^{\rm amb}$",
}
FOOTPRINTS = ("fiducial", "masked")
FOOTPRINT_TEX = {"fiducial": "fiducial", "masked": "v1.4.8 mask"}
FOOTPRINT_COLOR = {"fiducial": "C0", "masked": "C3"}
# Paper pure-EB scale cut, kept only for the analytic-covariance diagnostic.
PURE_THETA_MIN = 12.0
PURE_THETA_MAX = 83.0
HARTLAP_N_PURE = 2000
_ID_RE = re.compile(r"(?:glass_mock_|mock_)(\d{5})")


def _native(values):
    return np.asarray(values, dtype=float)


def _id_from_path(path: str | Path) -> str:
    match = _ID_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Could not recover a five-digit mock ID from {path}")
    return match.group(1)


def _load_footprint(pattern: str):
    """Return {mock_id: {key: array}} plus the shared theta grid for one glob."""

    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No pure-EB files matched {pattern!r}")

    vectors = {}
    theta = None
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in ("theta", *BLOCK_ORDER) if key not in data]
            if missing:
                raise KeyError(
                    f"{path} is missing pure-EB keys {missing}; keys={data.files}"
                )
            entry = {key: _native(data[key]) for key in BLOCK_ORDER}
            this_theta = _native(data["theta"])
        if this_theta.shape != (20,):
            raise ValueError(f"Expected 20 pure-EB reporting bins in {path}")
        if not all(np.all(np.isfinite(value)) for value in entry.values()):
            raise ValueError(f"Non-finite pure-EB vector in {path}")
        if theta is None:
            theta = this_theta
        elif not np.allclose(
            theta, this_theta, rtol=1e-3
        ):  # treecorr mean-theta jitters ~1e-5 per realization
            raise ValueError(f"{path} reports a different theta grid than its peers")
        vectors[_id_from_path(path)] = entry
    return vectors, theta


def _load_analytic_covariance(path: str | Path, theta: np.ndarray):
    """Slice the stored six-block covariance into per-statistic blocks."""

    with np.load(path, allow_pickle=False) as data:
        if "cov_pure_eb" not in data:
            raise KeyError(f"{path} has no cov_pure_eb key; keys={data.files}")
        covariance = _native(data["cov_pure_eb"])
    nbins = len(theta)
    if covariance.shape != (6 * nbins, 6 * nbins):
        raise ValueError(
            f"Expected a {6 * nbins}x{6 * nbins} pure-EB covariance, "
            f"found {covariance.shape}"
        )
    blocks = {}
    for index, key in enumerate(BLOCK_ORDER):
        start = index * nbins
        blocks[key] = covariance[start : start + nbins, start : start + nbins]
    return blocks


def _stack(vectors, ids, key):
    return np.array([vectors[mock_id][key] for mock_id in ids], dtype=float)


def _mean_statistics(sample):
    """Mean-vector test against the empirical covariance of the mean."""

    n_realizations, nbins = sample.shape
    mean = sample.mean(axis=0)
    std = sample.std(axis=0, ddof=1)
    standard_error = std / np.sqrt(n_realizations)
    covariance_mean = np.cov(sample, rowvar=False, ddof=1) / n_realizations
    chi2, pte, dof = compute_chi2_pte(mean, covariance_mean, n_samples=n_realizations)
    z_scores = mean / standard_error
    return {
        "mean": mean,
        "std": std,
        "standard_error": standard_error,
        "z": z_scores,
        "max_abs_z": float(np.max(np.abs(z_scores))),
        "chi2": float(chi2),
        "pte": float(pte),
        "dof": int(dof),
        "nbins": nbins,
    }


def _per_realization_chi2(sample, covariance, *, hartlap_n=None):
    """χ² of every realization against a single fixed covariance."""

    inverse = np.linalg.inv(covariance)
    dof = sample.shape[1]
    if hartlap_n is not None:
        inverse *= (hartlap_n - dof - 2) / (hartlap_n - 1)
    chi2 = np.einsum("ij,jk,ik->i", sample, inverse, sample)
    return chi2, stats.chi2.sf(chi2, dof), dof


def _per_realization_summary(values):
    """Return uncertainty and location summaries for per-realization values."""

    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sem": float(values.std(ddof=1) / np.sqrt(values.size)),
        "median": float(np.median(values)),
    }


def _paired_chi2_summary(masked, fiducial):
    """Summarize paired masked-minus-fiducial per-realization χ² values."""

    delta = masked - fiducial
    return {
        "delta_mean": float(delta.mean()),
        "delta_sem": float(delta.std(ddof=1) / np.sqrt(delta.size)),
        "fractional_shift_mean": float(delta.mean() / fiducial.mean()),
        "pearson_r": float(stats.pearsonr(masked, fiducial).statistic),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _plot(output_dir: Path, theta, results, analytic_chi2, label):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))

    for axis, keys in zip(axes[:2], (PRIMARY_STATS, SECONDARY_STATS)):
        axis.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        for key in keys:
            for footprint in FOOTPRINTS:
                summary = results[footprint][key]["mean_test"]
                # Offset the two footprints slightly so error bars stay legible.
                shift = 1.03 if footprint == "masked" else 0.97
                axis.errorbar(
                    theta * shift,
                    theta * summary["mean"],
                    yerr=theta * summary["standard_error"],
                    fmt="o" if key.startswith("xip") else "s",
                    ms=3,
                    lw=0.8,
                    capsize=2,
                    color=FOOTPRINT_COLOR[footprint],
                    mfc="none" if key.startswith("xim") else None,
                    label=f"{STAT_TEX[key]} {FOOTPRINT_TEX[footprint]}",
                )
        axis.set_xscale("log")
        axis.set_xlabel(r"$\theta$ [arcmin]")
        axis.set_ylabel(r"$\theta \times$ mean vector [arcmin]")
        axis.legend(fontsize=6)

    axes[0].set_title("Pure B modes", fontsize=8)
    axes[1].set_title("Ambiguous modes", fontsize=8)

    axis = axes[2]
    bins = np.histogram_bin_edges(
        np.concatenate([analytic_chi2[footprint] for footprint in FOOTPRINTS]),
        bins=25,
    )
    for footprint in FOOTPRINTS:
        axis.hist(
            analytic_chi2[footprint],
            bins=bins,
            histtype="step",
            lw=1.2,
            color=FOOTPRINT_COLOR[footprint],
            label=FOOTPRINT_TEX[footprint],
        )
    axis.set_xlabel(r"$\chi^2$ ($\xi_+^{B}$, analytic cov, 20 bins)")
    axis.set_ylabel("realizations")
    axis.set_title("Per-realization $\\chi^2$", fontsize=8)
    axis.legend(fontsize=6)

    fig.suptitle(f"Mask-geometry pure-B test: {label}", fontsize=9, y=1.02)
    fig.tight_layout()
    figure_path = output_dir / "figure.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {figure_path}", flush=True)
    return figure_path


def run_analysis(fiducial_glob, masked_glob, pure_cov, output_dir, label):
    fiducial_vectors, theta = _load_footprint(fiducial_glob)
    masked_vectors, theta_masked = _load_footprint(masked_glob)
    if not np.allclose(
        theta, theta_masked, rtol=1e-3
    ):  # mean-theta shifts slightly under the mask
        raise ValueError("The two footprints report different theta grids")

    ids = sorted(set(fiducial_vectors) & set(masked_vectors))
    if len(ids) < 3:
        raise ValueError(f"Only {len(ids)} matched realizations; nothing to test")
    n_realizations = len(ids)
    print(
        f"Matched {n_realizations} realizations "
        f"({len(fiducial_vectors)} fiducial, {len(masked_vectors)} masked)",
        f"({len(fiducial_vectors)} fiducial, {len(masked_vectors)} masked)",
        flush=True,
    )

    analytic_blocks = _load_analytic_covariance(pure_cov, theta)
    cut_mask = (theta >= PURE_THETA_MIN) & (theta <= PURE_THETA_MAX)

    samples = {
        "fiducial": {key: _stack(fiducial_vectors, ids, key) for key in BLOCK_ORDER},
        "masked": {key: _stack(masked_vectors, ids, key) for key in BLOCK_ORDER},
    }

    results = {footprint: {} for footprint in FOOTPRINTS}
    paired = {}
    analytic_chi2_xip = {}
    per_mock = {"mock_ids": np.asarray(ids)}

    for key in (*PRIMARY_STATS, *SECONDARY_STATS):
        for footprint in FOOTPRINTS:
            sample = samples[footprint][key]
            mean_test = _mean_statistics(sample)

            # Full-sample covariance is reused for every realization's χ²; the
            # vector under test is therefore part of its own covariance, which
            # biases the χ² slightly low.  Since both footprints are treated
            # identically, the comparison between them is unaffected.
            empirical = np.cov(sample, rowvar=False, ddof=1)
            chi2_empirical, _, dof_empirical = _per_realization_chi2(
                sample, empirical, hartlap_n=n_realizations
            )

            analytic_full = analytic_blocks[key]
            chi2_analytic, pte_analytic, dof_analytic = _per_realization_chi2(
                sample, analytic_full, hartlap_n=HARTLAP_N_PURE
            )
            cut = np.ix_(cut_mask, cut_mask)
            chi2_cut, pte_cut, dof_cut = _per_realization_chi2(
                sample[:, cut_mask], analytic_full[cut], hartlap_n=HARTLAP_N_PURE
            )
            empirical_summary = _per_realization_summary(chi2_empirical)
            analytic_summary = _per_realization_summary(chi2_analytic)
            pte_summary = _per_realization_summary(pte_analytic)
            cut_summary = _per_realization_summary(chi2_cut)
            pte_cut_summary = _per_realization_summary(pte_cut)

            results[footprint][key] = {
                "mean_test": mean_test,
                "chi2_empirical_mean": empirical_summary["mean"],
                "chi2_empirical_sem": empirical_summary["sem"],
                "chi2_empirical_median": empirical_summary["median"],
                "chi2_empirical_dof": int(dof_empirical),
                "chi2_analytic_mean": analytic_summary["mean"],
                "chi2_analytic_sem": analytic_summary["sem"],
                "chi2_analytic_median": analytic_summary["median"],
                "chi2_analytic_dof": int(dof_analytic),
                "pte_analytic_mean": pte_summary["mean"],
                "pte_analytic_sem": pte_summary["sem"],
                "pte_analytic_median": pte_summary["median"],
                "chi2_analytic_cut_mean": cut_summary["mean"],
                "chi2_analytic_cut_sem": cut_summary["sem"],
                "chi2_analytic_cut_median": cut_summary["median"],
                "chi2_analytic_cut_dof": int(dof_cut),
                "pte_analytic_cut_mean": pte_cut_summary["mean"],
                "pte_analytic_cut_sem": pte_cut_summary["sem"],
                "pte_analytic_cut_median": pte_cut_summary["median"],
                "_chi2_empirical": chi2_empirical,
            }
            per_mock[f"{footprint}_{key}_chi2"] = chi2_analytic
            per_mock[f"{footprint}_{key}_pte"] = pte_analytic
            per_mock[f"{footprint}_{key}_chi2_cut"] = chi2_cut
            per_mock[f"{footprint}_{key}_pte_cut"] = pte_cut
            if key == "xip_B":
                analytic_chi2_xip[footprint] = chi2_analytic

        ks = stats.ks_2samp(
            results["fiducial"][key]["_chi2_empirical"],
            results["masked"][key]["_chi2_empirical"],
        )

        # Cosmic variance cancels in the pair, so the difference isolates the
        # footprint change acting on identical underlying shear fields.
        difference = samples["masked"][key] - samples["fiducial"][key]
        difference_test = _mean_statistics(difference)
        paired[key] = {
            "difference_test": difference_test,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "paired_chi2": {
                "scale_cut": _paired_chi2_summary(
                    per_mock[f"masked_{key}_chi2_cut"],
                    per_mock[f"fiducial_{key}_chi2_cut"],
                ),
                "full_range": _paired_chi2_summary(
                    per_mock[f"masked_{key}_chi2"],
                    per_mock[f"fiducial_{key}_chi2"],
                ),
            },
        }

        for footprint in FOOTPRINTS:
            results[footprint][key].pop("_chi2_empirical")

        print(
            f"{key}: "
            f"fiducial χ²(mean)={results['fiducial'][key]['mean_test']['chi2']:.1f} "
            f"PTE={results['fiducial'][key]['mean_test']['pte']:.3f} | "
            f"masked χ²(mean)={results['masked'][key]['mean_test']['chi2']:.1f} "
            f"PTE={results['masked'][key]['mean_test']['pte']:.3f} | "
            f"paired χ²={difference_test['chi2']:.1f} "
            f"PTE={difference_test['pte']:.3f} | "
            f"KS p={ks.pvalue:.3g}",
            flush=True,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_mock_path = output_dir / "per_mock_chi2.npz"
    np.savez_compressed(per_mock_path, **per_mock)
    print(f"Saved {per_mock_path}", flush=True)
    figure_path = _plot(output_dir, theta, results, analytic_chi2_xip, label)

    evidence = {
        "spec": "mock_mask_geometry_analysis",
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "n_realizations": n_realizations,
        "mock_ids": ids,
        "theta": theta,
        "scale_cut": [PURE_THETA_MIN, PURE_THETA_MAX],
        "hartlap_n_analytic": HARTLAP_N_PURE,
        "inputs": {
            "fiducial_glob": fiducial_glob,
            "masked_glob": masked_glob,
            "pure_cov": str(pure_cov),
        },
        "footprints": results,
        "paired": paired,
        "figure": str(figure_path),
    }
    evidence_path = output_dir / "evidence.json"
    with open(evidence_path, "w") as stream:
        json.dump(_json_ready(evidence), stream, indent=2)
    print(f"Saved {evidence_path}", flush=True)
    return evidence


def _parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fiducial-glob",
        required=True,
        help="Glob for fiducial-footprint pure_eb npz files (quote it).",
    )
    parser.add_argument(
        "--masked-glob",
        required=True,
        help="Glob for v1.4.8-masked-footprint pure_eb npz files (quote it).",
    )
    parser.add_argument(
        "--pure-cov",
        required=True,
        help="Paper MC-propagated analytic pure-EB covariance npz (cov_pure_eb).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="mask-geometry")
    return parser.parse_args(argv)


def _standalone_main(argv=None):
    args = _parser(argv)
    run_analysis(
        fiducial_glob=args.fiducial_glob,
        masked_glob=args.masked_glob,
        pure_cov=args.pure_cov,
        output_dir=args.output_dir,
        label=args.label,
    )


if __name__ == "__main__":
    _standalone_main()
