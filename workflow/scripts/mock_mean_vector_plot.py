"""Mean B-mode data vector across GLASS mocks, per statistic.

Reads a campaign directory's per_mock_statistics.npz + evidence.json and
plots the across-realization mean of each B statistic in units of the
analytic (data) sigma, with both the mock-to-mock scatter and
sigma_emp/sqrt(N) error bars in the same units.
The y-axis therefore reads directly as "fraction of a data error bar":
a bias detectable at mock precision (error bars ~1/sqrt(N) ~ 0.05) can
still be negligible for the data (|mean| << 1).

Standalone:
    python mock_mean_vector_plot.py --campaign-dir <dir> [--label L]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    from plotting_utils import PAPER_MPLSTYLE  # type: ignore

    plt.style.use(PAPER_MPLSTYLE)
except Exception:  # pragma: no cover - style is cosmetic
    PAPER_MPLSTYLE = None


def _hartlap(n_real: int, p: int) -> float:
    return (n_real - p - 2) / (n_real - 1)


def _mean_chi2_pte(vectors: np.ndarray) -> tuple[float, float, int]:
    """Chi2/PTE of the mean against the empirical covariance of the mean."""
    from scipy import stats

    n_real, p = vectors.shape
    mean = vectors.mean(axis=0)
    cov_mean = np.cov(vectors, rowvar=False, ddof=1) / n_real
    inv = np.linalg.inv(cov_mean) * _hartlap(n_real, p)
    chi2 = float(mean @ inv @ mean)
    return chi2, float(stats.chi2.sf(chi2, p)), p


def _scale_averaged_stats(
    vectors: np.ndarray, sigma_ana: np.ndarray, bins: np.ndarray
) -> dict[str, object]:
    """Summarize per-mock scale averages and the largest mean bias."""
    normalized = vectors / sigma_ana
    results: dict[str, object] = {}
    for name, values in (("raw", vectors), ("normalized", normalized)):
        per_mock = values.mean(axis=1)
        mean = float(per_mock.mean())
        std = float(per_mock.std(ddof=1))
        stderr = std / np.sqrt(vectors.shape[0])
        results[name] = {
            "mean": mean,
            "std": std,
            "stderr": float(stderr),
            "z": float(mean / stderr),
        }

    mean_normalized = normalized.mean(axis=0)
    max_index = int(np.argmax(np.abs(mean_normalized)))
    results["max_abs_mean_normalized"] = float(abs(mean_normalized[max_index]))
    results["max_abs_mean_normalized_bin"] = bins[max_index].item()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    campaign = Path(args.campaign_dir)
    data = np.load(campaign / "per_mock_statistics.npz")
    with open(campaign / "evidence.json") as handle:
        evidence = json.load(handle)
    sig = evidence["sigma_ratios"]
    label = args.label or evidence.get("covariance_mode", {}).get(
        "label", campaign.name
    )

    n_real = data["Bn"].shape[0]

    panels = [
        (
            "Bn",
            np.arange(1, 21),
            np.asarray(sig["COSEBI_B_modes_1_to_20"]["analytic_sigma"]),
            "COSEBI mode $n$",
            r"$\langle B_n\rangle/\sigma_n^\mathrm{ana}$",
            None,
            [("$B_n$", "Bn", "o-")],
        ),
        (
            "pure",
            np.asarray(data["theta"]),
            None,
            r"$\theta$ [arcmin]",
            r"$\langle\xi^B\rangle/\sigma^\mathrm{ana}$",
            "log",
            [
                (r"$\xi_+^B$", "xip_B_full", "o-"),
                (r"$\xi_-^B$", "xim_B_full", "s-"),
            ],
        ),
        (
            "cl",
            np.asarray(data["ell_cut"]),
            np.asarray(sig["C_ell_BB_17_cut_bands"]["analytic_sigma"]),
            r"$\ell$",
            r"$\langle C_\ell^{BB}\rangle/\sigma_\ell^\mathrm{ana}$",
            None,
            [(r"$C_\ell^{BB}$", "cl_BB", "o-")],
        ),
    ]
    pure_sigma = {
        "xip_B_full": np.asarray(sig["xi_plus_B_20_bins"]["analytic_sigma"]),
        "xim_B_full": np.asarray(sig["xi_minus_B_20_bins"]["analytic_sigma"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
    all_stats: dict[str, dict[str, object]] = {}
    for ax, (name, x, sigma_ana, xlabel, ylabel, xscale, series) in zip(axes, panels):
        annotations = []
        for series_label, key, fmt in series:
            vectors = np.asarray(data[key])
            sig_ana = pure_sigma[key] if sigma_ana is None else sigma_ana
            mean = vectors.mean(axis=0) / sig_ana
            scatter = vectors.std(axis=0, ddof=1) / sig_ana
            err = scatter / np.sqrt(n_real)
            ax.errorbar(
                x,
                mean,
                yerr=scatter,
                fmt="none",
                ecolor="0.35",
                elinewidth=0.7,
                alpha=0.35,
            )
            ax.errorbar(
                x,
                mean,
                yerr=err,
                fmt=fmt,
                ms=3,
                lw=0.8,
                elinewidth=0.6,
                capsize=1.5,
                label=series_label,
            )
            chi2, pte, dof = _mean_chi2_pte(vectors)
            stat_name = {
                "Bn": "B_n",
                "xip_B_full": "xi_p_B",
                "xim_B_full": "xi_m_B",
                "cl_BB": "C_ell_BB",
            }[key]
            all_stats[stat_name] = {
                "n_bins": int(vectors.shape[1]),
                "bins": [value.item() for value in x],
                "scale_averaged": _scale_averaged_stats(vectors, sig_ana, x),
                "chi2": chi2,
                "pte": pte,
                "dof": dof,
            }
            annotations.append(
                rf"{series_label}: $\chi^2$={chi2:.1f}/{dof}, PTE={pte:.3g}"
            )
        ax.axhline(0.0, color="black", ls="--", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xscale:
            ax.set_xscale(xscale)
        legend_handles = [
            Line2D(
                [],
                [],
                color="0.35",
                lw=1.0,
                alpha=0.35,
                label=r"$\sigma_\mathrm{emp}/\sigma_\mathrm{ana}$",
            ),
            Line2D(
                [],
                [],
                color="0.35",
                marker="o",
                lw=1.0,
                markersize=3,
                label=r"$\sigma_\mathrm{emp}/(\sigma_\mathrm{ana}\sqrt{N})$",
            ),
        ]
        handles, _ = ax.get_legend_handles_labels()
        legend_handles = handles + legend_handles
        ax.legend(handles=legend_handles, fontsize=5.5)
        ax.text(
            0.03,
            0.03,
            "\n".join(annotations),
            transform=ax.transAxes,
            fontsize=5,
            va="bottom",
        )
        ax.grid(alpha=0.2)
    axes[0].set_xticks([1, 5, 10, 15, 20])
    fig.suptitle(
        rf"GLASS mock campaign ({label}): mean B vector in units of analytic $\sigma$"
        rf" (outer: $\sigma_\mathrm{{emp}}/\sigma_\mathrm{{ana}}$; "
        rf"inner: $\sigma_\mathrm{{emp}}/("
        rf"\sigma_\mathrm{{ana}}\sqrt{{N}})$, N={n_real})",
        fontsize=8,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(campaign / f"mean_vector.{ext}", dpi=250)
    print(f"Saved {campaign / 'mean_vector.png'}")

    stats_path = campaign / "mean_vector_stats.json"
    with open(stats_path, "w") as handle:
        json.dump(
            {"n_realizations": n_real, "statistics": all_stats},
            handle,
            indent=2,
        )
        handle.write("\n")

    for name, values in all_stats.items():
        chi2 = values["chi2"]
        pte = values["pte"]
        dof = values["dof"]
        scale = values["scale_averaged"]
        raw = scale["raw"]
        normalized = scale["normalized"]
        print(
            f"mean-vector {name}: "
            f"raw mean={raw['mean']:.6g}, std={raw['std']:.6g}, "
            f"stderr={raw['stderr']:.6g}, z={raw['z']:.6g}; "
            f"normalized mean={normalized['mean']:.6g}, "
            f"std={normalized['std']:.6g}, stderr={normalized['stderr']:.6g}, "
            f"z={normalized['z']:.6g}; "
            f"max |mean|/sigma_ana={scale['max_abs_mean_normalized']:.6g} "
            f"(bin {scale['max_abs_mean_normalized_bin']}); "
            f"chi2={chi2:.1f}/{dof} PTE={pte:.4g}"
        )


if __name__ == "__main__":
    main()
