"""Analyse the six Paper-II B-mode statistics across GLASS mocks.

This is a gather-only analysis.  It reads already-produced per-mock products,
filters to the complete subset that is available at invocation time, compares
the vectors with the paper's stored analytic covariances, and writes both
machine-readable evidence and compact paper-style figures.

The three covariance conventions are intentionally kept separate:

* COSEBIs: the paper 0.5--300 arcmin, 1000-bin covariance is transformed with
  the same ``COSEBIS.cosebis_covariance_from_xipm_covariance`` call used by the
  paper scripts; no Hartlap factor is applied.
* Pure ξ E/B: the stored 120 x 120 six-block covariance is sliced to the
  [ξ+^B, ξ-^B] blocks and the 12--83 arcmin cuts; Hartlap N=2000 is applied.
* C_l^BB: the stored 32 x 32 covariance is selected with the same powspace
  bin-edge logic as the paper scripts; no Hartlap factor is applied.  An
  optional ``--cl-cov-scale`` rescales this stored BB covariance for the
  mock's lower effective number density.

The script is dual-mode: Snakemake supplies directories and output paths via
``snakemake``; standalone execution discovers IDs from a glob or from the
configured 1--350 candidate set.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import treecorr
from astropy.io import fits
from cosmo_numba.B_modes.cosebis import COSEBIS
from scipy import stats

from sp_validation.b_modes import scale_cut_to_bins

try:
    from plotting_utils import (
        PAPER_MPLSTYLE,
        compute_chi2_pte,
        ell_bin_mask,
        get_powspace_bin_edges,
    )
except ImportError:  # pragma: no cover - only used in unusually small images
    PAPER_MPLSTYLE = None

    def compute_chi2_pte(data, covariance, n_samples=None):
        chi2 = float(data @ np.linalg.solve(covariance, data))
        dof = len(data)
        if n_samples is not None:
            chi2 *= (n_samples - dof - 2) / (n_samples - 1)
        return chi2, float(stats.chi2.sf(chi2, dof)), dof

    def get_powspace_bin_edges(ell_eff, lmin=8, lmax=2048, power=0.5):
        n_ell_bins = len(ell_eff)
        ells = np.arange(lmin, lmax + 1)
        bins_ell = np.power(
            np.linspace(lmin**power, lmax**power, n_ell_bins + 1), 1 / power
        )
        bpws = np.digitize(ells.astype(float), bins_ell) - 1
        bpws[0], bpws[-1] = 0, n_ell_bins - 1
        return (
            np.array([ells[bpws == i][0] for i in range(n_ell_bins)]),
            np.array([ells[bpws == i][-1] for i in range(n_ell_bins)]),
        )

    def ell_bin_mask(ell_eff, ell_min, ell_max, **kwargs):
        low, high = get_powspace_bin_edges(ell_eff, **kwargs)
        return (low >= ell_min) & (high <= ell_max)


try:
    if PAPER_MPLSTYLE:
        plt.style.use(PAPER_MPLSTYLE)
except OSError as exc:  # pragma: no cover - depends on checkout relocation
    print(f"Could not load paper matplotlib style ({exc}); using matplotlib defaults")


MOCK_IDS = [f"{i:05d}" for i in range(1, 351)]
STAT_NAMES = (
    "B_n<=6",
    "B_n<=20",
    "xi+^B",
    "xi-^B",
    "xi_tot^B",
    "C_ell^BB",
)
COSEBI_THETA_MIN = 250.0 ** (9.0 / 20.0)
COSEBI_THETA_MAX = 250.0 ** (16.0 / 20.0)
PURE_THETA_MIN = 12.0
PURE_THETA_MAX = 83.0
ELL_MIN = 300.0
ELL_MAX = 1600.0
HARTLAP_N_PURE = 2000
FIDUCIAL_NEFF = 4.957279270321334
MOCK_NEFF = 3.369700396483938
MOCKMATCHED_CL_COV_SCALE = (FIDUCIAL_NEFF / MOCK_NEFF) ** 2
STAT_TEX = {
    "B_n<=6": r"$B_{n\leq 6}$",
    "B_n<=20": r"$B_{n\leq 20}$",
    "xi+^B": r"$\xi_+^{B}$",
    "xi-^B": r"$\xi_-^{B}$",
    "xi_tot^B": r"$\xi_{\rm tot}^{B}$",
    "C_ell^BB": r"$C_\ell^{BB}$",
}
_ID_RE = re.compile(r"(?:glass_mock_|mock_)(\d{5})")


def _native(values):
    return np.asarray(values, dtype=float)


def _id_from_path(path: str | Path) -> str:
    match = _ID_RE.search(Path(path).name)
    if match is None:
        raise ValueError(f"Could not recover a five-digit mock ID from {path}")
    return match.group(1)


def _normalise_ids(ids):
    return sorted({str(mock_id).zfill(5) for mock_id in ids})


def _default_paths():
    repo = "/automnt/n17data/cdaley/unions/code/sp_validation"
    analysis = "results/glass_mock"
    cov_base = f"{repo}/cosmo_inference/data/covariance"
    cosebis_cov_base = (
        "covariance_SP_v1.4.6.3_leak_corr_A_g_minsep=0.5_maxsep=300.0_nbins=1000_masked"
    )
    return {
        "cosebis_dir": analysis,
        "pure_dir": analysis,
        "fine_dir": analysis,
        "cl_dir": "/n09data/guerrini/glass_mock_v1.4.6/results",
        "pseudo_cl_dir": analysis,
        "cosebis_xi_grid": (
            "/n17data/cdaley/unions/code/sp_validation/cosmo_val/output/"
            "SP_v1.4.6.3_leak_corr_xi_minsep=0.5_maxsep=300.0_"
            "nbins=1000_npatch=1.txt"
        ),
        "cosebis_cov": f"{cov_base}/{cosebis_cov_base}/{cosebis_cov_base}_processed.txt",
        "pure_cov": (
            "/automnt/n17data/cdaley/unions/analyses/shear_2d/bmodes_2d/"
            "results/paper_plots/intermediate/"
            "SP_v1.4.6.3_leak_corr_A_pure_eb_semianalytic.npz"
        ),
        "cl_cov": (
            "/n17data/cdaley/unions/code/sp_validation/cosmo_val/output/"
            "pseudo_cl_cov_SP_v1.4.6.3_leak_corr_blind=A_powspace_nbins=32.fits"
        ),
        "output_dir": "results/glass_mock/campaign",
    }


def _path_for(kind: str, directory: str | Path, mock_id: str) -> Path:
    directory = Path(directory)
    patterns = {
        "cosebis": f"cosebis_glass_mock_{mock_id}.npz",
        "pure": f"pure_eb_glass_mock_{mock_id}.npz",
        "fine": f"gg_glass_mock_{mock_id}_nbins=1000.fits",
        "cl": f"cl_glass_mock_{mock_id}_4096.npy",
        "pseudo_cl": f"pseudo_cl_glass_mock_{mock_id}_powspace_nbins=32.fits",
    }
    return directory / patterns[kind]


def _discover_ids(
    mock_ids=None,
    mock_glob=None,
    *,
    cosebis_dir="results/glass_mock",
):
    if mock_ids:
        return _normalise_ids(mock_ids)
    if mock_glob:
        return _normalise_ids(_id_from_path(path) for path in glob.glob(mock_glob))
    # Keep the candidate list explicit: the file checks below make a gather
    # invocation safe while the 350-mock campaign is still being produced.
    return _normalise_ids(MOCK_IDS)


def _load_cosebis(path: Path):
    with np.load(path, allow_pickle=False) as data:
        if "Bn" not in data:
            raise KeyError(f"{path} has no Bn key; keys={data.files}")
        bn = _native(data["Bn"])
        en = _native(data["En"]) if "En" in data else None
        theta_min = (
            float(np.asarray(data["theta_min_actual"]).squeeze())
            if "theta_min_actual" in data
            else None
        )
        theta_max = (
            float(np.asarray(data["theta_max_actual"]).squeeze())
            if "theta_max_actual" in data
            else None
        )
    if bn.shape != (20,):
        raise ValueError(f"Expected 20 COSEBI B modes in {path}, found {bn.shape}")
    return {
        "Bn": bn,
        "En": en,
        "stored_theta_min": theta_min,
        "stored_theta_max": theta_max,
        "source": str(path),
    }


def _stored_cosebis_is_fiducial(cosebis):
    """Allow bin-centre rounding, but reject the current 1--250 products."""

    tmin, tmax = cosebis["stored_theta_min"], cosebis["stored_theta_max"]
    if tmin is None or tmax is None:
        return False
    return abs(tmin - COSEBI_THETA_MIN) < 1.0 and abs(tmax - COSEBI_THETA_MAX) < 2.0


def _recompute_cosebis_from_fine(path: Path):
    """Recompute stale full-range mock COSEBIs on the paper fiducial cut."""

    gg = treecorr.GGCorrelation(
        min_sep=0.5, max_sep=500.0, nbins=1000, sep_units="arcmin"
    )
    gg.read(str(path))
    start, stop = scale_cut_to_bins(gg, COSEBI_THETA_MIN, COSEBI_THETA_MAX)
    inds = np.arange(start, stop)
    theta = _native(gg.meanr[inds])
    xip = _native(gg.xip[inds])
    xim = _native(gg.xim[inds])
    if len(theta) == 0:
        raise ValueError(f"No fine ξ bins in the COSEBI fiducial cut for {path}")
    cosebis = COSEBIS(float(theta.min()), float(theta.max()), 20, precision=120)
    en, bn = cosebis.cosebis_from_xipm(theta, xip, xim, parallel=True)
    return {
        "Bn": _native(bn),
        "En": _native(en),
        "stored_theta_min": float(theta.min()),
        "stored_theta_max": float(theta.max()),
        "source": str(path),
    }


def _load_or_recompute_cosebis(cosebis_path: Path, fine_path: Path):
    stored = _load_cosebis(cosebis_path)
    if _stored_cosebis_is_fiducial(stored):
        stored["scale_action"] = "used_stored_fiducial_vector"
        return stored
    if not fine_path.is_file():
        raise ValueError(
            f"{cosebis_path} stores a non-fiducial/unknown COSEBI cut "
            f"({stored['stored_theta_min']}, {stored['stored_theta_max']}); "
            f"fine ξ is unavailable for recomputation: {fine_path}"
        )
    print(
        f"COSEBI cut mismatch for {cosebis_path.name}: stored "
        f"[{stored['stored_theta_min']}, {stored['stored_theta_max']}], "
        f"recomputing at [{COSEBI_THETA_MIN:.6g}, {COSEBI_THETA_MAX:.6g}]",
        flush=True,
    )
    result = _recompute_cosebis_from_fine(fine_path)
    result["stored_theta_min"] = stored["stored_theta_min"]
    result["stored_theta_max"] = stored["stored_theta_max"]
    result["scale_action"] = "recomputed_from_fine_xi"
    return result


def _load_pure(path: Path):
    required = ("theta", "xip_B", "xim_B")
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(
                f"{path} is missing pure-EB keys {missing}; keys={data.files}"
            )
        result = {key: _native(data[key]) for key in required}
        for key in ("reporting_left_edges", "reporting_right_edges"):
            if key in data:
                result[key] = _native(data[key])
        if "integration_max_sep" in data:
            result["integration_max_sep"] = float(
                np.asarray(data["integration_max_sep"]).squeeze()
            )
    if result["theta"].shape != (20,):
        raise ValueError(f"Expected 20 pure-EB reporting bins in {path}")
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"Non-finite pure-EB vector in {path}")
    return result


def _load_cl(path: Path):
    array = _native(np.load(path, allow_pickle=False))
    if array.ndim != 2 or array.shape[0] < 5 or array.shape[1] != 32:
        raise ValueError(
            f"Expected external C_l array with at least 5x32 shape in {path}; "
            f"found {array.shape}"
        )
    # GLASS convention used by the paper: row 0=ell_eff, row 4=C_ell^BB.
    return {"ell": array[0], "BB": array[4], "source": str(path)}


def _load_pure_covariance(path: str | Path, theta: np.ndarray):
    with np.load(path, allow_pickle=False) as data:
        if "cov_pure_eb" not in data:
            raise KeyError(f"{path} has no cov_pure_eb key; keys={data.files}")
        covariance = _native(data["cov_pure_eb"])
    if covariance.shape != (120, 120):
        raise ValueError(
            f"Expected 120x120 pure-EB covariance, found {covariance.shape}"
        )
    nbins = len(theta)
    xip = covariance[2 * nbins : 3 * nbins, 2 * nbins : 3 * nbins]
    xim = covariance[3 * nbins : 4 * nbins, 3 * nbins : 4 * nbins]
    cross = covariance[2 * nbins : 3 * nbins, 3 * nbins : 4 * nbins]
    mask = (theta >= PURE_THETA_MIN) & (theta <= PURE_THETA_MAX)
    if int(mask.sum()) != 7:
        warnings.warn(
            f"Pure-EB [12, 83] selection found {mask.sum()} bins, expected 7",
            RuntimeWarning,
        )
    joint = np.block(
        [
            [xip[np.ix_(mask, mask)], cross[np.ix_(mask, mask)]],
            [cross.T[np.ix_(mask, mask)], xim[np.ix_(mask, mask)]],
        ]
    )
    return {
        "full": covariance,
        "xip_full": xip,
        "xim_full": xim,
        "cross_full": cross,
        "mask": mask,
        "xip_cut": xip[np.ix_(mask, mask)],
        "xim_cut": xim[np.ix_(mask, mask)],
        "cross_cut": cross[np.ix_(mask, mask)],
        "joint_cut": joint,
        "sigma_xip_full": np.sqrt(np.clip(np.diag(xip), 0, None)),
        "sigma_xim_full": np.sqrt(np.clip(np.diag(xim), 0, None)),
    }


def _load_cl_covariance(path: str | Path, ell: np.ndarray, scale=1.0):
    with fits.open(path) as hdul:
        covariance = _native(hdul["COVAR_BB_BB"].data)
    if covariance.shape != (32, 32):
        raise ValueError(f"Expected 32x32 C_l BB covariance, found {covariance.shape}")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"C_l covariance scale must be positive and finite, found {scale}"
        )
    # Mock matching assumes the BB covariance is noise dominated, so the
    # stored fiducial BB covariance is scaled by (sigma_e^2 / nbar) ratio^2.
    # Mixed signal-noise pieces would carry one power only; with BB signal=0,
    # the remaining caveat is mask-coupled EE×BB structure.
    covariance *= float(scale)
    mask = ell_bin_mask(ell, ELL_MIN, ELL_MAX)
    if int(mask.sum()) != 17:
        warnings.warn(
            f"Powspace [300, 1600] selection found {mask.sum()} bands, expected 17",
            RuntimeWarning,
        )
    return {
        "full": covariance,
        "scale": float(scale),
        "mask": mask,
        "cut": covariance[np.ix_(mask, mask)],
        "ell_low": get_powspace_bin_edges(ell)[0],
        "ell_high": get_powspace_bin_edges(ell)[1],
    }


def _load_cosebis_covariance(path_xi: str | Path, path_cov: str | Path):
    """Build the paper-identical COSEBI B covariance on the exact cut."""

    gg = treecorr.GGCorrelation(
        min_sep=0.5, max_sep=300.0, nbins=1000, sep_units="arcmin"
    )
    gg.read(str(path_xi))
    start, stop = scale_cut_to_bins(gg, COSEBI_THETA_MIN, COSEBI_THETA_MAX)
    inds = np.arange(start, stop)
    theta_cut = _native(gg.meanr[inds])
    covariance_xipm = _native(np.loadtxt(path_cov))
    nbins = len(gg.meanr)
    if covariance_xipm.shape != (2 * nbins, 2 * nbins):
        raise ValueError(
            f"Expected a {2 * nbins}x{2 * nbins} ξ covariance, "
            f"found {covariance_xipm.shape}"
        )
    covariance_indices = np.concatenate([inds, inds + nbins])
    cosebis = COSEBIS(float(theta_cut.min()), float(theta_cut.max()), 20, precision=120)
    covariance_cosebis = cosebis.cosebis_covariance_from_xipm_covariance(
        theta_cut,
        covariance_xipm[covariance_indices[:, None], covariance_indices],
    )
    covariance_B = covariance_cosebis[20:, 20:]
    sigma = np.sqrt(np.clip(np.diag(covariance_B), 0, None))
    print(
        "COSEBI analytic sigma_n (B, n=1..20): "
        + " ".join(f"{value:.8e}" for value in sigma),
        flush=True,
    )
    return {
        "full": covariance_B,
        "sigma": sigma,
        "theta_cut": theta_cut,
        "requested_cut": (COSEBI_THETA_MIN, COSEBI_THETA_MAX),
    }


def _compute_statistic(data, covariance, *, hartlap_n=None):
    return compute_chi2_pte(data, covariance, n_samples=hartlap_n)


def _statistics_for_mock(cosebis, pure, cl, covariances):
    bn = cosebis["Bn"]
    xip = pure["xip_B"]
    xim = pure["xim_B"]
    pure_mask = covariances["pure"]["mask"]
    xip_cut = xip[pure_mask]
    xim_cut = xim[pure_mask]
    xi_total = np.concatenate([xip_cut, xim_cut])
    cl_bb = cl["BB"][covariances["cl"]["mask"]]
    cosebis_cov = covariances["cosebis"]["full"]
    pure_cov = covariances["pure"]
    cl_cov = covariances["cl"]["cut"]

    stat_data = [bn[:6], bn, xip_cut, xim_cut, xi_total, cl_bb]
    stat_cov = [
        cosebis_cov[:6, :6],
        cosebis_cov,
        pure_cov["xip_cut"],
        pure_cov["xim_cut"],
        pure_cov["joint_cut"],
        cl_cov,
    ]
    hartlap = [None, None, HARTLAP_N_PURE, HARTLAP_N_PURE, HARTLAP_N_PURE, None]
    outputs = [
        _compute_statistic(data, covariance, hartlap_n=n_samples)
        for data, covariance, n_samples in zip(stat_data, stat_cov, hartlap)
    ]
    return {
        "Bn": bn,
        "xip_B_full": xip,
        "xim_B_full": xim,
        "xip_B_cut": xip_cut,
        "xim_B_cut": xim_cut,
        "xi_tot_B": xi_total,
        "cl_BB": cl_bb,
        "chi2": np.asarray([result[0] for result in outputs], dtype=float),
        "pte": np.asarray([result[1] for result in outputs], dtype=float),
        "dof": np.asarray([result[2] for result in outputs], dtype=int),
    }


def _json_number(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _json_array(values):
    return [_json_number(value) for value in np.asarray(values).ravel()]


def _sigma_evidence(label, empirical_data, analytic_sigma, labels=None):
    empirical_sigma = np.std(empirical_data, axis=0, ddof=1)
    analytic_sigma = np.asarray(analytic_sigma, dtype=float)
    ratio = empirical_sigma / analytic_sigma
    finite = ratio[np.isfinite(ratio)]
    summary = {
        "min": _json_number(np.min(finite)) if len(finite) else None,
        "max": _json_number(np.max(finite)) if len(finite) else None,
        "median": _json_number(np.median(finite)) if len(finite) else None,
    }
    return {
        "label": label,
        "labels": _json_array(labels) if labels is not None else None,
        "analytic_sigma": _json_array(analytic_sigma),
        "empirical_sigma": _json_array(empirical_sigma),
        "ratio": _json_array(ratio),
        "summary": summary,
    }


def _rank_columns(array):
    return np.apply_along_axis(stats.rankdata, 0, array)


def _correlation_matrix(array, method):
    if array.shape[0] < 2:
        return np.full((array.shape[1], array.shape[1]), np.nan)
    values = _rank_columns(array) if method == "spearman" else array
    return np.corrcoef(values, rowvar=False)


def _chi2_evidence(chi2, pte, dof):
    result = {}
    for index, name in enumerate(STAT_NAMES):
        values = np.asarray(chi2[:, index], dtype=float)
        ptes = np.asarray(pte[:, index], dtype=float)
        finite = np.isfinite(values) & np.isfinite(ptes)
        values, ptes = values[finite], ptes[finite]
        nu = int(dof[index])
        ks = stats.kstest(ptes, "uniform") if len(ptes) else None
        mean = float(np.mean(values)) if len(values) else np.nan
        variance = float(np.var(values, ddof=1)) if len(values) > 1 else np.nan
        effective = 2.0 * mean**2 / variance if variance > 0 else np.nan
        result[name] = {
            "n": int(len(values)),
            "dof": nu,
            "hartlap_n": HARTLAP_N_PURE if index in (2, 3, 4) else None,
            "chi2_mean": _json_number(mean),
            "chi2_variance": _json_number(variance),
            "expected_mean_nu": nu,
            "expected_variance_2nu": 2 * nu,
            "effective_dof": _json_number(effective),
            "effective_dof_from_mean": _json_number(mean),
            "effective_dof_from_variance": _json_number(variance / 2),
            "ks_pte_uniformity": {
                "statistic": _json_number(ks.statistic) if ks else None,
                "pvalue": _json_number(ks.pvalue) if ks else None,
            },
        }
    return result


def _correlation_evidence(pte, chi2):
    def nested_json(matrix):
        return [[_json_number(value) for value in row] for row in matrix]

    matrices = {}
    for label, values in (("pte", pte), ("chi2", chi2)):
        matrices[label] = {
            "stat_names": list(STAT_NAMES),
            "pearson": _json_array(_correlation_matrix(values, "pearson")),
            "spearman": _json_array(_correlation_matrix(values, "spearman")),
        }
        matrices[label]["shape"] = list(np.asarray(values).shape)
        # The flat arrays above are convenient for strict JSON consumers;
        # retain nested forms for human use and plotting provenance.
        matrices[label]["pearson_matrix"] = nested_json(
            _correlation_matrix(values, "pearson")
        )
        matrices[label]["spearman_matrix"] = nested_json(
            _correlation_matrix(values, "spearman")
        )
    return matrices


def _save_figure(fig, output_dir: Path, stem: str):
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_sigma_ratios(output_dir, arrays, covariances, theta, ell):
    cosebi_ratio = (
        np.std(arrays["Bn"], axis=0, ddof=1) / covariances["cosebis"]["sigma"]
    )
    xip_ratio = (
        np.std(arrays["xip_B_full"], axis=0, ddof=1)
        / covariances["pure"]["sigma_xip_full"]
    )
    xim_ratio = (
        np.std(arrays["xim_B_full"], axis=0, ddof=1)
        / covariances["pure"]["sigma_xim_full"]
    )
    cl_sigma = np.sqrt(np.clip(np.diag(covariances["cl"]["cut"]), 0, None))
    cl_ratio = np.std(arrays["cl_BB"], axis=0, ddof=1) / cl_sigma

    # Sampling error of an N-realization standard deviation:
    # sigma(s)/s ~= 1/sqrt(2(N-1)) => each ratio point carries that relative error.
    n_real = arrays["Bn"].shape[0]
    rel_err = 1.0 / np.sqrt(2.0 * (n_real - 1))
    eb_kw = dict(ms=3, lw=0.8, elinewidth=0.6, capsize=1.5)
    plotted_ratios = (cosebi_ratio, xip_ratio, xim_ratio, cl_ratio)
    plotted_content = np.concatenate(
        [
            np.asarray(ratio)[..., None]
            + np.asarray(ratio * rel_err)[..., None] * np.array([-1.0, 1.0])
            for ratio in plotted_ratios
        ]
    )
    finite_content = plotted_content[np.isfinite(plotted_content)]
    max_dev = np.max(np.abs(finite_content - 1.0))
    shared_ylim = (1.0 - 1.05 * max_dev, 1.0 + 1.05 * max_dev)

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
    axes[0].errorbar(
        np.arange(1, 21), cosebi_ratio, yerr=cosebi_ratio * rel_err, fmt="o-", **eb_kw
    )
    axes[0].set_xlabel("COSEBI mode $n$")
    axes[0].set_ylabel(r"$\sigma_\mathrm{emp}/\sigma_\mathrm{ana}$")
    axes[0].set_xticks([1, 5, 10, 15, 20])

    pure_theta = np.asarray(theta)
    axes[1].errorbar(
        pure_theta,
        xip_ratio,
        yerr=xip_ratio * rel_err,
        fmt="o-",
        label=r"$\xi_+^B$",
        **eb_kw,
    )
    axes[1].errorbar(
        pure_theta,
        xim_ratio,
        yerr=xim_ratio * rel_err,
        fmt="s-",
        label=r"$\xi_-^B$",
        **eb_kw,
    )
    axes[1].axvspan(PURE_THETA_MIN, PURE_THETA_MAX, color="0.85", zorder=0)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\theta$ [arcmin]")
    axes[1].legend(fontsize=6)

    ell_cut = np.asarray(ell)[covariances["cl"]["mask"]]
    axes[2].errorbar(ell_cut, cl_ratio, yerr=cl_ratio * rel_err, fmt="o-", **eb_kw)
    axes[2].axvspan(ELL_MIN, ELL_MAX, color="0.85", zorder=0)
    axes[2].set_xlabel(r"$\ell$")
    axes[2].set_xscale("squareroot" if PAPER_MPLSTYLE else "linear")

    for ax in axes:
        ax.axhline(1.0, color="black", ls="--", lw=0.8)
        ax.grid(alpha=0.2)
        ax.set_ylim(shared_ylim)
    fig.suptitle(
        r"GLASS mock covariance campaign: empirical/analytic $\sigma$", fontsize=9
    )
    _save_figure(fig, output_dir, "sigma_ratio")


def _plot_pte_uniformity(output_dir, pte):
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.0), squeeze=False)
    quantiles = (np.arange(1, len(pte) + 1) - 0.5) / len(pte)
    for index, (ax, name) in enumerate(zip(axes.flat, STAT_NAMES)):
        values = np.asarray(pte[:, index])
        ax.hist(values, bins=np.linspace(0, 1, 11), density=True, alpha=0.65)
        ax.axhline(1.0, color="black", ls="--", lw=0.8, label="Uniform")
        ax.set_xlim(0, 1)
        ax.set_xlabel("PTE")
        ax.set_ylabel("density")
        ks = stats.kstest(values, "uniform")
        ax.set_title(f"{STAT_TEX.get(name, name)}\nKS p={ks.pvalue:.3g}", fontsize=8)
        ax.grid(alpha=0.2)
        inset = ax.inset_axes([0.52, 0.16, 0.43, 0.34])
        sorted_values = np.sort(values)
        inset.plot(quantiles, sorted_values, ".", ms=2)
        inset.plot([0, 1], [0, 1], "k--", lw=0.6)
        inset.set_xlim(0, 1)
        inset.set_ylim(0, 1)
        inset.set_title("QQ", fontsize=6)
        inset.tick_params(labelsize=5)
    fig.suptitle("PTE uniformity and uniform QQ diagnostics", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir, "pte_uniformity_qq")


def _heatmap(ax, matrix, title):
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(
        range(len(STAT_NAMES)),
        [STAT_TEX.get(n, n) for n in STAT_NAMES],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(
        range(len(STAT_NAMES)), [STAT_TEX.get(n, n) for n in STAT_NAMES], fontsize=7
    )
    for row in range(len(STAT_NAMES)):
        for col in range(len(STAT_NAMES)):
            value = matrix[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title(title, fontsize=8)
    return image


def _plot_correlations(output_dir, pte, chi2):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    image = _heatmap(axes[0], _correlation_matrix(pte, "spearman"), "PTE Spearman")
    _heatmap(axes[1], _correlation_matrix(chi2, "spearman"), r"$\chi^2$ Spearman")
    fig.colorbar(image, ax=axes, shrink=0.8, label="correlation")
    fig.suptitle("Cross-statistic correlation of null-test outputs", fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_dir, "cross_statistic_correlation")


def run_campaign(
    *,
    mock_ids,
    cosebis_dir,
    pure_dir,
    fine_dir,
    cl_dir,
    pseudo_cl_dir,
    cosebis_xi_grid,
    cosebis_cov,
    pure_cov,
    cl_cov,
    output_dir,
    label="fiducial",
    cl_cov_scale=1.0,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_ids = _normalise_ids(mock_ids)
    complete_ids = []
    skipped = {}
    for mock_id in requested_ids:
        paths = {
            kind: _path_for(kind, directory, mock_id)
            for kind, directory in (
                ("cosebis", cosebis_dir),
                ("pure", pure_dir),
                ("fine", fine_dir),
                ("cl", cl_dir),
            )
        }
        missing = [
            str(path)
            for key, path in paths.items()
            if key != "fine" and not path.is_file()
        ]
        if missing:
            skipped[mock_id] = {"reason": "missing required input", "paths": missing}
        else:
            complete_ids.append(mock_id)
    if not complete_ids:
        raise RuntimeError(
            "No complete mocks found. Required products are per-mock COSEBIs, "
            "pure-EB npz, and external cl npy."
        )
    print(
        f"Campaign candidates={len(requested_ids)}, complete={len(complete_ids)}, "
        f"skipped={len(skipped)}",
        flush=True,
    )
    print(
        f"Covariance mode={label}, C_l BB covariance scale={float(cl_cov_scale):.8g}",
        flush=True,
    )

    first_pure = _load_pure(_path_for("pure", pure_dir, complete_ids[0]))
    pure_integration_max = first_pure.get("integration_max_sep")
    if pure_integration_max is not None and pure_integration_max > 300.0 + 1e-6:
        warnings.warn(
            "Pure-EB mock vectors use an integration grid beyond 300 arcmin, "
            "whereas the stored paper pure-EB covariance stops at 300 arcmin; "
            "the xi- comparison is not apples-to-apples.",
            RuntimeWarning,
        )
    pure_covariances = _load_pure_covariance(pure_cov, first_pure["theta"])
    cosebis_covariances = _load_cosebis_covariance(cosebis_xi_grid, cosebis_cov)
    first_cl = _load_cl(_path_for("cl", cl_dir, complete_ids[0]))
    cl_covariances = _load_cl_covariance(cl_cov, first_cl["ell"], scale=cl_cov_scale)
    covariances = {
        "pure": pure_covariances,
        "cosebis": cosebis_covariances,
        "cl": cl_covariances,
    }

    records = []
    stored_scale_metadata = {}
    ell_eff = first_cl["ell"]
    theta = first_pure["theta"]
    for mock_id in complete_ids:
        try:
            paths = {
                "cosebis": _path_for("cosebis", cosebis_dir, mock_id),
                "pure": _path_for("pure", pure_dir, mock_id),
                "fine": _path_for("fine", fine_dir, mock_id),
                "cl": _path_for("cl", cl_dir, mock_id),
            }
            cosebis = _load_or_recompute_cosebis(paths["cosebis"], paths["fine"])
            pure = _load_pure(paths["pure"])
            cl = _load_cl(paths["cl"])
            # theta is the pair-weighted meanr and legitimately varies per mock;
            # the binning (edges) is what must match the reference.
            edge_keys = ("reporting_left_edges", "reporting_right_edges")
            if all(key in first_pure and key in pure for key in edge_keys):
                for edge_key in edge_keys:
                    if not np.allclose(
                        pure[edge_key], first_pure[edge_key], rtol=1e-10
                    ):
                        raise ValueError(
                            f"pure-EB reporting bin edges differ from mock {complete_ids[0]}"
                        )
            elif not np.allclose(pure["theta"], theta, rtol=1e-3):
                raise ValueError(
                    f"pure-EB theta grid differs from mock {complete_ids[0]}"
                )
            if not np.allclose(cl["ell"], ell_eff, rtol=0, atol=1e-8):
                raise ValueError(f"C_l ell grid differs from mock {complete_ids[0]}")
            result = _statistics_for_mock(cosebis, pure, cl, covariances)
            if not all(np.all(np.isfinite(result[key])) for key in ("chi2", "pte")):
                raise ValueError("non-finite chi2/PTE")
            records.append(result)
            stored_scale_metadata[mock_id] = {
                "stored_theta_min": cosebis["stored_theta_min"],
                "stored_theta_max": cosebis["stored_theta_max"],
                "action": cosebis["scale_action"],
            }
        except Exception as exc:
            warnings.warn(f"Skipping mock {mock_id}: {exc}", RuntimeWarning)
            skipped[mock_id] = {"reason": str(exc)}

    if not records:
        raise RuntimeError("All candidate mocks failed validation; see warnings above")
    ids = [mock_id for mock_id in complete_ids if mock_id not in skipped]
    arrays = {
        key: np.stack([record[key] for record in records])
        for key in (
            "Bn",
            "xip_B_full",
            "xim_B_full",
            "xip_B_cut",
            "xim_B_cut",
            "xi_tot_B",
            "cl_BB",
        )
    }
    chi2 = np.stack([record["chi2"] for record in records])
    pte = np.stack([record["pte"] for record in records])
    dof = records[0]["dof"]
    statistics_path = output_dir / "per_mock_statistics.npz"
    np.savez(
        statistics_path,
        mock_ids=np.asarray(ids),
        stat_names=np.asarray(STAT_NAMES),
        theta=theta,
        theta_cut=theta[pure_covariances["mask"]],
        ell=ell_eff,
        ell_cut=ell_eff[cl_covariances["mask"]],
        cosebis_theta_cut=np.asarray(cosebis_covariances["theta_cut"]),
        **arrays,
        chi2=chi2,
        pte=pte,
        dof=dof,
    )
    print(f"Saved {statistics_path}", flush=True)

    pure_mask = pure_covariances["mask"]
    pure_cut_theta = theta[pure_mask]
    cl_mask = cl_covariances["mask"]
    cl_cut_ell = ell_eff[cl_mask]
    sigma_evidence = {
        "COSEBI_B_modes_1_to_20": _sigma_evidence(
            "COSEBI B modes 1--20",
            arrays["Bn"],
            cosebis_covariances["sigma"],
            labels=np.arange(1, 21),
        ),
        "xi_plus_B_20_bins": _sigma_evidence(
            "ξ+^B, all 20 reporting bins",
            arrays["xip_B_full"],
            pure_covariances["sigma_xip_full"],
            labels=theta,
        ),
        "xi_minus_B_20_bins": _sigma_evidence(
            "ξ−^B, all 20 reporting bins",
            arrays["xim_B_full"],
            pure_covariances["sigma_xim_full"],
            labels=theta,
        ),
        "xi_plus_B_7_cut_bins": _sigma_evidence(
            "ξ+^B, 12--83 arcmin cut",
            arrays["xip_B_cut"],
            np.sqrt(np.clip(np.diag(pure_covariances["xip_cut"]), 0, None)),
            labels=pure_cut_theta,
        ),
        "xi_minus_B_7_cut_bins": _sigma_evidence(
            "ξ−^B, 12--83 arcmin cut",
            arrays["xim_B_cut"],
            np.sqrt(np.clip(np.diag(pure_covariances["xim_cut"]), 0, None)),
            labels=pure_cut_theta,
        ),
        "C_ell_BB_17_cut_bands": _sigma_evidence(
            "C_l^BB, 17 bands fully inside 300--1600",
            arrays["cl_BB"],
            np.sqrt(np.clip(np.diag(cl_covariances["cut"]), 0, None)),
            labels=cl_cut_ell,
        ),
    }
    evidence = {
        "spec": "mock_covariance_campaign",
        "generated": datetime.now().isoformat(),
        "n_mocks": len(ids),
        "mock_ids": ids,
        "requested_mock_ids": requested_ids,
        "skipped": skipped,
        "scale_cuts": {
            "cosebi_requested_exact": [COSEBI_THETA_MIN, COSEBI_THETA_MAX],
            "pure_xip_and_xim": [PURE_THETA_MIN, PURE_THETA_MAX],
            "cl_bb": [ELL_MIN, ELL_MAX],
            "pure_bins_selected": int(pure_mask.sum()),
            "cl_bands_selected": int(cl_mask.sum()),
        },
        "covariance_mode": {
            "label": str(label),
            "cosebis_covariance": str(cosebis_cov),
            "pure_eb_covariance": str(pure_cov),
            "cl_bb_covariance": str(cl_cov),
            "cl_cov_scale": float(cl_cov_scale),
            "mockmatching_noise_density_ratio": FIDUCIAL_NEFF / MOCK_NEFF,
            "mockmatching_noise_variance_ratio": MOCKMATCHED_CL_COV_SCALE,
            "cl_cov_scale_description": (
                "Full stored C_l BB covariance multiplied by the supplied noise "
                "variance ratio. This is exact for the noise-noise term only; "
                "mixed signal-noise terms would scale by one power, but BB has "
                "zero signal here, leaving only mask-coupled EE×BB caveat terms."
            ),
        },
        "cosebi_scale_conventions": stored_scale_metadata,
        "pure_eb_input_grid": {
            "first_mock_integration_max_sep": pure_integration_max,
            "covariance_integration_max_sep": 300.0,
            "comparison_warning": (
                pure_integration_max is not None and pure_integration_max > 300.0 + 1e-6
            ),
        },
        "sigma_ratios": sigma_evidence,
        "chi2_calibration": _chi2_evidence(chi2, pte, dof),
        "correlations": _correlation_evidence(pte, chi2),
    }
    evidence_path = output_dir / "evidence.json"
    with evidence_path.open("w") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
    print(f"Saved {evidence_path}", flush=True)

    # One-mock sanity gate: it is diagnostic rather than a campaign veto,
    # since a single Gaussian realization can naturally land below PTE=0.05.
    first = records[0]
    finite = all(
        np.all(np.isfinite(first[key]))
        for key in ("chi2", "pte", "xip_B_cut", "xim_B_cut")
    )
    xip_zmax = float(
        np.max(
            np.abs(first["xip_B_cut"]) / np.sqrt(np.diag(pure_covariances["xip_cut"]))
        )
    )
    xim_zmax = float(
        np.max(
            np.abs(first["xim_B_cut"]) / np.sqrt(np.diag(pure_covariances["xim_cut"]))
        )
    )
    sanity_pass = finite and xip_zmax <= 2.5 and xim_zmax <= 2.5
    print(
        "One-mock sanity: "
        f"finite={finite}, max |xi+^B|/sigma={xip_zmax:.2f}, "
        f"max |xi-^B|/sigma={xim_zmax:.2f}, pass(~2sigma)={sanity_pass}",
        flush=True,
    )
    if not sanity_pass:
        warnings.warn(
            "One-mock pure-B sanity gate is outside the loose 2.5σ diagnostic",
            RuntimeWarning,
        )

    # Provenance check requested by the campaign brief.  Compare the first
    # dozen IDs for which the in-repo FITS product exists; the external array's
    # row 4 is the BB vector.
    compared = 0
    for mock_id in ids:
        if compared >= 12:
            break
        pseudo_path = _path_for("pseudo_cl", pseudo_cl_dir, mock_id)
        if not pseudo_path.is_file():
            continue
        with fits.open(pseudo_path) as hdul:
            pseudo_bb = _native(hdul["PSEUDO_CELL"].data["BB"])
        external_bb = _native(
            np.load(_path_for("cl", cl_dir, mock_id), allow_pickle=False)[4]
        )
        if pseudo_bb.shape != external_bb.shape:
            print(
                f"WARNING C_l provenance {mock_id}: shape {external_bb.shape} vs {pseudo_bb.shape}",
                flush=True,
            )
            continue
        difference = external_bb - pseudo_bb
        scale = np.maximum(np.abs(pseudo_bb), 1e-30)
        max_abs = float(np.max(np.abs(difference)))
        max_relative = float(np.max(np.abs(difference) / scale))
        rms = float(np.sqrt(np.mean(difference**2)))
        agreement = max_relative <= 1e-3 and np.allclose(
            external_bb, pseudo_bb, rtol=1e-5, atol=1e-14
        )
        print(
            f"C_l provenance {mock_id}: max_abs={max_abs:.4e}, rms={rms:.4e}, "
            f"max_relative={max_relative:.4e}, allclose={agreement}",
            flush=True,
        )
        if max_relative > 1e-3:
            print(
                f"WARNING: material external-vs-FITS BB disagreement for mock {mock_id}",
                flush=True,
            )
        compared += 1

    _plot_sigma_ratios(output_dir, arrays, covariances, theta, ell_eff)
    _plot_pte_uniformity(output_dir, pte)
    _plot_correlations(output_dir, pte, chi2)
    print(f"Wrote campaign figures to {output_dir}", flush=True)


def _parser(argv=None):
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock-ids",
        nargs="*",
        help="Explicit mock IDs; default is available subset of 1--350",
    )
    parser.add_argument("--mock-glob", help="Glob used to discover candidate mock IDs")
    parser.add_argument("--cosebis-dir", default=defaults["cosebis_dir"])
    parser.add_argument("--pure-dir", default=defaults["pure_dir"])
    parser.add_argument("--fine-dir", default=defaults["fine_dir"])
    parser.add_argument("--cl-dir", default=defaults["cl_dir"])
    parser.add_argument("--pseudo-cl-dir", default=defaults["pseudo_cl_dir"])
    parser.add_argument("--cosebis-xi-grid", default=defaults["cosebis_xi_grid"])
    parser.add_argument("--cosebis-cov", default=defaults["cosebis_cov"])
    parser.add_argument("--pure-cov", default=defaults["pure_cov"])
    parser.add_argument("--cl-cov", default=defaults["cl_cov"])
    parser.add_argument(
        "--cl-cov-scale",
        type=float,
        default=1.0,
        help="Multiplicative scale for the stored C_l BB covariance (default: 1).",
    )
    parser.add_argument(
        "--label",
        default="fiducial",
        help="Covariance suite label written to evidence.json (e.g. fiducial/mockmatched).",
    )
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    return parser.parse_args(argv)


def _standalone_main(argv=None):
    args = _parser(argv)
    ids = _discover_ids(args.mock_ids, args.mock_glob, cosebis_dir=args.cosebis_dir)
    run_campaign(
        mock_ids=ids,
        cosebis_dir=args.cosebis_dir,
        pure_dir=args.pure_dir,
        fine_dir=args.fine_dir,
        cl_dir=args.cl_dir,
        pseudo_cl_dir=args.pseudo_cl_dir,
        cosebis_xi_grid=args.cosebis_xi_grid,
        cosebis_cov=args.cosebis_cov,
        pure_cov=args.pure_cov,
        cl_cov=args.cl_cov,
        output_dir=args.output_dir,
        label=args.label,
        cl_cov_scale=args.cl_cov_scale,
    )


def _snakemake_main(smk):
    run_campaign(
        mock_ids=smk.params.mock_ids,
        cosebis_dir=smk.input.cosebis_dir,
        pure_dir=smk.input.pure_dir,
        fine_dir=smk.input.fine_dir,
        cl_dir=smk.input.cl_dir,
        pseudo_cl_dir=smk.input.pseudo_cl_dir,
        cosebis_xi_grid=smk.input.cosebis_xi_grid,
        cosebis_cov=smk.input.cosebis_cov,
        pure_cov=smk.input.pure_cov,
        cl_cov=smk.input.cl_cov,
        output_dir=smk.output.campaign,
        label=smk.params.label,
        cl_cov_scale=smk.params.cl_cov_scale,
    )


if "snakemake" in globals():
    _snakemake_main(snakemake)
else:
    _standalone_main()
