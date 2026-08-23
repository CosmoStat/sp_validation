"""Inspect fine-grid weights behind localized pure-EB covariance features.

This reproduces the deterministic part of the paper's MC propagation.  The
Schneider formulas are linear, but the implementation's Akima interpolation
has data-dependent slopes, so the numerical transform is not globally linear.
Without a baseline, this script therefore reports zero-baseline unit-vector
secants rather than claiming that ``w.T @ C @ w`` must equal the stored MC
covariance.  With ``--mean-fine``, it instead computes a local finite-difference
Jacobian around the production mean.  With ``--replay-production`` it replays
the production Gaussian draws exactly and compares the resulting covariance.

Example
-------
    python pure_eb_cov_spike_diagnostic.py \
        --pure-cov /path/SP_v1.4.6.3_leak_corr_A_pure_eb_semianalytic.npz \
        --fine-cov /path/covariance_processed.txt \
        --out pure_eb_cov_spike_diagnostic.png

For an exact production replay, also provide the saved 2N-vector ``mean_int``
from ``precompute_pure_eb_chunk.py`` and add ``--mean-fine mean_int.npy
--replay-production``.

The script requires the project container environment: cosmo_numba,
NumbaQuadpack, NumPy, and Matplotlib.  It intentionally does not sample from
the covariance unless ``--replay-production`` is requested, and does not modify
either input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_OUTPUT_NAMES = (
    "xip_E",
    "xim_E",
    "xip_B",
    "xim_B",
    "xip_amb",
    "xim_amb",
)
_SPIKE_TARGETS = {
    "xip_B": (55.6, 221.0),
    "xim_B": (2.0, 3.5),
}


def _reporting_binning(theta_int, n_reporting, min_sep, max_sep):
    """Return the production fine-to-reporting averaging matrix."""

    from scipy import sparse

    edges = np.logspace(np.log10(min_sep), np.log10(max_sep), n_reporting + 1)
    bin_indices = np.digitize(theta_int, edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < n_reporting)
    row_indices = bin_indices[valid]
    col_indices = np.where(valid)[0]
    matrix = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(n_reporting, len(theta_int)),
    )
    row_sums = np.asarray(matrix.sum(axis=1)).flatten()
    if np.any(row_sums == 0):
        empty = np.flatnonzero(row_sums).tolist()
        raise ValueError(f"Reporting bins have no fine-grid samples: {empty}")
    matrix = sparse.diags(1 / row_sums) @ matrix
    return matrix, edges, bin_indices


def _load_inputs(pure_cov_path, fine_cov_path):
    pure = np.load(pure_cov_path)
    required = {"theta", "theta_int", "cov_pure_eb"}
    missing = required.difference(pure.files)
    if missing:
        raise ValueError(f"{pure_cov_path} is missing keys: {sorted(missing)}")

    theta = np.ascontiguousarray(pure["theta"], dtype=float)
    theta_int = np.ascontiguousarray(pure["theta_int"], dtype=float)
    covariance = np.asarray(np.loadtxt(fine_cov_path), dtype=float)
    stored_covariance = np.asarray(pure["cov_pure_eb"], dtype=float)

    n_reporting = len(theta)
    n_fine = len(theta_int)
    expected_fine_shape = (2 * n_fine, 2 * n_fine)
    if covariance.shape != expected_fine_shape:
        raise ValueError(
            f"Fine covariance shape {covariance.shape} does not match "
            f"theta_int ({n_fine}): expected {expected_fine_shape}"
        )
    expected_output_shape = (6 * n_reporting, 6 * n_reporting)
    if stored_covariance.shape != expected_output_shape:
        raise ValueError(
            f"Stored pure covariance shape {stored_covariance.shape} does not "
            f"match theta ({n_reporting}): expected {expected_output_shape}"
        )
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(theta_int)):
        raise ValueError("theta and theta_int must be finite")
    if np.any(np.diff(theta) <= 0) or np.any(np.diff(theta_int) <= 0):
        raise ValueError("theta and theta_int must be strictly increasing")

    return theta, theta_int, covariance, stored_covariance


def _all_b_rows(theta):
    """All xip_B and xim_B rows, in output order."""

    n = len(theta)
    rows = []
    labels = []
    for output_name in ("xip_B", "xim_B"):
        base = _OUTPUT_NAMES.index(output_name) * n
        for index in range(n):
            rows.append(base + index)
            labels.append(f"{output_name} bin {index} ({theta[index]:.4g} arcmin)")
    return np.asarray(rows, dtype=int), labels


def _selected_rows(theta, neighbor_radius):
    """Select target bins and adjacent bins, preserving target order."""

    selected = []
    labels = []
    for output_name, targets in _SPIKE_TARGETS.items():
        for target in targets:
            center = int(np.argmin(np.abs(theta - target)))
            start = max(0, center - neighbor_radius)
            stop = min(len(theta), center + neighbor_radius + 1)
            for index in range(start, stop):
                row = _OUTPUT_NAMES.index(output_name) * len(theta) + index
                if row not in selected:
                    selected.append(row)
                    labels.append(
                        f"{output_name} bin {index} ({theta[index]:.4g} arcmin)"
                    )
    return np.asarray(selected, dtype=int), labels


def _transform_unit(
    unit,
    fine_index,
    *,
    theta,
    theta_int,
    reporting_matrix,
    tmin,
    tmax,
    pad_xim,
    pad_theta_max_decade,
    interp_order,
):
    """Push one fine-vector basis vector through the production transform."""

    n_fine = len(theta_int)
    xip_int = np.zeros(n_fine, dtype=float)
    xim_int = np.zeros(n_fine, dtype=float)
    if unit == "xip":
        xip_int[fine_index] = 1.0
    else:
        xim_int[fine_index] = 1.0

    return _transform_vector(
        xip_int,
        xim_int,
        theta=theta,
        theta_int=theta_int,
        reporting_matrix=reporting_matrix,
        tmin=tmin,
        tmax=tmax,
        pad_xim=pad_xim,
        pad_theta_max_decade=pad_theta_max_decade,
        interp_order=interp_order,
    )


def _transform_vector(
    xip_int,
    xim_int,
    *,
    theta,
    theta_int,
    reporting_matrix,
    tmin,
    tmax,
    pad_xim,
    pad_theta_max_decade,
    interp_order,
):
    """Apply the exact production rebinning and pure-EB transform."""

    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    xip = reporting_matrix @ xip_int
    xim = reporting_matrix @ xim_int
    modes = get_pure_EB_modes(
        theta=theta,
        xip=np.ascontiguousarray(xip),
        xim=np.ascontiguousarray(xim),
        theta_int=theta_int,
        xip_int=np.ascontiguousarray(xip_int),
        xim_int=np.ascontiguousarray(xim_int),
        tmin=tmin,
        tmax=tmax,
        parallel=False,
        pad_xim=pad_xim,
        pad_theta_max_decade=pad_theta_max_decade,
        interp_order=interp_order,
    )
    return np.concatenate(modes)


def extract_weight_rows(
    *,
    theta,
    theta_int,
    rows,
    min_sep,
    max_sep,
    pad_xim=True,
    pad_theta_max_decade=1.0,
    interp_order=5,
    baseline_fine=None,
    finite_difference_step=1e-8,
):
    """Extract zero-baseline secants or a local Jacobian for selected rows."""

    reporting_matrix, edges, bin_indices = _reporting_binning(
        theta_int, len(theta), min_sep, max_sep
    )
    n_fine = len(theta_int)
    weights = np.empty((len(rows), 2 * n_fine), dtype=float)

    # The paper calls the transform with tmin/tmax equal to the reporting-grid
    # endpoints reconstructed in precompute_pure_eb_chunk.py.
    tmin = float(edges[0])
    tmax = float(edges[-1])
    if baseline_fine is not None:
        baseline_fine = np.asarray(baseline_fine, dtype=float)
        if baseline_fine.shape != (2 * n_fine,):
            raise ValueError(
                f"baseline_fine must have shape {(2 * n_fine,)}, "
                f"found {baseline_fine.shape}"
            )
        base_xip = baseline_fine[:n_fine]
        base_xim = baseline_fine[n_fine:]
        base_output = _transform_vector(
            base_xip,
            base_xim,
            theta=theta,
            theta_int=theta_int,
            reporting_matrix=reporting_matrix,
            tmin=tmin,
            tmax=tmax,
            pad_xim=pad_xim,
            pad_theta_max_decade=pad_theta_max_decade,
            interp_order=interp_order,
        )
    else:
        base_xip = base_xim = base_output = None

    for column in range(2 * n_fine):
        if baseline_fine is None:
            unit = "xip" if column < n_fine else "xim"
            fine_index = column if unit == "xip" else column - n_fine
            transformed = _transform_unit(
                unit,
                fine_index,
                theta=theta,
                theta_int=theta_int,
                reporting_matrix=reporting_matrix,
                tmin=tmin,
                tmax=tmax,
                pad_xim=pad_xim,
                pad_theta_max_decade=pad_theta_max_decade,
                interp_order=interp_order,
            )
        else:
            plus = base_xip.copy()
            minus = base_xim.copy()
            if column < n_fine:
                plus[column] += finite_difference_step
            else:
                minus[column - n_fine] += finite_difference_step
            plus_output = _transform_vector(
                plus,
                minus,
                theta=theta,
                theta_int=theta_int,
                reporting_matrix=reporting_matrix,
                tmin=tmin,
                tmax=tmax,
                pad_xim=pad_xim,
                pad_theta_max_decade=pad_theta_max_decade,
                interp_order=interp_order,
            )
            plus = base_xip.copy()
            minus = base_xim.copy()
            if column < n_fine:
                plus[column] -= finite_difference_step
            else:
                minus[column - n_fine] -= finite_difference_step
            minus_output = _transform_vector(
                plus,
                minus,
                theta=theta,
                theta_int=theta_int,
                reporting_matrix=reporting_matrix,
                tmin=tmin,
                tmax=tmax,
                pad_xim=pad_xim,
                pad_theta_max_decade=pad_theta_max_decade,
                interp_order=interp_order,
            )
            transformed = (plus_output - minus_output) / (2.0 * finite_difference_step)
        weights[:, column] = transformed[rows]
        if (column + 1) % 100 == 0 or column + 1 == 2 * n_fine:
            print(f"Transformed {column + 1}/{2 * n_fine} fine basis vectors")

    return weights, reporting_matrix, edges, bin_indices, base_output


def _load_mean_fine(path, n_fine):
    """Load production's 2N fine-grid mean from .npy or text."""

    path = Path(path)
    values = np.load(path) if path.suffix == ".npy" else np.loadtxt(path)
    values = np.asarray(values, dtype=float).reshape(-1)
    expected = 2 * n_fine
    if values.shape != (expected,):
        raise ValueError(f"Expected {expected} fine-mean values, found {values.shape}")
    return values


def _replay_production(
    *,
    mean_fine,
    fine_cov,
    theta,
    theta_int,
    reporting_matrix,
    tmin,
    tmax,
    pad_xim,
    pad_theta_max_decade,
    interp_order,
    n_samples,
    n_chunks,
    seed,
):
    """Replay precompute_pure_eb_chunk.py's seeded sample stream exactly."""

    from tqdm import tqdm

    samples_per_chunk = n_samples // n_chunks
    transformed_chunks = []
    n_fine = len(theta_int)
    for chunk_id in range(n_chunks):
        start = chunk_id * samples_per_chunk
        stop = start + samples_per_chunk if chunk_id < n_chunks - 1 else n_samples
        rng = np.random.default_rng(seed=seed + chunk_id)
        samples = rng.multivariate_normal(mean_fine, fine_cov, size=stop - start)
        transformed = []
        for sample in tqdm(samples, desc=f"Replay chunk {chunk_id}"):
            transformed.append(
                _transform_vector(
                    sample[:n_fine],
                    sample[n_fine:],
                    theta=theta,
                    theta_int=theta_int,
                    reporting_matrix=reporting_matrix,
                    tmin=tmin,
                    tmax=tmax,
                    pad_xim=pad_xim,
                    pad_theta_max_decade=pad_theta_max_decade,
                    interp_order=interp_order,
                )
            )
        transformed_chunks.append(np.asarray(transformed))
    return np.cov(np.vstack(transformed_chunks).T)


def _plot_weights(theta_int, rows, labels, weights, out_path):
    """Plot selected rows, separating fine ξ+ and ξ− input weights."""

    n_fine = len(theta_int)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(rows)))
    for row_index, (label, color) in enumerate(zip(labels, colors)):
        axes[0].plot(
            theta_int,
            weights[row_index, :n_fine],
            color=color,
            linewidth=1.0,
            label=label,
        )
        axes[1].plot(
            theta_int,
            weights[row_index, n_fine:],
            color=color,
            linewidth=1.0,
            label=label,
        )

    axes[0].set_ylabel(r"weight on fine $\xi_+'(\theta')$")
    axes[1].set_ylabel(r"weight on fine $\xi_-'(\theta')$")
    axes[1].set_xlabel(r"fine $\theta'$ [arcmin]")
    for axis in axes:
        axis.set_xscale("log")
        axis.axhline(0.0, color="0.4", linewidth=0.6)
        axis.grid(True, which="both", alpha=0.2)
    axes[0].legend(loc="upper left", fontsize="x-small", ncol=2)
    fig.suptitle("Effective fine-grid weights for pure-EB output bins")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def diagnose(args):
    theta, theta_int, fine_cov, stored_cov = _load_inputs(args.pure_cov, args.fine_cov)
    if args.all_b_rows:
        rows, labels = _all_b_rows(theta)
    else:
        rows, labels = _selected_rows(theta, args.neighbor_radius)
    mean_fine = (
        _load_mean_fine(args.mean_fine, len(theta_int)) if args.mean_fine else None
    )
    if args.replay_production and mean_fine is None:
        raise ValueError("--replay-production requires --mean-fine")

    weights, reporting_matrix, edges, bin_indices, base_output = extract_weight_rows(
        theta=theta,
        theta_int=theta_int,
        rows=rows,
        min_sep=args.min_sep,
        max_sep=args.max_sep,
        pad_xim=not args.no_pad_xim,
        pad_theta_max_decade=args.pad_theta_max_decade,
        interp_order=args.interp_order,
        baseline_fine=mean_fine,
        finite_difference_step=args.finite_difference_step,
    )

    linearized = np.einsum("bi,ij,bj->b", weights, fine_cov, weights)
    stored = np.diag(stored_cov)[rows]
    linearized_error = (linearized - stored) / np.maximum(np.abs(stored), 1e-300)

    replayed = None
    replayed_error = None
    if args.replay_production:
        tmin = float(edges[0])
        tmax = float(edges[-1])
        replayed_cov = _replay_production(
            mean_fine=mean_fine,
            fine_cov=fine_cov,
            theta=theta,
            theta_int=theta_int,
            reporting_matrix=reporting_matrix,
            tmin=tmin,
            tmax=tmax,
            pad_xim=not args.no_pad_xim,
            pad_theta_max_decade=args.pad_theta_max_decade,
            interp_order=args.interp_order,
            n_samples=args.n_samples,
            n_chunks=args.n_chunks,
            seed=args.seed,
        )
        replayed = np.diag(replayed_cov)[rows]
        replayed_error = (replayed - stored) / np.maximum(np.abs(stored), 1e-300)

    mode = (
        "local finite-difference Jacobian"
        if mean_fine is not None
        else "zero-baseline unit secants"
    )
    print(f"\n{mode} covariance comparison:")
    if mean_fine is None:
        print(
            "NOTE: Akima interpolation is data-dependent; this zero-baseline "
            "secant is not an exact MC covariance reconstruction."
        )
    print(
        "row  output/bin/theta                         J Cfine J.T          "
        "stored             rel.err"
    )
    for row, label, prediction, reference, error in zip(
        rows, labels, linearized, stored, linearized_error
    ):
        print(
            f"{row:3d}  {label:<42} {prediction: .8e}  {reference: .8e}  {error: .3e}"
        )
    if replayed is not None:
        print("\nExact production replay comparison:")
        print(
            "row  output/bin/theta                         replayed             "
            "stored             rel.err"
        )
        for row, label, prediction, reference, error in zip(
            rows, labels, replayed, stored, replayed_error
        ):
            print(
                f"{row:3d}  {label:<42} {prediction: .8e}  "
                f"{reference: .8e}  {error: .3e}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_weights(theta_int, rows, labels, weights, out_path)
    weights_path = out_path.with_suffix(".npz")
    output = dict(
        theta=theta,
        theta_int=theta_int,
        reporting_edges=edges,
        fine_to_reporting=reporting_matrix.toarray(),
        fine_bin_indices=bin_indices,
        output_rows=rows,
        output_labels=np.asarray(labels),
        weights=weights,
        linearized_diagonal=linearized,
        stored_diagonal=stored,
        linearized_relative_error=linearized_error,
        weight_mode=mode,
        pad_xim=not args.no_pad_xim,
        pad_theta_max_decade=args.pad_theta_max_decade,
        interp_order=args.interp_order,
    )
    if base_output is not None:
        output["baseline_output"] = base_output
    if replayed is not None:
        output.update(
            replayed_diagonal=replayed,
            replayed_relative_error=replayed_error,
            n_samples=args.n_samples,
            n_chunks=args.n_chunks,
            seed=args.seed,
        )
    np.savez(weights_path, **output)
    print(f"Saved weight plot: {out_path}")
    print(f"Saved weights and covariance check: {weights_path}")


def _parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pure-cov", type=Path, required=True)
    parser.add_argument("--fine-cov", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("pure_eb_cov_spike_diagnostic.png")
    )
    parser.add_argument("--min-sep", type=float, default=1.0)
    parser.add_argument("--max-sep", type=float, default=250.0)
    parser.add_argument("--neighbor-radius", type=int, default=1)
    parser.add_argument(
        "--all-b-rows",
        action="store_true",
        help="Extract weight rows for ALL xip_B and xim_B reporting bins.",
    )
    parser.add_argument(
        "--mean-fine",
        type=Path,
        help="Production 2N fine mean (a .npy or text vector).",
    )
    parser.add_argument(
        "--finite-difference-step",
        type=float,
        default=1e-8,
        help="Central-difference step used with --mean-fine.",
    )
    parser.add_argument(
        "--replay-production",
        action="store_true",
        help="Replay the 2000 seeded production draws; requires --mean-fine.",
    )
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--n-chunks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pad-xim",
        action="store_true",
        help="Disable cosmo_numba's default one-decade xi- padding.",
    )
    parser.add_argument("--pad-theta-max-decade", type=float, default=1.0)
    parser.add_argument("--interp-order", type=int, default=5)
    return parser.parse_args(argv)


if __name__ == "__main__":
    diagnose(_parser())
