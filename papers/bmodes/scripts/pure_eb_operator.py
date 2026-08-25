"""Shared pure E/B operator: the Schneider-2022 transform as the paper applies it.

Both the MC covariance chunks (``precompute_pure_eb_chunk.py``) and the point
estimate (``gather_pure_eb_chunks.py``) must push ξ± through the *same*
operator, or the data vector and its covariance describe different statistics.
This module is that operator.

Two modes:

``transform_averaged=False`` (legacy, pointwise)
    Evaluate the transform at the 20 reporting ``meanr`` values, integrating
    over the fine grid. Produces the per-bin σ spikes documented in
    sp_validation 72ad0efc.

``transform_averaged=True`` (fiducial)
    Evaluate on the full fine integration grid, then pair-count-weighted
    average each E/B/ambiguous block into the reporting bins, matching the
    bin-averaged character of the estimator itself. Non-finite fine values are
    dropped with row renormalization; an emptied reporting bin is an error.
    Integration limits are the fine-grid ``meanr`` extent rather than the
    reporting edges: fine bins abutting the reporting limits leave fewer than
    ``interp_order + 1`` quadrature nodes, where the degree-5 spline is
    degenerate and silently returns ~1e160 garbage that passes a finite check.
    Note this defines the pure modes over the wider separation interval (a
    different ambiguous-mode content than the pointwise operator).
"""

import numpy as np
from scipy import sparse

EB_KEYS = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")


def load_xi(path, min_sep, max_sep, nbins, require_npairs=False):
    """Load ξ± from a TreeCorr text dump and recompute the log bin edges."""
    data = np.loadtxt(path, comments="#", max_rows=nbins)
    if data.shape[0] != nbins:
        raise ValueError(f"Expected {nbins} ξ rows in {path}, found {data.shape[0]}")
    if require_npairs and data.shape[1] <= 10:
        raise ValueError(f"Expected an npairs column at index 10 in {path}")
    bin_edges = np.logspace(np.log10(min_sep), np.log10(max_sep), nbins + 1)
    result = {
        "meanr": np.asarray(data[:, 1], dtype=float),
        "xip": np.asarray(data[:, 3], dtype=float),
        "xim": np.asarray(data[:, 4], dtype=float),
        "left_edges": bin_edges[:-1],
        "right_edges": bin_edges[1:],
    }
    if data.shape[1] > 10:
        result["npairs"] = np.asarray(data[:, 10], dtype=float)
    return result


def make_binning_matrix(reporting, integration, npairs=None):
    """Fine → reporting averaging matrix, pair-count weighted when npairs given."""
    n_int = len(integration["meanr"])
    if npairs is None:
        npairs = np.ones(n_int)
    npairs = np.asarray(npairs, dtype=float)
    if npairs.shape != integration["meanr"].shape:
        raise ValueError("npairs must have one entry per integration bin")
    if not np.all(np.isfinite(npairs)) or np.any(npairs < 0):
        raise ValueError("npairs must be finite and non-negative")

    reporting_edges = np.concatenate(
        [reporting["left_edges"], [reporting["right_edges"][-1]]]
    )
    bin_indices = np.digitize(integration["meanr"], reporting_edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < len(reporting["meanr"])) & (npairs > 0)
    row_indices = bin_indices[valid]
    col_indices = np.where(valid)[0]
    matrix = sparse.csr_matrix(
        (npairs[valid], (row_indices, col_indices)),
        shape=(len(reporting["meanr"]), n_int),
    )
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    if np.any(row_sums == 0):
        raise ValueError("Empty reporting bin in the fine → reporting binning matrix")
    return sparse.diags(1.0 / row_sums) @ matrix


def average_modes(modes, binning_matrix, fine_indices):
    """Average finite fine-grid modes into reporting bins, per block."""
    matrix = binning_matrix[:, fine_indices]
    averaged = []
    dropped = []
    for block_name, values in zip(EB_KEYS, modes):
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        dropped.append(int(np.count_nonzero(~finite)))
        block_matrix = matrix[:, finite]
        block_values = values[finite]
        row_sums = np.asarray(block_matrix.sum(axis=1)).ravel()
        if np.any(row_sums == 0):
            empty = np.flatnonzero(row_sums == 0).tolist()
            raise ValueError(
                f"All finite fine bins were dropped for {block_name} in "
                f"reporting bins {empty}"
            )
        block_matrix = sparse.diags(1.0 / row_sums) @ block_matrix
        averaged.append(np.asarray(block_matrix @ block_values).ravel())
    return tuple(averaged), np.asarray(dropped, dtype=int)


def fine_support(binning_matrix):
    """Indices of fine bins that carry weight into some reporting bin."""
    return np.flatnonzero(np.asarray(binning_matrix.sum(axis=0)).ravel() > 0)


def averaged_limits(theta_int):
    """Integration limits for the averaged operator: the fine-grid extent."""
    return float(np.min(theta_int)) * (1 - 1e-9), float(np.max(theta_int)) * (1 + 1e-9)


def pure_eb_modes(
    theta_rep,
    xip_rep,
    xim_rep,
    theta_int,
    xip_int,
    xim_int,
    tmin,
    tmax,
    transform_averaged,
    binning_matrix=None,
    fine_indices=None,
    **kwargs,
):
    """Return (xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb), dropped-counts."""
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    if not transform_averaged:
        modes = get_pure_EB_modes(
            theta=theta_rep,
            xip=xip_rep,
            xim=xim_rep,
            theta_int=theta_int,
            xip_int=xip_int,
            xim_int=xim_int,
            tmin=tmin,
            tmax=tmax,
            **kwargs,
        )
        return modes, np.zeros(len(EB_KEYS), dtype=int)

    if binning_matrix is None or fine_indices is None:
        raise ValueError("transform_averaged needs binning_matrix and fine_indices")

    tmin_avg, tmax_avg = averaged_limits(theta_int)
    modes = get_pure_EB_modes(
        theta=theta_int[fine_indices],
        xip=np.asarray(xip_int)[fine_indices],
        xim=np.asarray(xim_int)[fine_indices],
        theta_int=theta_int,
        xip_int=xip_int,
        xim_int=xim_int,
        tmin=tmin_avg,
        tmax=tmax_avg,
        parallel=True,
        **kwargs,
    )
    return average_modes(modes, binning_matrix, fine_indices)


def print_drop_counts(label, counts):
    details = ", ".join(f"{name}={count}" for name, count in zip(EB_KEYS, counts))
    print(f"Dropped non-finite {label}: {details}", flush=True)
