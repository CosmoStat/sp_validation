"""
B-mode analysis functions for weak lensing validation.

Pure E/B modes (Schneider et al. 2022) as a fixed linear operator on the fine
ξ± grid with their exact covariance, COSEBIs, and the χ²/PTE and plotting
helpers shared by both.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

_EB_KEYS = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")


def find_conservative_scale_cut_key(results, requested_scale_cut):
    """
    Find scale cut key that conservatively fits within requested range.

    Parameters
    ----------
    results : dict
        COSEBIs results dictionary with scale cut tuples as keys
    requested_scale_cut : tuple
        (min_theta, max_theta) requested by user

    Returns
    -------
    tuple
        Best matching scale cut key from results
    """
    min_req, max_req = requested_scale_cut

    # Find all keys that fit conservatively within the requested range
    valid_keys = [
        key for key in results.keys() if key[0] >= min_req and key[1] <= max_req
    ]

    if valid_keys:
        # Choose the largest scale cut (widest range) among valid ones
        return max(valid_keys, key=lambda k: k[1] - k[0])

    # If no conservative match, find closest by total distance
    return min(results.keys(), key=lambda k: abs(k[0] - min_req) + abs(k[1] - max_req))


def scale_cut_to_bins(gg, min_scale=None, max_scale=None):
    """
    Convert conservative scale cuts to bin indices.

    Conservative approach: excludes ANY bin with partial overlap with forbidden region.
    For example, with a bin [1,3] arcmin and cuts at 2 arcmin:
    - Both lower and upper cuts at 2 would exclude this bin

    Parameters
    ----------
    gg : treecorr.GGCorrelation
        Correlation function object with bin edges
    min_scale : float, optional
        Minimum angular scale (lower cut) - exclude bins extending below this
    max_scale : float, optional
        Maximum angular scale (upper cut) - exclude bins extending above this

    Returns
    -------
    start_bin : int
        First included bin index
    stop_bin : int
        Last included bin index + 1 (for slicing notation)
    """
    return bins_from_edges(gg.left_edges, gg.right_edges, min_scale, max_scale)


def bins_from_edges(left_edges, right_edges, min_scale=None, max_scale=None):
    """:func:`scale_cut_to_bins` on bare bin edges."""
    start_bin = (
        np.searchsorted(left_edges, min_scale, side="left")
        if min_scale is not None
        else 0
    )
    stop_bin = (
        np.searchsorted(right_edges, max_scale, side="right")
        if max_scale is not None
        else len(right_edges)
    )
    return start_bin, stop_bin


def bins_from_scale_cut(left_edges, right_edges, scale_cut):
    """Reporting bins inside a pure-E/B scale cut, as ``(start_bin, stop_bin)``.

    Each end of ``scale_cut`` snaps to the nearest reporting edge in log θ —
    the rule that snaps reporting edges onto the fine grid — so a cut placed
    on a nominal edge selects the same bins whichever fine grid the edges
    were snapped to. ``stop_bin`` is exclusive.
    """
    edges = np.append(left_edges, right_edges[-1])
    start_bin, stop_bin = (
        int(np.abs(np.log(edges) - np.log(cut)).argmin()) for cut in scale_cut
    )
    if stop_bin <= start_bin:
        raise RuntimeError(f"scale cut {scale_cut} selects no bins")
    return start_bin, stop_bin


def log_bin_edges(min_sep, max_sep, nbins):
    """TreeCorr ``Log`` bin edges — the grid a binning defines.

    SACC ξ± parts store bin centres, not edges, so a consumer working from a
    part reconstructs the edges from the binning it was measured on.
    """
    edges = np.geomspace(float(min_sep), float(max_sep), int(nbins) + 1)
    return edges[:-1], edges[1:]


def correlation_from_covariance(covariance):
    """
    Convert covariance matrix to correlation matrix.

    Parameters
    ----------
    covariance : numpy.ndarray
        Covariance matrix

    Returns
    -------
    numpy.ndarray
        Correlation matrix
    """
    stdev = np.sqrt(np.diag(covariance))
    return covariance / np.outer(stdev, stdev)


def hartlap_factor(npatch, dof):
    """Hartlap (2007) debiasing of an inverse covariance.

    ``(N - p - 2) / (N - 1)`` for a jackknife covariance from ``N = npatch``
    patches inverted over ``p = dof`` data points; exactly 1 for an analytic
    covariance, which ``npatch=None`` denotes.
    """
    return 1.0 if npatch is None else (npatch - dof - 2) / (npatch - 1)


def _reporting_binning(weight_int, edges_int, reporting_edges):
    """Reporting bins as unions of fine bins, and their pair-weighted average.

    Each requested reporting edge snaps to the nearest fine edge (in log θ), so
    every reporting bin is a whole number of fine bins. Row ``i`` of the
    ``(n_report, n_fine)`` matrix weights the fine bins of reporting bin ``i``
    by their TreeCorr pair weight ``Σ w_i w_j`` and sums to one. TreeCorr's ξ±
    and ``meanr`` are averages over pairs under that same weight, so each row
    reproduces what TreeCorr would measure on the snapped bin.

    Returns ``(binning, edges)``, ``edges`` being the snapped reporting edges.
    """
    weight_int = np.asarray(weight_int, dtype=float)
    edges_int = np.asarray(edges_int, dtype=float)
    reporting_edges = np.asarray(reporting_edges, dtype=float)
    if edges_int.shape != (weight_int.size + 1,):
        raise ValueError("edges_int must hold n_fine + 1 fine-grid edges")
    if reporting_edges.min() < edges_int[0] or reporting_edges.max() > edges_int[-1]:
        raise ValueError(
            f"reporting edges [{reporting_edges.min()}, {reporting_edges.max()}] "
            f"reach outside the fine grid [{edges_int[0]}, {edges_int[-1]}]"
        )
    snap = np.abs(np.log(reporting_edges)[:, None] - np.log(edges_int)).argmin(axis=1)
    if np.any(np.diff(snap) <= 0):
        raise ValueError(
            "reporting edges snap onto the same fine edge; the fine grid is too "
            f"coarse for them: {reporting_edges.tolist()} -> "
            f"{edges_int[snap].tolist()}"
        )
    binning = np.zeros((snap.size - 1, weight_int.size))
    for i, (lo, hi) in enumerate(zip(snap[:-1], snap[1:])):
        binning[i, lo:hi] = weight_int[lo:hi]
    weight = binning.sum(axis=1)
    if np.any(weight == 0):
        empty = np.flatnonzero(weight == 0).tolist()
        raise ValueError(f"reporting bins {empty} hold no integration-grid pairs")
    return binning / weight[:, None], edges_int[snap]


def _fixed_quadrature_operator(theta_int):
    """The Schneider (2022) transform at every fine node, ``_EB_KEYS`` order.

    The single call into cosmo_numba. Evaluating at every node of the fine
    grid, with ``[tmin, tmax]`` at its extent, makes the transform a function
    of the fine grid alone: the zero-padded ξ− window is extended from the
    evaluation grid. Returns the ``(6, n_fine, 2 * n_fine)`` stack of the six
    matrices acting on ``[xi_+; xi_-]``; rows whose integration support is too
    small near the grid edges are NaN.
    """
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_operator

    return np.stack(
        get_pure_EB_operator(
            theta_int,
            theta_int,
            tmin=theta_int[0] * (1 - 1e-9),
            tmax=theta_int[-1] * (1 + 1e-9),
            local_from_int=True,
        )
    )


def pure_eb_operator(theta_int, weight_int, edges_int, reporting_edges):
    """The pure-E/B estimator as one matrix on the fine ξ± grid.

    The Schneider et al. (2022) transform is evaluated with fixed-quadrature
    weights at the fine-grid nodes, integrating over the whole fine grid, and
    the six pure modes are then averaged into the reporting bins with TreeCorr
    pair weights. The reporting edges snap to the nearest fine edges, so each
    reporting bin is a union of fine bins. Both steps are linear and
    independent of the ξ± values, so the estimator is
    ``K = (I_6 ⊗ P) · M`` and

        [xip_E; xim_E; xip_B; xim_B; xip_amb; xim_amb] = K @ [xip_int; xim_int]

    with ``K`` of shape ``(6 * n_report, 2 * n_fine)``.

    Parameters
    ----------
    theta_int : array_like
        Fine (integration) grid, ascending and log-spaced — TreeCorr ``meanr``.
        The transform's ``[tmin, tmax]`` is its extent, so it must reach
        beyond the reporting range on both sides.
    weight_int : array_like
        TreeCorr pair weight ``Σ w_i w_j`` per fine bin (``gg.weight``), the
        averaging weights.
    edges_int : array_like
        The ``n_fine + 1`` fine-grid bin edges.
    reporting_edges : array_like
        Requested reporting-bin edges, ``n_report + 1`` of them.

    Returns
    -------
    operator : numpy.ndarray
        ``K``, shape ``(6 * n_report, 2 * n_fine)``.
    binning : numpy.ndarray
        ``P``, the ``(n_report, n_fine)`` pair-weighted average.
    edges : numpy.ndarray
        The reporting edges actually used, each a fine-grid edge.
    """
    theta_int = np.asarray(theta_int, dtype=float)
    if np.shape(weight_int) != theta_int.shape:
        raise ValueError("weight_int must have one entry per integration bin")
    binning, edges = _reporting_binning(weight_int, edges_int, reporting_edges)
    nodes = np.flatnonzero(binning.any(axis=0))
    # Only the rows P averages enter; NaN edge rows outside them are dropped,
    # since 0 · NaN would still poison the product.
    transform = _fixed_quadrature_operator(theta_int)[:, nodes]
    undetermined = {
        key: int(np.count_nonzero(~np.isfinite(rows).all(axis=1)))
        for key, rows in zip(_EB_KEYS, transform)
        if not np.isfinite(rows).all()
    }
    if undetermined:
        raise ValueError(
            "pure-E/B operator rows are under-determined (reporting bins too "
            f"close to the integration-grid edge): {undetermined}"
        )
    operator = np.vstack([binning[:, nodes] @ rows for rows in transform])
    return operator, binning, edges


def calculate_pure_eb_correlation(
    theta_int,
    xip_int,
    xim_int,
    weight_int,
    edges_int,
    cov_xi,
    reporting_edges,
    *,
    npatch=None,
):
    """Pure E/B modes and their exact covariance from fine-grid ξ±.

    The modes are :func:`pure_eb_operator` applied to ``[xip_int; xim_int]``,
    and the covariance is ``K C_xi Kᵀ`` — exact for whatever ξ± covariance is
    supplied, analytic or jackknife. ``npatch`` records which: the jackknife
    patch count behind ``cov_xi``, or ``None`` for an analytic covariance. It
    travels in the results and sets the Hartlap factor of every χ² built on
    them (:func:`hartlap_factor`).

    The reporting bins are unions of fine bins (the requested edges snap to
    the nearest fine edges), and their ``theta``, ``xip``/``xim`` and
    variances are the same pair-weighted average of the fine grid, so
    ``xi_± = E ± B + amb`` holds bin by bin and ``theta``/``xip``/``xim`` are
    what TreeCorr would measure on the snapped bins.

    Parameters
    ----------
    theta_int, xip_int, xim_int, weight_int : array_like
        Fine-grid ``meanr``, ξ±, and TreeCorr pair weight ``Σ w_i w_j``.
    edges_int : array_like
        The ``n_fine + 1`` fine-grid bin edges.
    cov_xi : array_like
        ``(2 n_fine, 2 n_fine)`` covariance of ``[xip_int; xim_int]``.
    reporting_edges : array_like
        Requested reporting-bin edges, ``n_report + 1`` of them.
    npatch : int, optional
        Jackknife patch count behind ``cov_xi``; ``None`` if it is analytic.

    Returns
    -------
    dict
        The six ``_EB_KEYS`` mode arrays, ``cov`` (in ``_EB_KEYS`` block
        order), ``npatch``, the reporting grid (``theta``, the snapped
        ``left_edges``/``right_edges``, ``xip``, ``xim``, ``var_xip``,
        ``var_xim``) and the fine-grid inputs (``theta_int``, ``xip_int``,
        ``xim_int``, ``weight_int``, ``edges_int``).
    """
    if npatch is not None and npatch < 2:
        raise ValueError(f"a jackknife covariance needs npatch > 1, not {npatch}")
    theta_int, xip_int, xim_int = (
        np.asarray(a, dtype=float) for a in (theta_int, xip_int, xim_int)
    )
    cov_xi = np.asarray(cov_xi, dtype=float)
    operator, binning, edges = pure_eb_operator(
        theta_int, weight_int, edges_int, reporting_edges
    )
    if cov_xi.shape != (operator.shape[1],) * 2:
        raise ValueError(
            f"cov_xi has shape {cov_xi.shape}; the fine grid needs "
            f"{(operator.shape[1],) * 2}"
        )

    n_report = len(edges) - 1
    modes = operator @ np.concatenate([xip_int, xim_int])
    n_fine = theta_int.size
    var_xip, var_xim = (
        np.einsum("ij,jk,ik->i", binning, block, binning)
        for block in (cov_xi[:n_fine, :n_fine], cov_xi[n_fine:, n_fine:])
    )
    results = {
        "theta": binning @ theta_int,
        "left_edges": edges[:-1],
        "right_edges": edges[1:],
        "xip": binning @ xip_int,
        "xim": binning @ xim_int,
        "var_xip": var_xip,
        "var_xim": var_xim,
        "theta_int": theta_int,
        "xip_int": xip_int,
        "xim_int": xim_int,
        "weight_int": np.asarray(weight_int, dtype=float),
        "edges_int": np.asarray(edges_int, dtype=float),
        "cov": operator @ cov_xi @ operator.T,
        "npatch": npatch,
    }
    for i, key in enumerate(_EB_KEYS):
        results[key] = modes[i * n_report : (i + 1) * n_report]

    # The B-mode block is what every χ² inverts; the ambiguous blocks are
    # ill-conditioned by construction.
    b_block = slice(2 * n_report, 4 * n_report)
    try:
        np.linalg.cholesky(results["cov"][b_block, b_block])
    except np.linalg.LinAlgError:
        warnings.warn(
            "B-mode covariance is not positive definite. "
            "Chi-squared statistics may be unreliable.",
            UserWarning,
        )

    return results


def covariance_label(npatch):
    """How a pure-E/B covariance was made, from its ``npatch`` record."""
    return "analytic" if npatch is None else f"jackknife ({npatch} patches)"


def calculate_cosebis(gg, nmodes=10, scale_cuts=None, cov_path=None):
    """
    Calculate COSEBIs modes from a correlation function for multiple scale cuts.

    Parameters
    ----------
    gg : treecorr.GGCorrelation
        Fine-binned correlation function object for COSEBIs calculation
    nmodes : int, optional
        Number of COSEBIs modes to compute
    scale_cuts : list of tuples, optional
        List of (min_theta, max_theta) scale cuts to apply. If None, uses full range
        as a single scale cut.
    cov_path : str, optional
        Path to theoretical covariance matrix (ξ±) which will be transformed to COSEBIs
        space

    Returns
    -------
    dict
        Dictionary with scale cut tuples as keys and results dictionaries as values.
        Each results dictionary contains 'En', 'Bn', 'cov', 'chi2_E', 'chi2_B',
        'pte_B', 'scale_cut', and 'mask' entries.
    """
    cov_xipm = np.loadtxt(cov_path) if cov_path is not None else gg.cov
    return cosebis_scan_from_xi(
        gg.meanr,
        gg.xip,
        gg.xim,
        cov_xipm,
        gg.left_edges,
        gg.right_edges,
        nmodes=nmodes,
        scale_cuts=scale_cuts,
        # A theory covariance has no jackknife realisations to debias.
        npatch=None if cov_path is not None else gg.npatch1,
    )


def cosebis_scan_from_xi(
    theta,
    xip,
    xim,
    cov_xipm,
    left_edges,
    right_edges,
    *,
    nmodes=10,
    scale_cuts=None,
    npatch=None,
):
    """COSEBIs over a set of scale cuts, from ξ± arrays and their covariance.

    The values-and-covariance seam of :func:`calculate_cosebis`, for callers
    holding a ξ± data vector rather than a TreeCorr ``GGCorrelation``. The
    COSEBIs covariance is the ξ± covariance carried through the same linear
    kernel as the modes, so no estimator re-run is involved. ``npatch`` is the
    jackknife realisation count behind ``cov_xipm``, which sets the Hartlap
    debiasing; leave it ``None`` for a theory covariance, which needs none.

    Returns the ``{scale_cut: result}`` mapping :func:`calculate_cosebis` returns.
    """
    from cosmo_numba.B_modes.cosebis import COSEBIS

    theta, xip, xim = (np.asarray(a) for a in (theta, xip, xim))
    cov_xipm = np.asarray(cov_xipm)
    if scale_cuts is None:
        scale_cuts = [(left_edges[0], right_edges[-1])]
    nbins = len(theta)
    hartlap_factor = 1 if npatch is None else (npatch - 2 * nmodes - 2) / (npatch - 1)

    all_results = {}

    # Loop over each scale cut
    for scale_cut in tqdm.tqdm(scale_cuts, desc="COSEBIs scale cuts"):
        min_theta, max_theta = scale_cut

        start_bin, stop_bin = bins_from_edges(
            left_edges, right_edges, min_theta, max_theta
        )
        inds = np.arange(start_bin, stop_bin)

        theta_cut, xip_cut, xim_cut = [arr[inds] for arr in [theta, xip, xim]]

        # Calculate COSEBIs E/B modes using actual theta range (per Axel's recommendation)
        # Use precision=120 (vs default 80) to avoid sympy root convergence failures
        # for high modes (11+). The error "try n < 80 or maxsteps > 50" requires higher precision.
        cosebis = COSEBIS(
            theta_min=np.min(theta_cut),
            theta_max=np.max(theta_cut),
            N_max=nmodes,
            precision=120,
        )
        En, Bn = cosebis.cosebis_from_xipm(theta_cut, xip_cut, xim_cut, parallel=True)

        # Extract covariance and transform to COSEBIs space
        cov_inds = np.concatenate([inds, inds + nbins])
        cov_cosebis = cosebis.cosebis_covariance_from_xipm_covariance(
            theta_cut, cov_xipm[cov_inds[:, None], cov_inds]
        )
        cov_E, cov_B = cov_cosebis[:nmodes, :nmodes], cov_cosebis[nmodes:, nmodes:]
        chi2_E, chi2_B = [
            hartlap_factor * (modes @ np.linalg.solve(cov, modes))
            for modes, cov in [(En, cov_E), (Bn, cov_B)]
        ]

        results = {
            "En": En,
            "Bn": Bn,
            "cov": cov_cosebis,
            "hartlap_factor": hartlap_factor,
            "chi2_E": chi2_E,
            "chi2_B": chi2_B,
            "pte_B": 1 - stats.chi2.cdf(chi2_B, nmodes),
            "scale_cut": (min_theta, max_theta),
            "inds": inds,
        }

        print(f"COSEBIs results [{min_theta:.1f}-{max_theta:.1f} arcmin]:")
        print(
            f"  chi2(E) = {chi2_E:.2f}, chi2(B) = {chi2_B:.2f}, "
            f"PTE(B) = {results['pte_B']:.4f}"
        )

        all_results[scale_cut] = results

    return all_results


def calculate_eb_statistics(results):
    """
    Calculate E/B mode statistics using 2D PTE analysis for all scale cut combinations.

    This function processes pure E/B mode results, calculates covariance blocks,
    and performs comprehensive 2D PTE analysis. The traditional "1D PTE" values
    are simply extracted from the 2D matrices at the full-range position.

    Parameters
    ----------
    results : dict
        Pure E/B results: the six mode arrays, the ``cov`` block, the reporting
        ``theta``, and ``npatch`` — the jackknife patch count behind the
        covariance, or ``None`` for an analytic one — which sets the Hartlap
        factor (:func:`hartlap_factor`)

    Returns
    -------
    dict
        Updated results dictionary with PTE matrices and statistics
    """
    nbins = len(results["theta"])
    npatch = results["npatch"]

    # Extract covariance blocks and standard deviations
    cov = results["cov"]
    for i, key in enumerate(_EB_KEYS):
        start, end = nbins * i, nbins * (i + 1)
        cov_block = cov[start:end, start:end]
        results[f"cov_{key}"] = cov_block
        results[f"std_{key}"] = np.sqrt(np.diag(cov_block))

    # Initialize PTE matrices
    pte_xip_B = np.full((nbins, nbins), np.nan)
    pte_xim_B = np.full((nbins, nbins), np.nan)
    pte_combined = np.full((nbins, nbins), np.nan)

    # Generate all valid (start, stop) combinations for scale cuts
    combinations = [
        (start, stop) for start in range(nbins) for stop in range(start, nbins + 1)
    ]

    for start_bin, stop_bin in combinations:
        nbins_eff = stop_bin - start_bin
        hartlap = hartlap_factor(npatch, nbins_eff)

        # Individual B-mode chi-squared calculations
        data_slices = [results[f"{xi}_B"][start_bin:stop_bin] for xi in ["xip", "xim"]]
        cov_slices = [
            results[f"cov_{xi}_B"][start_bin:stop_bin, start_bin:stop_bin]
            for xi in ["xip", "xim"]
        ]
        chi2_values = [
            hartlap * (data @ np.linalg.solve(cov, data))
            for data, cov in zip(data_slices, cov_slices)
        ]

        pte_xip_B[start_bin, stop_bin - 1] = stats.chi2.sf(chi2_values[0], nbins_eff)
        pte_xim_B[start_bin, stop_bin - 1] = stats.chi2.sf(chi2_values[1], nbins_eff)

        # Combined calculation with cross-correlation
        data_combined = np.concatenate(data_slices)

        # Build combined covariance matrix blocks
        xip_slice = slice(2 * nbins + start_bin, 2 * nbins + stop_bin)
        xim_slice = slice(3 * nbins + start_bin, 3 * nbins + stop_bin)
        cov_xip_block = cov[xip_slice, xip_slice]
        cov_xim_block = cov[xim_slice, xim_slice]
        cov_cross_block = cov[xip_slice, xim_slice]
        cov_combined = np.block(
            [[cov_xip_block, cov_cross_block], [cov_cross_block.T, cov_xim_block]]
        )
        chi2_combined = hartlap_factor(npatch, 2 * nbins_eff) * (
            data_combined @ np.linalg.solve(cov_combined, data_combined)
        )
        pte_combined[start_bin, stop_bin - 1] = stats.chi2.sf(
            chi2_combined, 2 * nbins_eff
        )

    # Store PTE matrices
    pte_data = {"xip_B": pte_xip_B, "xim_B": pte_xim_B, "combined": pte_combined}
    results["pte_matrices"] = pte_data

    return results


def plot_integration_vs_reporting(results, output_path, version):
    """
    Plot integration vs reporting scale comparison.

    Parameters
    ----------
    results : dict
        Pure E/B results carrying the fine grid (``theta_int``/``xip_int``/
        ``xim_int``) and its pair-weighted average into the reporting bins
        (``theta``/``xip``/``xim``, with the ``var_xip``/``var_xim`` the error
        bars use)
    output_path : str
        Output file path for the plot
    version : str
        Version string for plot title
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Configure plot data for both xi+ and xi- in a consolidated loop
    plot_configs = [
        ("+", "xip", r"$\theta \xi_+(\theta) \times 10^4$"),
        ("-", "xim", r"$\theta \xi_-(\theta) \times 10^4$"),
    ]

    data_configs = [
        ("_int", "k.", 3, 0.3, "Integration"),
        ("", ".", 12, 1, "Reporting"),
    ]

    for ax_idx, (xi_label, xi_attr, ylabel) in enumerate(plot_configs):
        for suffix, fmt, ms, alpha, label_type in data_configs:
            theta = results[f"theta{suffix}"]
            xi_val = results[f"{xi_attr}{suffix}"]
            var = results.get(f"var_{xi_attr}{suffix}")
            yerr = theta * np.sqrt(var) / 1e-4 if var is not None else None
            axs[ax_idx].errorbar(
                theta,
                theta * xi_val / 1e-4,
                yerr=yerr,
                fmt=fmt,
                ms=ms,
                alpha=alpha,
                capsize=3,
                ls="" if label_type == "Reporting" else None,
                label=(
                    rf"$\xi_{{{xi_label}}}$, {label_type}: "
                    rf"${theta[0]:.2g} < \theta < {theta[-1]:.4g}$, "
                    rf"{len(theta)} bins"
                ),
            )
        axs[ax_idx].set(
            xlabel=r"$\theta$ [arcmin]",
            ylabel=ylabel,
            xscale="log",
            title=rf"$\xi_{{{xi_label}}}(\theta)$",
        )
        axs[ax_idx].legend()

    plt.suptitle(f"{version}: Integration vs. Reporting", fontsize=23, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def _get_pte_from_scale_cut(pte_matrix, edges, scale_cut):
    """
    Extract PTE value from matrix at a scale cut (:func:`bins_from_scale_cut`).

    Parameters
    ----------
    pte_matrix : numpy.ndarray
        2D PTE matrix
    edges : tuple of numpy.ndarray
        ``(left_edges, right_edges)`` of the grid the matrix is indexed on
    scale_cut : tuple
        (min_scale, max_scale) angular range for scale cut

    Returns
    -------
    float
        PTE value for the given scale cut, or full-range PTE if scale_cut is None
    """
    left_edges, right_edges = edges
    nbins = len(left_edges)

    if scale_cut is None:
        # Return full-range PTE (first row, last column)
        return pte_matrix[0, nbins - 1]

    start_bin, stop_bin = bins_from_scale_cut(left_edges, right_edges, scale_cut)
    return pte_matrix[start_bin, stop_bin - 1]


def plot_pure_eb_correlations(
    results,
    output_path,
    version,
    fiducial_xip_scale_cut=None,
    fiducial_xim_scale_cut=None,
):
    """
    Plot pure E/B mode correlation functions.

    Parameters
    ----------
    results : dict
        Results dictionary containing E/B mode data and statistics
    output_path : str
        Output file path for the plot
    version : str
        Version string for plot title
    fiducial_xip_scale_cut : tuple, optional
        (min_scale, max_scale) for xi+ fiducial analysis, shown as gray regions
    fiducial_xim_scale_cut : tuple, optional
        (min_scale, max_scale) for xi- fiducial analysis, shown as gray regions
    """
    theta = results["theta"]
    edges = (results["left_edges"], results["right_edges"])
    nbins = len(theta)
    cov = results["cov"]

    # Calculate combined PTE using off-diagonal covariance blocks
    # Get scale cuts for both xi+ and xi-
    if fiducial_xip_scale_cut is not None:
        xip_start_bin, xip_stop_bin = bins_from_scale_cut(
            *edges, fiducial_xip_scale_cut
        )
    else:
        xip_start_bin, xip_stop_bin = 0, nbins

    if fiducial_xim_scale_cut is not None:
        xim_start_bin, xim_stop_bin = bins_from_scale_cut(
            *edges, fiducial_xim_scale_cut
        )
    else:
        xim_start_bin, xim_stop_bin = 0, nbins

    # Extract ξ+^B and ξ-^B data vectors for the scale cut ranges
    xip_B_data = results["xip_B"][xip_start_bin:xip_stop_bin]
    xim_B_data = results["xim_B"][xim_start_bin:xim_stop_bin]
    data_combined = np.concatenate([xip_B_data, xim_B_data])

    # Build combined covariance matrix from covariance blocks
    # ξ+^B block (2*nbins:3*nbins, 2*nbins:3*nbins)
    # ξ-^B block (3*nbins:4*nbins, 3*nbins:4*nbins)
    # ξ+^B-ξ-^B cross (2*nbins:3*nbins, 3*nbins:4*nbins)
    cov_xip_B = cov[
        2 * nbins + xip_start_bin : 2 * nbins + xip_stop_bin,
        2 * nbins + xip_start_bin : 2 * nbins + xip_stop_bin,
    ]
    cov_xim_B = cov[
        3 * nbins + xim_start_bin : 3 * nbins + xim_stop_bin,
        3 * nbins + xim_start_bin : 3 * nbins + xim_stop_bin,
    ]
    cov_cross = cov[
        2 * nbins + xip_start_bin : 2 * nbins + xip_stop_bin,
        3 * nbins + xim_start_bin : 3 * nbins + xim_stop_bin,
    ]

    cov_combined = np.block([[cov_xip_B, cov_cross], [cov_cross.T, cov_xim_B]])

    # Calculate combined chi-squared with Hartlap factor
    nbins_eff = len(xip_B_data) + len(xim_B_data)
    chi2_combined = hartlap_factor(results["npatch"], nbins_eff) * (
        data_combined.T @ np.linalg.solve(cov_combined, data_combined)
    )
    combined_pte = stats.chi2.sf(chi2_combined, nbins_eff)

    # Extract PTE values for fiducial scale cuts (or full range)
    xip_B_pte = _get_pte_from_scale_cut(
        results["pte_matrices"]["xip_B"], edges, fiducial_xip_scale_cut
    )
    xim_B_pte = _get_pte_from_scale_cut(
        results["pte_matrices"]["xim_B"], edges, fiducial_xim_scale_cut
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    scale_factor = 1e-4

    # Main correlation functions and decomposed modes
    plot_configs = [
        (
            "xip",
            "+",
            "var_xip",
            r"$\xi_{+}=\xi_{+}^{E}+\xi_{+}^{B}+\xi_{+}^{\mathrm{amb}}$",
            xip_B_pte,
        ),
        (
            "xim",
            "-",
            "var_xim",
            r"$\xi_{-}=\xi_{-}^{E}-\xi_{-}^{B}+\xi_{-}^{\mathrm{amb}}$",
            xim_B_pte,
        ),
    ]

    for ax_idx, (xi_type, xi_label, var_attr, main_label, pte_val) in enumerate(
        plot_configs
    ):
        # Plot main correlation function
        xi_val = results[xi_type]
        axs[ax_idx].errorbar(
            theta,
            theta * xi_val / scale_factor,
            yerr=theta * np.sqrt(results[var_attr]) / scale_factor,
            fmt="k.",
            capsize=3,
            label=main_label,
        )

        # Plot E/B/amb modes
        plot_data = [
            (f"{xi_type}_E", "g", 0.25, rf"$\xi_{{{xi_label}}}^{{E}}$"),
            (
                f"{xi_type}_B",
                "r",
                1.0,
                rf"$\xi_{{{xi_label}}}^{{B}}, {{\rm PTE}}="
                rf"{np.round(pte_val, 4)}$",
            ),
            (
                f"{xi_type}_amb",
                "purple",
                0.25,
                rf"$\xi_{{{xi_label}}}^{{\mathrm{{amb}}}}$",
            ),
        ]

        for key, color, alpha, label in plot_data:
            axs[ax_idx].errorbar(
                theta,
                theta * results[key] / scale_factor,
                yerr=theta * results[f"std_{key}"] / scale_factor,
                color=color,
                ls="",
                marker=".",
                alpha=alpha,
                capsize=3,
                label=label,
            )

        axs[ax_idx].set_ylabel(r"$\theta\xi\times10^{4}$")
        axs[ax_idx].set_title(rf"$\xi_{{{xi_label}}}$")

    # Configure both axes
    for ax in axs:
        ax.set(xscale="log", xlabel=r"$\theta$ [arcmin]")
        ax.axhline(0, alpha=0.3, color="k", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper left")
        ax.set_ylim(-0.5, 2)

    # Save the axis limits after data plotting but before adding gray regions
    original_xlims = [ax.get_xlim() for ax in axs]

    # Add fiducial scale cuts if provided - show excluded regions in gray
    scale_cuts = [(fiducial_xip_scale_cut, 0), (fiducial_xim_scale_cut, 1)]
    for scale_cut, ax_idx in scale_cuts:
        if scale_cut is not None:
            xlim = original_xlims[ax_idx]

            # The same bins the PTEs use
            start_bin, stop_bin = bins_from_scale_cut(*edges, scale_cut)

            # Show excluded regions based on bin edges used in PTE calculation
            # Lower exclusion: bins 0 to start_bin-1 are excluded
            if start_bin > 0:
                lower_exclusion_edge = edges[1][start_bin - 1]
                axs[ax_idx].axvspan(
                    xlim[0],
                    lower_exclusion_edge,
                    alpha=0.2,
                    color="gray",
                    label="Scale cuts" if ax_idx == 0 else None,
                )

            # Upper exclusion: bins stop_bin to end are excluded
            if stop_bin < len(edges[0]):
                upper_exclusion_edge = edges[0][stop_bin]
                axs[ax_idx].axvspan(
                    upper_exclusion_edge,
                    xlim[1],
                    alpha=0.2,
                    color="gray",
                    label="Scale cuts" if ax_idx == 0 and start_bin == 0 else None,
                )

            # Restore original axis limits
            axs[ax_idx].set_xlim(xlim)

    plt.suptitle(
        f"{version}: Pure Correlation Functions, Combined PTE = {combined_pte:.4f}",
        fontsize=23,
        y=0.95,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_cosebis_scale_cut_heatmap(
    cosebis_results, edges, version, output_path, fiducial_scale_cut=None
):
    """
    Create 2D heatmaps showing how COSEBIs statistics vary across different scale cuts.

    Parameters
    ----------
    cosebis_results : dict
        Dictionary with scale cut tuples as keys, containing 'chi2_E' and 'pte_B' values
    edges : tuple of numpy.ndarray
        ``(left_edges, right_edges)`` of the grid the scale cuts index
    version : str
        Version string for main title
    output_path : str
        Output file path for the plot
    fiducial_scale_cut : tuple, optional
        (min_scale, max_scale) for cross-hatching
    """
    left_edges, right_edges = edges
    nbins = len(left_edges)

    # Initialize matrices
    snrs = [np.sqrt(result["chi2_E"]) for result in cosebis_results.values()]
    ptes = [result["pte_B"] for result in cosebis_results.values()]

    rows = [start for start in range(nbins) for stop in range(start, nbins)]
    columns = [stop for start in range(nbins) for stop in range(start, nbins)]

    snr_matrix = np.full((nbins, nbins), np.nan)
    pte_matrix = np.full((nbins, nbins), np.nan)
    for row, column, snr, pte in zip(rows, columns, snrs, ptes):
        snr_matrix[row, column] = snr
        pte_matrix[row, column] = pte

    # Fill matrices from COSEBIs results
    for i in range(len(left_edges)):
        for j in range(i, len(right_edges)):
            scale_cut = (left_edges[i], right_edges[j])
            result = cosebis_results.get(scale_cut)

            if result is not None:
                snr_matrix[i, j] = np.sqrt(result["chi2_E"])
                pte_matrix[i, j] = result["pte_B"]

    # Create 1x2 subplot layout
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Set up colormaps
    mako_cmap, vlag_cmap = [
        sns.color_palette(name, as_cmap=True).copy() for name in ["mako", "vlag"]
    ]
    for cmap in (mako_cmap, vlag_cmap):
        cmap.set_bad(color="lightgray")

    # Left plot: E-mode SNR
    snr_max = np.nanmax(snr_matrix) if not np.all(np.isnan(snr_matrix)) else 1
    snr_plot_data = snr_matrix.T
    im1 = axs[0].imshow(
        snr_plot_data,
        origin="lower",
        aspect="auto",
        cmap=mako_cmap,
        vmin=0,
        vmax=snr_max,
        extent=[0, nbins, 0, nbins],
    )
    axs[0].set_title(r"E-mode SNR: $\sqrt{\chi^2_E}$")
    axs[0].set_xlabel("Lower scale cut")
    axs[0].set_ylabel("Upper scale cut")

    # Right plot: B-mode PTE with contours
    pte_plot_data = pte_matrix.T
    im2 = axs[1].imshow(
        pte_plot_data,
        origin="lower",
        aspect="auto",
        cmap=vlag_cmap,
        vmin=0,
        vmax=1,
        extent=[0, nbins, 0, nbins],
    )

    # Add contours to PTE plot
    cs = axs[1].contour(
        pte_plot_data,
        levels=[0.05, 0.95],
        colors="black",
        linewidths=1,
        extent=[0, nbins, 0, nbins],
    )
    axs[1].clabel(cs, inline=True, fontsize=8)

    axs[1].set_title("B-mode PTE")
    axs[1].set_xlabel("Lower scale cut")

    # Add fiducial scale cut cross-hatching if provided
    if fiducial_scale_cut is not None:
        min_scale, max_scale = fiducial_scale_cut
        start_bin, stop_bin = bins_from_edges(
            left_edges, right_edges, min_scale, max_scale
        )

        if stop_bin > start_bin and start_bin < nbins and stop_bin > 0:
            # Add cross-hatching at the fiducial scale cut matrix element
            rect_x = start_bin
            rect_y = stop_bin - 1
            rect_width = 1
            rect_height = 1

            for ax_idx, ax in enumerate(axs):
                ax.add_patch(
                    plt.Rectangle(
                        (rect_x, rect_y),
                        rect_width,
                        rect_height,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                        hatch="///",
                        alpha=0.8,
                        label="Fiducial scale cut" if ax_idx == 0 else None,
                    )
                )

    # Set angular scale ticks
    tick_indices = np.arange(0, nbins)
    x_tick_labels = [
        f"{left_edges[i]:.1f}" for i in tick_indices if i < len(left_edges)
    ]
    y_tick_labels = [
        f"{right_edges[i]:.1f}" for i in tick_indices if i < len(right_edges)
    ]
    x_tick_positions = tick_indices + 0.5
    y_tick_positions = tick_indices + 0.5

    for ax in axs:
        ax.set_xticks(x_tick_positions[: len(x_tick_labels)])
        ax.set_xticklabels(x_tick_labels, rotation=45)
        ax.set_yticks(y_tick_positions[: len(y_tick_labels)])
        ax.set_yticklabels(y_tick_labels)
        ax.set_aspect("equal")

    # Add colorbars to both plots
    colorbar_data = [(im1, "SNR"), (im2, "PTE")]
    for i, (im, label) in enumerate(colorbar_data):
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(label, rotation=270, labelpad=15)

    # Add legend if fiducial scale cut shown
    if fiducial_scale_cut is not None:
        axs[0].legend(loc="lower right", fontsize=8)

    plt.suptitle(f"{version}: COSEBIs Scale Cut Analysis", fontsize=23, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_pte_2d_heatmaps(
    results,
    version,
    output_path,
    fiducial_xip_scale_cut=None,
    fiducial_xim_scale_cut=None,
):
    """
    Plot 2D PTE matrix heatmaps for B-mode analysis.

    Parameters
    ----------
    results : dict
        Results dictionary containing PTE matrices
    version : str
        Version string for plot title
    output_path : str
        Output file path for the plot
    fiducial_xip_scale_cut : tuple, optional
        (min_scale, max_scale) for xi+ fiducial analysis, shown as cross-hatched
    fiducial_xim_scale_cut : tuple, optional
        (min_scale, max_scale) for xi- fiducial analysis, shown as cross-hatched
    """
    theta = results["theta"]
    edges = (results["left_edges"], results["right_edges"])
    nbins = len(theta)
    pte_xip_B = results["pte_matrices"]["xip_B"]
    pte_xim_B = results["pte_matrices"]["xim_B"]

    # Create 1x2 subplot layout
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Set up colormap
    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color="lightgray")
    contour_levels = [0.05, 0.95]

    # Left plot: xi+ PTE
    xip_plot_data = pte_xip_B.T
    axs[0].imshow(
        xip_plot_data,
        origin="lower",
        aspect="auto",
        cmap=vlag_cmap,
        vmin=0,
        vmax=1,
        extent=[0, nbins, 0, nbins],
    )
    cs1 = axs[0].contour(
        xip_plot_data,
        levels=contour_levels,
        colors="black",
        linewidths=1,
        extent=[0, nbins, 0, nbins],
    )
    axs[0].clabel(cs1, inline=True, fontsize=8)
    axs[0].set_title(r"$\xi_+^B$ PTE")
    axs[0].set_xlabel("Lower scale cut")
    axs[0].set_ylabel("Upper scale cut")

    # Right plot: xi- PTE
    xim_plot_data = pte_xim_B.T
    im2 = axs[1].imshow(
        xim_plot_data,
        origin="lower",
        aspect="auto",
        cmap=vlag_cmap,
        vmin=0,
        vmax=1,
        extent=[0, nbins, 0, nbins],
    )
    cs2 = axs[1].contour(
        xim_plot_data,
        levels=contour_levels,
        colors="black",
        linewidths=1,
        extent=[0, nbins, 0, nbins],
    )
    axs[1].clabel(cs2, inline=True, fontsize=8)
    axs[1].set_title(r"$\xi_-^B$ PTE")
    axs[1].set_xlabel("Lower scale cut")

    # Add fiducial scale cuts individually
    fiducial_scale_cuts = [fiducial_xip_scale_cut, fiducial_xim_scale_cut]
    for ax_idx, fiducial_scale_cut in enumerate(fiducial_scale_cuts):
        if fiducial_scale_cut is not None:
            start_bin, stop_bin = bins_from_scale_cut(*edges, fiducial_scale_cut)

            if stop_bin > start_bin and start_bin < nbins and stop_bin > 0:
                rect_x = start_bin
                rect_y = stop_bin - 1
                rect_width = 1
                rect_height = 1

                axs[ax_idx].add_patch(
                    plt.Rectangle(
                        (rect_x, rect_y),
                        rect_width,
                        rect_height,
                        fill=False,
                        edgecolor="black",
                        linewidth=2,
                        hatch="///",
                        alpha=0.8,
                        label="Fiducial scale cut" if ax_idx == 0 else None,
                    )
                )

    # Set angular scale ticks
    tick_indices = np.arange(0, nbins)
    x_tick_labels = [f"{edges[0][i]:.1f}" for i in tick_indices if i < len(edges[0])]
    y_tick_labels = [f"{edges[1][i]:.1f}" for i in tick_indices if i < len(edges[1])]
    x_tick_positions = tick_indices + 0.5
    y_tick_positions = tick_indices + 0.5

    for ax in axs:
        ax.set_xticks(x_tick_positions[: len(x_tick_labels)])
        ax.set_xticklabels(x_tick_labels, rotation=45)
        ax.set_yticks(y_tick_positions[: len(y_tick_labels)])
        ax.set_yticklabels(y_tick_labels)
        ax.set_aspect("equal")

    # Add colorbar
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label("PTE", rotation=270, labelpad=15)

    # Add legend if any fiducial scale cuts shown
    if any(cut is not None for cut in fiducial_scale_cuts):
        axs[0].legend(loc="lower right", fontsize=8)

    plt.suptitle(f"{version}: PTEs as a Function of Scale Cut", fontsize=23, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_eb_covariance_matrix(cov_matrix, label, output_path, version):
    """
    Plot E/B mode covariance matrix as correlation matrix.

    Parameters
    ----------
    cov_matrix : numpy.ndarray
        Covariance matrix from E/B mode analysis
    label : str
        How the covariance was made (:func:`covariance_label`)
    output_path : str
        Output file path for the plot
    version : str
        Version string for plot title
    """
    nbins = cov_matrix.shape[0] // 6

    fig, ax = plt.subplots(figsize=(9, 9))
    vlag_cmap = sns.color_palette("vlag", as_cmap=True)
    im = ax.matshow(correlation_from_covariance(cov_matrix), cmap=vlag_cmap)

    # Configure tick labels for E/B modes
    tick_positions = np.arange(nbins / 2, nbins * 6, nbins)
    tick_labels = [
        r"$\xi_+^{E}$",
        r"$\xi_-^{E}$",
        r"$\xi_+^{B}$",
        r"$\xi_-^{B}$",
        r"$\xi_+^{\rm amb}$",
        r"$\xi_-^{\rm amb}$",
    ]

    for ticks in (plt.xticks, plt.yticks):
        ticks(tick_positions, tick_labels)
        ticks(np.arange(0, nbins * 6 + 1, 20), minor=True)

    ax.tick_params(axis="both", which="major", length=0)
    im.set_clim(-1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax.set_title(f"{version}: {label} correlation matrix")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_cosebis_modes(results, version, output_path, fiducial_scale_cut=None):
    """
    Plot COSEBIs E/B mode correlation functions.

    Parameters
    ----------
    results : dict
        COSEBIs results dictionary containing E/B mode data
    version : str
        Version string for plot title
    output_path : str
        Output file path for the plot
    fiducial_scale_cut : tuple, optional
        (min_scale, max_scale) for fiducial analysis, shown for reference
    """
    plt.figure(figsize=(9, 9))

    # Create mode numbers array
    nmodes = len(results["En"])
    modes = np.arange(1, nmodes + 1)

    # Plot E-modes
    plt.errorbar(
        modes,
        results["En"],
        yerr=np.sqrt(np.diag(results["cov"][:nmodes, :nmodes])),
        label=f"COSEBIs E-modes; SNR = {np.sqrt(results['chi2_E']):.2f}",
    )

    # Plot B-modes
    plt.errorbar(
        modes,
        results["Bn"],
        yerr=np.sqrt(np.diag(results["cov"][nmodes:, nmodes:])),
        c="crimson",
        label=f"COSEBIs B-modes; PTE = {results['pte_B']:.3f}",
    )

    plt.axhline(0, ls="--", color="k")
    plt.legend()
    plt.xlabel("n (mode)")
    plt.ylabel("E_n, B_n")
    plt.ylim(-0.5e-10, 3e-10)

    # Add scale cut information to title - use actual scale cut from results
    scale_info = ""
    if "scale_cut" in results:
        actual_scale_cut = results["scale_cut"]
        scale_info = (
            f", scale cut: ({actual_scale_cut[0]:.1f}, {actual_scale_cut[1]:.1f})"
        )
    elif fiducial_scale_cut is not None:
        scale_info = f", scale cut: {fiducial_scale_cut}"

    plt.title(f"{version} COSEBIs E/B-modes{scale_info}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def save_pure_eb_results(results, output_path):
    """
    Save pure E/B mode data vectors and covariance to .npz.

    Parameters
    ----------
    results : dict
        Results dictionary from calculate_eb_statistics
    output_path : str
        Output .npz file path
    """
    # Data vectors and covariance
    save_dict = {
        key: results[key] for key in ("theta", "left_edges", "right_edges", "cov")
    }
    for key in _EB_KEYS:
        save_dict[key] = results[key]

    # PTE matrices
    for key, matrix in results.get("pte_matrices", {}).items():
        save_dict[f"pte_matrices_{key}"] = matrix

    # The jackknife patch count, stored only for a jackknife covariance.
    if results["npatch"] is not None:
        save_dict["npatch"] = np.array(results["npatch"])

    np.savez(output_path, **save_dict)
    print(f"Saved pure E/B results to {output_path}")


def _cosebis_result_to_dict(r, suffix=""):
    """Build save dict entries from a single COSEBIs result."""
    return {
        f"En{suffix}": r["En"],
        f"Bn{suffix}": r["Bn"],
        f"cov{suffix}": r["cov"],
        f"chi2_E{suffix}": np.array(r["chi2_E"]),
        f"chi2_B{suffix}": np.array(r["chi2_B"]),
        f"pte_B{suffix}": np.array(r["pte_B"]),
    }


def save_cosebis_results(results, output_path, fiducial_scale_cut=None):
    """
    Save COSEBIs data vectors and covariance to .npz.

    Parameters
    ----------
    results : dict
        COSEBIs results. Either a single-scale-cut dict with keys like
        'En', 'Bn', 'cov', or a multi-scale-cut dict with tuple keys.
    output_path : str
        Output .npz file path
    fiducial_scale_cut : tuple, optional
        If results has multiple scale cuts, save only this one.
        If None and results has multiple scale cuts, saves all.
    """
    is_multi = all(isinstance(k, tuple) for k in results)

    if is_multi and fiducial_scale_cut is not None:
        # Select the single best-matching scale cut
        key = find_conservative_scale_cut_key(results, fiducial_scale_cut)
        save_dict = _cosebis_result_to_dict(results[key])
        save_dict["scale_cut"] = np.array(key)
    elif is_multi:
        # Save all scale cuts with tagged keys
        save_dict = {}
        for sc, r in results.items():
            save_dict.update(_cosebis_result_to_dict(r, f"_{sc[0]}_{sc[1]}"))
        save_dict["scale_cuts"] = np.array(list(results.keys()))
    else:
        # Single scale cut
        save_dict = _cosebis_result_to_dict(results)
        if "scale_cut" in results:
            save_dict["scale_cut"] = np.array(results["scale_cut"])

    np.savez(output_path, **save_dict)
    print(f"Saved COSEBIs results to {output_path}")


def plot_cosebis_covariance_matrix(results, version, var_method, output_path):
    """
    Plot COSEBIs covariance matrix as correlation matrix.

    Parameters
    ----------
    results : dict
        COSEBIs results dictionary containing covariance matrix
    version : str
        Version string for plot title
    var_method : str
        Variance method used for the analysis
    output_path : str
        Output file path for the plot
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    # Convert to correlation matrix
    cov_EB = results["cov"]
    corr_EB = correlation_from_covariance(cov_EB)

    im = ax.imshow(corr_EB, cmap=sns.color_palette("vlag", as_cmap=True))

    nmodes = len(results["En"])
    for ticks in (plt.xticks, plt.yticks):
        ticks(
            np.arange(nmodes / 2, nmodes * 2, nmodes),
            ["E_n", "B_n"],
        )
        ticks(np.arange(0, nmodes * 2 + 1, nmodes), minor=True)

    clim = np.max(np.abs(corr_EB))
    im.set_clim(-clim, clim)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    ax.set_title(f"{version} COSEBIs E/B {var_method} correlation matrix")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
