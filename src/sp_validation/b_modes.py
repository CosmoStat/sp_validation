"""
B-mode analysis functions for weak lensing validation.

This module contains pure E/B mode decomposition, COSEBIs analysis,
and semi-analytical covariance calculations extracted from CosmologyValidation.
"""


import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import treecorr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import sparse, stats

from .cosmology import get_theo_xi


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
        key for key in results.keys()
        if key[0] >= min_req and key[1] <= max_req
    ]

    if valid_keys:
        # Choose the largest scale cut (widest range) among valid ones
        return max(valid_keys, key=lambda k: k[1] - k[0])

    # If no conservative match, find closest by total distance
    return min(results.keys(),
              key=lambda k: abs(k[0] - min_req) + abs(k[1] - max_req))


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
    nbins = len(gg.meanr)

    if min_scale is not None:
        # Conservative: exclude bins whose left edge is below min_scale
        start_bin = np.searchsorted(gg.left_edges, min_scale, side='left')
    else:
        start_bin = 0

    if max_scale is not None:
        # Conservative: exclude bins whose right edge is above max_scale
        stop_bin = np.searchsorted(gg.right_edges, max_scale, side='right')
    else:
        stop_bin = nbins

    return start_bin, stop_bin


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


def calculate_pure_eb_correlation(
    gg,
    gg_int,
    var_method="jackknife",
    cov_path_int=None,
    cosmo_cov=None,
    n_samples=1000,
    z_dist=None
):
    """
    Calculate pure E/B modes from correlation function objects.

    Parameters
    ----------
    gg : treecorr.GGCorrelation
        Correlation function for reporting binning (coarser binning for final results)
    gg_int : treecorr.GGCorrelation
        Correlation function for integration binning (fine binning for numerical
        integration)
    var_method : str, optional
        Variance method ("jackknife" or "bootstrap")
    cov_path_int : str, optional
        Path to integration covariance matrix for semi-analytical calculation
    cosmo_cov : pyccl.Cosmology, optional
        Cosmology for theoretical predictions in semi-analytical covariance
    n_samples : int, optional
        Number of Monte Carlo samples for semi-analytical covariance
    z_dist : 2D array, optional
        Redshift distribution;
        z_dist[:, 0] = z, z_dist[:, 1] = n(z)

    Returns
    -------
    dict
        Dictionary containing pure E/B mode results and covariance
    """
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    # Calculate min_sep and max_sep from gg object
    min_sep, max_sep = gg.left_edges[0], gg.right_edges[-1]

    def pure_EB(corrs):
        gg, gg_int = corrs
        return get_pure_EB_modes(
            theta=gg.meanr, xip=gg.xip, xim=gg.xim,
            theta_int=gg_int.meanr, xip_int=gg_int.xip, xim_int=gg_int.xim,
            tmin=min_sep, tmax=max_sep
        )

    # Initialize results dictionary with basic E/B mode data
    eb_keys = ["xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"]
    results = {"gg": gg, "gg_int": gg_int}
    results.update(dict(zip(eb_keys, pure_EB([gg, gg_int]))))

    if cov_path_int is not None:
        # Use semi-analytical covariance propagation
        print("Computing semi-analytical covariance for pure E/B modes")
        if z_dist is None:
            raise ValueError("z_dist must be provided for semi-analytical covariance")
        if cosmo_cov is None:
            raise ValueError(
                "cosmo_cov must be provided for semi-analytical covariance"
            )

        # Load covariance matrix
        cov_int = np.loadtxt(cov_path_int)

        # Set up integration binning and pre-compute binning matrix
        nbins_int, theta_int = len(gg_int.meanr), gg_int.meanr
        reporting_bin_edges = np.concatenate([
            gg.left_edges, [gg.right_edges[-1]]
        ])
        bin_indices = np.digitize(theta_int, reporting_bin_edges) - 1

        valid_mask = (bin_indices >= 0) & (bin_indices < len(gg.meanr))
        row_indices, col_indices = (
            bin_indices[valid_mask], np.where(valid_mask)[0]
        )

        binning_matrix = sparse.csr_matrix(
            (np.ones(len(row_indices)), (row_indices, col_indices)),
            shape=(len(gg.meanr), nbins_int)
        )
        row_sums = np.array(binning_matrix.sum(axis=1)).flatten()
        binning_matrix = sparse.diags(1/row_sums) @ binning_matrix

        # Generate theoretical xi+/xi- predictions and sample
        mean_int = np.concatenate(get_theo_xi(
            theta=theta_int, z=z_dist[:, 0], nz=z_dist[:, 1],
            backend="ccl", cosmo=cosmo_cov
        ))

        samples_int = np.random.multivariate_normal(
            mean_int, cov_int, size=n_samples
        )
        samples_int_xip = samples_int[:, :nbins_int]
        samples_int_xim = samples_int[:, nbins_int:]
        samples_rep_xip = (binning_matrix @ samples_int_xip.T).T
        samples_rep_xim = (binning_matrix @ samples_int_xim.T).T

        transformed_samples = [
            np.concatenate(get_pure_EB_modes(
                theta=gg.meanr, theta_int=gg_int.meanr,
                xip=samples_rep_xip[i], xim=samples_rep_xim[i],
                xip_int=samples_int_xip[i], xim_int=samples_int_xim[i],
                tmin=min_sep, tmax=max_sep
            ))
            for i in tqdm.tqdm(range(n_samples), desc="MC samples")
        ]

        # Store semi-analytical covariance results
        eb_samples = np.array(transformed_samples)
        results.update({"cov": np.cov(eb_samples.T), "eb_samples": eb_samples})
    else:
        # Use existing treecorr covariance estimation
        results["cov"] = treecorr.estimate_multi_cov(
            [gg, gg_int],
            var_method,
            func=lambda x: np.hstack(pure_EB(x)),
            cross_patch_weight="match" if var_method == "jackknife" else None,
        )

    # Validate covariance matrix
    try:
        np.linalg.cholesky(results["cov"])
    except np.linalg.LinAlgError:
        warnings.warn(
            "E/B mode covariance matrix is not positive definite. "
            "Chi-squared statistics may be unreliable.",
            UserWarning
        )

    return results


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
    from cosmo_numba.B_modes.cosebis import COSEBIS

    # Default to full range if no scale cuts provided
    if scale_cuts is None:
        scale_cuts = [(gg.left_edges[0], gg.right_edges[-1])]

    # Pre-compute values that don't change across scale cuts
    dtheta = gg.right_edges - gg.left_edges
    nbins = len(gg.meanr)

    # Load covariance matrix and calculate Hartlap factor once
    if cov_path is not None:
        print(f"Loading theoretical covariance from {cov_path}")
        cov_xipm = np.loadtxt(cov_path)
        hartlap_factor = 1  # Not defined for analytic covariances
    else:
        cov_xipm = gg.cov
        hartlap_factor = (gg.npatch - 2 * nmodes - 2) / (gg.npatch - 1)

    all_results = {}

    # Loop over each scale cut
    for scale_cut in tqdm.tqdm(scale_cuts, desc="COSEBIs scale cuts"):
        min_theta, max_theta = scale_cut

        # Apply scale cuts using scale_cut_to_bins for consistency
        start_bin, stop_bin = scale_cut_to_bins(gg, min_theta, max_theta)
        inds = np.arange(start_bin, stop_bin)

        theta_cut, dtheta_cut, xip_cut, xim_cut = [
            arr[inds] for arr in [gg.meanr, dtheta, gg.xip, gg.xim]
        ]

        # Calculate COSEBIs E/B modes using exact bin edges from indices
        cosebis = COSEBIS(
            theta_min=gg.left_edges[inds][0],
            theta_max=gg.right_edges[inds][-1],
            N_max=nmodes
        )
        En, Bn = cosebis.cosebis_from_xipm(theta_cut, dtheta_cut, xip_cut, xim_cut)

        # Extract covariance and transform to COSEBIs space
        cov_inds = np.concatenate([inds, inds + nbins])
        cov_cosebis = cosebis.cosebis_covariance_from_xipm_covariance(
            theta_cut, dtheta_cut, cov_xipm[cov_inds[:, None], cov_inds]
        )
        cov_E, cov_B = cov_cosebis[:nmodes, :nmodes], cov_cosebis[nmodes:, nmodes:]
        chi2_E, chi2_B = [
            hartlap_factor * (modes @ np.linalg.solve(cov, modes))
            for modes, cov in [(En, cov_E), (Bn, cov_B)]
        ]

        results = {
            'En': En, 'Bn': Bn, 'cov': cov_cosebis,
            'hartlap_factor': hartlap_factor, 'chi2_E': chi2_E, 'chi2_B': chi2_B,
            'pte_B': 1 - stats.chi2.cdf(chi2_B, nmodes),
            'scale_cut': (min_theta, max_theta), 'inds': inds
        }

        print(f"COSEBIs results [{min_theta:.1f}-{max_theta:.1f} arcmin]:")
        print(f"  chi2(E) = {chi2_E:.2f}, chi2(B) = {chi2_B:.2f}, "
              f"PTE(B) = {results['pte_B']:.4f}")

        all_results[scale_cut] = results

    return all_results


def calculate_eb_statistics(
    results,
    cov_path_int=None,
    n_samples=1000,
):
    """
    Calculate E/B mode statistics using 2D PTE analysis for all scale cut combinations.

    This function processes pure E/B mode results, calculates covariance blocks,
    and performs comprehensive 2D PTE analysis. The traditional "1D PTE" values
    are simply extracted from the 2D matrices at the full-range position.

    Parameters
    ----------
    results : dict
        Dictionary containing pure E/B mode results from calculate_pure_eb_correlation
    cov_path_int : str, optional
        Path to integration covariance matrix for semi-analytical calculation
    n_samples : int, optional
        Number of Monte Carlo samples used for semi-analytical covariance
    min_bins : int, optional
        Minimum number of bins required for valid PTE calculation

    Returns
    -------
    dict
        Updated results dictionary with PTE matrices and statistics
    """
    gg = results["gg"]
    nbins = gg.nbins
    npatch = gg.npatch1
    n_eff = n_samples if cov_path_int is not None else npatch

    # Extract covariance blocks and standard deviations
    eb_keys = ["xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb"]
    cov = results["cov"]
    for i, key in enumerate(eb_keys):
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
        (start, stop) for start in range(nbins)
        for stop in range(start, nbins + 1)
    ]

    for start_bin, stop_bin in combinations:
        nbins_eff = stop_bin - start_bin
        hartlap_factor = (n_eff - nbins_eff - 2) / (n_eff - 1)

        # Individual B-mode chi-squared calculations
        data_slices = [
            results[f"{xi}_B"][start_bin:stop_bin] for xi in ["xip", "xim"]
        ]
        cov_slices = [
            results[f"cov_{xi}_B"][start_bin:stop_bin, start_bin:stop_bin]
            for xi in ["xip", "xim"]
        ]
        chi2_values = [
            hartlap_factor * (data @ np.linalg.solve(cov, data))
            for data, cov in zip(data_slices, cov_slices)
        ]

        pte_xip_B[start_bin, stop_bin-1] = stats.chi2.sf(chi2_values[0], nbins_eff)
        pte_xim_B[start_bin, stop_bin-1] = stats.chi2.sf(chi2_values[1], nbins_eff)

        # Combined calculation with cross-correlation
        data_combined = np.concatenate(data_slices)

        # Build combined covariance matrix blocks
        xip_slice = slice(2*nbins + start_bin, 2*nbins + stop_bin)
        xim_slice = slice(3*nbins + start_bin, 3*nbins + stop_bin)
        cov_xip_block = cov[xip_slice, xip_slice]
        cov_xim_block = cov[xim_slice, xim_slice]
        cov_cross_block = cov[xip_slice, xim_slice]
        cov_combined = np.block([
            [cov_xip_block, cov_cross_block],
            [cov_cross_block.T, cov_xim_block]
        ])
        chi2_combined = hartlap_factor * (
            data_combined @ np.linalg.solve(cov_combined, data_combined)
        )
        pte_combined[start_bin, stop_bin-1] = stats.chi2.sf(chi2_combined, 2*nbins_eff)

    # Store PTE matrices
    pte_data = {"xip_B": pte_xip_B, "xim_B": pte_xim_B, "combined": pte_combined}
    results["pte_matrices"] = pte_data

    return results


def plot_integration_vs_reporting(gg, gg_int, output_path, version):
    """
    Plot integration vs reporting scale comparison.

    Parameters
    ----------
    gg : treecorr.GGCorrelation
        Reporting scale correlation function
    gg_int : treecorr.GGCorrelation
        Integration scale correlation function
    output_path : str
        Output file path for the plot
    version : str
        Version string for plot title
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Configure plot data for both xi+ and xi- in a consolidated loop
    plot_configs = [
        ('+', 'xip', 'varxip', r"$\theta \xi_+(\theta) \times 10^4$"),
        ('-', 'xim', 'varxim', r"$\theta \xi_-(\theta) \times 10^4$")
    ]

    data_configs = [
        (gg_int, 'k.', 3, 0.3, 'Integration'),
        (gg, '.', 12, 1, 'Reporting')
    ]

    for ax_idx, (xi_label, xi_attr, var_attr, ylabel) in enumerate(plot_configs):
        for data, fmt, ms, alpha, label_type in data_configs:
            xi_val = getattr(data, xi_attr)
            yerr = (
                data.meanr * np.sqrt(getattr(data, var_attr)) / 1e-4
                if hasattr(data, var_attr) and label_type == 'Reporting'
                else None
            )
            axs[ax_idx].errorbar(
                data.meanr, data.meanr * xi_val / 1e-4, yerr=yerr,
                fmt=fmt, ms=ms, alpha=alpha, capsize=3,
                ls='' if label_type == 'Reporting' else None,
                label=(
                    rf"$\xi_{{{xi_label}}}$, {label_type}: "
                    rf"${data.min_sep} < \theta < {data.max_sep}$, "
                    rf"{data.nbins} bins"
                )
            )
        axs[ax_idx].set(
            xlabel=r"$\theta$ [arcmin]",
            ylabel=ylabel,
            xscale="log",
            title=rf"$\xi_{{{xi_label}}}(\theta)$"
        )
        axs[ax_idx].legend()

    plt.suptitle(f"{version}: Integration vs. Reporting", fontsize=23, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def _get_pte_from_scale_cut(pte_matrix, gg, scale_cut):
    """
    Extract PTE value from matrix based on scale cut range using conservative logic.

    Parameters
    ----------
    pte_matrix : numpy.ndarray
        2D PTE matrix
    gg : treecorr.GGCorrelation
        Correlation function object with bin edges
    scale_cut : tuple
        (min_scale, max_scale) angular range for scale cut

    Returns
    -------
    float
        PTE value for the given scale cut, or full-range PTE if scale_cut is None
    """
    nbins = len(gg.meanr)

    if scale_cut is None:
        # Return full-range PTE (first row, last column)
        return pte_matrix[0, nbins-1]

    min_scale, max_scale = scale_cut

    # Use conservative scale_cut_to_bins helper
    start_bin, stop_bin = scale_cut_to_bins(gg, min_scale, max_scale)

    # Ensure valid range, otherwise fallback to full range
    if stop_bin <= start_bin or start_bin >= nbins or stop_bin <= 0:
        raise RuntimeError("Invalid scale cut range")
    return pte_matrix[start_bin, stop_bin-1]


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
    gg = results["gg"]
    nbins = gg.nbins
    cov = results["cov"]

    # Calculate combined PTE using off-diagonal covariance blocks
    # Get scale cuts for both xi+ and xi-
    if fiducial_xip_scale_cut is not None:
        xip_start_bin, xip_stop_bin = scale_cut_to_bins(gg, *fiducial_xip_scale_cut)
    else:
        xip_start_bin, xip_stop_bin = 0, nbins

    if fiducial_xim_scale_cut is not None:
        xim_start_bin, xim_stop_bin = scale_cut_to_bins(gg, *fiducial_xim_scale_cut)
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
    cov_xip_B = cov[2*nbins + xip_start_bin:2*nbins + xip_stop_bin,
                    2*nbins + xip_start_bin:2*nbins + xip_stop_bin]
    cov_xim_B = cov[3*nbins + xim_start_bin:3*nbins + xim_stop_bin,
                    3*nbins + xim_start_bin:3*nbins + xim_stop_bin]
    cov_cross = cov[2*nbins + xip_start_bin:2*nbins + xip_stop_bin,
                    3*nbins + xim_start_bin:3*nbins + xim_stop_bin]

    cov_combined = np.block([[cov_xip_B, cov_cross],
                            [cov_cross.T, cov_xim_B]])

    # Calculate combined chi-squared with Hartlap factor
    nbins_eff = len(xip_B_data) + len(xim_B_data)

    # Determine effective number of samples for Hartlap correction
    if "eb_samples" in results:  # Semi-analytical case
        n_eff = results["eb_samples"].shape[0]
    else:  # Jackknife case
        n_eff = gg.npatch1

    hartlap_factor = (n_eff - nbins_eff - 2) / (n_eff - 1)
    chi2_combined = hartlap_factor * (
        data_combined.T @ np.linalg.solve(cov_combined, data_combined)
    )
    combined_pte = stats.chi2.sf(chi2_combined, nbins_eff)

    # Extract PTE values for fiducial scale cuts (or full range)
    xip_B_pte = _get_pte_from_scale_cut(
        results["pte_matrices"]["xip_B"], gg, fiducial_xip_scale_cut
    )
    xim_B_pte = _get_pte_from_scale_cut(
        results["pte_matrices"]["xim_B"], gg, fiducial_xim_scale_cut
    )

    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    scale_factor = 1e-4

    # Main correlation functions and decomposed modes
    plot_configs = [
        ('xip', '+', 'varxip',
         r"$\xi_{+}=\xi_{+}^{E}+\xi_{+}^{B}+\xi_{+}^{\mathrm{amb}}$", xip_B_pte),
        ('xim', '-', 'varxim',
         r"$\xi_{-}=\xi_{-}^{E}-\xi_{-}^{B}+\xi_{-}^{\mathrm{amb}}$", xim_B_pte)
    ]

    for ax_idx, (xi_type, xi_label, var_attr, main_label, pte_val) in enumerate(
        plot_configs
    ):
        # Plot main correlation function
        xi_val = getattr(gg, xi_type)
        axs[ax_idx].errorbar(
            gg.meanr, gg.meanr * xi_val / scale_factor,
            yerr=gg.meanr * np.sqrt(getattr(gg, var_attr)) / scale_factor,
            fmt="k.", capsize=3, label=main_label
        )

        # Plot E/B/amb modes
        plot_data = [
            (f"{xi_type}_E", "g", 0.25, rf"$\xi_{{{xi_label}}}^{{E}}$"),
            (f"{xi_type}_B", "r", 1.0,
             rf"$\xi_{{{xi_label}}}^{{B}}, {{\rm PTE}}="
             rf"{np.round(pte_val, 4)}$"),
            (f"{xi_type}_amb", "purple", 0.25,
             rf"$\xi_{{{xi_label}}}^{{\mathrm{{amb}}}}$")
        ]

        for key, color, alpha, label in plot_data:
            axs[ax_idx].errorbar(
                gg.meanr, gg.meanr * results[key] / scale_factor,
                yerr=gg.meanr * results[f"std_{key}"] / scale_factor,
                color=color, ls="", marker=".", alpha=alpha, capsize=3, label=label
            )

        axs[ax_idx].set_ylabel(r"$\theta\xi\times10^{4}$")
        axs[ax_idx].set_title(rf"$\xi_{{{xi_label}}}$")

    # Configure both axes
    for ax in axs:
        ax.set(xscale="log", xlabel=r"$\theta$ [arcmin]")
        ax.axhline(0, alpha=0.3, color="k", linestyle="--", linewidth=0.5)
        ax.legend(loc="upper left")

    # Save the axis limits after data plotting but before adding gray regions
    original_xlims = [ax.get_xlim() for ax in axs]

    # Add fiducial scale cuts if provided - show excluded regions in gray
    scale_cuts = [(fiducial_xip_scale_cut, 0), (fiducial_xim_scale_cut, 1)]
    for scale_cut, ax_idx in scale_cuts:
        if scale_cut is not None:
            min_scale, max_scale = scale_cut
            xlim = original_xlims[ax_idx]

            # Use conservative scale_cut_to_bins helper for consistency
            start_bin, stop_bin = scale_cut_to_bins(gg, min_scale, max_scale)

            # Show excluded regions based on bin edges used in PTE calculation
            # Lower exclusion: bins 0 to start_bin-1 are excluded
            if start_bin > 0:
                lower_exclusion_edge = gg.right_edges[start_bin-1]
                axs[ax_idx].axvspan(
                    xlim[0], lower_exclusion_edge, alpha=0.2, color='gray',
                    label='Scale cuts' if ax_idx == 0 else None
                )

            # Upper exclusion: bins stop_bin to end are excluded
            if stop_bin < len(gg.left_edges):
                upper_exclusion_edge = gg.left_edges[stop_bin]
                axs[ax_idx].axvspan(
                    upper_exclusion_edge, xlim[1], alpha=0.2, color='gray',
                    label='Scale cuts' if ax_idx == 0 and start_bin == 0 else None
                )

            # Restore original axis limits
            axs[ax_idx].set_xlim(xlim)

    plt.suptitle(
        f"{version}: Pure Correlation Functions, Combined PTE = {combined_pte:.4f}",
        fontsize=23, y=0.95
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_cosebis_scale_cut_heatmap(
    cosebis_results, gg, version, output_path, fiducial_scale_cut=None
):
    """
    Create 2D heatmaps showing how COSEBIs statistics vary across different scale cuts.

    Parameters
    ----------
    cosebis_results : dict
        Dictionary with scale cut tuples as keys, containing 'chi2_E' and 'pte_B' values
    gg : treecorr.GGCorrelation
        Correlation function object for bin edges
    version : str
        Version string for main title
    output_path : str
        Output file path for the plot
    fiducial_scale_cut : tuple, optional
        (min_scale, max_scale) for cross-hatching
    """
    nbins = gg.nbins

    # Initialize matrices
    snrs = [np.sqrt(result["chi2_E"]) for result in cosebis_results.values()]
    ptes = [result["pte_B"] for result in cosebis_results.values()]

    rows = [start for start in range(nbins) for stop in range(start, nbins)]
    columns = [stop for start in range(nbins) for stop in range(start, nbins)]

    snr_matrix = np.full((nbins, nbins), np.nan)
    pte_matrix = np.full((nbins, nbins), np.nan)
    for (row, column, snr, pte) in zip(rows, columns, snrs, ptes):
        snr_matrix[row, column] = snr
        pte_matrix[row, column] = pte

    # Fill matrices from COSEBIs results
    for i in range(len(gg.left_edges)):
        for j in range(i, len(gg.right_edges)):
            scale_cut = (gg.left_edges[i], gg.right_edges[j])
            result = cosebis_results.get(scale_cut)

            if result is not None:
                snr_matrix[i, j] = np.sqrt(result['chi2_E'])
                pte_matrix[i, j] = result['pte_B']

    # Create 1x2 subplot layout
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Set up colormaps
    mako_cmap, vlag_cmap = [
        sns.color_palette(name, as_cmap=True).copy()
        for name in ["mako", "vlag"]
    ]
    for cmap in (mako_cmap, vlag_cmap):
        cmap.set_bad(color='lightgray')

    # Left plot: E-mode SNR
    snr_max = np.nanmax(snr_matrix) if not np.all(np.isnan(snr_matrix)) else 1
    snr_plot_data = snr_matrix.T
    im1 = axs[0].imshow(
        snr_plot_data, origin='lower', aspect='auto', cmap=mako_cmap,
        vmin=0, vmax=snr_max, extent=[0, nbins, 0, nbins]
    )
    axs[0].set_title(r'E-mode SNR: $\sqrt{\chi^2_E}$')
    axs[0].set_xlabel('Lower scale cut')
    axs[0].set_ylabel('Upper scale cut')

    # Right plot: B-mode PTE with contours
    pte_plot_data = pte_matrix.T
    im2 = axs[1].imshow(
        pte_plot_data, origin='lower', aspect='auto', cmap=vlag_cmap,
        vmin=0, vmax=1, extent=[0, nbins, 0, nbins]
    )

    # Add contours to PTE plot
    cs = axs[1].contour(
        pte_plot_data, levels=[0.05, 0.95], colors='black', linewidths=1,
        extent=[0, nbins, 0, nbins]
    )
    axs[1].clabel(cs, inline=True, fontsize=8)

    axs[1].set_title('B-mode PTE')
    axs[1].set_xlabel('Lower scale cut')

    # Add fiducial scale cut cross-hatching if provided
    if fiducial_scale_cut is not None:
        min_scale, max_scale = fiducial_scale_cut
        start_bin, stop_bin = scale_cut_to_bins(gg, min_scale, max_scale)

        if stop_bin > start_bin and start_bin < nbins and stop_bin > 0:
            # Add cross-hatching at the fiducial scale cut matrix element
            rect_x = start_bin
            rect_y = stop_bin - 1
            rect_width = 1
            rect_height = 1

            for ax_idx, ax in enumerate(axs):
                ax.add_patch(plt.Rectangle(
                    (rect_x, rect_y), rect_width, rect_height,
                    fill=False, edgecolor='black', linewidth=2,
                    hatch='///', alpha=0.8,
                    label='Fiducial scale cut' if ax_idx == 0 else None
                ))

    # Set angular scale ticks
    tick_indices = np.arange(0, nbins)
    x_tick_labels = [f'{gg.left_edges[i]:.1f}' for i in tick_indices
                     if i < len(gg.left_edges)]
    y_tick_labels = [f'{gg.right_edges[i]:.1f}' for i in tick_indices
                     if i < len(gg.right_edges)]
    x_tick_positions = tick_indices + 0.5
    y_tick_positions = tick_indices + 0.5

    for ax in axs:
        ax.set_xticks(x_tick_positions[:len(x_tick_labels)])
        ax.set_xticklabels(x_tick_labels, rotation=45)
        ax.set_yticks(y_tick_positions[:len(y_tick_labels)])
        ax.set_yticklabels(y_tick_labels)
        ax.set_aspect('equal')

    # Add colorbars to both plots
    colorbar_data = [(im1, 'SNR'), (im2, 'PTE')]
    for i, (im, label) in enumerate(colorbar_data):
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(label, rotation=270, labelpad=15)

    # Add legend if fiducial scale cut shown
    if fiducial_scale_cut is not None:
        axs[0].legend(loc='lower right', fontsize=8)

    plt.suptitle(f'{version}: COSEBIs Scale Cut Analysis', fontsize=23, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_pte_2d_heatmaps(
    results, version, output_path,
    fiducial_xip_scale_cut=None, fiducial_xim_scale_cut=None
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
    gg = results["gg"]
    nbins = gg.nbins
    pte_xip_B = results["pte_matrices"]["xip_B"]
    pte_xim_B = results["pte_matrices"]["xim_B"]

    # Create 1x2 subplot layout
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Set up colormap
    vlag_cmap = sns.color_palette("vlag", as_cmap=True).copy()
    vlag_cmap.set_bad(color='lightgray')
    contour_levels = [0.05, 0.95]

    # Left plot: xi+ PTE
    xip_plot_data = pte_xip_B.T
    axs[0].imshow(
        xip_plot_data, origin='lower', aspect='auto', cmap=vlag_cmap,
        vmin=0, vmax=1, extent=[0, nbins, 0, nbins]
    )
    cs1 = axs[0].contour(
        xip_plot_data, levels=contour_levels, colors='black', linewidths=1,
        extent=[0, nbins, 0, nbins]
    )
    axs[0].clabel(cs1, inline=True, fontsize=8)
    axs[0].set_title(r'$\xi_+^B$ PTE')
    axs[0].set_xlabel('Lower scale cut')
    axs[0].set_ylabel('Upper scale cut')

    # Right plot: xi- PTE
    xim_plot_data = pte_xim_B.T
    im2 = axs[1].imshow(
        xim_plot_data, origin='lower', aspect='auto', cmap=vlag_cmap,
        vmin=0, vmax=1, extent=[0, nbins, 0, nbins]
    )
    cs2 = axs[1].contour(
        xim_plot_data, levels=contour_levels, colors='black', linewidths=1,
        extent=[0, nbins, 0, nbins]
    )
    axs[1].clabel(cs2, inline=True, fontsize=8)
    axs[1].set_title(r'$\xi_-^B$ PTE')
    axs[1].set_xlabel('Lower scale cut')

    # Add fiducial scale cuts individually
    fiducial_scale_cuts = [fiducial_xip_scale_cut, fiducial_xim_scale_cut]
    for ax_idx, fiducial_scale_cut in enumerate(fiducial_scale_cuts):
        if fiducial_scale_cut is not None:
            min_scale, max_scale = fiducial_scale_cut
            start_bin, stop_bin = scale_cut_to_bins(gg, min_scale, max_scale)

            if stop_bin > start_bin and start_bin < nbins and stop_bin > 0:
                rect_x = start_bin
                rect_y = stop_bin - 1
                rect_width = 1
                rect_height = 1

                axs[ax_idx].add_patch(plt.Rectangle(
                    (rect_x, rect_y), rect_width, rect_height,
                    fill=False, edgecolor='black', linewidth=2,
                    hatch='///', alpha=0.8,
                    label='Fiducial scale cut' if ax_idx == 0 else None
                ))

    # Set angular scale ticks
    tick_indices = np.arange(0, nbins)
    x_tick_labels = [f'{gg.left_edges[i]:.1f}' for i in tick_indices
                     if i < len(gg.left_edges)]
    y_tick_labels = [f'{gg.right_edges[i]:.1f}' for i in tick_indices
                     if i < len(gg.right_edges)]
    x_tick_positions = tick_indices + 0.5
    y_tick_positions = tick_indices + 0.5

    for ax in axs:
        ax.set_xticks(x_tick_positions[:len(x_tick_labels)])
        ax.set_xticklabels(x_tick_labels, rotation=45)
        ax.set_yticks(y_tick_positions[:len(y_tick_labels)])
        ax.set_yticklabels(y_tick_labels)
        ax.set_aspect('equal')

    # Add colorbar
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label('PTE', rotation=270, labelpad=15)

    # Add legend if any fiducial scale cuts shown
    if any(cut is not None for cut in fiducial_scale_cuts):
        axs[0].legend(loc='lower right', fontsize=8)

    plt.suptitle(f'{version}: PTEs as a Function of Scale Cut', fontsize=23, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


def plot_eb_covariance_matrix(cov_matrix, var_method, output_path, version):
    """
    Plot E/B mode covariance matrix as correlation matrix.

    Parameters
    ----------
    cov_matrix : numpy.ndarray
        Covariance matrix from E/B mode analysis
    var_method : str
        Variance method used for the analysis
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
        r"$\xi_+^{E}$", r"$\xi_-^{E}$", r"$\xi_+^{B}$",
        r"$\xi_-^{B}$", r"$\xi_+^{\rm amb}$", r"$\xi_-^{\rm amb}$"
    ]

    for ticks in (plt.xticks, plt.yticks):
        ticks(tick_positions, tick_labels)
        ticks(np.arange(0, nbins * 6 + 1, 20), minor=True)

    ax.tick_params(axis="both", which="major", length=0)
    im.set_clim(-1, 1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax.set_title(f"{version}: {var_method} correlation matrix")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def plot_cosebis_modes(
    results,
    version,
    output_path,
    fiducial_scale_cut=None
):
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

    # Add scale cut information to title - use actual scale cut from results
    scale_info = ""
    if 'scale_cut' in results:
        actual_scale_cut = results['scale_cut']
        scale_info = (
            f", scale cut: ({actual_scale_cut[0]:.1f}, {actual_scale_cut[1]:.1f})"
        )
    elif fiducial_scale_cut is not None:
        scale_info = f", scale cut: {fiducial_scale_cut}"

    plt.title(f"{version} COSEBIs E/B-modes{scale_info}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


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

