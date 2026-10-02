"""STATISTICS.

:Name: statistics.py

:Description: Cosmology-independent statistical helpers (jackknife resampling,
              chi2/PTE, covariance<->correlation, OneCovariance reshaping).
              Extracted verbatim from the former basic.py.
"""

import itertools

import numpy as np
from scipy import stats


def jackknif_weighted_average2(
    data,
    weights,
    remove_size=0.1,
    n_realization=100,
):
    """Add docstring.

    ...

    """
    samp_size = len(data)
    keep_size_pc = 1 - remove_size

    if keep_size_pc < 0:
        raise ValueError("remove size should be in [0, 1]")

    subsamp_size = int(samp_size * keep_size_pc)

    all_ind = np.arange(samp_size)

    all_est = []
    for i in range(n_realization):
        sub_data_ind = np.random.choice(all_ind, subsamp_size)

        if sum(data[sub_data_ind]) == 0:
            all_est.append(np.nan)
        else:
            all_est.append(
                np.average(data[sub_data_ind], weights=weights[sub_data_ind])
            )

    all_est = np.array(all_est)

    return np.mean(all_est), np.std(all_est)


def corr_from_cov(cov):
    """Correlation matrix from a covariance matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix.

    Returns
    -------
    numpy.ndarray
        Correlation matrix.

    """
    std_dev = np.sqrt(np.diag(cov))
    return cov / np.outer(std_dev, std_dev)


def chi2_and_pte(data_vector, cov, verbose=False):
    """Chi-squared, reduced chi-squared and PTE for a data vector.

    The data vector is assumed to be zero-mean under the null hypothesis,
    so ``chi2 = d^T C^-1 d``.

    Parameters
    ----------
    data_vector : numpy.ndarray
        Data vector.
    cov : numpy.ndarray
        Covariance matrix of the data vector.
    verbose : bool, optional
        If ``True``, print the statistics; default is ``False``.

    Returns
    -------
    tuple
        ``(chi2, reduced_chi2, pte)``.

    """
    # Solve the linear system rather than forming C^-1 explicitly (more stable,
    # cheaper); use the survival function rather than 1 - cdf (avoids
    # catastrophic cancellation in the high-chi2 / low-PTE tail).
    chi2 = data_vector @ np.linalg.solve(cov, data_vector)
    dof = len(data_vector)
    reduced_chi2 = chi2 / dof
    pte = stats.chi2.sf(chi2, dof)
    if verbose:
        print(f"Chi2: {chi2:.4f}")
        print(f"Reduced Chi2: {reduced_chi2:.4f}")
        print(f"PTE: {pte:.4f}")
    return chi2, reduced_chi2, pte


def cov_from_one_covariance(cov_one_cov, gaussian=True):
    """Reshape a OneCovariance ``covariance_list`` table into a matrix.

    Parameters
    ----------
    cov_one_cov : numpy.ndarray
        Flat OneCovariance output (e.g. from ``covariance_list_..._Cell.dat``),
        with one row per ``(i, j)`` element pair.
    gaussian : bool, optional
        If ``True`` use the Gaussian-only column, otherwise the
        Gaussian+non-Gaussian column; default is ``True``.

    Returns
    -------
    numpy.ndarray
        Square covariance matrix.

    """
    # Get the ell_bins and tomo_bins for each covariance entry
    ell1 = cov_one_cov[:, 1]
    ell2 = cov_one_cov[:, 2]
    tomoi = cov_one_cov[:, 5].astype(int)
    tomoj = cov_one_cov[:, 6].astype(int)
    tomok = cov_one_cov[:, 7].astype(int)
    tomol = cov_one_cov[:, 8].astype(int)

    # Get the values to save in the covariance
    cov_col = 10 if gaussian else 9
    values = cov_one_cov[:, cov_col]

    # Map the ell bins to and index
    ell_bins = np.unique(ell1)
    n_ell_bins = len(ell_bins)
    ell_to_idx = {ell: idx for idx, ell in enumerate(ell_bins)}

    # Map the tomo bin pairs to an index
    tomo_bin_ids = np.unique(tomoi)
    tomo_bin_pairs = list(itertools.combinations_with_replacement(tomo_bin_ids, 2))
    n_spectra = len(tomo_bin_pairs)
    pair_to_idx = {pair: idx for idx, pair in enumerate(tomo_bin_pairs)}

    # Initialize the covariance matrix
    cov_size = n_ell_bins * n_spectra
    cov = np.zeros((cov_size, cov_size))

    # Get the tomo bin pair indices
    # the ordering of OneCovariance is the same than itertools
    a_idx = np.fromiter(
        (pair_to_idx[(bin_i, bin_j)] for bin_i, bin_j in zip(tomoi, tomoj)),
        dtype=int,
        count=len(tomoi),
    )
    b_idx = np.fromiter(
        (pair_to_idx[(bin_k, bin_l)] for bin_k, bin_l in zip(tomok, tomol)),
        dtype=int,
        count=len(tomok),
    )

    # Get the ell bin indices
    ell1_idx = np.fromiter(
        (ell_to_idx[ell] for ell in ell1), dtype=int, count=len(ell1)
    )
    ell2_idx = np.fromiter(
        (ell_to_idx[ell] for ell in ell2), dtype=int, count=len(ell2)
    )

    # Get the row and col
    row = a_idx * n_ell_bins + ell1_idx
    col = b_idx * n_ell_bins + ell2_idx

    # Assign the values
    cov[row, col] = values
    cov[col, row] = values  # Symmetrize

    return cov
