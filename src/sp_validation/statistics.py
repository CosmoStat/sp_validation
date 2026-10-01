"""STATISTICS.

:Name: statistics.py

:Description: Cosmology-independent statistical helpers (jackknife resampling,
              jackknife patch centres, chi2/PTE, covariance<->correlation,
              OneCovariance reshaping).
              Extracted verbatim from the former basic.py.
"""

import numpy as np
from scipy import stats

#: Depth of the ball-tree layers that seed the jackknife k-means.
PATCH_MIN_TOP = 6


def jackknife_patch_centers(cat, npatch, seed=0):
    """Seeded k-means jackknife patch centres for a TreeCorr catalogue.

    Seeding the k-means does not by itself fix the patches. TreeCorr starts the
    k-means from the top layers of a ball tree whose depth, unless ``min_top``
    is given, grows with the OpenMP thread count (``Field._determine_top``),
    and ``Catalog(npatch=..., rng=...)`` offers no way to set it. Pinning that
    depth makes the centres a function of the catalogue's positions, weights
    and ``seed`` alone, on every machine.

    Parameters
    ----------
    cat : treecorr.Catalog
        Catalogue with spherical (RA, Dec) positions.
    npatch : int
        Number of patches.
    seed : int, optional
        Seed for the k-means initialisation.

    Returns
    -------
    numpy.ndarray
        Patch centres, to pass as ``patch_centers`` to ``treecorr.Catalog``.
    """
    field = cat.getNField(
        min_top=PATCH_MIN_TOP,
        max_top=int.bit_length(npatch) - 1,
        coords="spherical",
    )
    _, centers = field.run_kmeans(npatch, rng=np.random.default_rng(seed))
    return centers


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
    n_bins = np.sqrt(cov_one_cov.shape[0]).astype(int)
    cov = np.zeros((n_bins, n_bins))
    index_value = 10 if gaussian else 9
    for i in range(n_bins):
        for j in range(n_bins):
            cov[i, j] = cov_one_cov[i * n_bins + j, index_value]
    return cov
