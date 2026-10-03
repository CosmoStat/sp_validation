"""STATISTICS.

:Name: statistics.py

:Description: Cosmology-independent statistical helpers (jackknife resampling,
              jackknife patch centres, chi2/PTE, calibrated min-PTE across
              many null tests, covariance<->correlation, OneCovariance reshaping).
"""

import itertools
from dataclasses import dataclass

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


def _min_pte(ptes, two_sided):
    """Minimum PTE across the last axis, optionally two-sided."""
    ptes = np.asarray(ptes, dtype=float)
    if not np.all((ptes >= 0.0) & (ptes <= 1.0)):
        raise ValueError("PTEs must be finite and lie in [0, 1]")
    if two_sided:
        ptes = 2.0 * np.minimum(ptes, 1.0 - ptes)
    return ptes.min(axis=-1)


def wilson_interval(count, n, level=0.68):
    """Two-sided Wilson score interval for a binomial fraction ``count / n``."""
    z = stats.norm.ppf(0.5 + level / 2.0)
    fraction = count / n
    denominator = 1.0 + z**2 / n
    centre = (fraction + z**2 / (2.0 * n)) / denominator
    half_width = (
        z * np.sqrt(fraction * (1.0 - fraction) / n + z**2 / (4.0 * n**2)) / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def effective_number_of_tests(threshold, alpha):
    """Number of independent tests ``k`` with ``1 - (1 - threshold)^k = alpha``.

    For ``k`` independent uniform PTEs, the ``alpha``-quantile of their minimum
    is ``1 - (1 - alpha)^(1/k)``; inverting that for a calibrated threshold
    gives the effective number of independent tests it corresponds to.
    """
    return np.log1p(-alpha) / np.log1p(-np.asarray(threshold, dtype=float))


@dataclass(frozen=True)
class MinPTECalibration:
    """Global min-PTE null-test threshold calibrated on noise-only mocks.

    Attributes
    ----------
    alpha : float
        Global false-positive rate the threshold is calibrated to.
    two_sided : bool
        Whether each PTE ``p`` entered as ``2 min(p, 1 - p)``.
    mock_min_pte : numpy.ndarray
        Minimum PTE across statistics for each mock, shape ``(n_mocks,)``.
    threshold : float
        ``alpha``-quantile of ``mock_min_pte``: a data vector whose minimum
        PTE falls below it fails the global null test at level ``alpha``.
    threshold_interval : tuple of float
        Distribution-free interval on ``threshold`` at confidence ``level``.
    k_eff : float
        Effective number of independent tests implied by ``threshold``.
    k_eff_interval : tuple of float
        ``k_eff`` mapped from ``threshold_interval``.
    level : float
        Confidence level of the intervals.
    """

    alpha: float
    two_sided: bool
    mock_min_pte: np.ndarray
    threshold: float
    threshold_interval: tuple
    k_eff: float
    k_eff_interval: tuple
    level: float

    @property
    def n_mocks(self):
        """Number of mocks the calibration rests on."""
        return self.mock_min_pte.size

    def global_pte(self, data_ptes):
        """Global p-value of a data vector's PTEs against the mocks.

        Parameters
        ----------
        data_ptes : array_like
            The data's PTE for each statistic, ordered as the mock columns.

        Returns
        -------
        tuple
            ``(p, interval)``: the fraction of mocks whose minimum PTE is at
            most the data's, and its Wilson interval at ``level``.
        """
        data_min = _min_pte(data_ptes, self.two_sided)
        count = int(np.count_nonzero(self.mock_min_pte <= data_min))
        return count / self.n_mocks, wilson_interval(count, self.n_mocks, self.level)


def calibrate_min_pte(mock_ptes, alpha=0.05, two_sided=False, level=0.68):
    """Calibrate a global threshold on the minimum PTE across statistics.

    Testing many correlated statistics at a fixed per-test level inflates the
    false-positive rate by an unknown amount. Taking the minimum PTE across
    statistics for each noise-only mock and reading off its ``alpha``-quantile
    gives a threshold with exactly that global rate, whatever the correlations
    between statistics and whatever miscalibration of the individual PTEs.

    Parameters
    ----------
    mock_ptes : array_like
        PTEs of noise-only realisations, shape ``(n_mocks, n_stats)``; the
        columns can be any statistics, redshift-bin pairs or scale cuts.
    alpha : float, optional
        Global false-positive rate; default is ``0.05``.
    two_sided : bool, optional
        If ``True``, test each PTE ``p`` as ``2 min(p, 1 - p)`` so that
        anomalously small statistics also fail; default is ``False``.
    level : float, optional
        Confidence level of the reported intervals; default is ``0.68``.

    Returns
    -------
    MinPTECalibration
        The threshold, its interval, the effective number of independent
        tests, and :meth:`MinPTECalibration.global_pte` for the data.

    Notes
    -----
    The interval on the threshold uses order statistics: the number of mocks
    below the true ``alpha``-quantile is ``Binomial(n_mocks, alpha)``, and the
    ``(1 -/+ level) / 2`` quantiles of that count index the mocks bracketing it.
    """
    mock_ptes = np.asarray(mock_ptes, dtype=float)
    if mock_ptes.ndim != 2:
        raise ValueError("mock_ptes must have shape (n_mocks, n_stats)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    minima = _min_pte(mock_ptes, two_sided)
    n = minima.size
    ordered = np.sort(minima)
    tail = (1.0 - level) / 2.0
    lower = int(np.clip(stats.binom.ppf(tail, n, alpha), 1, n)) - 1
    upper = int(np.clip(stats.binom.ppf(1.0 - tail, n, alpha) + 1, 1, n)) - 1
    threshold = float(np.quantile(minima, alpha))
    interval = (float(ordered[lower]), float(ordered[upper]))
    k_eff = effective_number_of_tests([threshold, *interval], alpha)
    return MinPTECalibration(
        alpha=alpha,
        two_sided=two_sided,
        mock_min_pte=minima,
        threshold=threshold,
        threshold_interval=interval,
        k_eff=float(k_eff[0]),
        k_eff_interval=(float(k_eff[2]), float(k_eff[1])),
        level=level,
    )
