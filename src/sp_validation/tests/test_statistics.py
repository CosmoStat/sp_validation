"""VALUE-DRIFT CHARACTERIZATION TESTS FOR statistics.py.

This module pins the numeric behaviour of the cosmology-independent
statistical helpers in :mod:`sp_validation.statistics`. The inputs are
fully deterministic (seeded RNG, no cluster data) and the outputs are
committed as literals with a tight ``rtol``. A refactor that changes the
numbers must turn this file red.

:Author: cdaley

"""

import numpy as np
import numpy.testing as npt

from sp_validation.statistics import jackknif_weighted_average2

def test_jackknif_weighted_average2_mean_and_error():
    """Pin the jackknife weighted average + error for a seeded RNG draw.

    WHAT IS PINNED: ``jackknif_weighted_average2`` draws ``n_realization``
    bootstrap-style subsamples (size = (1 - remove_size) * N, sampled WITH
    replacement via ``np.random.choice``) and returns
    (mean over realizations, std over realizations) of the per-subsample
    weighted average. With ``np.random.seed`` fixed before the call, the
    draws are deterministic, so both numbers are pinned as literals
    obtained from an actual run.

    WHY TEETH: the function depends on the data values, the weights, and
    the sampling. The companion assertion changes a single weight (under
    the same seed and therefore the same index draws) and asserts the mean
    moves, proving the weighting is load-bearing and not ignored.
    """
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    np.random.seed(1234)
    mean, err = jackknif_weighted_average2(
        data, weights, remove_size=0.1, n_realization=50
    )

    npt.assert_allclose(mean, _PINNED_JK_MEAN, rtol=1e-10)
    npt.assert_allclose(err, _PINNED_JK_ERR, rtol=1e-10)

    # TEETH: same seed (same index draws) but a perturbed weight -> the
    # weighted average changes, so the returned mean must differ.
    weights_perturbed = weights.copy()
    weights_perturbed[0] = 100.0
    np.random.seed(1234)
    mean_p, _ = jackknif_weighted_average2(
        data, weights_perturbed, remove_size=0.1, n_realization=50
    )
    assert not np.isclose(mean, mean_p)


# Literals pinned from an observed run inside the container; see the test
# docstring for the seed and parameters that reproduce them.
_PINNED_JK_MEAN = 6.248279984721161
_PINNED_JK_ERR = 0.971349020459367
