"""TreeCorr position–shear estimator and its joint jackknife covariance.

The caller constructs both catalogues using the same patch centres.  This module
is deliberately independent of catalogue layout and does not generate randoms.
"""

import numpy as np
import treecorr


def measure(
    lenses,
    sources,
    *,
    randoms=None,
    min_sep=1.0,
    max_sep=250.0,
    nbins=20,
    num_threads=None,
):
    """Measure tangential and cross shear, optionally subtracting random lenses.

    All catalogues must use the same spatial patch centres.  The returned
    covariance has block order (tangential bins, cross bins), including the
    covariance between the two and, when supplied, the random subtraction.
    Separations are in arcminutes.
    """
    centers = sources.patch_centers
    if len(centers) < 2:
        raise ValueError("jackknife requires at least two source patches")
    for catalogue in (lenses,) if randoms is None else (lenses, randoms):
        other = catalogue.patch_centers
        if other.shape != centers.shape or not np.allclose(other, centers):
            raise ValueError("all catalogues must share the source patch centres")

    config = dict(
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
        sep_units="arcmin",
        var_method="jackknife",
    )
    correlations = []
    for positions in (lenses,) if randoms is None else (lenses, randoms):
        corr = treecorr.NGCorrelation(config)
        corr.process(positions, sources, num_threads=num_threads)
        correlations.append(corr)

    def vector(corrs):
        result = np.concatenate((corrs[0].xi, corrs[0].xi_im))
        if len(corrs) == 2:
            result = result - np.concatenate((corrs[1].xi, corrs[1].xi_im))
        return result

    values = vector(correlations)
    covariance = treecorr.estimate_multi_cov(
        correlations, "jackknife", func=vector, cross_patch_weight="simple"
    )
    error = np.sqrt(np.maximum(np.diag(covariance), 0))
    dg = correlations[0]
    result = dict(
        theta=np.exp(dg.meanlogr),
        xi_t=values[:nbins],
        xi_x=values[nbins:],
        error_t=error[:nbins],
        error_x=error[nbins:],
        covariance=covariance,
        npairs=dg.npairs.copy(),
        weight=dg.weight.copy(),
    )
    if randoms is not None:
        result.update(
            random_npairs=correlations[1].npairs.copy(),
            random_weight=correlations[1].weight.copy(),
        )
    return result
