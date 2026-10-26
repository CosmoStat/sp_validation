"""Non-cosmological weighted pair counts for real-space shape noise.

TreeCorr auto counts are unordered. OneCovariance's xi_pm noise is
4*s**4/N_ordered, where s is the per-component dispersion, so supply
N_ordered = 2*W**2/Q, not raw npairs or W. No shape columns are accessed.
"""

from pathlib import Path

import numpy as np
import treecorr

from sp_validation.io import Catalogue


def ordered_effective_pairs(weight, weight_squared):
    """Convert unordered W=sum(w_i*w_j), Q=sum((w_i*w_j)**2) to 2 W²/Q.

    Empty/non-finite bins cannot define a covariance; fail rather than
    silently falling back to analytic counts.
    """
    weight = np.asarray(weight, dtype=float)
    weight_squared = np.asarray(weight_squared, dtype=float)
    if (
        weight.shape != weight_squared.shape
        or not np.all(np.isfinite(weight))
        or not np.all(np.isfinite(weight_squared))
        or np.any(weight <= 0)
        or np.any(weight_squared <= 0)
    ):
        raise ValueError("Pair counts require matching, finite, positive W and Q")
    return 2 * weight**2 / weight_squared


def measure_pair_counts(
    catalogue,
    output,
    *,
    min_sep,
    max_sep,
    nbins,
    ra_col="RA",
    dec_col="Dec",
    w_col="w",
    num_threads=1,
):
    """Cache W, Q and raw counts using positions and weights only.

    Uses the xi rule's log binning, spherical metric and min_top=6.
    The adjacent two-column .txt has nominal log-bin centres in arcmin,
    never mean pair separations: OneCovariance must not rebin W²/Q.
    """
    with Catalogue(str(catalogue)) as cat:
        data = cat.read(columns=(ra_col, dec_col, w_col))
    weights = np.asarray(data[w_col], dtype=float)
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.all(np.isfinite(data[ra_col]))
        or not np.all(np.isfinite(data[dec_col]))
    ):
        raise ValueError("Positions and non-negative weights must be finite")
    config = dict(
        min_sep=float(min_sep),
        max_sep=float(max_sep),
        nbins=int(nbins),
        sep_units="arcmin",
        min_top=6,
        num_threads=int(num_threads),
    )
    counts = []
    for power in (1, 2):
        cat = treecorr.Catalog(
            ra=data[ra_col],
            dec=data[dec_col],
            w=weights**power,
            ra_units="deg",
            dec_units="deg",
        )
        nn = treecorr.NNCorrelation(config)
        nn.process(cat)
        counts.append(nn.weight.copy())
        print(f"Completed NN weights^{power}", flush=True)
    W, Q = counts
    ordered = ordered_effective_pairs(W, Q)
    edges = np.geomspace(float(min_sep), float(max_sep), int(nbins) + 1)
    theta = np.sqrt(edges[:-1] * edges[1:])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        theta=theta,
        edges=edges,
        npairs=nn.npairs,
        weight=W,
        weight_squared=Q,
        ordered_effective_pairs=ordered,
    )
    np.savetxt(
        output.with_suffix(".txt"),
        np.column_stack((theta, ordered)),
        header="theta_arcmin N_ordered_effective",
    )
