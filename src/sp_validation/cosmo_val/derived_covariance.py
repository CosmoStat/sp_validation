"""Covariance propagation for derived tomographic ξ± statistics."""

import numpy as np
from scipy.linalg import block_diag

from .. import sacc_io
from .sacc_writers import onecov_xi_covariance


def xi_pairs(part):
    """Source-bin pairs in the order in which the ξ± part holds them."""
    pairs = []
    for dp in part.data:
        if dp.data_type == sacc_io.XI_PLUS:
            pair = tuple(int(t.removeprefix("source_")) for t in dp.tracers)
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def xi_covariance(part, path=None):
    """Pair-major covariance and jackknife patch count, or analytic covariance."""
    if path is not None:
        return onecov_xi_covariance(part, np.loadtxt(path)), None
    if part.covariance is None:
        raise ValueError("the integration-grid ξ± part has no covariance")
    return part.covariance.dense, int(part.metadata.get("npatch", 1))


def propagate(operators, covariance, *, cosebis=False):
    """Propagate all cross-pair blocks, with the COSEBIs E×B-zero convention."""
    operator = block_diag(*operators)
    result = operator @ covariance @ operator.T
    if cosebis:
        e, b, offset = [], [], 0
        for op in operators:
            n = op.shape[0] // 2
            e.extend(range(offset, offset + n))
            b.extend(range(offset + n, offset + 2 * n))
            offset += 2 * n
        result[np.ix_(e, b)] = 0
        result[np.ix_(b, e)] = 0
    return result
