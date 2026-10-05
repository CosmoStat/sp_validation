"""Logarithmic angular-grid contracts, usable by the host workflow."""

import math


def validate_nested_grids(reporting, integration):
    """Require reporting bins to be exact unions of uniform fine bins.

    Both grids use ``min_sep``, ``max_sep`` and ``nbins`` in the same units.
    Return the integer oversampling factor. The tolerance allows floating-point
    round-off in logarithms, not snapping a requested edge onto a nearby one.
    """

    def grid(config):
        lo, hi, n = (
            float(config["min_sep"]),
            float(config["max_sep"]),
            int(config["nbins"]),
        )
        if config.get("bin_type", "Log") != "Log":
            raise ValueError("nested ξ grids require logarithmic binning")
        if not (0 < lo < hi and n > 0 and n == float(config["nbins"])):
            raise ValueError(
                "log grids require 0 < min_sep < max_sep and positive integer nbins"
            )
        return lo, hi, n, math.log(hi / lo) / n

    rlo, _, rn, rb = grid(reporting)
    flo, _, fn, fb = grid(integration)
    factor = rb / fb
    offset = math.log(rlo / flo) / fb
    tol = 1e-10
    prefix = "integration grid must nest into reporting grid: "
    if factor < 1 or not math.isclose(factor, round(factor), abs_tol=tol, rel_tol=0):
        raise ValueError(
            prefix
            + f"reporting/fine bin_size ratio {factor:.12g} is not a positive integer"
        )
    if not math.isclose(offset, round(offset), abs_tol=tol, rel_tol=0):
        raise ValueError(
            prefix + f"reporting min_sep lies at non-integer fine edge {offset:.12g}"
        )
    if offset < -tol or offset + rn * factor > fn + tol:
        raise ValueError(prefix + "the fine grid does not cover every reporting edge")
    return int(round(factor))
