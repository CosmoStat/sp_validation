"""Tests for the FootprintPlotter dedup.

``FootprintPlotter`` lives in ``cs_util``; sp_validation imports it at each
call site rather than keeping its own (previously stale) copy. These tests
guard that contract and exercise the canonical class's map building on a tiny
input.

:Author: cdaley

"""

import numpy as np
from cs_util.plots import FootprintPlotter


def test_footprintplotter_not_exposed_by_sp_validation_plots():
    """sp_validation.plots must not redefine or re-export FootprintPlotter."""
    import sp_validation.plots as sp_plots

    assert not hasattr(sp_plots, "FootprintPlotter"), (
        "FootprintPlotter should be imported from cs_util at each call site, "
        "not re-exported via sp_validation.plots"
    )


def test_cosmo_val_uses_cs_util_footprintplotter():
    """cosmo_val references the canonical cs_util class via its cs_plots alias."""
    from sp_validation import cosmo_val

    assert cosmo_val.cs_plots.FootprintPlotter is FootprintPlotter
    # the bare name must no longer leak into cosmo_val's namespace
    assert not hasattr(cosmo_val, "FootprintPlotter")


def test_create_hsp_map_preserves_counts():
    """create_hsp_map bins (ra, dec) into a HealSparseMap of per-pixel counts."""
    plotter = FootprintPlotter(nside_coverage=32, nside_map=64)

    # two coincident points (one pixel, count 2) + one elsewhere (count 1)
    ra = np.array([10.0, 10.0, 200.0])
    dec = np.array([-5.0, -5.0, 30.0])

    hsp_map = plotter.create_hsp_map(ra, dec)
    # create_hsp_map uses a NaN sentinel, so only the populated pixels carry
    # counts; the rest of each allocated coverage block reads back as NaN.
    counts = hsp_map[hsp_map.valid_pixels]
    counts = counts[~np.isnan(counts)]

    assert counts.sum() == len(ra)  # every object counted once
    assert counts.max() == 2        # the coincident pixel
    assert len(counts) == 2         # two distinct populated pixels
