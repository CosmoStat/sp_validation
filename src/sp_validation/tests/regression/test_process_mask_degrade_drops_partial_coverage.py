"""Degrading a binary HealSparse mask must preserve the fractional coverage."""

import importlib.util
from pathlib import Path

import healpy as hp
import healsparse as hsp
import numpy as np
import pytest

import sp_validation

REPO = Path(__file__).resolve().parents[4]
# Use the selected package's checkout during cross-branch verification.
PACKAGE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if PACKAGE_REPO != REPO:
    REPO = PACKAGE_REPO
SCRIPT = REPO / "workflow" / "scripts" / "process_mask.py"

NSIDE_COV, NSIDE_FINE, NSIDE_COARSE = 32, 1024, 64
NCHILD = (NSIDE_FINE // NSIDE_COARSE) ** 2  # 256 fine pixels per coarse pixel


def _load_process_mask():
    if not SCRIPT.exists():
        pytest.skip("process_mask.py missing; requires the tomography branch workflow")
    spec = importlib.util.spec_from_file_location("process_mask", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _toy_mask(bit_packed):
    """Cover pixel 0 fully and pixels 1, 2, 3 by 1/2, 1/4, 3/4.

    These are coarse NEST pixels; everything else is outside the footprint.
    """
    if bit_packed:
        m = hsp.HealSparseMap.make_empty(
            NSIDE_COV, NSIDE_FINE, dtype=bool, bit_packed=True
        )
    else:
        m = hsp.HealSparseMap.make_empty(NSIDE_COV, NSIDE_FINE, dtype=bool)
    fracs = {0: 1.0, 1: 0.5, 2: 0.25, 3: 0.75}
    fill = np.concatenate(
        [np.arange(p * NCHILD, p * NCHILD + int(f * NCHILD)) for p, f in fracs.items()]
    )
    m[fill] = True
    ring = hp.nest2ring(NSIDE_COARSE, np.array(list(fracs)))
    return m, ring, np.array(list(fracs.values())), len(fill)


@pytest.mark.parametrize("bit_packed", [True, False], ids=["bitpacked", "bool"])
@pytest.mark.xfail(
    strict=True,
    reason="#387: mean degradation ignores uncovered children",
)
def test_degraded_mask_keeps_fractional_coverage_and_area(bit_packed):
    """Protect fractional coverage and area at footprint edges and holes.

    process_mask.degrade_mask turns the nside-131072 UNIONS bit-packed boolean
    mask into a coarse float mask whose values feed the mask power spectrum, and
    calculate_effective_area sums it into the survey area. For a binary mask the
    correct coarse value is the covered fraction of each coarse pixel's children
    (missing/sentinel children count as 0), and the correct area is the number of
    covered fine pixels times the fine pixel area -- both known by construction
    here. HealSparse's reduction='mean' averages only *valid* children, so any
    partially covered coarse pixel gets 1.0 and every footprint edge and hole is
    counted in full."""
    pm = _load_process_mask()
    m, ring, expected_frac, n_fine = _toy_mask(bit_packed)

    nest_mask = pm.degrade_mask(m, NSIDE_COARSE, verbose=False)
    area = pm.calculate_effective_area(nest_mask, NSIDE_COARSE)

    expected_area = n_fine * hp.nside2pixarea(NSIDE_FINE, degrees=True)
    nest_idx = hp.ring2nest(NSIDE_COARSE, ring)
    got_frac = nest_mask[nest_idx]
    assert np.allclose(got_frac, expected_frac), (
        f"coarse mask values {got_frac.tolist()} != covered fractions "
        f"{expected_frac.tolist()}"
    )
    assert area == pytest.approx(expected_area, rel=1e-10), (
        f"effective area {area:.6f} deg2 != {expected_area:.6f} deg2 "
        f"(covered fine pixels x fine pixel area)"
    )
