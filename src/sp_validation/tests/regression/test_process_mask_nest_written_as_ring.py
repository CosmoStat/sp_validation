"""Protect sky positions and power spectra in the mask-processing round trip."""

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
SCRIPTS = REPO / "workflow" / "scripts"


def _load(name):
    path = SCRIPTS / f"{name}.py"
    if not path.exists():
        pytest.skip(f"{path.name} missing; requires the tomography branch workflow")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NSIDE_HI, NSIDE_LO = 64, 16


def _cap_fixture():
    """Full-sky-valid HealSparse map: 1 inside a 20 deg cap, 0 elsewhere."""
    m = hsp.HealSparseMap.make_empty(
        nside_coverage=8, nside_sparse=NSIDE_HI, dtype=np.float64
    )
    vec = hp.ang2vec(60.0, 30.0, lonlat=True)
    inside_nest = hp.query_disc(NSIDE_HI, vec, np.radians(20.0), nest=True)
    allpix = np.arange(hp.nside2npix(NSIDE_HI))
    vals = np.zeros(allpix.size)
    vals[inside_nest] = 1.0
    m[allpix] = vals
    # Independent reference in RING ordering, built directly with healpy.
    ref_hi = np.zeros(allpix.size)
    ref_hi[hp.query_disc(NSIDE_HI, vec, np.radians(20.0), nest=False)] = 1.0
    ref_lo = hp.ud_grade(ref_hi, NSIDE_LO, order_in="RING", order_out="RING")
    return m, ref_lo


@pytest.mark.xfail(
    strict=True,
    reason="#387: NESTED mask pixels are written with a RING header",
)
def test_downgraded_mask_written_in_ring_order_matches_sky(tmp_path):
    """process_mask must write a map whose pixels sit where the mask is on the sky.

    HealSparse's ``valid_pixels`` are NESTED indices, while process_mask writes the
    array with ``hp.write_map(nest=False)`` and analyze_mask_power_spectrum reads it
    back as RING. A 20 deg cap built at nside 64 and degraded (mean) to nside 16 must
    therefore read back equal to the same cap built directly in RING with healpy and
    ``ud_grade``d to nside 16 (every pixel is valid, so the HealSparse mean equals the
    ud_grade mean), and its anafast C_ell must match the reference's. Area (a sum)
    is ordering-invariant, so it is checked to agree either way.
    """
    pm = _load("process_mask")
    ps = _load("analyze_mask_power_spectrum")
    m, ref = _cap_fixture()

    arr = pm.degrade_mask(m, NSIDE_LO, verbose=False)
    out = tmp_path / "mask.fits"
    pm.save_healpix_mask(arr, str(out), NSIDE_LO, verbose=False)
    try:
        got = ps.load_healpix_mask(str(out), verbose=False)
    except TypeError:  # healpy without read_map(verbose=)
        got = hp.read_map(str(out))

    np.testing.assert_allclose(got.sum(), ref.sum(), rtol=1e-12)  # area unaffected

    n_wrong = int(np.sum(~np.isclose(got, ref)))
    _, cl_got = ps.calculate_power_spectrum(got, verbose=False)
    cl_ref = hp.anafast(ref, lmax=3 * NSIDE_LO - 1)
    assert n_wrong == 0, (
        f"{n_wrong}/{ref.size} pixels misplaced after NEST->RING round trip; "
        f"C_2 ratio got/ref = {cl_got[2] / cl_ref[2]:.3f}, "
        f"C_10 ratio = {cl_got[10] / cl_ref[10]:.3f}"
    )
    np.testing.assert_allclose(cl_got, cl_ref, rtol=1e-6, atol=1e-12)
