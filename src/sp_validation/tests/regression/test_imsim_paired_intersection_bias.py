"""Paired (+g/-g intersection) m-bias vs the per-branch metacal calibration.

Synthetic fixture with a known answer: two image-sim branches (+g, -g) share
intrinsic shapes and positions, carry the same pixel-noise realisation (as paired
shear sims do), and are each calibrated by the package's own ``metacal`` +
``get_calibrated_m_c`` with a weak shape-dependent size cut (so each branch
carries a selection response).
The shear response of the toy measurement is exactly 1 and the selection
response is measured metacal-style, so the per-branch calibration is correct
and the residual multiplicative bias of the calibrated survey is zero by
construction. ``ImageSimMBias`` with ``pair_match=True`` must recover that.
"""

import numpy as np
import pytest

from sp_validation.calibration import get_calibrated_m_c, metacal
from sp_validation.image_sims import ImageSimMBias

N = 2_000_000
G_IN = 0.02
STEP = 0.01
SIGMA_INT = 0.25
SIGMA_NOISE = 0.15
SIZE_SLOPE = 0.3  # T responds to the measured shape: R_sel ~ -0.04
REL_SIZE_MIN = 0.9


def _branch(e_int, size, g1_true, noise):
    """Calibrated (e1, e2) and selected indices of one branch, via metacal."""
    e1_ns = e_int + g1_true + noise
    n = len(e_int)
    cols = {}
    for suffix, h1, h2 in [
        ("NOSHEAR", 0, 0),
        ("1P", STEP, 0),
        ("1M", -STEP, 0),
        ("2P", 0, STEP),
        ("2M", 0, -STEP),
    ]:
        g1 = e1_ns + h1
        cols[f"NGMIX_FLAGS_{suffix}"] = np.zeros(n, np.int16)
        cols[f"NGMIX_G1_{suffix}"] = g1
        cols[f"NGMIX_G2_{suffix}"] = np.full(n, h2, float)
        cols[f"NGMIX_FLUX_{suffix}"] = np.full(n, 100.0)
        cols[f"NGMIX_FLUX_ERR_{suffix}"] = np.ones(n)
        cols[f"NGMIX_T_{suffix}"] = size * (1 + SIZE_SLOPE * g1)
        cols[f"NGMIX_T_ERR_{suffix}"] = np.full(n, 0.01)
        cols[f"NGMIX_T_PSF_RECONV_{suffix}"] = np.ones(n)
    cols["NGMIX_G1_ERR_NOSHEAR"] = np.full(n, 0.01)
    cols["NGMIX_G2_ERR_NOSHEAR"] = np.full(n, 0.01)
    arr = np.empty(n, dtype=[(k, v.dtype) for k, v in cols.items()])
    for k, v in cols.items():
        arr[k] = v
    del cols
    mc = metacal(
        arr,
        np.ones(n, bool),
        step=STEP,
        snr_min=5,
        snr_max=500,
        rel_size_min=REL_SIZE_MIN,
        rel_size_max=1e9,
        size_corr_ell=False,
        global_R_weight=None,
    )
    e, _, _, _, _, _ = get_calibrated_m_c(mc, additive_correction=False)
    return e, mc.mask_dict["ns"], mc.R_selection[0, 0]


def _estimator(pair_match):
    cfg = {
        "grids_dir": ".",
        "num": 1,
        "shear_amplitude": G_IN,
        "match_radius_deg": 1e-4,
        "pair_match": pair_match,
        "w_cols": ["none"],
        "n_bootstrap": 20,
        "bootstrap_seed": 1,
    }
    return ImageSimMBias(cfg)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="#386: matched intersection has the wrong selection response",
)
def test_paired_m_bias_zero_when_each_branch_is_correctly_calibrated():
    """The paired estimator must report m = 0 on correctly calibrated sims.

    ``calibrate_comprehensive_cat.py`` calibrates each sheared branch with the
    response (R_shear + R_sel) of *that branch's* selected sample, and
    ``ImageSimMBias._m_c_pair`` with ``pair_match=True`` then averages
    (e_+ - e_-)/2g over only the objects selected in *both* branches.
    Intersection membership depends on the sign of the shear (and, with
    independent noise, on both noise draws), so it is a different population
    from the one whose response was removed: the shape-dependent selection is
    applied twice, and the paired mean carries a spurious m ~ -R_sel even
    though each branch is unbiased. Here the truth is m = 0 (R_shear = 1,
    metacal R_sel exact up to finite-sample noise), and the unpaired estimator
    on the very same catalogues is asserted to recover it; a paired m that
    differs from the unpaired one by more than 4 combined bootstrap sigma
    is the defect.
    """
    rng = np.random.default_rng(3)
    e_int = rng.normal(0, SIGMA_INT, N)
    size = rng.lognormal(0, 0.4, N)
    ra = (np.arange(N) % 1000) * 0.01
    dec = (np.arange(N) // 1000) * 0.01 - 10.0

    paired, unpaired = _estimator(True), _estimator(False)
    noise = rng.normal(0, SIGMA_NOISE, N)
    r_sel = {}
    for name, g in (("1p2z", G_IN), ("1m2z", -G_IN)):
        e, sel, r_sel[name] = _branch(e_int, size, g, noise)
        cat = {
            "ra": ra[sel],
            "dec": dec[sel],
            "e1": e[0],
            "e2": e[1],
            "w": {"none": np.ones(len(sel))},
        }
        paired.cats[name] = cat
        unpaired.cats[name] = cat

    m_p, sig_p, _, _ = paired._m_c_pair("1p2z", "1m2z", 0, verbose=False)["none"]
    m_u, sig_u, _, _ = unpaired._m_c_pair("1p2z", "1m2z", 0, verbose=False)["none"]

    # Control: the per-branch calibration is right, so the unpaired m is ~0.
    assert abs(m_u) < 4 * sig_u, f"control failed: unpaired m={m_u:+.5f}+-{sig_u:.5f}"
    # The paired estimator targets the same quantity, so it must agree with 0
    # (and with the unpaired value) within the combined bootstrap error.
    sig = np.hypot(sig_p, sig_u)
    assert abs(m_p - m_u) < 4 * sig, (
        f"paired (intersection) m = {m_p:+.5f} +- {sig_p:.5f} vs unpaired m on "
        f"the same catalogues = {m_u:+.5f} +- {sig_u:.5f} (truth 0): difference "
        f"{m_p - m_u:+.5f} = {abs(m_p - m_u) / sig:.1f} sigma; per-branch R_sel = "
        f"{r_sel['1p2z']:+.4f}/{r_sel['1m2z']:+.4f}"
    )
