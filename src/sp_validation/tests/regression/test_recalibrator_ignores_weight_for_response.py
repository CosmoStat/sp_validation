"""Recalibration must use the statistic's weights for both response and bias."""

import numpy as np
import pytest
from astropy.io import fits

from sp_validation.calibration import get_calibrate_e_from_cat

G_TRUE = 0.02


def _write_catalogue(path):
    # Opposite weight preferences: (response, w_des, w_iv).
    # Intrinsic +/- pairs and patches +/-g make the global additive bias zero.
    response, w_des, w_iv, intrinsic, shear = [], [], [], [], []
    for patch_shear in (+G_TRUE, -G_TRUE):
        for r, wd, wi in ((0.5, 1.0, 3.0), (1.5, 3.0, 1.0)):
            for e in (+0.1, -0.1):
                response.append(r)
                w_des.append(wd)
                w_iv.append(wi)
                intrinsic.append(e)
                shear.append(patch_shear)
    response, w_des, w_iv, intrinsic, shear = map(
        np.array, (response, w_des, w_iv, intrinsic, shear)
    )
    zeros = np.zeros_like(response)
    columns = [
        fits.Column(name="e1_uncal", array=response * shear + intrinsic, format="D"),
        fits.Column(name="e2_uncal", array=zeros, format="D"),
        fits.Column(name="w_des", array=w_des, format="D"),
        fits.Column(name="w_iv", array=w_iv, format="D"),
        fits.Column(name="R_g11", array=response, format="D"),
        fits.Column(name="R_g12", array=zeros, format="D"),
        fits.Column(name="R_g21", array=zeros, format="D"),
        fits.Column(name="R_g22", array=response, format="D"),
    ]
    header = fits.Header()
    for key in ("R_S11", "R_S12", "R_S21", "R_S22"):
        header[key] = 0.0
    with fits.HDUList(
        [fits.PrimaryHDU(header=header), fits.BinTableHDU.from_columns(columns)]
    ) as hdul:
        hdul.writeto(path)
    return {"des": w_des, "iv": w_iv}, shear > 0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#395: recalibration averages response unweighted",
)
@pytest.mark.parametrize("weight_type", ["des", "iv"])
def test_recalibrated_weighted_shear_uses_weighted_response(tmp_path, weight_type):
    """The weighted recalibrated patch shear must recover g=0.02 exactly.

    Patch P has responses 0.5 and 1.5 with opposite w_des/w_iv preferences.
    Intrinsic +/- pairs in patches with opposite shears make the global
    weighted mean ellipticity zero, so c=0. The patch's weighted uncalibrated
    mean is exactly <R>_w*g. Dividing by the matching response (des: 1.25,
    iv: 0.75) recovers g; the unweighted response 1.0 gives 1.25*g or 0.75*g.
    """
    path = tmp_path / "cat.fits"
    weights, in_patch = _write_catalogue(path)
    w = weights[weight_type]
    g1, _ = get_calibrate_e_from_cat(str(path), weight_type=weight_type)
    g_hat = np.average(g1[in_patch], weights=w[in_patch])
    assert g_hat == pytest.approx(G_TRUE, rel=0, abs=1e-12), (
        f"weight_type={weight_type!r}: weighted patch shear {g_hat:.6f} "
        f"!= truth {G_TRUE}; response and additive bias must share weights"
    )
