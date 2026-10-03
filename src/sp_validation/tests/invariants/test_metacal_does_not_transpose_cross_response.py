"""Metacal must preserve both cross-derivatives in its per-object response.

The total response R = R_shear + R_selection, with R_ab = d<e_a>/d gamma_b
for both terms. Applying the inverse full response to the no-shear
ellipticity must recover that ellipticity.
"""

import numpy as np
import pytest
from astropy.table import Table

from sp_validation.calibration import get_calibrated_quantities, metacal

pytestmark = [pytest.mark.fast, pytest.mark.decision("calibration.response_estimator")]


_VARIANTS = ("1M", "1P", "2M", "2P", "NOSHEAR")
_N = 8
_RESPONSE = np.array([[0.7, 0.1], [0.2, 0.9]])
# Per-object no-shear ellipticities; distinct per component so that pairing
# the wrong component with a shear branch changes the selection response.
_NOSHEAR = np.stack(
    [0.05 + 0.001 * np.arange(_N), -0.02 - 0.002 * np.arange(_N)], axis=1
)
# Objects rejected by flags in one sheared branch only: this produces a
# non-zero selection response with R12_s != R21_s.
_REJECTED = {"1M": 0, "2P": 1}


def _build_ngmix_catalog(step):
    """Build a catalogue with known non-diagonal shear and selection response."""
    shifts = {
        "NOSHEAR": np.zeros(2),
        "1P": step * _RESPONSE[:, 0],
        "1M": -step * _RESPONSE[:, 0],
        "2P": step * _RESPONSE[:, 1],
        "2M": -step * _RESPONSE[:, 1],
    }
    columns = {}
    for variant in _VARIANTS:
        ellipticity = _NOSHEAR + shifts[variant]
        flags = np.zeros(_N, dtype=int)
        if variant in _REJECTED:
            flags[_REJECTED[variant]] = 1
        columns[f"NGMIX_G1_{variant}"] = ellipticity[:, 0]
        columns[f"NGMIX_G2_{variant}"] = ellipticity[:, 1]
        columns[f"NGMIX_FLAGS_{variant}"] = flags
        columns[f"NGMIX_FLUX_{variant}"] = np.full(_N, 50.0)
        columns[f"NGMIX_FLUX_ERR_{variant}"] = np.ones(_N)
        columns[f"NGMIX_T_{variant}"] = np.ones(_N)
        columns[f"NGMIX_T_ERR_{variant}"] = np.full(_N, 0.1)
        columns[f"NGMIX_T_PSF_RECONV_{variant}"] = np.ones(_N)

    columns["NGMIX_G1_ERR_NOSHEAR"] = np.full(_N, 0.25)
    columns["NGMIX_G2_ERR_NOSHEAR"] = np.full(_N, 0.25)
    return Table(columns)


def _expected_selection_response(step):
    """R_s[a, b] = (<e_a^ns>_{b+} - <e_a^ns>_{b-}) / 2h over selected objects."""
    selected = {}
    for variant in ("1P", "1M", "2P", "2M"):
        keep = np.ones(_N, dtype=bool)
        if variant in _REJECTED:
            keep[_REJECTED[variant]] = False
        selected[variant] = keep
    r_sel = np.empty((2, 2))
    for b in range(2):
        plus, minus = selected[f"{b + 1}P"], selected[f"{b + 1}M"]
        r_sel[:, b] = (_NOSHEAR[plus].mean(0) - _NOSHEAR[minus].mean(0)) / (2 * step)
    return r_sel


def test_metacal_does_not_transpose_cross_response():
    """Statistical false-alarm probability is 0 for this deterministic identity;
    ``atol=1e-12`` allows only floating-point error on the specified fixture."""
    for step in (0.01, 0.02):
        expected_response = _RESPONSE + _expected_selection_response(step)
        data = _build_ngmix_catalog(step)
        mcal = metacal(
            data,
            np.ones(len(data), dtype=bool),
            step=step,
            size_corr_ell=False,
            global_R_weight=None,
        )

        np.testing.assert_allclose(
            mcal.R,
            expected_response,
            rtol=0,
            atol=1e-12,
            err_msg=f"Metacal response transposed a cross-derivative at step {step}",
        )

        g_cal, e_ns, _, _ = get_calibrated_quantities(mcal)
        np.testing.assert_allclose(
            expected_response @ g_cal,
            e_ns,
            rtol=0,
            atol=1e-12,
            err_msg=(
                f"Inverse response did not recover no-shear ellipticity at step {step}"
            ),
        )
