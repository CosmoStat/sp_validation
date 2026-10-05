"""ShapePipe G*_ERR columns store standard deviations, not variances."""

import numpy as np
import pytest

from sp_validation.calibration import metacal


def _catalogue():
    fields = []
    for shear in ("NOSHEAR", "1M", "1P", "2M", "2P"):
        fields.extend(
            (f"NGMIX_{name}_{shear}", "f8")
            for name in (
                "FLAGS",
                "G1",
                "G2",
                "FLUX",
                "FLUX_ERR",
                "T",
                "T_ERR",
                "T_PSF_RECONV",
            )
        )
    fields.extend((f"NGMIX_G{i}_ERR_NOSHEAR", "f8") for i in (1, 2))
    data = np.zeros(2, dtype=fields)
    for shear in ("NOSHEAR", "1M", "1P", "2M", "2P"):
        data[f"NGMIX_FLUX_{shear}"] = 50.0
        data[f"NGMIX_FLUX_ERR_{shear}"] = 1.0
        data[f"NGMIX_T_{shear}"] = 1.0
        data[f"NGMIX_T_PSF_RECONV_{shear}"] = 1.0
    # ERR columns store sqrt(diag(g_cov)), not diag(g_cov).
    data["NGMIX_G1_ERR_NOSHEAR"] = [0.05, 0.20]
    data["NGMIX_G2_ERR_NOSHEAR"] = [0.10, 0.30]
    # Exactly diagonal per-object shear responses 0.6 I and 1.0 I.
    for component in (1, 2):
        data[f"NGMIX_G{component}_{component}P"] = [0.006, 0.010]
        data[f"NGMIX_G{component}_{component}M"] = [-0.006, -0.010]
    return data


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#395: shape-error sigma is added to a variance",
)
@pytest.mark.parametrize("mask", [None, np.array([False, True])], ids=["all", "masked"])
def test_iv_weights_square_both_shape_error_standard_deviations(mask):
    """Protect weights from adding standard deviations to intrinsic variance.

    ShapePipe stores G1_ERR/G2_ERR as square roots of covariance diagonals.
    For sigma_eps=0.34 and errors (0.05,0.10)/(0.20,0.30), the variances are
    2*0.34**2+0.05**2+0.10**2=0.2437 and
    2*0.34**2+0.20**2+0.30**2=0.3612. Their reciprocals are the correct
    inverse-variance weights, including after masking.
    """
    _, _, actual = metacal.get_variance_ivweights(_catalogue(), 0.34, mask=mask)
    expected = 1.0 / np.array([0.2437, 0.3612])
    if mask is not None:
        expected = expected[mask]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#395: shape-error sigma biases weighted response",
)
def test_global_shear_response_uses_squared_shape_error_weights():
    """Protect the applied response from the same dimensional weight error.

    Both objects pass every sheared selection, so selection response is zero.
    Their finite-difference shear responses are 0.6 I and 1.0 I. With
    global_R_weight='w', the correct diagonal response is
    (0.6/0.2437+1/0.3612)/(1/0.2437+1/0.3612), using intrinsic plus
    measurement variances, rather than standard deviations.
    """
    result = metacal(
        _catalogue(),
        np.ones(2, dtype=bool),
        sigma_eps=0.34,
        global_R_weight="w",
        size_corr_ell=False,
    )
    np.testing.assert_allclose(result.R_selection, np.zeros((2, 2)), atol=1e-14)
    expected_scalar = (0.6 * 0.3612 + 1.0 * 0.2437) / (0.3612 + 0.2437)
    np.testing.assert_allclose(
        result.R, np.eye(2) * expected_scalar, rtol=1e-12, atol=1e-14
    )
