"""A failed leakage fit must not return a coefficient of -99."""

from types import SimpleNamespace

import numpy as np
import pytest

from sp_validation.catalog_builders import compute_PSF_leakage


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#382: a failed leakage fit returns alpha=-99",
)
def test_leakage_fit_failure_does_not_return_minus_99_sentinel():
    """A failed fit must surface, rather than corrupt released ellipticities.

    calibrate_comprehensive_cat.py applies returned alpha directly as
    e_leak_corrected=e-alpha*e_PSF. Eight galaxies with identical size ratios
    make pd.qcut fail because bin edges cannot be distinct. Correct behaviour
    is to raise, or return a genuine finite leakage estimate with |alpha|<1.
    The -99 sentinel instead turns e1=0.1 and e1_PSF=0.02 into 2.08 and must
    never be applied to the catalogue.
    """
    n = 8
    data = np.zeros(n, dtype=[("e1_PSF", "f8"), ("e2_PSF", "f8")])
    data["e1_PSF"] = 0.02
    data["e2_PSF"] = -0.01
    cat_gal = {
        "w_des": np.ones(n),
        "NGMIX_T_PSF_RECONV_NOSHEAR": np.ones(n),
        "NGMIX_T_NOSHEAR": np.ones(n),  # Tied size ratios: qcut fails.
        "snr": np.linspace(20, 80, n),
    }
    shear = np.vstack([np.full(n, 0.1), np.full(n, -0.03)])
    mask = SimpleNamespace(_mask=np.ones(n, dtype=bool))
    try:
        alpha_1, alpha_2 = compute_PSF_leakage(
            cat_gal, shear, data, mask, np.ones(n, dtype=bool), num_bins=4
        )
    except (ValueError, RuntimeError):
        # qcut failure (or an explicit fit failure) surfaced to the caller.
        return
    e1_corr = shear[0] - np.asarray(alpha_1) * cat_gal["e1_PSF"]
    assert np.all(np.abs(np.asarray(alpha_1)) < 1) and np.all(
        np.abs(np.asarray(alpha_2)) < 1
    ), (
        f"leakage fit failed silently: alpha_1={alpha_1!r}, alpha_2={alpha_2!r}; "
        f"released e1_leak_corrected={e1_corr[0]:.4f} from e1={shear[0, 0]}"
    )
