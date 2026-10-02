"""The leakage trend fit must weight bins by inverse variance, not its square."""

import numpy as np
import pytest
import statsmodels.api as sm

from sp_validation import calibration


def _synthetic_catalogue(n=120_000, seed=7):
    """Misspecified alpha(SNR,R) with heteroscedastic noise across bins."""
    rng = np.random.default_rng(seed)
    snr = np.exp(rng.uniform(np.log(8), np.log(200), n))
    size_ratio = rng.uniform(0.3, 0.9, n)
    e1_psf, e2_psf = rng.normal(0, 0.03, (2, n))
    alpha = 0.05 + 0.8 * np.exp(-snr / 12) + 0.3 * size_ratio**3
    sigma = 0.1 * (1 + 60 / snr)
    return {
        "e1": alpha * e1_psf + rng.normal(0, sigma),
        "e2": alpha * e2_psf + rng.normal(0, sigma),
        "e1_PSF": e1_psf,
        "e2_PSF": e2_psf,
        "snr": snr,
        "w_des": np.ones(n),
        "NGMIX_T_PSF_RECONV_NOSHEAR": np.ones(n),
        "NGMIX_T_NOSHEAR": 1 / size_ratio - 1,
    }


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#383: trend weights bins by 1/error**4",
)
def test_leakage_trend_fit_weights_bins_by_inverse_variance(monkeypatch):
    """Protect the trend coefficients that determine per-object leakage.

    First-pass estimates alpha_i +/- sigma_i are fitted with a five-term
    trend in SNR and size ratio. A chi-square fit minimises
    sum_i (X_i beta-alpha_i)**2/sigma_i**2: least-squares rows must be scaled
    by 1/sigma_i, not 1/sigma_i**2. The latter weights residuals by 1/sigma**4.
    Spies preserve all real WLS/lstsq computations and record the first-pass
    estimates and trend design. An independent statsmodels WLS with weights
    1/sigma**2 supplies the correct bin-centre trend. The synthetic model is
    deliberately misspecified, so unequal bin errors expose wrong weighting.
    """
    wls_records = []
    real_wls = sm.WLS

    class SpyWLS(real_wls):
        def fit(self, *args, **kwargs):
            result = super().fit(*args, **kwargs)
            wls_records.append((result.params[1], np.sqrt(result.cov_params()[1, 1])))
            return result

    monkeypatch.setattr(calibration.sm, "WLS", SpyWLS)
    lstsq_records = []
    real_lstsq = np.linalg.lstsq

    def spy_lstsq(design, response, *args, **kwargs):
        result = real_lstsq(design, response, *args, **kwargs)
        lstsq_records.append((np.array(design), np.array(response), result[0]))
        return result

    monkeypatch.setattr(np.linalg, "lstsq", spy_lstsq)
    num_bins = 5
    calibration.get_alpha_leakage_per_object(_synthetic_catalogue(), num_bins)
    first = np.array(wls_records[: 2 * num_bins**2])
    alpha1, err1 = first[0::2, 0], first[0::2, 1]
    design_scaled, response_scaled, poly_code = lstsq_records[0]
    scale = design_scaled[:, 0]  # The intercept column is the row scale.
    design = design_scaled / scale[:, None]
    np.testing.assert_allclose(response_scaled / scale, alpha1, rtol=1e-10)
    poly_ref = real_wls(alpha1, design, weights=1 / err1**2).fit().params
    trend_code, trend_ref = design @ poly_code, design @ poly_ref
    np.testing.assert_allclose(trend_code, trend_ref, rtol=0, atol=1e-6)
