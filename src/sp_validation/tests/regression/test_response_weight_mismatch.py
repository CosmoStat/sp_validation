"""Global metacal response must use the weights of the shear statistic."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from sp_validation import calibration
from sp_validation.calibration import metacal

GAMMA = np.array([0.02, -0.01])
STEP = 0.01
POPULATIONS = [(0.6, 1.0), (0.9, 3.0)]  # (response, statistic weight)
N_PER_POP = 400  # Even: intrinsic ellipticities come in +/- pairs.


def _repository_root():
    root = Path(__file__).resolve().parents[4]
    # A cross-check can import another checkout via PYTHONPATH while keeping
    # this test file. Use that checkout's configuration as well as its code.
    source_root = Path(calibration.__file__).resolve().parents[2]
    if source_root != root and (source_root / "pyproject.toml").is_file():
        return source_root
    return root


def _synthetic_catalogue():
    """ngmix-style catalogue with known responses and zero selection response."""
    rng = np.random.default_rng(1)
    names = ["1M", "1P", "2M", "2P", "NOSHEAR"]
    fields = []
    for name in names:
        fields.extend(
            [
                (f"NGMIX_FLAGS_{name}", "i4"),
                (f"NGMIX_G1_{name}", "f8"),
                (f"NGMIX_G2_{name}", "f8"),
                (f"NGMIX_FLUX_{name}", "f8"),
                (f"NGMIX_FLUX_ERR_{name}", "f8"),
                (f"NGMIX_T_{name}", "f8"),
                (f"NGMIX_T_ERR_{name}", "f8"),
                (f"NGMIX_T_PSF_RECONV_{name}", "f8"),
            ]
        )
    fields += [("NGMIX_G1_ERR_NOSHEAR", "f8"), ("NGMIX_G2_ERR_NOSHEAR", "f8")]
    ntot = N_PER_POP * len(POPULATIONS)
    data = np.zeros(ntot, dtype=fields)
    response = np.concatenate([np.full(N_PER_POP, r) for r, _ in POPULATIONS])
    w_des = np.concatenate([np.full(N_PER_POP, w) for _, w in POPULATIONS])
    intrinsic = np.empty((2, ntot))
    for k in range(len(POPULATIONS)):
        half = rng.normal(0, 0.25, size=(2, N_PER_POP // 2))
        intrinsic[:, k * N_PER_POP : (k + 1) * N_PER_POP] = np.concatenate(
            [half, -half], axis=1
        )
    g_ns = intrinsic + response * GAMMA[:, None]
    shifts = {
        "NOSHEAR": (0.0, 0.0),
        "1P": (STEP, 0.0),
        "1M": (-STEP, 0.0),
        "2P": (0.0, STEP),
        "2M": (0.0, -STEP),
    }
    for name, (d1, d2) in shifts.items():
        data[f"NGMIX_G1_{name}"] = g_ns[0] + response * d1
        data[f"NGMIX_G2_{name}"] = g_ns[1] + response * d2
        data[f"NGMIX_FLUX_{name}"] = 50.0
        data[f"NGMIX_FLUX_ERR_{name}"] = 1.0  # SNR 50: inside all cuts.
        data[f"NGMIX_T_{name}"] = 1.5
        data[f"NGMIX_T_PSF_RECONV_{name}"] = 1.0  # Size ratio inside all cuts.
    data["NGMIX_G1_ERR_NOSHEAR"] = 0.01
    data["NGMIX_G2_ERR_NOSHEAR"] = 0.01
    return data, w_des


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#395: global response and statistic weights differ",
)
def test_w_des_weighted_shear_unbiased_with_committed_global_response_weight():
    """Protect multiplicative calibration of production w_des statistics.

    mask_v1.X.6.yaml sets the weight used to average the global response;
    cosmo_val uses w_des with shear.R=1. A weighted shear estimator is unbiased
    only when the applied response is averaged with the statistic's weights.
    Intrinsic shapes cancel in pairs and selection response is zero here.
    Responses 0.6/0.9 with w_des=1/3 give <R>_w_des=0.825, while equal
    measurement errors give <R>_w_iv=0.75. The calibrated weighted mean must
    equal gamma, not 1.1*gamma.
    """
    root = _repository_root()
    config = yaml.safe_load((root / "config/calibration/mask_v1.X.6.yaml").read_text())[
        "metacal"
    ]
    stat_config = yaml.safe_load((root / "cosmo_val/cat_config.yaml").read_text())
    # Check the production statistic's weight, rather than an unused template.
    shear_config = stat_config["SP_v1.4.6.3"]["shear"]
    assert shear_config["w_col"] == "w_des"
    assert shear_config["R"] == 1.0
    data, w_des = _synthetic_catalogue()
    gal = metacal(
        data,
        np.ones(len(data), bool),
        snr_min=config["gal_snr_min"],
        snr_max=config["gal_snr_max"],
        rel_size_min=config["gal_rel_size_min"],
        rel_size_max=config["gal_rel_size_max"],
        size_corr_ell=config["gal_size_corr_ell"],
        sigma_eps=config["sigma_eps_prior"],
        global_R_weight=config["global_R_weight"],
    )
    assert len(gal.mask_dict["ns"]) == len(data)
    np.testing.assert_allclose(gal.R_selection, 0, atol=1e-12)
    g_corr, _, _, mask, _, _ = calibration.get_calibrated_m_c(
        gal, additive_correction=False
    )
    gamma_hat = np.array([np.average(g_corr[c], weights=w_des[mask]) for c in (0, 1)])
    assert np.allclose(gamma_hat, GAMMA, rtol=1e-6), (
        f"w_des-weighted shear biased: gamma_hat={gamma_hat}, gamma={GAMMA}, "
        f"m={gamma_hat / GAMMA - 1}; global_R_weight="
        f"{config['global_R_weight']!r}, applied diag={np.diag(gal.R)}, "
        "expected statistic response=0.825"
    )
