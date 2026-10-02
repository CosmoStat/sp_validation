"""Synthetic end-to-end regression for uncertainty in branch calibration responses."""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import image_sims
from sp_validation.calibration import get_calibrated_m_c
from sp_validation.image_sims import ImageSimMBias

N_OBJECTS = 96
N_REALISATIONS = 192
N_BOOTSTRAP = 256
G = 0.02
R_TRUE = 0.6
SIGMA_R = 0.3
SIGMA_E = 0.004


def _branch_hdus(raw, response):
    """Use the production calibration, and preserve its sufficient statistics."""
    n = len(raw[0])
    global_response = np.diag(response.mean(axis=1))
    meta = SimpleNamespace(
        R=global_response,
        ns={"g1": raw[0], "g2": raw[1], "w": np.ones(n)},
        mask_dict={"ns": np.ones(n, dtype=bool)},
    )
    calibrated, *_ = get_calibrated_m_c(meta, additive_correction=False)
    # The reference does not use the production calibration calculation.
    np.testing.assert_allclose(calibrated, raw / response.mean(axis=1)[:, None])
    columns = {
        "RA": 100.0 + np.arange(n) * 0.001,
        "Dec": np.full(n, 20.0),
        "e1": calibrated[0],
        "e2": calibrated[1],
        "e1_uncal": raw[0],
        "e2_uncal": raw[1],
        "R_g11": response[0],
        "R_g22": response[1],
        "R_g12": np.zeros(n),
        "R_g21": np.zeros(n),
    }
    header = fits.Header()
    for i in range(2):
        for j in range(2):
            suffix = f"{i + 1}{j + 1}"
            header[f"R_{suffix}"] = global_response[i, j]
            header[f"R_G{suffix}"] = global_response[i, j]
            header[f"R_S{suffix}"] = 0.0
    header["c1"] = 0.0
    header["c2"] = 0.0
    return fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.BinTableHDU.from_columns(
                [fits.Column(name=k, format="D", array=v) for k, v in columns.items()]
            ),
        ]
    )


@pytest.mark.parametrize("component", [0, 1], ids=["g1", "g2"])
@pytest.mark.xfail(
    strict=True,
    reason="#386: bootstrap holds noisy branch responses fixed",
)
def test_m_bias_bootstrap_propagates_branch_response_noise(component, monkeypatch):
    """Protect the reported m error against conditioning on a noisy response.

    Each branch measures raw e = +/-g*R_true + independent measurement noise,
    and estimates its response from independent noisy per-object R_g values;
    selection response is exactly zero and every object has one matched partner.
    Thus m is the difference of two ratios of sample means, whose repeat-run
    scatter includes both numerator and denominator noise, even with no intrinsic
    shape noise; bootstrapping both together must recover that scatter within
    [0.85, 1.2], whereas resampling already-calibrated e holds denominators fixed.
    An independent ratio-of-means bootstrap checks that this tolerance is valid.
    """
    rng = np.random.default_rng(70831 + component)
    draw = np.random.default_rng(913).integers(0, N_OBJECTS, (N_BOOTSTRAP, N_OBJECTS))
    values, reported_errors, reference_errors = [], [], []
    catalogues = {}
    directory = "synthetic-memory"
    # Only replace storage IO: production calibration, FITS loader, matcher and
    # estimator all execute normally, without hundreds of tiny filesystem writes.
    monkeypatch.setattr(
        image_sims.fits, "open", lambda path, **kwargs: catalogues[str(path)]
    )
    config = {
        "grids_dir": directory,
        "num": 1,
        "branches": ["plus", "minus"],
        "pairs": [{"plus": "plus", "minus": "minus", "component": component}],
        "shear_amplitude": G,
        "match_radius_deg": 1e-6,
        "pair_match": True,
        "w_cols": ["none"],
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": 913,
    }
    for _ in range(N_REALISATIONS):
        raw = rng.normal(0.0, SIGMA_E, (2, 2, N_OBJECTS))
        raw[0, component] += G * R_TRUE
        raw[1, component] -= G * R_TRUE
        response = rng.normal(R_TRUE, SIGMA_R, (2, 2, N_OBJECTS))
        for branch, e, r in zip(["plus", "minus"], raw, response):
            path = f"{directory}/{branch}_grid_1/shape_catalog_cut_ngmix.fits"
            catalogues[path] = _branch_hdus(e, r)
        estimator = ImageSimMBias(config)
        estimator.load_catalogs(verbose=False)
        result = estimator.run(verbose=False)
        values.append(result[f"m{component + 1}"])
        reported_errors.append(result[f"m{component + 1}_err"])
        # Independent reference: paired resample of BOTH raw e and R_g.
        ep, em = raw[:, component]
        rp, rm = response[:, component]
        expected_m = (ep.mean() / rp.mean() - em.mean() / rm.mean()) / (2 * G) - 1
        assert values[-1] == pytest.approx(expected_m, abs=1e-12)
        m_boot = (
            ep[draw].mean(axis=1) / rp[draw].mean(axis=1)
            - em[draw].mean(axis=1) / rm[draw].mean(axis=1)
        ) / (2 * G) - 1
        reference_errors.append(m_boot.std())
    scatter = np.std(values, ddof=1)
    reported = np.mean(reported_errors)
    reference = np.mean(reference_errors)
    # Independent first-order propagation; curvature corrections are small at n=96.
    analytic = np.sqrt((SIGMA_E / G) ** 2 + SIGMA_R**2) / (
        R_TRUE * np.sqrt(2 * N_OBJECTS)
    )
    assert 0.85 <= scatter / analytic <= 1.2, (
        "Synthetic ensemble misses analytic scatter"
    )
    assert 0.85 <= scatter / reference <= 1.2, (
        "Joint bootstrap misses repeat-run scatter"
    )
    ratio = scatter / reported
    assert 0.85 <= ratio <= 1.2, (
        f"m{component + 1} uncertainty omits branch-response noise: "
        f"repeat-run scatter={scatter:.6f}, mean reported m_err={reported:.6f}, "
        f"scatter/reported={ratio:.3f} (required [0.85, 1.2]); "
        f"joint raw-e/R_g bootstrap={reference:.6f}, analytic={analytic:.6f}"
    )
