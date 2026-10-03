"""The m/c bootstrap errors must include variation in raw shape and response.

With common seeded draws, both returned errors must equal independent
raw-shape/response ratio-bootstrap standard deviations.
"""

import numpy as np
import pytest
from astropy.io import fits

from sp_validation.image_sims import ImageSimMBias

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("shear_bias_simulations.mbias_uncertainty"),
]

G_IN = 0.02
N_OBJECTS = 8
N_BOOTSTRAP = 32
BOOTSTRAP_SEED = 913
RTOL = 1e-10
ATOL = 1e-12


def _write_branch(path, raw_e, response, ra, dec):
    """Write raw and calibrated shapes plus per-object response columns."""
    calibrated_e = raw_e / response.mean(axis=1)[:, None]
    columns = {
        "RA": ra,
        "Dec": dec,
        "e1": calibrated_e[0],
        "e2": calibrated_e[1],
        "e1_uncal": raw_e[0],
        "e2_uncal": raw_e[1],
        "R_g11": response[0],
        "R_g22": response[1],
        "R_g12": np.zeros(len(ra)),
        "R_g21": np.zeros(len(ra)),
    }
    header = fits.Header()
    for i in range(2):
        for j in range(2):
            suffix = f"{i + 1}{j + 1}"
            value = response[i].mean() if i == j else 0.0
            header[f"R_{suffix}"] = value
            header[f"R_G{suffix}"] = value
            header[f"R_S{suffix}"] = 0.0
    header["c1"] = 0.0
    header["c2"] = 0.0
    hdus = [
        fits.PrimaryHDU(header=header),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name=name, format="D", array=value)
                for name, value in columns.items()
            ]
        ),
    ]
    fits.HDUList(hdus).writeto(path)


def _ratio_bootstrap_errors(raw_plus, raw_minus, r_plus, r_minus, draws):
    """Calculate m/c errors from independently resampled raw e and response."""
    errors = {}
    for component in range(2):
        ep = raw_plus[component][draws].mean(axis=1) / r_plus[component][draws].mean(
            axis=1
        )
        em = raw_minus[component][draws].mean(axis=1) / r_minus[component][draws].mean(
            axis=1
        )
        m_boot = (ep - em) / (2 * G_IN) - 1
        c_boot = (ep + em) / 2
        errors[f"m{component + 1}_err"] = np.std(m_boot, ddof=0)
        errors[f"c{component + 1}_err"] = np.std(c_boot, ddof=0)
    return errors


def _fixed_response_errors(raw_plus, raw_minus, r_plus, r_minus, draws):
    """Calculate the same errors after fixing each branch's full-sample response."""
    calibrated_plus = raw_plus / r_plus.mean(axis=1)[:, None]
    calibrated_minus = raw_minus / r_minus.mean(axis=1)[:, None]
    errors = {}
    for component in range(2):
        ep = calibrated_plus[component][draws].mean(axis=1)
        em = calibrated_minus[component][draws].mean(axis=1)
        m_boot = (ep - em) / (2 * G_IN) - 1
        c_boot = (ep + em) / 2
        errors[f"m{component + 1}_err"] = np.std(m_boot, ddof=0)
        errors[f"c{component + 1}_err"] = np.std(c_boot, ddof=0)
    return errors


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#386: the loader drops raw e/response "
        "inputs, so bootstrap draws hold the branch response fixed"
    ),
)
def test_mbias_bootstrap_resamples_response_with_raw_shape(tmp_path):
    """With fixed draws, the identity has statistical false-alarm probability 0;
    the stated tolerances allow only floating-point error."""
    noise_1 = np.array([0.001, -0.001, 0.002, -0.002, 0.003, -0.003, 0.004, -0.004])
    noise_2 = np.array([-0.002, 0.001, 0.003, -0.004, 0.002, -0.001, 0.004, -0.003])
    raw_plus = np.array([G_IN * 0.7 + noise_1, G_IN * 0.45 + noise_2])
    raw_minus = np.array([-G_IN * 0.7 + noise_1[::-1], -G_IN * 0.45 + noise_2[::-1]])
    r_plus = np.array(
        [
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            [0.65, 0.95, 0.55, 1.05, 0.75, 0.85, 1.15, 0.45],
        ]
    )
    r_minus = np.array(
        [
            [1.1, 0.9, 0.7, 0.5, 0.4, 0.6, 0.8, 1.0],
            [1.0, 0.6, 0.9, 0.5, 1.1, 0.7, 0.8, 0.4],
        ]
    )
    ra = 100.0 + np.arange(N_OBJECTS) * 0.01
    dec = np.full(N_OBJECTS, 20.0)
    branches = {
        "plus": (raw_plus, r_plus),
        "minus": (raw_minus, r_minus),
    }
    for branch, (raw_e, response) in branches.items():
        directory = tmp_path / f"{branch}_grid_1"
        directory.mkdir()
        _write_branch(
            directory / "shape_catalog_cut_ngmix.fits", raw_e, response, ra, dec
        )

    config = {
        "grids_dir": str(tmp_path),
        "num": 1,
        "branches": ["plus", "minus"],
        "pairs": [
            {"plus": "plus", "minus": "minus", "component": 0},
            {"plus": "plus", "minus": "minus", "component": 1},
        ],
        "shear_amplitude": G_IN,
        "match_radius_deg": 1e-5,
        "pair_match": True,
        "w_cols": [None],
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    estimator = ImageSimMBias(config)
    estimator.load_catalogs(verbose=False)
    result = estimator.run(verbose=False)

    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, N_OBJECTS, (N_BOOTSTRAP, N_OBJECTS)
    )
    expected = _ratio_bootstrap_errors(raw_plus, raw_minus, r_plus, r_minus, draws)
    fixed = _fixed_response_errors(raw_plus, raw_minus, r_plus, r_minus, draws)
    for key, expected_error in expected.items():
        tolerance = ATOL + RTOL * abs(expected_error)
        assert abs(expected_error - fixed[key]) > 10 * tolerance, (
            f"fixture does not distinguish joint raw e/response resampling for {key}: "
            f"joint={expected_error:.16g}, fixed-response={fixed[key]:.16g}, "
            f"10*tolerance={10 * tolerance:.3g}"
        )

    for key, expected_error in expected.items():
        np.testing.assert_allclose(
            result[key],
            expected_error,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{key} must include jointly resampled raw shapes and response",
        )
