"""UNIT TESTS FOR THE IMAGE-SIMULATION m/c ESTIMATOR.

Exercise ``sp_validation.image_sims.ImageSimMBias`` -- the multiplicative and
additive shear-bias estimator used by the image-simulation workflow -- and the
``sp_validation.catalog.match_catalogs_radec`` helper it relies on.

The estimator recovers ``m`` and ``c`` from five calibrated catalogues named
``1z2z`` (reference, no input shear), ``1p2z``/``1m2z`` (input shear
``g1 = +-|g|``) and ``1z2p``/``1z2m`` (``g2 = +-|g|``). Objects are matched to
the reference by RA/Dec, then, per component,

    m = (<e_+> - <e_->) / (2 |g|) - 1 ,   c = (<e_+> + <e_->) / 2 .

We build synthetic catalogues in which the measured ellipticity is exactly
``e = (1 + m_true) g_in + c_true`` at shared positions, so the recovered m/c
must equal the injected values to machine precision -- an analytic check of
the estimator maths that needs no pipeline run.

:Author: cdaley

"""

import numpy as np
import numpy.testing as npt
from astropy.io import fits

from sp_validation.catalog import match_catalogs_radec
from sp_validation.image_sims import ImageSimMBias

# Injected truth, shared across the synthetic-recovery test.
A = 0.02  # input shear amplitude |g|
M_TRUE = 0.05  # multiplicative bias (same for both components)
C1_TRUE = 0.001  # additive bias, component 1
C2_TRUE = -0.002  # additive bias, component 2
N_GAL = 2000


def _write_cat(path, ra, dec, e1, e2, w):
    """Write a minimal calibrated shape catalogue (RA, Dec, e1, e2, w_des)."""
    cols = [
        fits.Column(name=name, array=arr, format="D")
        for name, arr in (
            ("RA", ra),
            ("Dec", dec),
            ("e1", e1),
            ("e2", e2),
            ("w_des", w),
        )
    ]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(
        path, overwrite=True
    )


def _make_grid(grids_dir, num):
    """Create the five sheared/reference catalogues with a known m/c."""
    rng = np.random.default_rng(0)
    ra = 30.0 + rng.uniform(0, 0.1, N_GAL)
    dec = rng.uniform(0, 0.1, N_GAL)
    w = np.ones(N_GAL)
    zero = np.zeros(N_GAL)

    def e(g_in):
        return (1 + M_TRUE) * g_in + zero

    sims = {
        "1z2z": (C1_TRUE + zero, C2_TRUE + zero),
        "1p2z": (e(+A) + C1_TRUE, C2_TRUE + zero),
        "1m2z": (e(-A) + C1_TRUE, C2_TRUE + zero),
        "1z2p": (C1_TRUE + zero, e(+A) + C2_TRUE),
        "1z2m": (C1_TRUE + zero, e(-A) + C2_TRUE),
    }
    for name, (e1, e2) in sims.items():
        sim_dir = grids_dir / f"{name}_grid_{num}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        _write_cat(sim_dir / "cat.fits", ra, dec, e1, e2, w)


def test_match_catalogs_radec_identity():
    """Identical positions match one-to-one; a shifted object drops out."""
    ra = np.array([30.0, 30.01, 30.02])
    dec = np.array([10.0, 10.01, 10.02])
    # Second catalogue = first, but the last object nudged well past threshold.
    ra2, dec2 = ra.copy(), dec.copy()
    ra2[2] += 1.0
    idx1, idx2 = match_catalogs_radec(ra, dec, ra2, dec2, thresh_deg=0.0002)
    npt.assert_array_equal(idx2, [0, 1])
    npt.assert_array_equal(idx1, [0, 1])


def test_mbias_recovers_injected_values(tmp_path):
    """ImageSimMBias recovers the injected m/c to machine precision."""
    num = 7
    _make_grid(tmp_path, num)
    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_col": "w_des",
        "n_bootstrap": 50,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)

    npt.assert_allclose(res["m1"], M_TRUE, atol=1e-9)
    npt.assert_allclose(res["m2"], M_TRUE, atol=1e-9)
    npt.assert_allclose(res["c1"], C1_TRUE, atol=1e-9)
    npt.assert_allclose(res["c2"], C2_TRUE, atol=1e-9)
    # Bootstrap errors are non-negative and finite.
    for key in ("m1_err", "m2_err", "c1_err", "c2_err"):
        assert np.isfinite(res[key]) and res[key] >= 0
