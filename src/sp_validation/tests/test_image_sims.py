"""UNIT TESTS FOR THE IMAGE-SIMULATION m/c ESTIMATOR.

Exercise ``sp_validation.image_sims.ImageSimMBias`` -- the multiplicative and
additive shear-bias estimator used by the image-simulation workflow -- and the
``sp_validation.catalog.match_catalogs_radec`` helper it relies on.

The estimator recovers ``m`` and ``c`` from five calibrated catalogues named
``1z2z`` (reference, no input shear), ``1p2z``/``1m2z`` (input shear
``g1 = +-|g|``) and ``1z2p``/``1z2m`` (``g2 = +-|g|``). The +g and -g sims are
matched to *each other* by RA/Dec -- same galaxies, opposite input shear -- and
the bias is the object-paired ("pool") average

    m = <(e_+ - e_-) / (2 |g|) - 1> ,   c = <(e_+ + e_-) / 2> ,

so the intrinsic shape cancels object-by-object in ``m``.

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
        "w_cols": ["w_des"],
        "n_bootstrap": 50,
        "pair_match": True,
        "bootstrap_seed": 42,
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
    # Self-describing results: the primary scheme is mirrored at the top level
    # *and* lives under ``weights[scheme]``, and the two agree exactly.
    assert list(res["weights"]) == ["w_des"]
    for key in ("m1", "m1_err", "c1", "c1_err", "m2", "m2_err", "c2", "c2_err"):
        assert res[key] == res["weights"]["w_des"][key]


def test_mbias_multiple_weight_schemes_share_draws(tmp_path):
    """Multi-scheme runs key results per scheme; the primary mirrors the first.

    ``none`` (unit weights) and a real weight column are computed in one run.
    With uniform per-object weights in the synthetic grid the two schemes give
    the *same* m/c (the weighting is a no-op), and the shared bootstrap indices
    make even the errors identical -- the property that lets a scheme
    comparison be a clean weighting comparison. The first entry (``none``) is
    the primary result surfaced at the top level.
    """
    num = 8
    _make_grid(tmp_path, num)
    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["none", "w_des"],
        "n_bootstrap": 50,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)

    assert list(res["weights"]) == ["none", "w_des"]
    # Primary (first) scheme mirrored at the top level.
    for key in ("m1", "m1_err", "c1", "c1_err", "m2", "m2_err", "c2", "c2_err"):
        assert res[key] == res["weights"]["none"][key]
    # Uniform grid weights make the schemes agree bit-for-bit, errors included
    # (shared bootstrap draws).
    assert res["weights"]["none"] == res["weights"]["w_des"]


def test_mbias_deprecated_w_col_still_runs(tmp_path):
    """A pre-``w_cols`` config with the scalar ``w_col`` still runs.

    The deprecated single-scheme key is honoured as ``[w_col]`` when ``w_cols``
    is absent, so a legacy run config keeps working and produces the same
    single-scheme result as the ``w_cols=[w_col]`` spelling.
    """
    num = 9
    _make_grid(tmp_path, num)
    base = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "n_bootstrap": 50,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    res_dep = ImageSimMBias({**base, "w_col": "w_des"})
    res_dep.load_catalogs(verbose=False)
    out_dep = res_dep.run(verbose=False)

    res_new = ImageSimMBias({**base, "w_cols": ["w_des"]})
    res_new.load_catalogs(verbose=False)
    out_new = res_new.run(verbose=False)

    assert list(out_dep["weights"]) == ["w_des"]
    assert out_dep["weights"] == out_new["weights"]


def test_mbias_pool_cancels_shape_noise(tmp_path):
    """The paired estimator cancels intrinsic shape noise in m.

    With realistic per-galaxy intrinsic ellipticity (sigma_e ~ 0.3) shared
    between the +g and -g sims plus small independent measurement noise, the
    object-paired difference cancels the intrinsic shape, so sigma(m) is set by
    the measurement noise (~1e-2), not the shape noise. An *unpaired* estimator
    (differencing two independently-drawn means) would instead return
    sigma(m) ~ sigma_e / (2 |g| sqrt(N)) -- an order of magnitude larger. We
    assert the recovered error sits well below that shape-noise floor, which is
    the property the pooling exists to deliver.
    """
    num = 3
    rng = np.random.default_rng(1)
    ra = 30.0 + rng.uniform(0, 0.1, N_GAL)
    dec = rng.uniform(0, 0.1, N_GAL)
    w = np.ones(N_GAL)
    sigma_e, sigma_meas = 0.3, 0.01
    e1_int = rng.normal(0, sigma_e, N_GAL)  # intrinsic shape, shared across sims
    e2_int = rng.normal(0, sigma_e, N_GAL)

    def measured(g1_in, g2_in):
        """Measured ellipticity = intrinsic + (1 + m) * input shear + noise."""
        e1 = e1_int + (1 + M_TRUE) * g1_in + rng.normal(0, sigma_meas, N_GAL)
        e2 = e2_int + (1 + M_TRUE) * g2_in + rng.normal(0, sigma_meas, N_GAL)
        return e1, e2

    sims = {
        "1z2z": measured(0, 0),
        "1p2z": measured(+A, 0),
        "1m2z": measured(-A, 0),
        "1z2p": measured(0, +A),
        "1z2m": measured(0, -A),
    }
    for name, (e1, e2) in sims.items():
        sim_dir = tmp_path / f"{name}_grid_{num}"
        sim_dir.mkdir(parents=True, exist_ok=True)
        _write_cat(sim_dir / "cat.fits", ra, dec, e1, e2, w)

    config = {
        "grids_dir": str(tmp_path),
        "num": num,
        "catalog_name": "cat.fits",
        "shear_amplitude": A,
        "match_radius_deg": 0.0002,
        "w_cols": ["w_des"],
        "n_bootstrap": 200,
        "pair_match": True,
        "bootstrap_seed": 42,
    }
    mb = ImageSimMBias(config)
    mb.load_catalogs(verbose=False)
    res = mb.run(verbose=False)

    shape_noise_floor = sigma_e / (2 * A * np.sqrt(N_GAL))  # the unpaired error
    for comp in (1, 2):
        # m recovered within a few sigma of truth...
        assert abs(res[f"m{comp}"] - M_TRUE) < 5 * res[f"m{comp}_err"]
        # ...and its error is far below what an unpaired estimator would give.
        assert res[f"m{comp}_err"] < 0.1 * shape_noise_floor
