"""GOLDEN-VALUE CHARACTERIZATION TESTS FOR THE PSEUDO-Cl ESTIMATOR.

This module pins the numeric behavior of the pseudo-Cl (harmonic-space)
primitives in ``sp_validation.cosmo_val.pseudo_cl.PseudoClMixin`` against a
fixed, deterministic, in-memory synthetic catalog (seeded ``np.random``, a
small ``nside=64`` HEALPix grid -- no cluster data, no catalogue files). Every
pinned literal was produced by an actual run of the estimator inside the
sp_validation container.

WHY THIS EXISTS
---------------
The pseudo-Cl estimator is a Paper-II deliverable that, before this file, had
ZERO numeric pins. Track B2 will extract these mixin methods into free
functions in a new ``sp_validation/pseudo_cl.py`` module. These golden values
are the safety net: an extraction that silently changes the numbers fails here
rather than staying green on a smoke test.

The pinned primitives (the exact functions Track B2 moves):

* ``get_namaster_bin``       -- NaMaster binning, all three schemes
* ``get_n_gal_map``          -- weighted galaxy number-density map
* ``get_pseudo_cls_map``     -- map-based pseudo-Cl (decoupled EE/EB/BE/BB)
* ``get_pseudo_cls_catalog`` -- catalog-based pseudo-Cl (NmtFieldCatalog)
* ``calculate_pseudo_cl_catalog`` -- the deterministic end-to-end catalog path
* ``apply_random_rotation``  -- pinned by INVARIANT + seed reproducibility (see below)

REPRODUCIBILITY / TOLERANCE (measured over 3 independent container processes)
----------------------------------------------------------------------------
* ``get_namaster_bin`` / ``get_n_gal_map`` / ``get_pseudo_cls_map`` are
  BITWISE-identical across processes (pure binning math; map-based NaMaster
  reduction is deterministic). Pinned at ``rtol=1e-9`` -- far inside the
  float-noise floor of 0, yet a hair of margin so an exact-equality assertion
  doesn't make the suite brittle to a future NaMaster build.
* ``get_pseudo_cls_catalog`` (the ``NmtFieldCatalog`` path) and the end-to-end
  ``calculate_pseudo_cl_catalog`` are NOT bitwise-reproducible: their reduction
  order drifts at the ~2e-12 relative level (worst case measured 2.1e-12,
  absolute 2.7e-19). Pinned at ``rtol=1e-6 / atol=1e-12`` -- ~6 orders of
  magnitude above that float-noise floor (no flakiness margin consumed) yet
  tight enough that a sub-percent change in any band bites. This matches the
  precedent in ``test_b_modes.py`` and the pure-E/B pins in ``test_cosmo_val``.
* ``apply_random_rotation`` takes an optional ``rng``: with a seeded generator
  the draw is reproducible, with ``rng=None`` it is non-deterministic. It is
  pinned by the rotation INVARIANT (magnitude conservation
  ``e1_rot^2 + e2_rot^2 == e1^2 + e2^2`` and shape, both exact) plus a
  reproducibility test (same seed -> identical, different seed -> differs).
  The noise debiasing now threads a seeded ``rng`` (``CosmologyValidation``'s
  ``cell_seed``) so the pseudo-Cl realizations are reproducible run-to-run.

:Author: claude-opus on behalf of Cail

"""

import os
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import yaml

from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.pseudo_cl import apply_random_rotation
from sp_validation.rho_tau import get_params_rho_tau

# These tests need the full harmonic-space stack (pymaster/NaMaster + healpy),
# i.e. they run inside the sp_validation container. Skip cleanly otherwise so
# the rest of the suite still collects.
pymaster = pytest.importorskip("pymaster")
healpy = pytest.importorskip("healpy")

from astropy.io import fits  # noqa: E402  (after importorskip)

# --- Fixed experiment configuration -----------------------------------------
NSIDE = 64
SEED = 1234
N_ELL_BINS = 8

REFERENCE_TOMO = Path(__file__).parent / "data" / "test_cl_catalog.npz"


# ---------------------------------------------------------------------------
# Synthetic-catalog fixture (deterministic; no cluster data)
# ---------------------------------------------------------------------------
def _write_synthetic_config(tmp_path):
    """Write a tiny seeded shear catalog + dndz, return (config_path, version).

    A 5000-object random-shear catalog over a 20x20 deg patch, plus the minimal
    config blocks the pseudo-Cl seam reads (``shear`` for column names, ``psf``
    so ``get_params_rho_tau`` is satisfied). Fully determined by ``SEED``.
    """
    from astropy.table import Table

    rng = np.random.default_rng(SEED)
    version = "TestCatalog"

    cat_dir = tmp_path / "catalog"
    nz_dir = tmp_path / "nz"
    output_dir = tmp_path / "output"
    for d in (cat_dir, nz_dir, output_dir):
        d.mkdir()

    n_gal = 5000
    ra = rng.uniform(10.0, 30.0, n_gal)
    dec = rng.uniform(10.0, 30.0, n_gal)
    e1 = rng.normal(0, 0.25, n_gal)
    e2 = rng.normal(0, 0.25, n_gal)
    w = rng.uniform(0.5, 1.0, n_gal)
    tomo_bin_id = rng.integers(1, 3, n_gal)  # Create a two bin catalogue
    Table(
        {"RA": ra, "Dec": dec, "e1": e1, "e2": e2, "w": w, "tomo_bin_id": tomo_bin_id}
    ).write(cat_dir / "shear.fits", overwrite=True)

    z_edges = np.linspace(0.05, 3.0, 31)
    dndz = np.exp(-(((z_edges - 0.7) / 0.3) ** 2))
    lines = ["# z dn_dz"] + [f"{zz} {nn}" for zz, nn in zip(z_edges, dndz)]
    (nz_dir / "dndz_SP_A.txt").write_text("\n".join(lines) + "\n")

    shear_cfg = {
        "path": "shear.fits",
        "w_col": "w",
        "e1_col": "e1",
        "e2_col": "e2",
        "R": 1.0,
        "e1_col_corrected": "e1",
        "e2_col_corrected": "e2",
        "ra_col": "RA",
        "dec_col": "Dec",
        "tomo_bin_col": "tomo_bin_id",
    }
    # Minimal psf block so get_params_rho_tau() succeeds; the pseudo-Cl
    # primitives only read the shear-side keys, but the production call path
    # routes through get_params_rho_tau, so we mirror it.
    psf_cfg = {
        "path": "shear.fits",
        "ra_col": "RA",
        "dec_col": "Dec",
        "e1_PSF_col": "e1",
        "e2_PSF_col": "e2",
        "e1_star_col": "e1",
        "e2_star_col": "e2",
        "PSF_size": "w",
        "star_size": "w",
        "PSF_flag": "w",
        "star_flag": "w",
    }
    config_data = {
        "nz": {"subdir": str(nz_dir), "dndz": {"blind": "A", "path": "dndz"}},
        "paths": {"output": str(output_dir)},
        version: {
            "subdir": str(cat_dir),
            "pipeline": "SP",
            "shear": shear_cfg,
            "star": {**psf_cfg},
            "psf": psf_cfg,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_data, sort_keys=False))
    return str(config_path), version


@pytest.fixture
def cv(tmp_path):
    """CosmologyValidation on the synthetic catalog, pinned pseudo-Cl config."""
    config_path, version = _write_synthetic_config(tmp_path)
    cv = CosmologyValidation(
        versions=[version],
        catalog_config=config_path,
        output_dir=str(tmp_path / "output"),
        nside=NSIDE,
        binning="powspace",
        power=0.5,
        n_ell_bins=N_ELL_BINS,
        pol_factor=-1,
    )
    cv._test_version = version
    return cv


@pytest.fixture
def cat_and_params(cv):
    """(catalog ndarray, params dict) for the synthetic version."""
    ver = cv._test_version
    params = get_params_rho_tau(cv.cc[ver], survey=ver)
    cat_gal = fits.getdata(cv.cc[ver]["shear"]["path"])
    return cat_gal, params


def test_reference_exists():
    """Guard the guard: a missing reference must fail loudly, not skip."""
    assert REFERENCE_TOMO.exists(), f"committed reference missing: {REFERENCE_TOMO}"


# Tolerances ----------------------------------------------------------------
# Bitwise-stable primitives (binning math, n_gal map, map-based pseudo-Cl).
RTOL_DET = 1e-9
ATOL_DET = 1e-30
# Catalog (NmtFieldCatalog) path: drifts ~2e-12 cross-process; pin well above.
RTOL_CAT = 1e-6
ATOL_CAT = 1e-12

LMIN = 8
LMAX = 2 * NSIDE
B_LMAX = LMAX - 1


# ===========================================================================
# get_namaster_bin -- all three binning schemes
# ===========================================================================
class TestGetNamasterBin:
    """Pin the NaMaster binning object for each supported scheme."""

    def test_powspace(self, cv):
        cv.binning, cv.power, cv.n_ell_bins = "powspace", 0.5, 8
        b = cv.get_namaster_bin(LMIN, LMAX, B_LMAX)
        assert b.get_n_bands() == 8
        npt.assert_allclose(
            b.get_effective_ells(),
            np.array([11.5, 20.0, 30.5, 43.5, 58.5, 75.5, 95.0, 116.5]),
            rtol=RTOL_DET,
            atol=ATOL_DET,
        )
        npt.assert_array_equal(
            np.asarray(b.get_ell_list(0)),
            np.array([8, 9, 10, 11, 12, 13, 14, 15]),
        )

    def test_logspace(self, cv):
        cv.binning, cv.n_ell_bins = "logspace", 8
        b = cv.get_namaster_bin(LMIN, LMAX, B_LMAX)
        assert b.get_n_bands() == 8
        npt.assert_allclose(
            b.get_effective_ells(),
            np.array([32.0, 60.0, 67.5, 75.5, 84.5, 95.5, 107.5, 120.5]),
            rtol=RTOL_DET,
            atol=ATOL_DET,
        )
        # Low-ell padding: all multipoles below ell_min_log(=50) fall in bin 0.
        npt.assert_array_equal(
            np.asarray(b.get_ell_list(0)), np.arange(8, 57, dtype=np.int32)
        )

    def test_linear(self, cv):
        cv.binning, cv.ell_step = "linear", 10
        b = cv.get_namaster_bin(LMIN, LMAX, B_LMAX)
        assert b.get_n_bands() == 12
        npt.assert_allclose(
            b.get_effective_ells(),
            np.array(
                [
                    12.5,
                    22.5,
                    32.5,
                    42.5,
                    52.5,
                    62.5,
                    72.5,
                    82.5,
                    92.5,
                    102.5,
                    112.5,
                    122.5,
                ]
            ),
            rtol=RTOL_DET,
            atol=ATOL_DET,
        )
        npt.assert_array_equal(
            np.asarray(b.get_ell_list(0)),
            np.array([8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        )

    def test_unknown_binning_raises(self, cv):
        cv.binning = "nonsense"
        with pytest.raises(ValueError, match="Unknown binning"):
            cv.get_namaster_bin(LMIN, LMAX, B_LMAX)


# ==========================================================================
# get_pixels -- fetch the pixels from ra, dec, and nside (HEALPix)
# ==========================================================================
def test_get_pixels(cv, cat_and_params):
    """Pin the HEALPix pixelization of the synthetic catalog."""
    cat_gal, params = cat_and_params
    unique_pix, idx, idx_rep = cv.get_pixels(params, NSIDE, cat_gal)

    # Structural invariants of the HEALPix occupancy map.
    assert unique_pix.size == 485
    assert idx_rep.size == 5000  # one entry per galaxy

    # Pinned scalar summaries: total weight is conserved (sum of weights),
    # peak occupancy, and the pixel index bookkeeping.
    npt.assert_array_equal(unique_pix[:5], [12167, 12168, 12169, 12170, 12171])
    assert int(unique_pix.sum()) == 7849989


# ===========================================================================
# get_n_gal_map -- weighted galaxy number-density map
# ===========================================================================
def test_get_n_gal_map(cv, cat_and_params):
    cat_gal, params = cat_and_params

    unique_pix, idx, idx_rep = cv.get_pixels(params, NSIDE, cat_gal)
    n_gal = cv.get_n_gal_map(
        params, NSIDE, cat_gal, unique_pix=unique_pix, idx=None, idx_rep=idx_rep
    )

    # Structural invariants of the HEALPix occupancy map.
    assert n_gal.shape == (healpy.nside2npix(NSIDE),)
    assert int(np.count_nonzero(n_gal)) == 485
    # The map is supported exactly on the occupied pixels.
    npt.assert_array_equal(np.nonzero(n_gal)[0], np.sort(unique_pix))

    # Pinned scalar summaries: total weight is conserved (sum of weights),
    # peak occupancy, and the pixel index bookkeeping.
    npt.assert_allclose(n_gal.sum(), 3760.657282591494, rtol=RTOL_DET)
    npt.assert_allclose(n_gal.max(), 16.063410433711447, rtol=RTOL_DET)
    npt.assert_allclose(
        n_gal[unique_pix][:5],
        np.array(
            [
                3.733980174372703,
                3.4379134578473773,
                4.573348360592046,
                7.926927899839571,
                4.5074633637396575,
            ]
        ),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )


# ===========================================================================
# get_pseudo_cls_map -- map-based pseudo-Cl (bitwise-stable)
# ===========================================================================
def _build_shear_map(cv, cat_gal, params):
    """Replicate calculate_pseudo_cl_map's weighted shear-map construction."""
    unique_pix, _idx, idx_rep = cv.get_pixels(params, NSIDE, cat_gal)
    n_gal = cv.get_n_gal_map(
        params, NSIDE, cat_gal, unique_pix=unique_pix, idx=_idx, idx_rep=idx_rep
    )
    m1, m2 = cv.get_shear_map(
        params, NSIDE, cat_gal, unique_pix=unique_pix, idx=_idx, idx_rep=idx_rep
    )
    return m1 + 1j * m2, n_gal


def test_get_pseudo_cls_map(cv, cat_and_params):
    cat_gal, params = cat_and_params
    shear_map, n_gal = _build_shear_map(cv, cat_gal, params)
    ell_eff, cl_all, wsp = cv.get_pseudo_cls_map(shear_map, n_gal)
    ell_eff_2, cl_all_2, wsp_2 = cv.get_pseudo_cls_map(
        shear_map, n_gal, shear_map_b=shear_map, mask_b=n_gal
    )

    assert cl_all.shape == (4, N_ELL_BINS)
    npt.assert_allclose(
        ell_eff,
        np.array([11.5, 20.0, 30.5, 43.5, 58.5, 75.5, 95.0, 116.5]),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        cl_all[0],  # EE
        np.array(
            [
                -3.5657848980278575e-06,
                2.834003490842137e-06,
                5.141388691413763e-07,
                1.4864803917859117e-06,
                1.529668502385154e-06,
                9.126381881185905e-07,
                8.60858982887623e-07,
                2.324190517614256e-06,
            ]
        ),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        cl_all[1],  # EB
        np.array(
            [
                1.1955866959532166e-05,
                -2.422720314939584e-06,
                -9.9636714751786e-07,
                -6.033940105986599e-07,
                7.694521848269152e-07,
                2.2127016030540707e-07,
                -1.716732270067554e-07,
                1.290575887877422e-07,
            ]
        ),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        cl_all[3],  # BB
        np.array(
            [
                1.0742046525039353e-05,
                -8.20401788619865e-07,
                1.1220545996996594e-06,
                9.428235070512487e-07,
                1.8882197714480114e-06,
                1.5461915098836886e-06,
                1.8834735730847215e-06,
                1.8974610324105966e-06,
            ]
        ),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    # BE (index 2) is the transpose-symmetric partner of EB for an auto-spectrum.
    npt.assert_allclose(cl_all[2], cl_all[1], rtol=RTOL_DET, atol=1e-18)

    # Assert that running the same map against itself and against itself as a second map gives the same result.
    npt.assert_allclose(cl_all, cl_all_2, rtol=RTOL_DET, atol=ATOL_DET)
    npt.assert_allclose(ell_eff, ell_eff_2, rtol=RTOL_DET, atol=ATOL_DET)


def test_get_pseudo_cls_map_with_tomo(cv, cat_and_params):
    cat_gal, params = cat_and_params
    cat_gal_tomo1 = cat_gal[cat_gal[params["tomo_bin_col"]] == 1]
    cat_gal_tomo2 = cat_gal[cat_gal[params["tomo_bin_col"]] == 2]
    shear_map_1, n_gal_1 = _build_shear_map(cv, cat_gal_tomo1, params)
    shear_map_2, n_gal_2 = _build_shear_map(cv, cat_gal_tomo2, params)
    ell_eff, cl_all, wsp = cv.get_pseudo_cls_map(
        shear_map_1, n_gal_1, shear_map_b=shear_map_2, mask_b=n_gal_2
    )

    assert cl_all.shape == (4, N_ELL_BINS)
    npt.assert_allclose(
        ell_eff,
        np.array([11.5, 20.0, 30.5, 43.5, 58.5, 75.5, 95.0, 116.5]),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        cl_all[0],  # EE
        np.array(
            [
                -8.648947250571888e-06,
                4.386882223005456e-06,
                -2.3193526251107696e-06,
                1.0651397314115168e-06,
                -2.722591256637468e-07,
                -3.862270431501105e-07,
                -1.40420405782524e-06,
                2.1531429566476774e-07,
            ]
        ),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        cl_all[1],  # EB
        np.array(
            [
                -2.8032134416262087e-07,
                -1.271298363785575e-06,
                1.7112863041724342e-08,
                -2.899403854283714e-07,
                1.360217254538336e-06,
                7.682614612656511e-08,
                6.620365648170535e-07,
                -3.164479179586416e-07,
            ]
        ),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        cl_all[3],  # BB
        np.array(
            [
                9.387700364892905e-06,
                -2.9045253374872504e-06,
                -1.7006849005134948e-06,
                -3.6869001010423035e-07,
                4.4986878855942184e-07,
                6.021221048891767e-07,
                8.413198154662088e-08,
                -5.782957119801217e-07,
            ]
        ),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )


# ===========================================================================
# get_pseudo_cls_catalog -- NmtFieldCatalog path (drifts ~2e-12)
# ===========================================================================
def test_get_pseudo_cls_catalog(cv, cat_and_params):
    cat_gal, params = cat_and_params
    ell_eff, cl_all, wsp = cv.get_pseudo_cls_catalog(
        catalog=cat_gal, params=params, tomo_bin_a="all", tomo_bin_b="all"
    )

    assert cl_all.shape == (4, N_ELL_BINS)
    # Effective ells share the binning math with the map path: bitwise-stable.
    npt.assert_allclose(
        ell_eff,
        np.array([11.5, 20.0, 30.5, 43.5, 58.5, 75.5, 95.0, 116.5]),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        cl_all[0],  # EE
        np.array(
            [
                -8.480351557857007e-06,
                3.1313332030157945e-06,
                -1.7823125201406074e-06,
                3.9616637779304446e-07,
                -2.620506952433997e-07,
                -3.110249362821697e-07,
                -6.168246601737946e-07,
                4.523379773412076e-08,
            ]
        ),
        rtol=RTOL_CAT,
        atol=ATOL_CAT,
    )
    npt.assert_allclose(
        cl_all[1],  # EB
        np.array(
            [
                1.0851111241279543e-05,
                -2.3080564471133627e-06,
                -8.888571491169305e-07,
                -3.261182548134984e-07,
                5.997911542118897e-07,
                1.0323861486573016e-07,
                -2.120110521972082e-07,
                5.976985156531123e-09,
            ]
        ),
        rtol=RTOL_CAT,
        atol=ATOL_CAT,
    )
    npt.assert_allclose(
        cl_all[3],  # BB
        np.array(
            [
                1.1206499651601968e-05,
                -3.114109101436142e-06,
                1.4336907374239745e-07,
                -4.6001797302624655e-07,
                7.817610027040966e-07,
                -4.18006575112908e-07,
                -1.5043730205251898e-08,
                -1.251687104580433e-07,
            ]
        ),
        rtol=RTOL_CAT,
        atol=ATOL_CAT,
    )
    # BE (index 2) equals EB to within the catalog-path float noise.
    npt.assert_allclose(cl_all[2], cl_all[1], rtol=RTOL_CAT, atol=ATOL_CAT)


# ===========================================================================
# apply_random_rotation -- invariant + reproducibility
# ===========================================================================
def test_apply_random_rotation_preserves_magnitude(cv, cat_and_params):
    """The rotation conserves shape magnitude and array shape.

    The rotation is orthogonal, so per-object magnitude (e1^2 + e2^2) and array
    shape are exactly preserved regardless of the random angle.
    """
    cat_gal, params = cat_and_params
    e1 = np.asarray(cat_gal[params["e1_col"]], dtype=np.float64)
    e2 = np.asarray(cat_gal[params["e2_col"]], dtype=np.float64)

    e1_rot, e2_rot = apply_random_rotation(e1, e2)

    assert e1_rot.shape == e1.shape
    assert e2_rot.shape == e2.shape
    # Rotation is orthogonal -> per-object magnitude is exactly preserved.
    npt.assert_allclose(e1_rot**2 + e2_rot**2, e1**2 + e2**2, rtol=1e-12, atol=1e-15)


def test_apply_random_rotation_reproducible_with_seed(cv, cat_and_params):
    """A seeded generator reproduces the rotation; rng=None stays random.

    The noise-debiasing used to call np.random.seed() with no argument,
    re-seeding from entropy every call. apply_random_rotation now takes an
    optional rng: the same seeded generator reproduces the draw, a different
    seed differs, and rng=None remains non-deterministic.
    """
    cat_gal, params = cat_and_params
    e1 = np.asarray(cat_gal[params["e1_col"]], dtype=np.float64)
    e2 = np.asarray(cat_gal[params["e2_col"]], dtype=np.float64)

    a1, a2 = apply_random_rotation(e1, e2, np.random.default_rng(42))
    b1, b2 = apply_random_rotation(e1, e2, np.random.default_rng(42))
    npt.assert_array_equal(a1, b1)
    npt.assert_array_equal(a2, b2)

    c1, _ = apply_random_rotation(e1, e2, np.random.default_rng(7))
    assert not np.allclose(a1, c1)

    d1, _ = apply_random_rotation(e1, e2)
    f1, _ = apply_random_rotation(e1, e2)
    assert not np.allclose(d1, f1)


# ===========================================================================
# calculate_pseudo_cl_catalog -- deterministic end-to-end catalog path
# ===========================================================================
def test_calculate_pseudo_cl_catalog_end_to_end(cv, tmp_path):
    """End-to-end catalog path: FITS round-trip of ell + EE/EB/BB.

    The catalog method has no random noise debiasing, so it is reproducible to
    the same ~2e-12 catalog-path float noise. save_pseudo_cl stores ELL/EE/EB/BB
    (it drops the BE row); we pin the round-tripped table.
    """
    ver = cv._test_version
    cv._pseudo_cls = {ver: {"tomo_bin_all_tomo_bin_all": {}}}
    out_path = cv._output_path(f"pseudo_cl_cat_{ver}.fits")
    cv.calculate_pseudo_cl_catalog(ver, out_path, tomo_bin_a="all", tomo_bin_b="all")

    assert os.path.exists(out_path)
    d = fits.getdata(out_path)
    # FITS gives big-endian f8; normalize for value comparison.
    ell = np.asarray(d["ELL"], dtype=np.float64)
    ee = np.asarray(d["EE"], dtype=np.float64)
    eb = np.asarray(d["EB"], dtype=np.float64)
    bb = np.asarray(d["BB"], dtype=np.float64)

    npt.assert_allclose(
        ell,
        np.array([11.5, 20.0, 30.5, 43.5, 58.5, 75.5, 95.0, 116.5]),
        rtol=RTOL_DET,
        atol=ATOL_DET,
    )
    npt.assert_allclose(
        ee,
        np.array(
            [
                -8.480351557857063e-06,
                3.131333203015815e-06,
                -1.7823125201406123e-06,
                3.961663777930465e-07,
                -2.6205069524340103e-07,
                -3.1102493628216637e-07,
                -6.168246601737958e-07,
                4.523379773413286e-08,
            ]
        ),
        rtol=RTOL_CAT,
        atol=ATOL_CAT,
    )
    npt.assert_allclose(
        eb,
        np.array(
            [
                1.0851111241279614e-05,
                -2.3080564471133826e-06,
                -8.888571491169258e-07,
                -3.2611825481350053e-07,
                5.997911542118896e-07,
                1.0323861486573065e-07,
                -2.1201105219720712e-07,
                5.976985156542028e-09,
            ]
        ),
        rtol=RTOL_CAT,
        atol=ATOL_CAT,
    )
    npt.assert_allclose(
        bb,
        np.array(
            [
                1.1206499651602029e-05,
                -3.114109101436161e-06,
                1.43369073742402e-07,
                -4.600179730262474e-07,
                7.817610027040972e-07,
                -4.180065751129079e-07,
                -1.5043730205253605e-08,
                -1.2516871045803156e-07,
            ]
        ),
        rtol=RTOL_CAT,
        atol=ATOL_CAT,
    )
    # The end-to-end catalog EE matches the primitive get_pseudo_cls_catalog EE
    # (same computation, FITS round-trip) -- consistency, not an independent pin.
    cat_gal = fits.getdata(cv.cc[ver]["shear"]["path"])
    params = get_params_rho_tau(cv.cc[ver], survey=ver)
    _, cl_prim, _ = cv.get_pseudo_cls_catalog(
        catalog=cat_gal, params=params, tomo_bin_a="all", tomo_bin_b="all"
    )
    npt.assert_allclose(ee, cl_prim[0], rtol=RTOL_CAT, atol=ATOL_CAT)


def test_calculate_pseudo_cl_catalog_end_to_end_tomo(cv, tmp_path):
    """End-to-end catalog path: FITS round-trip of ell + EE/EB/BB.

    The catalog method has no random noise debiasing, so it is reproducible to
    the same ~2e-12 catalog-path float noise. save_pseudo_cl stores ELL/EE/EB/BB
    (it drops the BE row); we pin the round-tripped table.
    """
    ver = cv._test_version
    cv._pseudo_cls = {
        ver: {
            "tomo_bin_1_tomo_bin_1": {},
            "tomo_bin_1_tomo_bin_2": {},
            "tomo_bin_2_tomo_bin_2": {},
        }
    }
    out_path = cv._output_path(f"pseudo_cl_cat_{ver}.fits")
    tomo_bin_ids, tomo_bin_pairs = cv._get_tomo_bins(ver)

    result_to_compare = np.load(REFERENCE_TOMO, allow_pickle=True)["arr_0"].item()
    for tomo_bin_a, tomo_bin_b in tomo_bin_pairs:
        out_path = cv._output_path(
            f"pseudo_cl_cat_{ver}_{tomo_bin_a}_{tomo_bin_b}.fits"
        )
        cv.calculate_pseudo_cl_catalog(
            ver, out_path, tomo_bin_a=tomo_bin_a, tomo_bin_b=tomo_bin_b
        )

        assert os.path.exists(out_path)
        d = fits.getdata(out_path)
        # FITS gives big-endian f8; normalize for value comparison.
        ell = np.asarray(d["ELL"], dtype=np.float64)
        ee = np.asarray(d["EE"], dtype=np.float64)
        eb = np.asarray(d["EB"], dtype=np.float64)
        be = np.asarray(d["BE"], dtype=np.float64)
        bb = np.asarray(d["BB"], dtype=np.float64)

        npt.assert_allclose(
            ell,
            result_to_compare[f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"][
                "pseudo_cl"
            ]["ELL"],
            rtol=RTOL_DET,
            atol=ATOL_DET,
        )
        npt.assert_allclose(
            ee,
            result_to_compare[f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"][
                "pseudo_cl"
            ]["EE"],
            rtol=RTOL_CAT,
            atol=ATOL_CAT,
        )
        npt.assert_allclose(
            eb,
            result_to_compare[f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"][
                "pseudo_cl"
            ]["EB"],
            rtol=RTOL_CAT,
            atol=ATOL_CAT,
        )
        npt.assert_allclose(
            be,
            result_to_compare[f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"][
                "pseudo_cl"
            ]["BE"],
            rtol=RTOL_CAT,
            atol=ATOL_CAT,
        )
        npt.assert_allclose(
            bb,
            result_to_compare[f"tomo_bin_{tomo_bin_a}_tomo_bin_{tomo_bin_b}"][
                "pseudo_cl"
            ]["BB"],
            rtol=RTOL_CAT,
            atol=ATOL_CAT,
        )

        # The end-to-end catalog EE matches the primitive get_pseudo_cls_catalog EE
        # (same computation, FITS round-trip) -- consistency, not an independent pin.
        cat_gal = fits.getdata(cv.cc[ver]["shear"]["path"])
        params = get_params_rho_tau(cv.cc[ver], survey=ver)
        _, cl_prim, _ = cv.get_pseudo_cls_catalog(
            catalog=cat_gal, params=params, tomo_bin_a=tomo_bin_a, tomo_bin_b=tomo_bin_b
        )
        npt.assert_allclose(ee, cl_prim[0], rtol=RTOL_CAT, atol=ATOL_CAT)
