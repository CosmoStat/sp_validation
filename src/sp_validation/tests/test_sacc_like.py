"""Equality tests: the sp_validation SACC-likelihood shim vs CosmoSIS ``2pt_like``.

PR 7 adopts CosmoSIS's native ``SaccClLikelihood`` for the ξ± inference path,
through the shim :mod:`sp_validation.sacc_like_unions` (which fixes the upstream
arcmin→rad gap and adds an ordering guard). The contract is *in-process module
equality*: run the shimmed ``sacc_like`` on the analysis SACC and CosmoSIS's
``2pt_like`` on the PR-3 converter's 2pt-FITS against an identical synthetic
theory DataBlock, and require the same χ², log-likelihood, theory vector and
post-cut point count. The prototype (``sacc-like-probe/probe_equality.py``)
observed *exact* equality (Δχ²=0, Δtheory=0), so the equality tests assert
``array_equal`` / rtol=1e-12.

Environment
-----------
Both engines are pure-python CosmoSIS module files (no CAMB, no compiled CSL
modules), so they run in the shared venv with ``cosmosis`` installed. They need a
checkout of the CosmoSIS Standard Library, located via the ``CSL_DIR`` env var;
the module is skipped when cosmosis is absent (CI image) or ``CSL_DIR`` is unset
or missing. Recipe::

    git clone --depth 1 https://github.com/joezuntz/cosmosis-standard-library CSL
    CSL_DIR=/path/to/CSL  pytest ... test_sacc_like.py

The pyproject ``cosmosis`` extra pins ``cosmosis>=3.25`` for the engine itself
(inert in CI, whose Dockerfile installs only ``[test,glass,blinding]``).
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

cosmosis = pytest.importorskip("cosmosis")

# The upstream CSL checkout carrying likelihood/sacc + likelihood/2pt. Skip the
# whole module (not error) when it is not configured, mirroring the cosmo_numba
# env-gated precedent.
_CSL_DIR = os.environ.get("CSL_DIR")
if not _CSL_DIR or not Path(_CSL_DIR, "likelihood", "sacc").is_dir():
    pytest.skip(
        "CSL_DIR unset or has no likelihood/sacc — set CSL_DIR to a CosmoSIS "
        "Standard Library checkout to run the sacc_like equality tests",
        allow_module_level=True,
    )

from cosmosis.datablock import DataBlock, option_section  # noqa: E402

from sp_validation import sacc_io, twopoint_convert  # noqa: E402

# Reuse the angular-bin count from the converter tests so the two suites' single-
# bin shapes stay in lockstep. (The χ²-dynamics builders here need a covariance
# commensurate with the data, which the converter's byte-compare _sacc is not.)
from sp_validation.tests.test_twopoint_convert import N_ANG  # noqa: E402

CSL = Path(_CSL_DIR)
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)

# The like_name both engines are configured with, so both write identical block
# keys (<like_name>_CHI2, _LIKE, _theory) — the parity the design mandates.
LIKE_NAME = "2pt_like"


# ---------------------------------------------------------------------------
# Shared engine harness
# ---------------------------------------------------------------------------
def _load_module_file(path, name):
    """Import a CosmoSIS module file (``setup``/``execute``/``cleanup``) by path.

    ``likelihood/sacc`` and ``likelihood/2pt`` are added to ``sys.path`` so the
    upstream modules resolve their siblings (``sacc_likelihoods``, ``spec_tools``,
    ``twopoint_cosmosis``, …). Loading ``2pt_like.py`` runs its
    ``build_module()`` at import (module-level ``setup, execute, cleanup``);
    ``sacc_like_unions.py`` defines those functions directly.
    """
    import sys

    for sub in ("likelihood/sacc", "likelihood/2pt"):
        p = str(CSL / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _theory_block():
    """A DataBlock carrying the synthetic ξ± theory both engines interpolate.

    Mirrors the probe: the theory θ grid is in *radians* (CosmoSIS convention),
    and the ξ+/ξ− predictions are smooth power laws sampled on it. Both engines
    build a spline over this grid and evaluate it at each data point's angular
    tag — so as long as both see the same block, the interpolated theory (and
    hence χ²) must agree.
    """
    theta_arcmin = np.geomspace(0.3, 400.0, 300)

    def t_xip(th):
        return 2e-4 * (th / 10.0) ** -0.8

    def t_xim(th):
        return 1e-4 * (th / 10.0) ** -0.5

    b = DataBlock()
    for section, f in (("shear_xi_plus", t_xip), ("shear_xi_minus", t_xim)):
        b[section, "theta"] = theta_arcmin * ARCMIN_TO_RAD
        b[section, "bin_1_1"] = f(theta_arcmin)
        b[section, "is_auto"] = True
        b[section, "nbin_a"] = 1
        b[section, "nbin_b"] = 1
        b[section, "sample_a"] = "nz_source"
        b[section, "sample_b"] = "nz_source"
        b[section, "sep_name"] = "theta"
        b[section, "save_name"] = ""
    return b


def _run(mod, options):
    """Run a CosmoSIS likelihood module file standalone; return (like, chi2, theory, n).

    Builds the options DataBlock (module options live under ``option_section``),
    calls ``setup`` → ``execute`` on a fresh theory block, and reads the standard
    Gaussian-likelihood outputs back out under the shared ``LIKE_NAME`` keys.
    """
    opt = DataBlock()
    for key, value in options.items():
        opt[option_section, key] = value
    config = mod.setup(opt)
    block = _theory_block()
    mod.execute(block, config)
    like = block["likelihoods", f"{LIKE_NAME}_LIKE"]
    chi2 = block["data_vector", f"{LIKE_NAME}_CHI2"]
    theory = block["data_vector", f"{LIKE_NAME}_theory"]
    return like, chi2, np.asarray(theory), len(theory)


# ---------------------------------------------------------------------------
# The two engine module files + their options
# ---------------------------------------------------------------------------
_SHIM_PATH = Path(__file__).resolve().parents[1] / "sacc_like_unions.py"


@pytest.fixture(scope="module")
def m_shim():
    return _load_module_file(_SHIM_PATH, "sacc_like_unions_test")


@pytest.fixture(scope="module")
def m_2pt():
    return _load_module_file(CSL / "likelihood/2pt/2pt_like.py", "twopt_like_test")


@pytest.fixture(scope="module")
def m_raw_sacc():
    return _load_module_file(CSL / "likelihood/sacc/sacc_like.py", "raw_sacc_like_test")


def _shim_opts(sacc_path, **extra):
    return {
        "csl_dir": str(CSL),
        "data_file": sacc_path,
        "data_sets": "galaxy_shear_xi_plus galaxy_shear_xi_minus",
        "like_name": LIKE_NAME,
        **extra,
    }


def _twopt_opts(fits_path, **extra):
    return {
        "data_file": fits_path,
        "data_sets": "XI_PLUS XI_MINUS",
        "covmat_name": "COVMAT",
        "like_name": LIKE_NAME,
        "gaussian_covariance": False,
        "cut_zeros": False,
        **extra,
    }


def _realistic_sacc(seed=0, *, xip=None):
    """A single-bin ξ± SACC with data + covariance sized like the real product.

    The converter-test ``_sacc`` builder uses a covariance ~14 orders of
    magnitude larger than the ξ values (fine for byte-comparing the converter,
    where covariance *content* is irrelevant), which makes every χ² collapse to
    numerical zero — no teeth for the unit-gap tripwire. This builder instead
    lays down realistic ξ± power laws and a covariance ~ ``(0.1·|ξ|)²`` (the
    probe's recipe), so χ² is O(1)-scale and the raw-vs-shim gap is visible.

    ``xip`` overrides the ξ+ values (perturbation teeth).
    """
    rng = np.random.default_rng(seed)
    n = N_ANG
    theta = np.geomspace(1.0, 250.0, n)  # arcmin
    z = np.linspace(0.01, 3.0, 200)
    nz = z**2 * np.exp(-((z / 0.5) ** 1.5))

    xip_vals = 2e-4 * (theta / 10.0) ** -0.8 * (1 + 0.05 * rng.standard_normal(n))
    xim_vals = 1e-4 * (theta / 10.0) ** -0.5 * (1 + 0.05 * rng.standard_normal(n))
    if xip is not None:
        xip_vals = xip

    s = sacc_io.new_sacc({0: (z, nz)}, {"catalogue_version": "test"})
    sacc_io.add_xi(s, (0, 0), theta, xip_vals, xim_vals, grid="coarse")

    sig = 0.1 * np.abs(np.concatenate([xip_vals, xim_vals]))
    a = rng.standard_normal((2 * n, 3 * n))
    cov = (a @ a.T / (3 * n)) * np.outer(sig, sig) * 0.3 + np.diag(sig**2)
    s.add_covariance(cov)
    return s, theta


def _write_pair(tmp_path, seed=0, name="probe", *, xip=None):
    """Write a realistic analysis SACC and its PR-3 converter 2pt-FITS.

    Returns ``(sacc_path, fits_path)``. The SACC carries the arcmin ξ± tags the
    shim converts; the FITS is the byte-compatible product ``2pt_like`` reads.
    """
    s, _theta = _realistic_sacc(seed, xip=xip)
    sacc_path = str(tmp_path / f"{name}.sacc")
    sacc_io.save(s, sacc_path)
    fits_path = str(tmp_path / f"{name}_2pt.fits")
    twopoint_convert.sacc_to_twopoint_fits(sacc_io.load(sacc_path), fits_path, n_bins=1)
    return sacc_path, fits_path


# ---------------------------------------------------------------------------
# 1. Core equality: shim on SACC ≡ 2pt_like on converter FITS
# ---------------------------------------------------------------------------
def test_shimmed_equals_2pt_like_exact(tmp_path, m_shim, m_2pt):
    """The shimmed sacc_like and 2pt_like agree exactly on the same data+theory.

    Same synthetic theory block, the SACC through the shim vs the converter FITS
    through 2pt_like: χ², log-likelihood and the theory vector must match to
    numerical precision (the probe observed exact equality). This is the PR's
    central contract — the native path reproduces the validated converter path.
    """
    sacc_path, fits_path = _write_pair(tmp_path, seed=0)

    like_s, chi2_s, theory_s, n_s = _run(m_shim, _shim_opts(sacc_path))
    like_t, chi2_t, theory_t, n_t = _run(m_2pt, _twopt_opts(fits_path))

    assert n_s == n_t == 2 * N_ANG
    np.testing.assert_allclose(chi2_s, chi2_t, rtol=1e-12)
    np.testing.assert_allclose(like_s, like_t, rtol=1e-12)
    np.testing.assert_allclose(theory_s, theory_t, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Tripwire: raw upstream sacc_like is broken on arcmin tags
# ---------------------------------------------------------------------------
def test_upstream_unit_gap_tripwire(tmp_path, m_raw_sacc, m_2pt):
    """RAW upstream sacc_like (no shim) gives a wildly wrong χ² on arcmin tags.

    Documents and guards the arcmin→rad gap the shim fixes: with raw ``theta``
    tags the theory spline is evaluated outside its (radian) grid and returns 0,
    collapsing χ² to dᵀC⁻¹d. We assert the relative χ² difference against
    2pt_like exceeds 10 (the probe saw Δχ²≈3184).

    IF THIS TEST EVER FAILS: upstream ``sacc_like`` has fixed its units — the
    shim's arcmin→rad conversion is now redundant and the shim can be retired.
    """
    sacc_path, fits_path = _write_pair(tmp_path, seed=0)

    _like_r, chi2_raw, _theory_r, _n_r = _run(m_raw_sacc, _shim_opts(sacc_path))
    _like_t, chi2_t, _theory_t, _n_t = _run(m_2pt, _twopt_opts(fits_path))

    rel = abs(chi2_raw - chi2_t) / abs(chi2_t)
    assert rel > 10, (
        f"raw sacc_like χ²={chi2_raw:.6g} is within 10× of 2pt_like χ²={chi2_t:.6g} "
        f"(rel diff {rel:.3g}) — the upstream unit gap appears fixed; retire the shim"
    )


# ---------------------------------------------------------------------------
# 3. Scale cuts equivalent through both engines
# ---------------------------------------------------------------------------
def test_scale_cuts_equivalent(tmp_path, m_shim, m_2pt):
    """The same arcmin scale cuts give the same post-cut N and χ² on both engines.

    Cuts are expressed in the each engine's grammar but the SAME numeric arcmin
    values (shim cuts run before the arcmin→rad conversion, so they take arcmin
    just like 2pt_like's angle_range). ξ+ ∈ [10, 200], ξ− ∈ [20, 200] arcmin.
    """
    sacc_path, fits_path = _write_pair(tmp_path, seed=1)

    shim_cuts = _shim_opts(
        sacc_path,
        **{
            "angle_range_galaxy_shear_xi_plus_source_0_source_0": np.array(
                [10.0, 200.0]
            ),
            "angle_range_galaxy_shear_xi_minus_source_0_source_0": np.array(
                [20.0, 200.0]
            ),
        },
    )
    twopt_cuts = _twopt_opts(
        fits_path,
        **{
            "angle_range_XI_PLUS_1_1": np.array([10.0, 200.0]),
            "angle_range_XI_MINUS_1_1": np.array([20.0, 200.0]),
        },
    )

    _like_s, chi2_s, _theory_s, n_s = _run(m_shim, shim_cuts)
    _like_t, chi2_t, _theory_t, n_t = _run(m_2pt, twopt_cuts)

    assert n_s == n_t
    assert n_s < 2 * N_ANG  # the cuts actually removed points
    np.testing.assert_allclose(chi2_s, chi2_t, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Perturbation moves both engines identically
# ---------------------------------------------------------------------------
def test_perturbation_moves_both_identically(tmp_path, m_shim, m_2pt):
    """Perturbing one data value shifts both engines' χ² by the identical amount.

    Teeth: build the base pair and a pair whose first ξ+ value is bumped, and
    require Δχ²(shim) == Δχ²(2pt_like). If either engine ignored the perturbed
    point (e.g. a misaligned data vector), the deltas would diverge.
    """
    # Base pair, then a pair whose first ξ+ value is bumped. Rebuild the base ξ+
    # from the same seed so only the one perturbed entry differs.
    base_s, theta = _realistic_sacc(seed=2)
    base_xip = np.array(
        [p.value for p in base_s.data if p.data_type == sacc_io.XI_PLUS]
    )
    sacc_b, fits_b = _write_pair(tmp_path, seed=2, name="base")

    pert_xip = base_xip.copy()
    pert_xip[0] += 5e-5
    sacc_p, fits_p = _write_pair(tmp_path, seed=2, name="pert", xip=pert_xip)

    _l, chi2_shim_b, _t, _n = _run(m_shim, _shim_opts(sacc_b))
    _l, chi2_shim_p, _t, _n = _run(m_shim, _shim_opts(sacc_p))
    _l, chi2_2pt_b, _t, _n = _run(m_2pt, _twopt_opts(fits_b))
    _l, chi2_2pt_p, _t, _n = _run(m_2pt, _twopt_opts(fits_p))

    d_shim = chi2_shim_p - chi2_shim_b
    d_2pt = chi2_2pt_p - chi2_2pt_b
    assert abs(d_shim) > 0  # the perturbation actually moved χ²
    np.testing.assert_allclose(d_shim, d_2pt, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Ordering guard raises on a pair-major tomographic file
# ---------------------------------------------------------------------------
def test_ordering_guard_raises_pair_major_tomographic(tmp_path, m_shim):
    """A pair-major 2-bin SACC trips the shim's ordering guard at setup.

    Inserting ξ± per pair — (0,0), then (0,1), then (1,1) — lays the data vector
    out pair-major ([ξ+;ξ−] per pair), while ``get_data_types()`` groups the
    theory loop type-major (all ξ+ pairs, then all ξ− pairs). The two orders
    disagree (verified empirically), so the guard must raise a ValueError
    mentioning the ordering hazard rather than silently mis-comparing.
    """
    theta = np.geomspace(1.0, 250.0, N_ANG)  # arcmin
    z = np.linspace(0.01, 3.0, 200)
    nz = z**2 * np.exp(-((z / 0.5) ** 1.5))
    xip = np.ones(N_ANG) * 1e-4
    xim = np.ones(N_ANG) * 1e-4
    s = sacc_io.new_sacc({0: (z, nz), 1: (z, nz)})
    for pair in [(0, 0), (0, 1), (1, 1)]:
        sacc_io.add_xi(s, pair, theta, xip, xim, grid="coarse")
    s.add_covariance(np.eye(len(s.mean)))
    sacc_path = str(tmp_path / "pair_major.sacc")
    sacc_io.save(s, sacc_path)

    with pytest.raises(ValueError, match="ordering"):
        m_shim.setup(_as_option_block(_shim_opts(sacc_path)))


def _as_option_block(options):
    """Build a raw options DataBlock (keys under ``option_section``) for setup()."""
    opt = DataBlock()
    for key, value in options.items():
        opt[option_section, key] = value
    return opt


# ---------------------------------------------------------------------------
# 6. theta conversion scoped to real-category types (COSEBIs untouched)
# ---------------------------------------------------------------------------
def test_theta_conversion_scoped_to_real_types(tmp_path, m_shim):
    """The shim converts ξ ``theta`` tags but leaves cosebi ``n`` tags untouched.

    Build a SACC carrying both ξ± (real) and COSEBIs (a non-real ``cosebis``
    category with integer ``n`` tags). After the shim's ``build_data``, the ξ
    ``theta`` tags must be scaled arcmin→rad (so they equal the original arcmin
    values times the conversion factor), and the cosebi ``n`` tags must be
    numerically unchanged.
    """
    # Build ξ± + COSEBIs, then attach the covariance last (sacc forbids adding
    # points after add_covariance).
    theta = np.geomspace(1.0, 250.0, N_ANG)  # arcmin
    z = np.linspace(0.01, 3.0, 200)
    nz = z**2 * np.exp(-((z / 0.5) ** 1.5))
    n_modes = 5
    En = np.arange(1.0, n_modes + 1)
    Bn = np.arange(1.0, n_modes + 1) * 0.1

    s = sacc_io.new_sacc({0: (z, nz)})
    sacc_io.add_xi(
        s, (0, 0), theta, np.ones(N_ANG) * 1e-4, np.ones(N_ANG) * 1e-4, grid="coarse"
    )
    sacc_io.add_cosebis(s, (0, 0), En, Bn, scale_cut=(1.0, 250.0))
    s.add_covariance(np.eye(len(s.mean)))

    sacc_path = str(tmp_path / "with_cosebis.sacc")
    sacc_io.save(s, sacc_path)

    # data_sets keeps the cosebis in (so we can check its tags survive); cosebi's
    # section/category resolve from sacc_like's default_sections, so build_data
    # needs no extra ini config. Only setup() runs (build_data); the theory loop
    # (which would want a cosebi theory block) runs at execute, not here.
    config = m_shim.setup(
        _as_option_block(
            _shim_opts(
                sacc_path,
                data_sets=(
                    "galaxy_shear_xi_plus galaxy_shear_xi_minus "
                    "galaxy_shear_cosebi_ee galaxy_shear_cosebi_bb"
                ),
            )
        )
    )

    xi_thetas = [
        p.tags["theta"] for p in config.sacc_data.data if p.data_type == sacc_io.XI_PLUS
    ]
    cosebi_ns = [
        p.tags["n"] for p in config.sacc_data.data if p.data_type == sacc_io.COSEBI_EE
    ]
    # ξ theta converted to radians (original arcmin × ARCMIN_TO_RAD).
    np.testing.assert_allclose(
        np.sort(xi_thetas), np.sort(theta * ARCMIN_TO_RAD), rtol=1e-12
    )
    # cosebi n tags untouched (still the integer modes 1..n_modes).
    np.testing.assert_array_equal(np.sort(cosebi_ns), np.arange(1, n_modes + 1))


# ---------------------------------------------------------------------------
# 7. Real-data equality (candide-gated)
# ---------------------------------------------------------------------------
_REALDATA = (
    Path("/automnt/n17data/cdaley/unions/code/sp_validation/cosmo_inference/data")
    / "SP_v1.4.6_leak_corr_A_minsep=1.0_maxsep=250.0_nbins=20_npatch=1"
    / "cosmosis_SP_v1.4.6_leak_corr_A_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.fits"
)


@pytest.mark.skipif(
    not _REALDATA.exists(), reason=f"real 2pt-FITS not on disk: {_REALDATA}"
)
def test_realdata_shim_equals_2pt_like(tmp_path, m_shim, m_2pt):
    """On a real product, the shim on its SACC equals 2pt_like on the FITS.

    Builds a ξ-only analysis SACC from the real 2pt-FITS's own ξ± values and
    covariance sub-block (via the converter test's ``_sacc_from_2pt_fits``, then
    strip to ξ±), writes it, converts it to a plain-ξ FITS, and runs both engines
    against the synthetic theory block. Scoping to ξ± isolates the shear
    likelihood equality on the true (20-point-per-sign) data-vector shape — the
    IA-only inference scope this PR targets — and keeps ``2pt_like``'s
    ``twopoint.from_fits`` from tripping over the real file's separate
    COVMAT_CELL / τ blocks.
    """
    from astropy.io import fits

    from sp_validation.tests.test_twopoint_convert_realdata import _sacc_from_2pt_fits

    with fits.open(_REALDATA) as hdul:
        full_s, _rho_hdu, _tau_hdu = _sacc_from_2pt_fits(hdul)

    # Rebuild a ξ-only SACC: same n(z), ξ± values and ξ± covariance sub-block.
    source = sacc_io.source_name(0)
    z, nz = sacc_io.get_nz(full_s, 0)
    theta, xip, xim = sacc_io.get_xi(full_s, (0, 0), grid="coarse")
    xi_idx = np.concatenate(
        [
            full_s.indices(sacc_io.XI_PLUS, (source, source)),
            full_s.indices(sacc_io.XI_MINUS, (source, source)),
        ]
    )
    xi_cov = full_s.covariance.dense[np.ix_(xi_idx, xi_idx)]

    s = sacc_io.new_sacc({0: (z, nz)})
    sacc_io.add_xi(s, (0, 0), theta, xip, xim, grid="coarse")
    s.add_covariance(xi_cov)

    sacc_path = str(tmp_path / "real_xi.sacc")
    sacc_io.save(s, sacc_path)
    fits_path = str(tmp_path / "real_xi_2pt.fits")
    twopoint_convert.sacc_to_twopoint_fits(sacc_io.load(sacc_path), fits_path, n_bins=1)

    like_s, chi2_s, theory_s, n_s = _run(m_shim, _shim_opts(sacc_path))
    like_t, chi2_t, theory_t, n_t = _run(m_2pt, _twopt_opts(fits_path))

    assert n_s == n_t
    assert n_s == 2 * len(theta)
    np.testing.assert_allclose(chi2_s, chi2_t, rtol=1e-10)
    np.testing.assert_allclose(like_s, like_t, rtol=1e-10)
    np.testing.assert_allclose(theory_s, theory_t, rtol=1e-10)
