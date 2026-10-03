"""Regression tests: the GLASS mock generator must not reuse on-disk caches
(matter shell spectra, downgraded survey mask) built for a different cosmology
or a different input mask.

Runs against whichever worktree is first on PYTHONPATH: develop ships
``scripts/glass_mock/make_unions_glass_sim.py`` (caches inline in
``Sky.galaxies_simulation``), the tomography branch ships
``scripts/glass_mock/make_glass_sim.py`` (``Sky.read_mask`` /
``Sky.get_cl_matter_shells``).
"""

import importlib.util
import sys
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

glass = pytest.importorskip("glass")
hp = pytest.importorskip("healpy")
pytest.importorskip("fitsio")
pytest.importorskip("camb")

import sp_validation  # noqa: E402
from sp_validation.glass_mock import (  # noqa: E402
    GlassMockConfig,
    build_camb_params,
    matter_shell_cls,
)

ROOT = Path(__file__).resolve().parents[4]
# During a cross-check, use the scripts belonging to the imported checkout.
CODE_ROOT = Path(sp_validation.__file__).resolve().parents[2]
if CODE_ROOT != ROOT:
    ROOT = CODE_ROOT
SCRIPTS = ROOT / "scripts" / "glass_mock"
NSIDE = 8


def _load(name):
    path = SCRIPTS / name
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"maker_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    # Compile directly so loading the CLI cannot write a __pycache__ file.
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return mod


TOMO = _load("make_glass_sim.py")
DEVELOP = None if TOMO is not None else _load("make_unions_glass_sim.py")
if TOMO is None and DEVELOP is None:
    pytest.skip(
        "GLASS driver missing (including tomography branch driver)",
        allow_module_level=True,
    )


def _shells(cfg):
    return glass.linear_windows(np.linspace(0.0, cfg.zmax, 4))


class _Captured(Exception):
    pass


def _make_sky(mod, cfg, out, mask_path):
    sky = mod.Sky.__new__(mod.Sky)
    sky.config = cfg
    sky.pars = build_camb_params(cfg)
    sky.path = str(out)
    sky.root = str(out / "results")
    (out / "results").mkdir(parents=True, exist_ok=True)
    sky.path_mask = str(mask_path)
    sky.limber = True
    sky.rng = np.random.default_rng(0)
    return sky


def _run_generator(sky, monkeypatch):
    """Return (cls, mask) the generator would use for this Sky."""
    if TOMO is not None:
        with redirect_stdout(StringIO()):
            return sky.get_cl_matter_shells(_shells(sky.config)), sky.read_mask()
    # develop: caches are inline in galaxies_simulation; stop it at the first
    # GLASS call after both caches are resolved and read its locals.
    # Use the same tiny shells for the cached and fresh calculations; this
    # does not alter either cache or the cosmology used to compute its values.
    monkeypatch.setattr(DEVELOP, "build_shells", lambda cfg, pars: _shells(cfg))

    def capture(fields, cls, *args, **kwargs):
        frame = sys._getframe(1).f_locals
        raise _Captured(frame["cls"], frame["unions_mask"])

    monkeypatch.setattr(glass, "solve_gaussian_spectra", capture)
    with redirect_stdout(StringIO()), pytest.raises(_Captured) as exc:
        sky.galaxies_simulation()
    return exc.value.args


@pytest.fixture
def masks(tmp_path):
    full = np.ones(hp.nside2npix(NSIDE))
    half = full.copy()
    half[: half.size // 2] = 0.0
    a, b = tmp_path / "mask_full.fits", tmp_path / "mask_half.fits"
    hp.write_map(str(a), full, overwrite=True, dtype=np.float64)
    hp.write_map(str(b), half, overwrite=True, dtype=np.float64)
    return a, b, half


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#381: shell spectra cache ignores changes to cosmology",
)
def test_shell_cls_cache_ignores_changed_sigma8(tmp_path, masks, monkeypatch):
    """A second run into the same output directory with sigma8 raised from
    0.8102 to 0.90 must use matter spectra for sigma8=0.90. The expected value is
    a fresh ``matter_shell_cls`` call at the new cosmology on identical shells;
    C_ell scales ~sigma8^2 (ratio ~0.81), so a cache keyed only on nside, which
    returns the first run's spectra, is far outside rtol=1e-6.
    """
    mod = TOMO or DEVELOP
    full, _, _ = masks
    cfg = GlassMockConfig(nside=NSIDE, dx=500.0, zmax=0.3, sigma8=0.8102)
    out = tmp_path / "out"
    _run_generator(_make_sky(mod, cfg, out, full), monkeypatch)

    cfg2 = replace(cfg, sigma8=0.90)
    sky2 = _make_sky(mod, cfg2, out, full)
    used, _ = _run_generator(sky2, monkeypatch)
    kw = {"limber": True} if TOMO is not None else {}
    fresh = matter_shell_cls(cfg2, sky2.pars, _shells(cfg2), **kw)
    ratio = np.asarray(used[-1])[4] / np.asarray(fresh[-1])[4]
    assert np.allclose(np.asarray(used[-1]), np.asarray(fresh[-1]), rtol=1e-6), (
        f"generator used stale spectra at sigma8=0.90: used/fresh C_4 = {ratio:.4f}"
    )


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#381: downgraded mask cache ignores changes to input mask",
)
def test_mask_cache_ignores_changed_input_mask(tmp_path, masks, monkeypatch):
    """A second run into the same output directory with a half-sky input mask
    must use that footprint. At equal input and target nside the downgrade is
    the identity, so the expected mask is the half-sky map itself (384 of 768
    pixels); a cache keyed only on nside returns the first run's full-sky mask.
    """
    mod = TOMO or DEVELOP
    full, half_path, half = masks
    cfg = GlassMockConfig(nside=NSIDE, dx=500.0, zmax=0.3)
    out = tmp_path / "out"
    _run_generator(_make_sky(mod, cfg, out, full), monkeypatch)

    _, used = _run_generator(_make_sky(mod, cfg, out, half_path), monkeypatch)
    assert int(np.sum(used > 0)) == int(half.sum()), (
        f"generator used stale mask: {int(np.sum(used > 0))} occupied pixels, "
        f"requested footprint has {int(half.sum())}"
    )
