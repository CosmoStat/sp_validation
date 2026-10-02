"""Numbered GLASS mocks must be independent realisations.

Run the committed mock-generation CLI from the imported checkout: develop has
``make_unions_glass_sim.py``, tomography has ``make_glass_sim.py``.
Only CAMB parameter building is stubbed: this exercises argument parsing and
RNG construction in ``Sky``, never map generation or survey I/O.
"""

import importlib.util
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sp_validation

ROOT = Path(__file__).resolve().parents[4]
# Use the CLI belonging to the imported checkout for a tomography cross-check.
CODE_ROOT = Path(sp_validation.__file__).resolve().parents[2]
if CODE_ROOT != ROOT:
    ROOT = CODE_ROOT


def _load_script(monkeypatch):
    for name in ("glass", "fitsio", "camb"):
        pytest.importorskip(name)
    for branch, name in (
        ("tomo", "make_glass_sim.py"),
        ("develop", "make_unions_glass_sim.py"),
    ):
        path = ROOT / "scripts" / "glass_mock" / name
        if path.is_file():
            break
    else:
        pytest.skip("GLASS driver missing (including tomography branch driver)")
    spec = importlib.util.spec_from_file_location(f"glass_cli_{branch}", path)
    mod = importlib.util.module_from_spec(spec)
    # Loading the CLI must not write bytecode outside tmp_path.
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    # Cosmology calculations are unrelated to the seed; leave Sky.rng untouched.
    pars = SimpleNamespace(
        H0=0, omch2=0, ombh2=0, omegam=0, InitPower=SimpleNamespace(As=0, ns=0)
    )
    monkeypatch.setattr(mod, "build_camb_params", lambda config: pars)
    monkeypatch.setattr(mod, "camb_sigma8", lambda params: 0.0)
    return branch, mod


def _argv(branch, tmp_path, number):
    if branch == "develop":
        return ["prog", "-N", str(number), "-p", str(tmp_path)]
    nz = ROOT / "config/glass_mock/test_data/redshift_distribution_tomo.txt"
    if not nz.is_file():
        pytest.skip("tomography branch's test redshift distribution is missing")
    nbins = np.loadtxt(nz).shape[1] - 1
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"nside: 32\nnbins: {nbins}\nn_arcmin2: 4.0\nsigma_e: 0.26\nbias: 1.0\n"
        f"nz_path: {nz}\nmask_path: {tmp_path}/mask.fits\n"
        f"output_path: {tmp_path}\noutput_prefix: t\n"
    )
    return ["prog", "-N", str(number), "-c", str(cfg)]


def _first_draws(branch, mod, tmp_path, number, monkeypatch):
    monkeypatch.setattr(sys, "argv", _argv(branch, tmp_path, number))
    with redirect_stdout(StringIO()):
        sky = mod.Sky()
    return sky.n_sim, sky.rng.standard_normal(8)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#389: mock number does not enter the RNG seed",
)
def test_numbered_mocks_with_default_seed_are_distinct_realisations(
    tmp_path, monkeypatch
):
    """Mocks 1 and 2 generated with ``-N 1`` / ``-N 2`` must not share random draws.

    The CLI exposes ``-N/--number`` as the mock index that labels the output file
    (``*_glass_sim_0000N_*``), and every stochastic step (``glass.generate``, galaxy
    positions, redshifts, shape noise) draws from ``Sky.rng``. An ensemble of mocks
    used for a sample covariance or a PTE distribution needs independent members,
    so two different mock numbers under the same (default) ``-s`` must yield
    different RNG streams. Two independent standard-normal 8-vectors coincide
    with probability zero, so equality of the first draws proves the realisations
    are identical copies.
    """
    branch, mod = _load_script(monkeypatch)
    n1, d1 = _first_draws(branch, mod, tmp_path, 1, monkeypatch)
    n2, d2 = _first_draws(branch, mod, tmp_path, 2, monkeypatch)
    assert n1 != n2, "mock number should change the output label"
    assert not np.array_equal(d1, d2), (
        f"[{branch}] mocks {n1} and {n2} (default seed) draw identical random "
        f"streams: first draws {d1[:3]} == {d2[:3]}; -N only renames the file"
    )
