"""Tests for the Snakemake blind-at-birth wiring (issues #247/#252, PR #253).

Two seams are covered here, both independent of a live cluster:

1. **The path helpers in ``workflow/common.py``** must stay in lockstep with
   ``sp_validation.blinding`` — common mirrors ``init_paths`` / ``part_paths`` by
   hand (to keep the DAG build from importing the heavy blinding module), so a
   drift between them would silently mis-wire ``blind_part``. These tests are the
   guard.
2. **The data-run fail-closed assembly**: ``assemble_sacc`` must refuse an
   unblinded ``type='data'`` part and succeed once every part is concealed under
   one commitment — the terminal custody gate of #252.

A candide-only test additionally asserts the blinding subgraph resolves in the
cosmo_val DAG dry-run.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sp_validation import blinding
from sp_validation import sacc_io as sio
from sp_validation.cosmo_val import sacc_writers as sw


def _repo_root():
    return next(
        p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
    )


def _load_module(rel_path, name):
    """Import a workflow module/script by file path (off the package path)."""
    path = _repo_root() / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _load_module("workflow/common.py", "wf_common")
asm = _load_module("workflow/scripts/assemble_sacc.py", "assemble_sacc")


# --------------------------------------------------------------------------- #
# 1. common.py path helpers mirror sp_validation.blinding (drift guard)
# --------------------------------------------------------------------------- #
_STEMS = [
    "SP_v1.4.6.3_xi_reporting_minsep=1.0_maxsep=250.0_nbins=20_npatch=100",
    "SP_v1.4.6.3_leak_corr_xi_integration",
    "pseudo_cl_SP_v1.4.6.3_blind=A_powspace_nbins=32",
    "pseudo_cl_SP_v1.4.6.3_leak_corr_blind=A_powspace_nbins=32",
]


@pytest.mark.parametrize("stem", _STEMS)
def test_blinded_path_mirrors_blinding_part_paths(stem):
    part = f"/out/{stem}.sacc"
    assert common.blinded_path(part) == blinding.part_paths(part)["blinded"]


def test_blind_state_paths_mirror_blinding_init_paths():
    version = "SP_v1.4.6.3_leak_corr"
    common_paths = common.blind_state_paths(version)
    ref = blinding.init_paths(common.blind_state_dir(version))
    assert common_paths == ref


@pytest.mark.parametrize(
    "stem,expected",
    [
        (_STEMS[0], "SP_v1.4.6.3"),
        (_STEMS[1], "SP_v1.4.6.3_leak_corr"),
        (_STEMS[2], "SP_v1.4.6.3"),
        (_STEMS[3], "SP_v1.4.6.3_leak_corr"),
    ],
)
def test_version_of_extracts_catalogue_version(stem, expected):
    assert common.version_of(stem) == expected


def test_version_of_raises_without_version():
    with pytest.raises(ValueError, match="no catalogue version"):
        common.version_of("cosebis_no_version_here")


def test_blindable_part_switches_on_run_type(monkeypatch):
    part = "/out/SP_v1.4.6.3_xi_integration.sacc"
    monkeypatch.setattr(common, "RUN_TYPE", "data")
    assert common.blindable_part(part) == common.blinded_path(part)
    monkeypatch.setattr(common, "RUN_TYPE", "mock")
    assert common.blindable_part(part) == part


# --------------------------------------------------------------------------- #
# 2. Data-run fail-closed assembly (#252 terminal custody gate)
# --------------------------------------------------------------------------- #
META = {"catalogue_version": "vSYNTH", "npatch": 1}
# Two arbitrary-but-consistent hex stamps standing in for a real blind's
# sha256(seed) / config digest; the assembly only checks they agree across parts.
_COMMIT = "a" * 64
_DIGEST = "b" * 64


def _nz():
    return np.linspace(0.01, 2.0, 40), np.random.default_rng(0).uniform(0.1, 1.0, 40)


def _spd(n, seed):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _data_parts(tmp_path, *, conceal, one_plaintext=False):
    """Write the five per-statistic parts as ``type='data'``.

    ``conceal`` stamps every part with the shared blind (concealed=True). With
    ``one_plaintext`` the ξ± reporting part is left unconcealed — a blinded /
    plaintext mix the assembly must refuse.
    """
    nz = {0: _nz()}
    theta = np.geomspace(1.0, 100.0, 6)

    xi = sw.xi_to_sacc(
        nz, META, theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="reporting"
    )
    xi.add_covariance(_spd(len(xi.mean), 1))
    co = sw.cosebis_to_sacc(
        nz,
        META,
        {
            "En": np.arange(1, 6) * 1e-6,
            "Bn": np.arange(1, 6) * 1e-7,
            "cov": _spd(10, 3),
        },
        (1.0, 100.0),
    )
    eb_arrays = {k: np.arange(6) * (i + 1) * 1e-6 for i, k in enumerate(sio.PURE_KEYS)}
    eb = sw.pure_eb_to_sacc(nz, META, theta, eb_arrays, covariance=_spd(36, 4))
    rho = {"theta": theta}
    tau = {"theta": theta}
    rng = np.random.default_rng(5)
    for k in sw.RHO_K:
        for s in ("p", "m"):
            rho[f"rho_{k}_{s}"] = rng.normal(size=6) * 1e-6
            rho[f"varrho_{k}_{s}"] = rng.uniform(1e-14, 1e-13, 6)
    for k in sw.TAU_K:
        for s in ("p", "m"):
            tau[f"tau_{k}_{s}"] = rng.normal(size=6) * 1e-6
            tau[f"vartau_{k}_{s}"] = rng.uniform(1e-14, 1e-13, 6)
    rt = sw.rho_tau_to_sacc(nz, META, rho, tau)

    parts = {"xi_reporting": xi, "cosebis": co, "pure_eb": eb, "rho_tau": rt}
    paths = {}
    for name, part in parts.items():
        if conceal and not (one_plaintext and name == "xi_reporting"):
            blinding._stamp_provenance(part, _COMMIT, "A", _DIGEST)
        p = tmp_path / f"{name}.sacc"
        sio.save(part, str(p), type="data")
        paths[name] = str(p)
    return paths


def test_data_assemble_fails_closed_on_unblinded_part(tmp_path):
    """A data run refuses to assemble an unconcealed real part (fail closed)."""
    paths = _data_parts(tmp_path, conceal=False)
    with pytest.raises(ValueError, match="refusing to load an unblinded"):
        asm.assemble_sacc(
            "vSYNTH", paths, str(tmp_path / "vSYNTH.sacc"), placeholder_var=1.0
        )


def test_data_assemble_passes_on_blinded_parts(tmp_path):
    """With every part concealed under one blind, the data-run assembly succeeds
    and stamps the shared commitment on the terminal file."""
    paths = _data_parts(tmp_path, conceal=True)
    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc("vSYNTH", paths, str(out), placeholder_var=1.0)
    assert s.metadata["concealed"] is True
    assert s.metadata["blind_commitment"] == _COMMIT
    assert s.metadata["blind_config_digest"] == _DIGEST
    # Round-trips through the fail-closed load gate without an escape hatch.
    assert sio.load(str(out)).metadata["concealed"] is True


def test_data_assemble_refuses_blinded_plaintext_mix(tmp_path):
    """A concealed ξ± beside a plaintext one is a custody violation — refuse."""
    paths = _data_parts(tmp_path, conceal=True, one_plaintext=True)
    # The plaintext ξ± reporting part fails the load gate first (data + not
    # concealed), so the mix can never even reach assembly.
    with pytest.raises(ValueError, match="refusing to load an unblinded"):
        asm.assemble_sacc(
            "vSYNTH", paths, str(tmp_path / "vSYNTH.sacc"), placeholder_var=1.0
        )


def test_assert_consistent_blind_rejects_divergent_commitments(tmp_path):
    """Two ξ± parts blinded under different commitments must never combine."""
    nz = {0: _nz()}
    theta = np.geomspace(1.0, 100.0, 6)
    a = sw.xi_to_sacc(
        nz, META, theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="reporting"
    )
    b = sw.xi_to_sacc(
        nz, META, theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="integration"
    )
    blinding._stamp_provenance(a, _COMMIT, "A", _DIGEST)
    blinding._stamp_provenance(b, "c" * 64, "A", _DIGEST)
    with pytest.raises(ValueError, match="different blind commitments"):
        blinding.assert_consistent_blind([a, b])


# --------------------------------------------------------------------------- #
# 3. The blinding subgraph resolves in the cosmo_val DAG (candide-only)
# --------------------------------------------------------------------------- #
requires_candide_data = pytest.mark.skipif(
    not Path("/n17data/cdaley/unions").exists(),
    reason="candide-local workflow config/data (/n17data) absent — off-cluster",
)


@requires_candide_data
def test_blinding_subgraph_in_cosmo_val_dry_run():
    """A data-run cosmo_val assemble pulls blind_init + blind_part, and binds the
    ξ± / pseudo-Cℓ parts to their *_blinded siblings."""
    env = os.environ | {"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
    env.pop("SNAKEMAKE_PROFILE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "assemble_sacc_all",
            "--dry-run",
            "--cores",
            "1",
            "--configfile",
            "config/config.yaml",
        ],
        cwd=_repo_root() / "papers/cosmo_val",
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    out = result.stdout
    assert "rule blind_init:" in out, out
    assert "rule blind_part:" in out, out
    # assemble consumes the blinded ξ± reporting and pseudo-Cℓ. The integration
    # ξ± is not gathered into the terminal file (per the #247 ruling), but the
    # COSEBIs / pure-E/B consumers now bind the blinded integration part (via
    # blindable_part) to re-derive their concealed E-modes, so blind_part enters
    # the subgraph for it too.
    assert (
        "_xi_reporting_minsep=1.0_maxsep=250.0_nbins=20_npatch=100_blinded.sacc" in out
    )
    assert "_xi_integration_blinded.sacc" in out
    assert "_blind=A_powspace_nbins=32_blinded.sacc" in out
