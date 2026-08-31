"""Tests for the Snakemake blind-at-birth wiring, independent of a live cluster.

Covers ``workflow/common.py``'s part-path and run-type helpers, and the
data-run fail-closed assembly: ``assemble_sacc`` must refuse an unblinded
``type='data'`` part and succeed once every part is concealed under one
commitment. A candide-only test additionally asserts the blinding subgraph
resolves in the cosmo_val DAG dry-run.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import types
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
# 1. common.py part-path and run-type helpers
# --------------------------------------------------------------------------- #
_STEMS = [
    "SP_v1.4.6.3_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=100",
    "SP_v1.4.6.3_leak_corr_xi_minsep=0.08_maxsep=300_nbins=1000_npatch=1",
    "pseudo_cl_analysis_SP_v1.4.6.3_powspace_nbins=32",
    "pseudo_cl_analysis_SP_v1.4.6.3_leak_corr_powspace_nbins=32",
]


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
    part = "/out/SP_v1.4.6.3_xi_minsep=0.08_maxsep=300_nbins=1000_npatch=1.sacc"
    monkeypatch.setattr(common, "RUN_TYPE", "data")
    assert common.blindable_part(part) == common.blinded_path(part)
    monkeypatch.setattr(common, "RUN_TYPE", "mock")
    assert common.blindable_part(part) == part


# --------------------------------------------------------------------------- #
# 2. Data-run fail-closed assembly (#252 terminal custody gate)
# --------------------------------------------------------------------------- #
META = {"catalogue_version": "vSYNTH", "npatch": 1}
# Two arbitrary-but-consistent hex stamps standing in for a real blind's
# seed commitment / config digest; the assembly only checks they agree across parts.
_COMMIT = "a" * 64
_DIGEST = "b" * 64


def _nz():
    return np.linspace(0.01, 2.0, 40), np.random.default_rng(0).uniform(0.1, 1.0, 40)


def _spd(n, seed):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _data_parts(tmp_path, *, conceal, one_plaintext=False, run_type="data"):
    """Write the five per-statistic parts, stamped ``type=run_type``.

    ``conceal`` stamps every part with the shared blind (concealed=True). With
    ``one_plaintext`` the ξ± reporting part is left unconcealed — a blinded /
    plaintext mix the assembly must refuse. ``run_type='mock'`` writes the parts
    as a mock campaign's producers do, which is the only way an unconcealed
    blindable part is allowed through the assembly.
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
        sio.save(part, str(p), type=run_type)
        paths[name] = str(p)
    return paths


def test_data_assemble_fails_closed_on_unblinded_part(tmp_path):
    """A data run refuses to assemble an unconcealed real part (fail closed)."""
    paths = _data_parts(tmp_path, conceal=False)
    with pytest.raises(ValueError, match="refusing to load an unblinded"):
        asm.assemble_sacc("vSYNTH", paths, str(tmp_path / "vSYNTH.sacc"))


def test_data_assemble_passes_on_blinded_parts(tmp_path):
    """With every part concealed under one blind, the data-run assembly succeeds
    and stamps the shared commitment on the terminal file."""
    paths = _data_parts(tmp_path, conceal=True)
    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc("vSYNTH", paths, str(out))
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
        asm.assemble_sacc("vSYNTH", paths, str(tmp_path / "vSYNTH.sacc"))


def test_data_assemble_runs_behind_the_custody_guard(tmp_path, monkeypatch):
    """The custody guard is reached *through* the production assembly.

    Every part here is concealed, so every part clears the fail-closed load
    gate — the earlier tests all stop there. Only
    ``blinding.assert_consistent_blind`` can catch what is wrong with this
    assembly: the install's Smokescreen draws shifts under a different scheme
    than the one that made the blind, so it could never unblind what it is
    about to write. That ``assemble_sacc`` raises is the check that it runs
    through :func:`sacc_io.gather` like every other assembly path, rather than
    reimplementing the custody wrapper around its own assembler.
    """
    paths = _data_parts(tmp_path, conceal=True)
    monkeypatch.setattr(blinding, "draw_scheme", lambda: 99)
    with pytest.raises(ValueError, match="DRAW_SCHEME"):
        asm.assemble_sacc("vSYNTH", paths, str(tmp_path / "vSYNTH.sacc"))


def test_data_assemble_stamps_the_draw_scheme_on_the_terminal_file(tmp_path):
    """The assembled file carries the blind's draw scheme, like its parts."""
    paths = _data_parts(tmp_path, conceal=True)
    out = tmp_path / "vSYNTH.sacc"
    asm.assemble_sacc("vSYNTH", paths, str(out))
    assert sio.load(str(out)).metadata["blind_draw_scheme"] == blinding.draw_scheme()


def test_mock_assemble_succeeds_without_a_blind(tmp_path):
    """A mock campaign assembles its plaintext parts — the gate's mock branch is
    reachable from real producers, not only from test fixtures.

    Every part writer stamps the campaign's run type (CosmologyValidation's
    ``run_type``, ``run_2pcf``'s ``run_type=`` / ``--run-type``), so a ``mock`` campaign's parts declare themselves mocks and
    ``assert_consistent_blind`` lets them through unconcealed. The same parts
    stamped ``type='data'`` fail closed — that is
    ``test_data_assemble_fails_closed_on_unblinded_part``.
    """
    paths = _data_parts(tmp_path, conceal=False, run_type="mock")
    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc("vSYNTH", paths, str(out))
    assert "concealed" not in s.metadata
    # No escape hatch: a mock part is not gated by the fail-closed loader.
    assert sio.load(str(out)).metadata["type"] == "mock"


def test_rho_tau_part_is_stamped_concealed_from_the_commitment(tmp_path):
    """ρ/τ carries no cosmological vector, but a data run's assembly still opens
    it through the fail-closed load gate — so its writer must stamp it.

    This drives the real writer (``PSFSystematicsMixin.rho_tau_to_sacc_part``)
    with the commitment the ``rho_tau_stats`` rule binds on a data run, and
    checks the emitted part loads without the escape hatch. Without the stamp a
    data run's ``assemble_sacc`` dies on the ρ/τ part before custody is ever
    checked.
    """
    from sp_validation.cosmo_val.core import CosmologyValidation
    from sp_validation.cosmo_val.psf_systematics import PSFSystematicsMixin

    root = tmp_path / "blind"
    blind_dir = root / "vSYNTH"
    blind_dir.mkdir(parents=True)
    commitment = blinding.blind_init(str(blind_dir), log=lambda *_: None)["commitment"]
    theta = np.geomspace(1.0, 100.0, 6)
    rng = np.random.default_rng(11)
    rho = {"theta": theta}
    tau = {"theta": theta}
    for k in sw.RHO_K:
        for sign in ("p", "m"):
            rho[f"rho_{k}_{sign}"] = rng.normal(size=6) * 1e-6
            rho[f"varrho_{k}_{sign}"] = rng.uniform(1e-14, 1e-13, 6)
    for k in sw.TAU_K:
        for sign in ("p", "m"):
            tau[f"tau_{k}_{sign}"] = rng.normal(size=6) * 1e-6
            tau[f"vartau_{k}_{sign}"] = rng.uniform(1e-14, 1e-13, 6)

    class _Writer(PSFSystematicsMixin):
        """The writer's collaborators, stubbed — the method under test is real."""

        run_type = "data"
        blind_root = str(root)
        # The real per-version resolution is part of what this test covers.
        commitment_path = CosmologyValidation.commitment_path

        def sacc_nz(self, version):
            return {0: _nz()}

        def sacc_metadata(self, version):
            return dict(META)

        def print_magenta(self, *args, **kwargs):
            pass

    out_dir = tmp_path / "rho_tau_stats"
    out_dir.mkdir()
    _Writer().rho_tau_to_sacc_part(
        "vSYNTH",
        str(out_dir),
        "vSYNTH",
        types.SimpleNamespace(rho_stats=rho),
        types.SimpleNamespace(tau_stats=tau),
    )

    written = sio.load(str(out_dir / "rho_tau_vSYNTH.sacc"))
    assert written.metadata["concealed"] is True
    assert written.metadata["blind_draw_scheme"] == blinding.draw_scheme()
    with open(commitment, encoding="utf-8") as f:
        committed = json.load(f)
    assert written.metadata["blind_commitment"] == committed["seed_commitment"]
    assert written.metadata["blind_config_digest"] == committed["config_digest"]


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
    assert "_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=100_blinded.sacc" in out
    assert "_xi_minsep=0.08_maxsep=300_nbins=1000_npatch=1_blinded.sacc" in out
    assert "pseudo_cl_analysis_SP_v1.4.6.3_powspace_nbins=32_blinded.sacc" in out
    # Every blindable plaintext part is temp() on a data run, the analysis
    # pseudo-Cℓ included (its own rule exists so it can be).
    assert "Would remove temporary output" in out, out
    assert re.search(
        r"Would remove temporary output \S*pseudo_cl_analysis_\S+\.sacc", out
    ), out
