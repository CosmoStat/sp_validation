"""Integration tests for the ``assemble_sacc.py`` workflow script.

The pure assembler (``sacc_writers.assemble_analysis_sacc``) is covered in
``test_sacc_writers.py``. This file exercises the *script seam* the DAG uses:
``assemble_sacc.assemble_sacc`` loads per-statistic ``.sacc`` part *files* in
CANONICAL order, injects the born-cov-less ξ± / pseudo-Cℓ blocks (real CosmoCov
/ NaMaster covariance, or a flagged diagonal placeholder), and writes one
``{version}.sacc`` whose points and covariance blocks land in canonical order.

The script lives under ``workflow/scripts`` (off the package path); it is loaded
by file path exactly as the lightcone/ASTRA CLI path imports it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sp_validation import sacc_io as sio
from sp_validation.cosmo_val import sacc_writers as sw


def _load_assemble_module():
    """Import ``workflow/scripts/assemble_sacc.py`` by file path."""
    repo_root = next(
        p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
    )
    path = repo_root / "workflow" / "scripts" / "assemble_sacc.py"
    spec = importlib.util.spec_from_file_location("assemble_sacc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


asm = _load_assemble_module()


def _nz(seed=0, n=40):
    rng = np.random.default_rng(seed)
    return np.linspace(0.01, 2.0, n), rng.uniform(0.1, 1.0, n)


def _spd(n, seed):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _theta(n=6):
    return np.geomspace(1.0, 100.0, n)


META = {"catalogue_version": "vSYNTH", "npatch": 1}


def _write_parts(tmp_path, *, with_pseudo_cl=True, cov_less=("xi_coarse",)):
    """Write per-statistic parts to disk; return the ``{name: path}`` mapping.

    Parts named in ``cov_less`` are written without a covariance (mimicking the
    born-cov-less ξ± coarse / pseudo-Cℓ parts); the rest carry their own block.
    """
    nz = {0: _nz()}
    theta = _theta()
    ell = np.array([30.0, 60.0, 90.0])

    class _Wsp:
        def get_bandpower_windows(self):
            w = np.zeros((4, 3, 4, 20))
            for out in range(4):
                for b in range(3):
                    w[out, b, out, b * 6 : b * 6 + 6] = 1.0
            return w

    xi = sw.xi_to_sacc(
        nz, META, theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    if "xi_coarse" not in cov_less:
        xi.add_covariance(_spd(len(xi.mean), 1))

    cl_all = np.vstack(
        [np.arange(3) * 1e-9, np.arange(3) * 2e-9, np.zeros(3), np.arange(3) * 3e-9]
    )
    cl = sw.pseudo_cl_to_sacc(
        nz,
        META,
        ell,
        cl_all,
        _Wsp(),
        covariance=None if "pseudo_cl" in cov_less else _spd(9, 2),
    )

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

    eb_arrays = {
        key: np.arange(6) * (i + 1) * 1e-6 for i, key in enumerate(sio.PURE_KEYS)
    }
    eb = sw.pure_eb_to_sacc(nz, META, theta, eb_arrays, covariance=_spd(36, 4))

    rho = {"theta": theta}
    tau = {"theta": theta}
    rng = np.random.default_rng(5)
    for k in sw.RHO_K:
        for suffix in ("p", "m"):
            rho[f"rho_{k}_{suffix}"] = rng.normal(size=6) * 1e-6
            rho[f"varrho_{k}_{suffix}"] = rng.uniform(1e-14, 1e-13, 6)
    for k in sw.TAU_K:
        for suffix in ("p", "m"):
            tau[f"tau_{k}_{suffix}"] = rng.normal(size=6) * 1e-6
            tau[f"vartau_{k}_{suffix}"] = rng.uniform(1e-14, 1e-13, 6)
    rt = sw.rho_tau_to_sacc(nz, META, rho, tau)

    parts = {
        "xi_coarse": xi,
        "pseudo_cl": cl,
        "cosebis": co,
        "pure_eb": eb,
        "rho_tau": rt,
    }
    if not with_pseudo_cl:
        parts.pop("pseudo_cl")

    paths = {}
    for name, part in parts.items():
        p = tmp_path / f"{name}.sacc"
        sio.save(part, str(p), type="mock")
        paths[name] = str(p)
    return paths


def test_assemble_sacc_placeholder_canonical_order(tmp_path):
    """The cov-less ξ± part gets a placeholder; every point is covered and the
    blocks land in canonical order (ξ±, pseudo-Cℓ, COSEBIs, pure-E/B, ρ, τ)."""
    paths = _write_parts(tmp_path, cov_less=("xi_coarse",))
    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc("vSYNTH", paths, str(out), placeholder_var=1.0)
    assert out.exists()
    assert type(s.covariance).__name__ == "FullCovariance"
    assert s.covariance.dense.shape == (len(s.mean), len(s.mean))

    # Canonical insertion order: the first data types are ξ+ then ξ−.
    types_in_order = [dp.data_type for dp in s.data]
    assert types_in_order[0] == sio.XI_PLUS
    assert sio.XI_MINUS in types_in_order
    # ξ appears before pseudo-Cℓ before COSEBIs before pure-E/B before ρ/τ.
    first = {t: types_in_order.index(t) for t in set(types_in_order)}
    assert first[sio.XI_PLUS] < first[sio.CL_EE] < first[sio.COSEBI_EE]
    assert first[sio.COSEBI_EE] < first[sio.PURE_TYPES["xip_E"]]
    assert first[sio.PURE_TYPES["xip_E"]] < first[sio.RHO_PLUS.format(k=0)]
    assert first[sio.RHO_PLUS.format(k=0)] < first[sio.TAU_PLUS.format(k=0)]

    # The ξ± block is the placeholder diagonal (variance 1.0 on its own points).
    tr = ("source_0", "source_0")
    xi_idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    dense = s.covariance.dense
    assert np.allclose(np.diag(dense[np.ix_(xi_idx, xi_idx)]), 1.0)
    # ...and it does not bleed into the neighbouring COSEBIs block (cross zero).
    co_idx = np.concatenate(
        [s.indices(sio.COSEBI_EE, tr), s.indices(sio.COSEBI_BB, tr)]
    )
    assert np.allclose(dense[np.ix_(xi_idx, co_idx)], 0.0)


def test_assemble_sacc_injects_real_xi_covariance(tmp_path):
    """A CosmoCov ξ covariance .txt is loaded into the cov-less ξ± block."""
    paths = _write_parts(tmp_path, cov_less=("xi_coarse",))
    # ξ± part has 12 points ([ξ+; ξ−] over 6 θ); supply a matching cov .txt.
    xi_cov = _spd(12, 21)
    cov_path = tmp_path / "xi_cov.txt"
    np.savetxt(str(cov_path), xi_cov)

    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc("vSYNTH", paths, str(out), xi_cov=str(cov_path))
    tr = ("source_0", "source_0")
    xi_idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    assert np.allclose(s.covariance.dense[np.ix_(xi_idx, xi_idx)], xi_cov)


def test_assemble_sacc_injects_pseudo_cl_covariance(tmp_path):
    """The NaMaster cov FITS (COVAR_EE_EE/BB_BB/EB_EB) → block-diagonal pseudo-Cℓ
    block (the live default: ξ± placeholder + real pseudo-Cℓ cov)."""
    from astropy.io import fits

    paths = _write_parts(tmp_path, cov_less=("xi_coarse", "pseudo_cl"))
    # pseudo-Cℓ part is 3 ell × {EE, BB, EB} = 9 points; per-spectrum 3×3 blocks.
    ee, bb, eb = _spd(3, 31), _spd(3, 32), _spd(3, 33)
    cov_fits = tmp_path / "pseudo_cl_cov.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(ee, name="COVAR_EE_EE"),
            fits.ImageHDU(bb, name="COVAR_BB_BB"),
            fits.ImageHDU(eb, name="COVAR_EB_EB"),
        ]
    ).writeto(str(cov_fits))

    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc(
        "vSYNTH", paths, str(out), pseudo_cl_cov=str(cov_fits), placeholder_var=1.0
    )
    tr = ("source_0", "source_0")
    cl_idx = np.concatenate(
        [s.indices(sio.CL_EE, tr), s.indices(sio.CL_BB, tr), s.indices(sio.CL_EB, tr)]
    )
    dense = s.covariance.dense
    expected = np.zeros((9, 9))
    expected[0:3, 0:3], expected[3:6, 3:6], expected[6:9, 6:9] = ee, bb, eb
    assert np.allclose(dense[np.ix_(cl_idx, cl_idx)], expected)
    # ξ± stays the placeholder; the two blocks don't bleed into each other.
    xi_idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    assert np.allclose(np.diag(dense[np.ix_(xi_idx, xi_idx)]), 1.0)
    assert np.allclose(dense[np.ix_(xi_idx, cl_idx)], 0.0)


def test_assemble_sacc_missing_cov_raises(tmp_path):
    """A cov-less part with no injected block and no placeholder fails loudly."""
    paths = _write_parts(tmp_path, cov_less=("xi_coarse",))
    out = tmp_path / "vSYNTH.sacc"
    with pytest.raises(ValueError, match="carries no covariance"):
        asm.assemble_sacc("vSYNTH", paths, str(out))


def test_assemble_sacc_respects_pseudo_cl_toggle(tmp_path):
    """With pseudo_cl absent, assembly still succeeds and omits the Cℓ points."""
    paths = _write_parts(tmp_path, with_pseudo_cl=False, cov_less=("xi_coarse",))
    assert "pseudo_cl" not in paths
    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc("vSYNTH", paths, str(out), placeholder_var=1.0)
    tr = ("source_0", "source_0")
    assert len(s.indices(sio.CL_EE, tr)) == 0
    # Round-trips as a valid FullCovariance over the remaining points.
    s2 = sio.load(str(out))
    assert type(s2.covariance).__name__ == "FullCovariance"
    assert s2.covariance.dense.shape == (len(s2.mean), len(s2.mean))


def test_assemble_sacc_expected_part_missing_raises(tmp_path):
    """A typo'd input keyword drops a part from part_paths; the expected list
    catches it rather than silently omitting the statistic."""
    paths = _write_parts(tmp_path, cov_less=("xi_coarse",))
    # Simulate a rule-input typo: cosebis wired under the wrong key.
    paths["cosebi"] = paths.pop("cosebis")
    out = tmp_path / "vSYNTH.sacc"
    with pytest.raises(ValueError, match="expected parts \\['cosebis'\\] missing"):
        asm.assemble_sacc(
            "vSYNTH",
            paths,
            str(out),
            expected=["xi_coarse", "pseudo_cl", "cosebis", "pure_eb", "rho_tau"],
            placeholder_var=1.0,
        )


def test_assemble_sacc_expected_rejects_unknown_name(tmp_path):
    """A typo in the expected list itself is rejected (not a valid statistic)."""
    paths = _write_parts(tmp_path, cov_less=("xi_coarse",))
    out = tmp_path / "vSYNTH.sacc"
    with pytest.raises(ValueError, match="not assemblable statistics"):
        asm.assemble_sacc(
            "vSYNTH", paths, str(out), expected=["cosebi"], placeholder_var=1.0
        )
