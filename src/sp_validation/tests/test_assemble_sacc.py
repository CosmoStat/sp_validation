"""Integration tests for the ``assemble_sacc.py`` workflow script.

The pure assembler (``sacc_writers.assemble_analysis_sacc``) is covered in
``test_sacc_writers.py``. This file exercises the *script seam* the DAG uses:
``assemble_sacc.assemble_sacc`` loads per-statistic ``.sacc`` part *files* in
CANONICAL order, injects the born-cov-less ξ± / pseudo-Cℓ blocks from the real
CosmoCov ``.txt`` and NaMaster covariance FITS, and writes one
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


def _xi_cov_txt(tmp_path, n=12, seed=21):
    """A CosmoCov-format ξ± covariance: the dense ``_processed.txt`` matrix.

    Same writer/format ``covariance_process`` emits and ``--xi-cov`` reads, at
    the synthetic parts' 12-point ([ξ+; ξ−] over 6 θ) size. Returns
    ``(path, matrix)``.
    """
    cov = _spd(n, seed)
    path = tmp_path / "xi_cov_processed.txt"
    np.savetxt(str(path), cov)
    return str(path), cov


def _pseudo_cl_cov_fits(tmp_path, n=3):
    """A NaMaster covariance FITS: one HDU per spectrum. Returns (path, blocks)."""
    from astropy.io import fits

    blocks = {"EE": _spd(n, 31), "BB": _spd(n, 32), "EB": _spd(n, 33)}
    path = tmp_path / "pseudo_cl_cov.fits"
    fits.HDUList(
        [fits.PrimaryHDU()]
        + [fits.ImageHDU(block, name=f"COVAR_{k}_{k}") for k, block in blocks.items()]
    ).writeto(str(path))
    return str(path), blocks


def _write_parts(tmp_path, *, with_pseudo_cl=True, cov_less=("xi_reporting",)):
    """Write per-statistic parts to disk; return the ``{name: path}`` mapping.

    Parts named in ``cov_less`` are written without a covariance (mimicking the
    born-cov-less ξ± reporting / pseudo-Cℓ parts); the rest carry their own block.
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
        nz, META, theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="reporting"
    )
    if "xi_reporting" not in cov_less:
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
        "xi_reporting": xi,
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


def test_assemble_sacc_canonical_order(tmp_path):
    """Every point is covered and the blocks land in canonical order
    (ξ±, pseudo-Cℓ, COSEBIs, pure-E/B, ρ, τ)."""
    paths = _write_parts(tmp_path, cov_less=("xi_reporting",))
    cov_path, xi_cov = _xi_cov_txt(tmp_path)
    cl_cov_path, _blocks = _pseudo_cl_cov_fits(tmp_path)
    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc(
        "vSYNTH", paths, str(out), xi_cov=cov_path, pseudo_cl_cov=cl_cov_path
    )
    assert out.exists()
    assert type(s.covariance).__name__ == "BlockDiagonalCovariance"
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

    # The ξ± block is the injected CosmoCov matrix on its own points.
    tr = ("source_0", "source_0")
    xi_idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    dense = s.covariance.dense
    assert np.allclose(dense[np.ix_(xi_idx, xi_idx)], xi_cov)
    # ...and it does not bleed into the neighbouring COSEBIs block (cross zero).
    co_idx = np.concatenate(
        [s.indices(sio.COSEBI_EE, tr), s.indices(sio.COSEBI_BB, tr)]
    )
    assert np.allclose(dense[np.ix_(xi_idx, co_idx)], 0.0)


def test_injected_xi_covariance_replaces_the_parts_own(tmp_path):
    """The analytic ξ± covariance wins over the estimate the part was born with.

    The reporting part carries the jackknife it was measured with — useful as a
    diagnostic, but the analysis file takes the CosmoCov block.
    """
    paths = _write_parts(tmp_path, cov_less=())  # ξ± born with its own jackknife
    cov_path, xi_cov = _xi_cov_txt(tmp_path)
    cl_cov_path, _blocks = _pseudo_cl_cov_fits(tmp_path)

    born = sio.load(paths["xi_reporting"], allow_unblinded=True).covariance.dense
    assert not np.allclose(born, xi_cov)  # the two are distinguishable

    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc(
        "vSYNTH", paths, str(out), xi_cov=cov_path, pseudo_cl_cov=cl_cov_path
    )
    tr = ("source_0", "source_0")
    xi_idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    assert np.allclose(s.covariance.dense[np.ix_(xi_idx, xi_idx)], xi_cov)


def test_assemble_sacc_injects_pseudo_cl_covariance(tmp_path):
    """The NaMaster cov FITS (COVAR_EE_EE/BB_BB/EB_EB) → block-diagonal pseudo-Cℓ
    block, beside the injected CosmoCov ξ± block (the live default)."""
    paths = _write_parts(tmp_path, cov_less=("xi_reporting", "pseudo_cl"))
    cov_path, xi_cov = _xi_cov_txt(tmp_path)
    # pseudo-Cℓ part is 3 ell × {EE, BB, EB} = 9 points; per-spectrum 3×3 blocks.
    cov_fits, blocks = _pseudo_cl_cov_fits(tmp_path)
    ee, bb, eb = blocks["EE"], blocks["BB"], blocks["EB"]

    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc(
        "vSYNTH", paths, str(out), xi_cov=cov_path, pseudo_cl_cov=cov_fits
    )
    tr = ("source_0", "source_0")
    cl_idx = np.concatenate(
        [s.indices(sio.CL_EE, tr), s.indices(sio.CL_BB, tr), s.indices(sio.CL_EB, tr)]
    )
    dense = s.covariance.dense
    expected = np.zeros((9, 9))
    expected[0:3, 0:3], expected[3:6, 3:6], expected[6:9, 6:9] = ee, bb, eb
    assert np.allclose(dense[np.ix_(cl_idx, cl_idx)], expected)
    # ξ± carries its own CosmoCov block; the two don't bleed into each other.
    xi_idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    assert np.allclose(dense[np.ix_(xi_idx, xi_idx)], xi_cov)
    assert np.allclose(dense[np.ix_(xi_idx, cl_idx)], 0.0)


def test_missing_injected_covariance_raises(tmp_path):
    """A statistic whose covariance is external cannot fall back to its own."""
    paths = _write_parts(tmp_path, cov_less=())  # every part born with a block
    out = tmp_path / "vSYNTH.sacc"
    with pytest.raises(ValueError, match="takes its analysis covariance from"):
        asm.assemble_sacc("vSYNTH", paths, str(out))


def test_assemble_sacc_respects_pseudo_cl_toggle(tmp_path):
    """With pseudo_cl absent, assembly still succeeds and omits the Cℓ points."""
    paths = _write_parts(tmp_path, with_pseudo_cl=False, cov_less=("xi_reporting",))
    assert "pseudo_cl" not in paths
    cov_path, _xi_cov = _xi_cov_txt(tmp_path)
    out = tmp_path / "vSYNTH.sacc"
    s = asm.assemble_sacc("vSYNTH", paths, str(out), xi_cov=cov_path)
    tr = ("source_0", "source_0")
    assert len(s.indices(sio.CL_EE, tr)) == 0
    # Round-trips as a valid BlockDiagonalCovariance over the remaining points.
    s2 = sio.load(str(out))
    assert type(s2.covariance).__name__ == "BlockDiagonalCovariance"
    assert s2.covariance.dense.shape == (len(s2.mean), len(s2.mean))


def test_assemble_sacc_expected_part_missing_raises(tmp_path):
    """A typo'd input keyword drops a part from part_paths; the expected list
    catches it rather than silently omitting the statistic."""
    paths = _write_parts(tmp_path, cov_less=("xi_reporting",))
    # Simulate a rule-input typo: cosebis wired under the wrong key.
    paths["cosebi"] = paths.pop("cosebis")
    cov_path, _xi_cov = _xi_cov_txt(tmp_path)
    out = tmp_path / "vSYNTH.sacc"
    with pytest.raises(ValueError, match="expected parts \\['cosebis'\\] missing"):
        asm.assemble_sacc(
            "vSYNTH",
            paths,
            str(out),
            expected=["xi_reporting", "pseudo_cl", "cosebis", "pure_eb", "rho_tau"],
            xi_cov=cov_path,
        )


def test_assemble_sacc_expected_rejects_unknown_name(tmp_path):
    """A typo in the expected list itself is rejected (not a valid statistic)."""
    paths = _write_parts(tmp_path, cov_less=("xi_reporting",))
    cov_path, _xi_cov = _xi_cov_txt(tmp_path)
    out = tmp_path / "vSYNTH.sacc"
    with pytest.raises(ValueError, match="not assemblable statistics"):
        asm.assemble_sacc(
            "vSYNTH", paths, str(out), expected=["cosebi"], xi_cov=cov_path
        )
