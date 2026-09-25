"""End to end: one blinded catalogue through the rule scripts, then its reveal.

A synthetic catalogue is declared three ways (``TOY`` blinded, ``TOY_OPEN``
unblinded, ``TOY_MOCK`` mock). A blind is drawn for ``TOY`` with ``blinding
init``; the rule scripts run as Snakemake runs them (``runpy`` with a
``snakemake`` object) for ``TOY`` and ``TOY_leak_corr``: both ξ± grids, a
two-bin writer, pseudo-Cℓ on an nside-32 NaMaster workspace, ρ/τ, COSEBIs,
pure-E/B and assembly. The blind is then revealed, the declaration flipped, the
chain re-run, and the audit must prove blinded − true = shift(seed) on every
part the reveal archived.
"""

import json
import os
import runpy
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from _synthetic import write_synthetic_catalogs

from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio
from sp_validation.cosmo_val import CosmologyValidation

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "workflow" / "scripts"

GRIDS = {
    "reporting": {"min_sep": 5.0, "max_sep": 60.0, "nbins": 6, "npatch": 1},
    "integration": {"min_sep": 1.0, "max_sep": 150.0, "nbins": 120, "npatch": 1},
}
SCALE_CUT = [12.0, 60.0]
PARTS = ("xi_reporting", "pseudo_cl", "cosebis", "pure_eb", "rho_tau")


class Named(list):
    """The shape of Snakemake's ``input``/``output``/``params``: a list with names."""

    def __init__(self, **items):
        super().__init__(items.values())
        self._items = items

    def __getitem__(self, key):
        return self._items[key] if isinstance(key, str) else super().__getitem__(key)

    def __getattr__(self, name):
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, key, default=None):
        return self._items.get(key, default)


def run_rule(script, *, input=None, output=None, params=None):
    """Run a rule script as Snakemake runs a job: outputs removed, then ``script:``."""
    for path in (output or {}).values():
        Path(path).unlink(missing_ok=True)
    smk = types.SimpleNamespace(
        input=Named(**(input or {})),
        output=Named(**(output or {})),
        params=Named(**(params or {})),
    )
    runpy.run_path(
        str(SCRIPTS / script), init_globals={"snakemake": smk}, run_name="__main__"
    )


def declared(cat_config, version):
    """The custody a host Snakemake resolves for ``version``."""
    return cu.custody_of(
        yaml.safe_load(Path(cat_config).read_text()),
        version,
        registry=cu.registry_of(cat_config),
    )


def _namaster_part_inputs():
    """``(ell_eff, cl_all, workspace)`` on an nside-32 full-sky workspace."""
    import healpy as hp
    import pymaster as nmt

    nside = 32
    npix = hp.nside2npix(nside)
    field = nmt.NmtField(np.ones(npix), [np.zeros(npix), np.zeros(npix)], spin=2)
    binning = nmt.NmtBin.from_nside_linear(nside, 8)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, binning)
    ell = binning.get_effective_ells()
    ee = 1e-8 * (ell / 30.0) ** -1.2
    return ell, np.array([ee, 0.01 * ee, 0.01 * ee, 0.02 * ee]), workspace


def _rho_tau_handlers():
    rng = np.random.default_rng(5)
    theta = np.geomspace(5.0, 60.0, 6)
    rho, tau = {"theta": theta}, {"theta": theta}
    for k in range(6):
        for sfx in ("p", "m"):
            rho[f"rho_{k}_{sfx}"] = rng.normal(size=6) * 1e-6
            rho[f"varrho_{k}_{sfx}"] = rng.uniform(1e-14, 1e-13, 6)
    for k in (0, 2, 5):
        for sfx in ("p", "m"):
            tau[f"tau_{k}_{sfx}"] = rng.normal(size=6) * 1e-6
            tau[f"vartau_{k}_{sfx}"] = rng.uniform(1e-14, 1e-13, 6)
    return types.SimpleNamespace(rho_stats=rho), types.SimpleNamespace(tau_stats=tau)


def _covariances(root):
    """CosmoCov-format ξ± covariances on both grids, and a NaMaster Cℓ FITS."""
    from astropy.io import fits

    paths = {}
    for grid, b in GRIDS.items():
        paths[grid] = root / f"cov_{grid}.txt"
        np.savetxt(paths[grid], np.diag(np.full(2 * b["nbins"], 1e-10)))
    nbp = len(_namaster_part_inputs()[0])
    paths["pseudo_cl"] = root / "pseudo_cl_cov.fits"
    fits.HDUList(
        [fits.PrimaryHDU()]
        + [
            fits.ImageHDU(np.eye(nbp) * 1e-20, name=name)
            for name in ("COVAR_EE_EE", "COVAR_BB_BB", "COVAR_EB_EB")
        ]
    ).writeto(paths["pseudo_cl"], overwrite=True)
    return paths


def run_chain(cat_config, version, out, cov, *, grid_parts=None):
    """The cosmo_val chain for one version, into ``out``; returns the part paths."""
    out.mkdir(parents=True, exist_ok=True)
    token = declared(cat_config, version).token
    xi = {}
    for grid, b in GRIDS.items():
        xi[grid] = out / f"{version}_xi_{grid}.sacc"
        binning = (
            f"minsep={b['min_sep']}_maxsep={b['max_sep']}_nbins={b['nbins']}"
            f"_npatch={b['npatch']}"
        )
        run_rule(
            "run_2pcf.py",
            output={
                "txt": str(out / f"{version}_xi_{binning}.txt"),
                "sacc": str(xi[grid]),
            },
            params={
                "ver": version,
                **b,
                "cat_config": str(cat_config),
                "output_dir": str(out),
                "grid": grid,
                "custody": token,
            },
        )
    if grid_parts == "only":
        return xi

    cv = CosmologyValidation(
        versions=[version], catalog_config=str(cat_config), output_dir=str(out)
    )
    paths = {
        "xi_reporting": xi["reporting"],
        "pseudo_cl": out / f"pseudo_cl_{version}.sacc",
        "cosebis": out / f"{version}_cosebis.sacc",
        "pure_eb": out / f"{version}_pure_eb.sacc",
        "rho_tau": out / f"rho_tau_{version}.sacc",
    }
    cv.pseudo_cl_to_sacc_part(
        version, str(paths["pseudo_cl"]), *_namaster_part_inputs()
    )
    cv.rho_tau_to_sacc_part(version, str(out), version, *_rho_tau_handlers())

    integ = GRIDS["integration"]
    run_rule(
        "cv_cosebis.py",
        input={"xi": str(xi["integration"]), "cov": str(cov["integration"])},
        output={
            "npz": str(out / f"{version}_cosebis.npz"),
            "sacc": str(paths["cosebis"]),
            "figure_modes": str(out / f"{version}_cosebis_modes.png"),
            "figure_covariance": str(out / f"{version}_cosebis_cov.png"),
            "figure_scalecut_ptes": str(out / f"{version}_cosebis_ptes.png"),
        },
        params={
            "version": version,
            "min_sep": integ["min_sep"],
            "max_sep": integ["max_sep"],
            "nbins": integ["nbins"],
            "nmodes": 5,
            "scale_cuts": [SCALE_CUT],
            "fiducial_scale_cut": SCALE_CUT,
        },
    )
    rep = GRIDS["reporting"]
    run_rule(
        "cv_pure_eb.py",
        input={
            "xi_reporting": str(xi["reporting"]),
            "xi_integration": str(xi["integration"]),
            "cov_integration": str(cov["integration"]),
        },
        output={
            "npz": str(out / f"{version}_pure_eb.npz"),
            "sacc": str(paths["pure_eb"]),
            "figure_integration_vs_reporting": str(out / f"{version}_eb_ivr.png"),
            "figure_xis": str(out / f"{version}_eb_xis.png"),
            "figure_ptes": str(out / f"{version}_eb_ptes.png"),
            "figure_covariance": str(out / f"{version}_eb_cov.png"),
        },
        params={
            "version": version,
            "min_sep": rep["min_sep"],
            "max_sep": rep["max_sep"],
            "nbins": rep["nbins"],
            "n_samples": 10,
            "cosmo_params": {"transfer_function": "eisenstein_hu"},
            "fiducial_scale_cut": SCALE_CUT,
        },
    )
    assemble(cat_config, version, out, paths, cov, token)
    return paths


def assemble(cat_config, version, out, paths, cov, token):
    run_rule(
        "assemble_sacc.py",
        input={
            **{k: str(v) for k, v in paths.items()},
            "xi_cov": str(cov["reporting"]),
            "pseudo_cl_cov": str(cov["pseudo_cl"]),
        },
        output={"sacc": str(out / f"{version}.sacc")},
        params={
            "version": version,
            "expected": list(PARTS),
            "custody": token,
            "cat_config": str(cat_config),
        },
    )


def stamps(root):
    """``{relative path: stamp}`` of every SACC under ``root``."""
    return {
        str(p.relative_to(root)): cu.read_stamp(sio.load(p).metadata).stamp
        for p in sorted(root.rglob("*.sacc"))
        if "revealed" not in p.relative_to(root).parts
    }


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """The catalogue config, a blind for TOY, and the covariances the rules read."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import cv_runner
    import matplotlib

    # Rule jobs line-buffer their own streams; under pytest they are captured.
    monkeypatch.setattr(cv_runner, "_unbuffer_streams", lambda: None)
    # The figures render with matplotlib's own text engine, whatever LaTeX
    # setup the invoking user's matplotlibrc asks for.
    monkeypatch.setitem(matplotlib.rcParams, "text.usetex", False)
    params, _ = write_synthetic_catalogs(
        tmp_path,
        n_gal=20000,
        coherent_shear=True,
        with_psf=True,
        catalogues={"TOY": None, "TOY_OPEN": "unblinded", "TOY_MOCK": "mock"},
    )
    (tmp_path / "fast.json").write_text(
        json.dumps({"theory": {"transfer_function": "eisenstein_hu"}})
    )
    cat_config = Path(params["catalog_config"])
    bd.main(
        [
            "init",
            "toy",
            "TOY",
            "--cat-config",
            str(cat_config),
            "--config",
            str(tmp_path / "fast.json"),
        ]
    )
    return types.SimpleNamespace(
        root=tmp_path, cat_config=cat_config, cov=_covariances(tmp_path)
    )


def test_a_blinded_catalogue_from_birth_to_audit(toy, monkeypatch):
    out = toy.root / "cosmo_val"
    blinded = declared(toy.cat_config, "TOY")
    assert blinded.status == "blinded"

    # --- a blinded run -------------------------------------------------------
    for version in ("TOY", "TOY_leak_corr"):
        run_chain(toy.cat_config, version, out, toy.cov)
    born = stamps(out)
    assert len(born) == 2 * (2 + len(PARTS)), sorted(born)
    assert all(s == blinded.stamp for s in born.values()), born  # one commitment
    assert (
        bd.main(["verify", str(out / "TOY.sacc"), "--cat-config", str(toy.cat_config)])
        == 0
    )

    # A two-bin writer conceals every pair by passing the custody.
    two_bin = sio.new_sacc(
        {
            0: sio.get_nz(sio.load(out / "TOY_xi_reporting.sacc"), 0),
            1: sio.get_nz(sio.load(out / "TOY_xi_reporting.sacc"), 0),
        },
    )
    theta = np.geomspace(5.0, 60.0, 6)
    for pair in ((0, 0), (0, 1), (1, 1)):
        sio.add_xi(two_bin, pair, theta, 1e-5 / theta, 1e-6 / theta, grid="tomo")
    written = sio.save(two_bin, out / "two_bin.sacc", custody=blinded)
    assert np.all(np.asarray(written.mean) != np.asarray(two_bin.mean))

    # The true vector of TOY is TOY_OPEN's, the same files declared unblinded.
    open_parts = run_chain(
        toy.cat_config, "TOY_OPEN", toy.root / "open", toy.cov, grid_parts="only"
    )
    with pytest.raises(ValueError, match="stamps"):
        assemble(
            toy.cat_config,
            "TOY",
            toy.root / "swap",
            {
                "xi_reporting": open_parts["reporting"],
                **{
                    k: out / Path(v).name
                    for k, v in {
                        "pseudo_cl": "pseudo_cl_TOY.sacc",
                        "cosebis": "TOY_cosebis.sacc",
                        "pure_eb": "TOY_pure_eb.sacc",
                        "rho_tau": "rho_tau_TOY.sacc",
                    }.items()
                },
            },
            toy.cov,
            blinded.token,
        )
    plain = sio.load(open_parts["reporting"])
    for key in cu.STAMP_KEYS:
        plain.metadata.pop(key, None)
    with pytest.raises(ValueError, match="not copies"):
        sio.save(
            plain,
            out / "laundered.sacc",
            derived_from=[sio.load(out / "TOY_xi_reporting.sacc")],
        )
    assert not (out / "laundered.sacc").exists()

    # --- an interrupted birth leaves no part ---------------------------------
    from sp_validation.cosmo_val import sacc_writers

    def after_compute(*args, **kwargs):
        raise RuntimeError("interrupted after the measurement")

    with monkeypatch.context() as m:
        m.setattr(sacc_writers, "xi_to_sacc", after_compute)
        with pytest.raises(RuntimeError):
            run_chain(
                toy.cat_config,
                "TOY",
                toy.root / "interrupted",
                toy.cov,
                grid_parts="only",
            )
    assert not list((toy.root / "interrupted").glob("*.sacc"))

    calls = []
    real_factor = bd._factor

    def second_block_fails(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted inside conceal")
        return real_factor(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(bd, "_factor", second_block_fails)
        with pytest.raises(RuntimeError, match="inside conceal"):
            sio.save(
                two_bin, toy.root / "interrupted" / "two_bin.sacc", custody=blinded
            )
    assert not (toy.root / "interrupted" / "two_bin.sacc").exists()

    # --- the reveal: archive, flip the declaration, re-measure, audit --------
    (out / "two_bin.sacc").unlink()
    assert (
        bd.main(
            ["reveal", "toy", "--root", str(out), "--cat-config", str(toy.cat_config)]
        )
        == 0
    )
    archive = out / "revealed" / "toy"
    assert sorted(
        str(p.relative_to(archive)) for p in archive.rglob("*.sacc")
    ) == sorted(born)
    assert not list(out.glob("*.sacc"))

    config = yaml.safe_load(toy.cat_config.read_text())
    config["TOY"]["blinding"] = "unblinded"
    toy.cat_config.write_text(yaml.safe_dump(config, sort_keys=False))
    true = declared(toy.cat_config, "TOY")
    assert true.token == "unblinded:TOY"

    for version in ("TOY", "TOY_leak_corr"):
        run_chain(toy.cat_config, version, out, toy.cov)
    assert all(s == true.stamp for s in stamps(out).values())

    report = bd.audit("toy", archive=archive, true_root=out, cat_config=toy.cat_config)
    assert report["ok"], json.dumps(report, indent=1, default=str)
    assert set(report["parts"]) == set(born)
    assert (
        abs(report["shift"]["S8"]) <= 0.075 and abs(report["shift"]["Omega_m"]) <= 0.1
    )

    revealed = toy.cat_config.parent / "blinds" / "toy" / "revealed.json"
    published = revealed.read_text()
    os.chmod(revealed, 0o644)
    revealed.write_text(json.dumps({"seed": "not-the-seed"}))
    assert not bd.audit(
        "toy", archive=archive, true_root=out, cat_config=toy.cat_config
    )["ok"]
    revealed.write_text(published)


def test_a_mock_never_opens_a_blind(toy, monkeypatch):
    def refuse(custody):
        raise AssertionError(f"a mock opened a blind: {custody}")

    monkeypatch.setattr(bd, "open_blind", refuse)
    out = toy.root / "mock"
    run_chain(toy.cat_config, "TOY_MOCK", out, toy.cov)
    mock = declared(toy.cat_config, "TOY_MOCK")
    found = stamps(out)
    assert len(found) == 2 + len(PARTS)
    assert all(s == mock.stamp for s in found.values()), found
