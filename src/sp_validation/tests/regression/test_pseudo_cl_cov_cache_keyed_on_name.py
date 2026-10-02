"""Pseudo-Cl covariance must reflect current inputs, not an old native file."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import healpy as hp
import numpy as np
import pytest
import yaml
from astropy.io import fits

import sp_validation.cosmo_val.pseudo_cl as pcm
from sp_validation.cosmo_val import CosmologyValidation

VER = "SP_v1.4.6.3"
NSIDE = 16
NBINS = 4
POLS = ["EE", "EB", "BE", "BB"]
ROOT = Path(__file__).resolve().parents[4]
# Config schemas differ: use the active branch's config when testing a checkout.
ACTIVE_ROOT = Path(pcm.__file__).resolve().parents[3]
if ACTIVE_ROOT != ROOT:
    ROOT = ACTIVE_ROOT


@pytest.fixture(autouse=True)
def _unwrap_theory_dict(monkeypatch):
    """Isolate caching from the separate cs_util theory-return incompatibility.

    The external theory backend returns {"W1xW1": array}; develop expects the
    array. Only adapt that external boundary, leaving all cache decisions and
    file reads/writes untouched. Tomography already handles the dictionary.
    """
    if hasattr(pcm, "get_theo_c_ell"):
        orig = pcm.get_theo_c_ell

        def unwrapped(*a, **k):
            out = orig(*a, **k)
            return out["W1xW1"] if isinstance(out, dict) else out

        monkeypatch.setattr(pcm, "get_theo_c_ell", unwrapped)


@pytest.fixture
def cat_config(tmp_path):
    """1536-galaxy catalogue on the northern NSIDE=16 pixel centres and a
    smooth n(z); small enough that a NaMaster covariance takes seconds."""
    base_path = ROOT / "cosmo_val" / "cat_config.yaml"
    if not base_path.exists():
        pytest.skip(f"Repository catalogue configuration missing: {base_path.name}")
    base = yaml.safe_load(base_path.read_text())
    ra, dec = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)), lonlat=True)
    sel = dec > 0
    rng = np.random.default_rng(9)
    t = np.zeros(
        sel.sum(),
        dtype=[
            ("RA", "f8"),
            ("Dec", "f8"),
            ("e1", "f8"),
            ("e2", "f8"),
            ("w_des", "f8"),
        ],
    )
    t["RA"], t["Dec"] = ra[sel], dec[sel]
    t["e1"] = rng.normal(0, 0.01, len(t))
    t["e2"] = rng.normal(0, 0.01, len(t))
    t["w_des"] = 1.0
    fits.BinTableHDU(t).writeto(tmp_path / "tiny.fits")
    z = np.linspace(0.01, 3, 180)
    np.savetxt(
        tmp_path / "nz.txt", np.column_stack([z, z**2 * np.exp(-((z / 0.6) ** 1.5))])
    )
    entry = copy.deepcopy(base[VER])
    entry["subdir"] = str(tmp_path)
    entry["shear"]["path"] = str(tmp_path / "tiny.fits")
    entry["shear"]["redshift_path"] = str(tmp_path / "nz.txt")
    entry["shear"]["w_col"] = "w_des"
    cfg = {
        VER: entry,
        "nz": {"subdir": str(tmp_path), "dndz": {"path": "nz.txt"}},
        "paths": {"output": str(tmp_path / "out")},
    }
    path = tmp_path / "cat_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _ee_cov(output_dir, cat_config, sig8):
    cv = CosmologyValidation(
        [VER],
        catalog_config=str(cat_config),
        output_dir=str(output_dir),
        nside=NSIDE,
        n_ell_bins=NBINS,
        cosmo_params={"sig8": sig8},
    )
    if hasattr(cv, "calculate_pseudo_cl_inka_cov"):  # tomography branch
        cv.calculate_pseudo_cl_inka_cov(compute_tomography=False)
        hdul = cv._pseudo_cls[VER]["cov_iNKA_non_tomo"]
    else:  # develop
        cv.calculate_pseudo_cl_eb_cov()
        hdul = cv._pseudo_cls[VER]["cov"]
    try:
        return np.array(hdul["COVAR_EE_EE"].data)
    finally:
        hdul.close()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#384: pseudo-Cl covariance cache ignores cosmology",
)
def test_pseudo_cl_cov_cache_ignores_cosmology(tmp_path, cat_config):
    """The Gaussian pseudo-Cl covariance scales with the fiducial C_ell, so a
    request at sigma_8 = 1.0 must return the sigma_8 = 1.0 covariance whatever
    an earlier run at sigma_8 = 0.65 left in the same output directory. The
    expected value is computed independently of any cache, in a fresh
    directory; a cache that hits on a name lacking the cosmology returns the
    0.65 covariance instead."""
    shared = tmp_path / "shared"
    _ee_cov(shared, cat_config, 0.65)
    reused = _ee_cov(shared, cat_config, 1.0)
    fresh = _ee_cov(tmp_path / "fresh", cat_config, 1.0)
    ratio = np.diag(fresh) / np.diag(reused)
    assert np.allclose(reused, fresh, rtol=1e-6, atol=0), (
        "sigma_8=1.0 covariance in a directory that held a sigma_8=0.65 run "
        f"differs from a fresh sigma_8=1.0 computation: diag(fresh)/diag(returned) "
        f"= {np.round(ratio, 4).tolist()}"
    )


def _load_script():
    path = ROOT / "workflow" / "scripts" / "generate_pseudo_cl_cov.py"
    spec = importlib.util.spec_from_file_location("gen_pcl_cov", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#384: covariance rule adopts an undeclared stale native file",
)
def test_pseudo_cl_cov_rule_adopts_stale_native_file(tmp_path, cat_config):
    """The pseudo_cl_cov Snakemake rule declares a tagged output, but the
    script renames the untagged native pseudo_cl_cov_{ver}.fits into it.
    Snakemake deletes only declared outputs, so a native file left in the
    shared COSMO_VAL tree by an earlier run must not become this job's
    output. The planted file carries a sentinel (all 7.0, right shape), which
    no Gaussian covariance of this fixture can equal."""
    if not hasattr(CosmologyValidation, "calculate_pseudo_cl_eb_cov"):
        pytest.skip(
            "Tomography branch removed calculate_pseudo_cl_eb_cov; its generator "
            "still calls the develop API (separate script/primitive mismatch)."
        )
    out_dir = tmp_path / "cosmo_val_output"
    out_dir.mkdir()
    stale = fits.HDUList([fits.PrimaryHDU()])
    for a in POLS:
        for b in POLS:
            stale.append(
                fits.ImageHDU(np.full((NBINS, NBINS), 7.0), name=f"COVAR_{a}_{b}")
            )
    stale.writeto(out_dir / f"pseudo_cl_cov_{VER}.fits")

    output = out_dir / f"pseudo_cl_cov_{VER}_powspace_nbins={NBINS}.fits"
    smk = SimpleNamespace(
        output=SimpleNamespace(pseudo_cl_cov=str(output)),
        params={
            "version": VER,
            "cat_config": str(cat_config),
            "nside": NSIDE,
            "npatch": 1,
            "cosmo_params": {
                "Omega_m": 0.315,
                "Omega_b": 0.049,
                "h": 0.674,
                "sigma_8": 0.81,
                "n_s": 0.965,
            },
            "binning": "powspace",
            "nbins": NBINS,
            "power": 0.5,
        },
    )
    _load_script()._from_snakemake(smk)
    with fits.open(output) as hdul:
        ee = np.array(hdul["COVAR_EE_EE"].data)
    assert not np.all(ee == 7.0), (
        "declared output pseudo_cl_cov_..._powspace_nbins=4.fits is the stale "
        f"native file, unchanged: COVAR_EE_EE = {ee.tolist()}"
    )
