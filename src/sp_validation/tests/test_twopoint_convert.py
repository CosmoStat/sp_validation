"""Byte-compare tests for the SACC -> 2pt-FITS converter.

The converter (:mod:`sp_validation.twopoint_convert`) must reproduce the CosmoSIS
2pt-FITS that ``cosmo_inference/scripts/cosmosis_fitting.py`` assembles today,
so the inference chain (``2pt_like`` and Sacha Guerrini's rho/tau
``2pt_like_xi_sys`` fork) runs untouched behind it. The strongest possible check
is *byte* equality, and astropy writes FITS deterministically, so that is what we
assert: build a reference with the current script's own HDU-builder functions on
deterministic synthetic inputs, build a SACC from those same inputs via
:mod:`sp_validation.sacc_io`, convert it, and compare the two files byte for byte.

Three configurations pin the three product shapes today's ``__main__`` emits:

1. **plain xi** -- PRIMARY, NZ_SOURCE, COVMAT, XI_PLUS, XI_MINUS.
2. **xi + pseudo-Cl** -- adds COVMAT_CELL and CELL_EE (the harmonic block
   ``2pt_like`` reads; the script builds CELL_BB and discards it, so the
   converter does too).
3. **xi + rho/tau** -- adds the blocked tau covariance (TAU_0_PLUS / TAU_2_PLUS,
   with the tau_0<->tau_2 cross-correlation the truncated CosmoCov tau
   covariance carries) and the verbatim RHO_STATS table.

The rho/tau product needs the rho/tau *sidecar* HDUs: the RHO_STATS table carries
per-mode ``varrho_*`` variances the analysis SACC does not store, so the
converter copies them from the sidecar exactly as today's assembly does. A teeth
test pins that a perturbed input moves the output.

The reference builder imports ``cosmosis_fitting.py`` by path (it is a script,
not a package module), skipping cleanly if a dependency is missing -- the same
loader pattern as ``test_cosmosis_fitting.py``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import sacc_io, twopoint_convert

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "cosmo_inference"
    / "scripts"
    / "cosmosis_fitting.py"
)


def _load_cf():
    """Import cosmosis_fitting.py by path; skip cleanly if a dep is missing."""
    if not _SCRIPT.exists():
        pytest.skip(f"cosmosis_fitting.py not found at {_SCRIPT}")
    spec = importlib.util.spec_from_file_location("cosmosis_fitting", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - container has numpy/astropy
        pytest.importorskip(getattr(exc, "name", "") or "cosmosis_fitting_dependency")
        raise
    return module


cf = _load_cf()


# --- deterministic synthetic inputs -----------------------------------------
#
# One tomographic bin (today's analysis). N_ANG angular bins on an ascending
# theta grid; N_ELL bandpowers; a 200-point n(z) on a uniform z grid (the DES
# NZDATA table needs a uniform Z_MID axis). The xi covariance is a full
# (2*N_ANG) matrix (xi+/xi- cross-block nonzero, as CosmoCov produces); the tau
# covariance is a full (3*N_ANG) matrix that the assembly truncates to its first
# 2 statistics (tau_0, tau_2).

N_ANG = 5
N_ELL = 8
SOURCE = sacc_io.source_name(0)
PSF = sacc_io.PSF_TRACER


def _spd(n, seed):
    """A symmetric positive-definite (n, n) matrix, seeded and recognizable."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _inputs(seed=0):
    """Deterministic synthetic statistics + covariances for one bin pair."""
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.uniform(1.0, 250.0, N_ANG))
    ell = np.sort(rng.uniform(30.0, 3000.0, N_ELL))
    z = np.linspace(0.0125, 0.4875, 200)
    return {
        "theta": theta,
        "ell": ell,
        "z": z,
        "nz": np.exp(-((z - 0.25) ** 2) / 0.02),
        "xip": rng.uniform(1e-6, 1e-4, N_ANG),
        "xim": rng.uniform(1e-6, 1e-4, N_ANG),
        "cl_ee": rng.uniform(1e-10, 1e-8, N_ELL),
        "cl_bb": rng.uniform(1e-11, 1e-9, N_ELL),
        "cl_eb": rng.uniform(-1e-11, 1e-11, N_ELL),
        "tau0p": rng.uniform(1e-6, 1e-5, N_ANG),
        "tau2p": rng.uniform(1e-6, 1e-5, N_ANG),
        "tau0m": rng.uniform(1e-6, 1e-5, N_ANG),
        "tau2m": rng.uniform(1e-6, 1e-5, N_ANG),
        "xi_cov": _spd(2 * N_ANG, seed + 1),
        "cl_cov": _spd(N_ELL, seed + 2),
        "tau_cov_full": _spd(3 * N_ANG, seed + 3),
    }


# --- reference sidecar files + HDUs (the current script's own builders) ------


def _rho_sidecar_hdu(theta, seed=7):
    """A rho-stats BinTableHDU with the 25-column layout (values + variances)."""
    rng = np.random.default_rng(seed)
    columns = [fits.Column(name="theta", format="D", array=theta)]
    for k in range(6):
        for suffix in ("_p", "_m"):
            columns.append(
                fits.Column(
                    name=f"rho_{k}{suffix}",
                    format="D",
                    array=rng.uniform(-1e-3, 1e-3, len(theta)),
                )
            )
            columns.append(
                fits.Column(
                    name=f"varrho_{k}{suffix}",
                    format="D",
                    array=rng.uniform(1e-15, 1e-12, len(theta)),
                )
            )
    return fits.BinTableHDU.from_columns(fits.ColDefs(columns))


def _tau_sidecar_hdu(theta, tau0p, tau2p):
    """A tau-stats BinTableHDU with theta + tau_0_p + tau_2_p columns."""
    columns = [
        fits.Column(name="theta", format="D", array=theta),
        fits.Column(name="tau_0_p", format="D", array=tau0p),
        fits.Column(name="tau_2_p", format="D", array=tau2p),
    ]
    return fits.BinTableHDU.from_columns(fits.ColDefs(columns))


def _reference_fits(tmp_path, inp, *, cl=False, rho_tau=False):
    """Build the reference 2pt-FITS with cosmosis_fitting.py's own functions.

    Reproduces the exact ``__main__`` HDU list for the requested configuration,
    which is the assembly the converter must match byte for byte.
    """
    nz_txt = tmp_path / "nz.txt"
    np.savetxt(nz_txt, np.column_stack([inp["z"], inp["nz"]]))
    cov_txt = tmp_path / "cov_xi.txt"
    np.savetxt(cov_txt, inp["xi_cov"])

    nz_hdu = cf.nz_to_fits(str(nz_txt))
    xip_hdu = cf._create_2pt_hdu(inp["xip"], inp["theta"], "XI_PLUS", "G+R", "G+R")
    xim_hdu = cf._create_2pt_hdu(inp["xim"], inp["theta"], "XI_MINUS", "G-R", "G-R")

    hdu_list = [fits.PrimaryHDU(), nz_hdu]

    if rho_tau:
        tau_cov_npy = tmp_path / "cov_tau.npy"
        np.save(tau_cov_npy, inp["tau_cov_full"])
        cov_hdu = cf.covdat_to_fits(str(cov_txt), filename_cov_tau=str(tau_cov_npy))
    else:
        cov_hdu = cf.covdat_to_fits(str(cov_txt), filename_cov_tau=None)
    hdu_list.append(cov_hdu)

    if cl:
        cl_block = np.zeros((5, N_ELL))
        cl_block[0], cl_block[1], cl_block[4] = inp["ell"], inp["cl_ee"], inp["cl_bb"]
        cl_npy = tmp_path / "cl.npy"
        np.save(cl_npy, cl_block)
        cl_cov_npy = tmp_path / "cl_cov.npy"
        np.save(cl_cov_npy, inp["cl_cov"])
        ell_r, cl_ee_r, cl_bb_r = cf.load_pseudo_cl(str(cl_npy))
        cl_ee_hdu, _cl_bb_hdu = cf.cl_to_fits(ell_r, cl_ee_r, cl_bb_r)
        cov_cl_hdu = cf.cov_cl_to_fits(str(cl_cov_npy), cov_hdu="COVAR_FULL")
        hdu_list.append(cov_cl_hdu)

    hdu_list.extend([xip_hdu, xim_hdu])
    if cl:
        hdu_list.append(cl_ee_hdu)

    if rho_tau:
        rho_path = tmp_path / "rho.fits"
        fits.HDUList([fits.PrimaryHDU(), _rho_sidecar_hdu(inp["theta"])]).writeto(
            rho_path, overwrite=True
        )
        tau_path = tmp_path / "tau.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                _tau_sidecar_hdu(inp["theta"], inp["tau0p"], inp["tau2p"]),
            ]
        ).writeto(tau_path, overwrite=True)
        rho_hdu = cf.rho_to_fits(str(rho_path), theta=inp["theta"])
        tau0_hdu, tau2_hdu = cf.tau_to_fits(str(tau_path), theta=inp["theta"])
        hdu_list.extend([tau0_hdu, tau2_hdu, rho_hdu])

    out = tmp_path / "reference.fits"
    fits.HDUList(hdu_list).writeto(out, overwrite=True)
    return out


# --- the SACC each configuration is built from ------------------------------


def _sacc(inp, *, cl=False, rho_tau=False):
    """Build the analysis SACC the converter reads, matching ``_inputs``.

    The covariance is laid out to match the reference exactly: the xi block is
    the full (2*N_ANG) matrix; the Cl block is the EE bandpower covariance; the
    tau blocks carry the tau_0<->tau_2 cross-correlation from the truncated
    CosmoCov tau covariance. Blocks not consumed by the 2pt-FITS (Cl BB/EB, tau
    minus) get an identity block so ``add_covariance`` sees a full matrix.
    """
    s = sacc_io.new_sacc({0: (inp["z"], inp["nz"])})
    sacc_io.add_xi(s, (0, 0), inp["theta"], inp["xip"], inp["xim"], grid="coarse")
    if cl:
        sacc_io.add_pseudo_cl(
            s,
            (0, 0),
            inp["ell"],
            inp["cl_ee"],
            inp["cl_bb"],
            inp["cl_eb"],
            window_ells=np.arange(2, 102),
            window_weights=np.random.default_rng(9).uniform(0, 1, (100, N_ELL)),
        )
    if rho_tau:
        sacc_io.add_tau(s, (0, 0), 0, inp["theta"], inp["tau0p"], inp["tau0m"])
        sacc_io.add_tau(s, (0, 0), 2, inp["theta"], inp["tau2p"], inp["tau2m"])

    n = len(s.mean)
    full = np.zeros((n, n))
    ip = s.indices(sacc_io.XI_PLUS, (SOURCE, SOURCE))
    im = s.indices(sacc_io.XI_MINUS, (SOURCE, SOURCE))
    xi_all = np.concatenate([ip, im])
    full[np.ix_(xi_all, xi_all)] = inp["xi_cov"]

    if cl:
        iee = s.indices(sacc_io.CL_EE, (SOURCE, SOURCE))
        full[np.ix_(iee, iee)] = inp["cl_cov"]
        for dtype in (sacc_io.CL_BB, sacc_io.CL_EB):
            idx = s.indices(dtype, (SOURCE, SOURCE))
            full[np.ix_(idx, idx)] = np.eye(N_ELL)
    if rho_tau:
        t0p = s.indices(sacc_io.TAU_PLUS.format(k=0), (SOURCE, PSF))
        t2p = s.indices(sacc_io.TAU_PLUS.format(k=2), (SOURCE, PSF))
        tau_pp = np.concatenate([t0p, t2p])
        # The joint [tau_0+; tau_2+] block is the truncated CosmoCov tau
        # covariance -- cross-correlation kept, matching covdat_to_fits.
        full[np.ix_(tau_pp, tau_pp)] = inp["tau_cov_full"][: 2 * N_ANG, : 2 * N_ANG]
        for dtype in (sacc_io.TAU_MINUS.format(k=0), sacc_io.TAU_MINUS.format(k=2)):
            idx = s.indices(dtype, (SOURCE, PSF))
            full[np.ix_(idx, idx)] = np.eye(N_ANG)
    s.add_covariance(full)
    return s


def _sidecar_hdus(tmp_path, inp):
    """Return the (rho_hdu, tau_hdu) sidecar input HDUs for the rho/tau product."""
    rho_path = tmp_path / "rho_in.fits"
    fits.HDUList([fits.PrimaryHDU(), _rho_sidecar_hdu(inp["theta"])]).writeto(
        rho_path, overwrite=True
    )
    tau_path = tmp_path / "tau_in.fits"
    fits.HDUList(
        [fits.PrimaryHDU(), _tau_sidecar_hdu(inp["theta"], inp["tau0p"], inp["tau2p"])]
    ).writeto(tau_path, overwrite=True)
    with fits.open(rho_path) as r, fits.open(tau_path) as t:
        return r[1].copy(), t[1].copy()


# =============================================================================
# Byte-compare: the three product shapes
# =============================================================================


def test_plain_xi_byte_equal(tmp_path):
    """Plain-xi product: converter matches cosmosis_fitting.py byte for byte."""
    inp = _inputs(seed=0)
    reference = _reference_fits(tmp_path, inp)
    s = _sacc(inp)
    out = tmp_path / "converted.fits"
    twopoint_convert.sacc_to_twopoint_fits(s, str(out), n_bins=1)
    assert out.read_bytes() == reference.read_bytes()


def test_xi_cl_byte_equal(tmp_path):
    """xi + pseudo-Cl product: COVMAT_CELL + CELL_EE reproduced byte for byte."""
    inp = _inputs(seed=10)
    reference = _reference_fits(tmp_path, inp, cl=True)
    s = _sacc(inp, cl=True)
    out = tmp_path / "converted.fits"
    twopoint_convert.sacc_to_twopoint_fits(s, str(out), n_bins=1)
    assert out.read_bytes() == reference.read_bytes()


def test_xi_rho_tau_byte_equal(tmp_path):
    """xi + rho/tau product: blocked tau covariance + verbatim RHO_STATS match."""
    inp = _inputs(seed=20)
    reference = _reference_fits(tmp_path, inp, rho_tau=True)
    s = _sacc(inp, rho_tau=True)
    rho_hdu, tau_hdu = _sidecar_hdus(tmp_path, inp)
    out = tmp_path / "converted.fits"
    twopoint_convert.sacc_to_twopoint_fits(
        s, str(out), rho_stats_hdu=rho_hdu, tau_stats_hdu=tau_hdu, n_bins=1
    )
    assert out.read_bytes() == reference.read_bytes()


# =============================================================================
# Structural + teeth checks
# =============================================================================


def test_tau_covariance_keeps_tau0_tau2_cross(tmp_path):
    """The tau covariance block couples tau_0 and tau_2 (not block-diagonal).

    covdat_to_fits truncates the 3-statistic CosmoCov tau covariance to its
    first 2 blocks and lays it in as ONE contiguous block, so tau_0<->tau_2
    cross-terms survive. A block-diagonal shortcut would zero them; pin that
    the converter keeps them.
    """
    inp = _inputs(seed=20)
    s = _sacc(inp, rho_tau=True)
    rho_hdu, tau_hdu = _sidecar_hdus(tmp_path, inp)
    out = tmp_path / "converted.fits"
    twopoint_convert.sacc_to_twopoint_fits(
        s, str(out), rho_stats_hdu=rho_hdu, tau_stats_hdu=tau_hdu, n_bins=1
    )
    with fits.open(out) as hdul:
        cov = hdul["COVMAT"].data
    # tau_0 block rows 2*N_ANG..3*N_ANG, tau_2 block 3*N_ANG..4*N_ANG.
    cross = cov[2 * N_ANG : 3 * N_ANG, 3 * N_ANG : 4 * N_ANG]
    expected = inp["tau_cov_full"][:N_ANG, N_ANG : 2 * N_ANG]
    assert np.array_equal(cross, expected)
    assert np.any(cross != 0.0)


def test_perturbed_xi_changes_output(tmp_path):
    """Teeth: a changed xi+ input must change the converted data vector."""
    inp = _inputs(seed=0)
    s = _sacc(inp)
    out = tmp_path / "base.fits"
    twopoint_convert.sacc_to_twopoint_fits(s, str(out), n_bins=1)
    with fits.open(out) as hdul:
        base_xip = hdul["XI_PLUS"].data["VALUE"].copy()

    inp2 = _inputs(seed=0)
    inp2["xip"] = inp2["xip"] + 1.0
    s2 = _sacc(inp2)
    out2 = tmp_path / "perturbed.fits"
    twopoint_convert.sacc_to_twopoint_fits(s2, str(out2), n_bins=1)
    with fits.open(out2) as hdul:
        new_xip = hdul["XI_PLUS"].data["VALUE"]

    assert not np.array_equal(base_xip, new_xip)
    assert np.array_equal(new_xip, inp2["xip"])


def test_rho_tau_sidecars_required_together(tmp_path):
    """Supplying only one of the rho/tau sidecars is a loud error."""
    inp = _inputs(seed=20)
    s = _sacc(inp, rho_tau=True)
    rho_hdu, _tau_hdu = _sidecar_hdus(tmp_path, inp)
    with pytest.raises(ValueError, match="together"):
        twopoint_convert.sacc_to_twopoint_fits(
            s, str(tmp_path / "x.fits"), rho_stats_hdu=rho_hdu, n_bins=1
        )
