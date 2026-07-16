"""Candide-local byte-compare of the converter against real 2pt-FITS products.

Skipped unless a real product exists on disk (candide only; never committed).
It closes the loop end to end on real data: take a real CosmoSIS 2pt-FITS, build
an analysis SACC from its own contents, convert that SACC back to a 2pt-FITS, and
byte-compare.

The reference is *not* the on-disk file directly. The committed on-disk products
were written by an older ``cosmosis_fitting.py`` (they carry a CELL_BB HDU and
order COVMAT before NZ_SOURCE); the converter reproduces the *current* script,
which the PR-4 migration will use to regenerate them. So the meaningful contract
-- "the converter equals the current writer" -- is tested by running the current
``cosmosis_fitting.py`` builders on the same real contents and byte-comparing the
converter against that. For transparency the test also records the direct diff
against the stale on-disk file, and asserts only that it differs *by whole HDUs*
(the extra CELL_BB), not in any shared block -- i.e. the drift is purely the
known HDU-set change, with no silent data corruption.

Observed on 2026-07-10 for ``SP_v1.4.6_leak_corr`` and a ``glass_mock`` sibling:
converter == current-script byte for byte; converter vs on-disk differs only by
the CELL_BB HDU.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import sacc_io, twopoint_convert

_DATA = Path("/automnt/n17data/cdaley/unions/code/sp_validation/cosmo_inference/data")
_REAL_FILES = {
    "SP_v1.4.6_leak_corr": _DATA
    / "SP_v1.4.6_leak_corr_A_minsep=1.0_maxsep=250.0_nbins=20_npatch=1"
    / "cosmosis_SP_v1.4.6_leak_corr_A_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.fits",
    "glass_mock_00001": _DATA / "glass_mock_00001" / "cosmosis_glass_mock_00001.fits",
}

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "cosmo_inference"
    / "scripts"
    / "cosmosis_fitting.py"
)


def _load_cf():
    spec = importlib.util.spec_from_file_location("cosmosis_fitting", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sacc_from_2pt_fits(hdul):
    """Build an analysis SACC from a real CosmoSIS 2pt-FITS's own contents.

    Reads the NZ, ξ±, pseudo-Cℓ (EE/BB) and blocked covariance back out of the
    product and lays them into the standard SACC layout — the inverse direction
    the converter then undoes. τ± minus and Cℓ EB are not in the 2pt-FITS, so
    they are stored as zeros with identity covariance sub-blocks (the converter
    consumes only the ``+``/EE parts).
    """
    z = hdul["NZ_SOURCE"].data["Z_MID"].astype(float)
    nz = hdul["NZ_SOURCE"].data["BIN1"].astype(float)
    theta = hdul["XI_PLUS"].data["ANG"].astype(float)
    xip = hdul["XI_PLUS"].data["VALUE"].astype(float)
    xim = hdul["XI_MINUS"].data["VALUE"].astype(float)
    n = len(theta)

    ell = hdul["CELL_EE"].data["ANG"].astype(float)
    cl_ee = hdul["CELL_EE"].data["VALUE"].astype(float)
    cl_bb = hdul["CELL_BB"].data["VALUE"].astype(float)
    n_ell = len(ell)

    covmat = hdul["COVMAT"].data.astype(float)
    covmat_cell = hdul["COVMAT_CELL"].data.astype(float)
    xi_cov = covmat[: 2 * n, : 2 * n]
    tau_joint = covmat[2 * n : 4 * n, 2 * n : 4 * n]

    tau0p = hdul["TAU_0_PLUS"].data["VALUE"].astype(float)
    tau2p = hdul["TAU_2_PLUS"].data["VALUE"].astype(float)

    s = sacc_io.new_sacc({0: (z, nz)})
    sacc_io.add_xi(s, (0, 0), theta, xip, xim, grid="reporting")
    sacc_io.add_pseudo_cl(
        s,
        (0, 0),
        ell,
        cl_ee,
        cl_bb,
        np.zeros(n_ell),
        window_ells=np.arange(2, 102),
        window_weights=np.ones((100, n_ell)),
    )
    sacc_io.add_tau(s, (0, 0), 0, theta, tau0p, np.zeros(n))
    sacc_io.add_tau(s, (0, 0), 2, theta, tau2p, np.zeros(n))

    source, psf = sacc_io.source_name(0), sacc_io.PSF_TRACER
    idx = {
        "xi": np.concatenate(
            [
                s.indices(sacc_io.XI_PLUS, (source, source)),
                s.indices(sacc_io.XI_MINUS, (source, source)),
            ]
        ),
        "ee": s.indices(sacc_io.CL_EE, (source, source)),
        "bb": s.indices(sacc_io.CL_BB, (source, source)),
        "eb": s.indices(sacc_io.CL_EB, (source, source)),
        "t0p": s.indices(sacc_io.TAU_PLUS.format(k=0), (source, psf)),
        "t0m": s.indices(sacc_io.TAU_MINUS.format(k=0), (source, psf)),
        "t2p": s.indices(sacc_io.TAU_PLUS.format(k=2), (source, psf)),
        "t2m": s.indices(sacc_io.TAU_MINUS.format(k=2), (source, psf)),
    }
    full = np.zeros((len(s.mean), len(s.mean)))
    full[np.ix_(idx["xi"], idx["xi"])] = xi_cov
    full[np.ix_(idx["ee"], idx["ee"])] = covmat_cell
    tau_pp = np.concatenate([idx["t0p"], idx["t2p"]])
    full[np.ix_(tau_pp, tau_pp)] = tau_joint
    for key in ("bb", "eb"):
        full[np.ix_(idx[key], idx[key])] = np.eye(n_ell)
    for key in ("t0m", "t2m"):
        full[np.ix_(idx[key], idx[key])] = np.eye(n)
    s.add_covariance(full)

    tau_sidecar = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(name="theta", format="D", array=theta),
                fits.Column(name="tau_0_p", format="D", array=tau0p),
                fits.Column(name="tau_2_p", format="D", array=tau2p),
            ]
        )
    )
    return s, hdul["RHO_STATS"].copy(), tau_sidecar


def _current_script_reference(cf, hdul, tmp_path):
    """Reference: run the current cosmosis_fitting.py on the real file's contents."""
    z = hdul["NZ_SOURCE"].data["Z_MID"].astype(float)
    nz = hdul["NZ_SOURCE"].data["BIN1"].astype(float)
    theta = hdul["XI_PLUS"].data["ANG"].astype(float)
    xip = hdul["XI_PLUS"].data["VALUE"].astype(float)
    xim = hdul["XI_MINUS"].data["VALUE"].astype(float)
    n = len(theta)
    ell = hdul["CELL_EE"].data["ANG"].astype(float)
    cl_ee = hdul["CELL_EE"].data["VALUE"].astype(float)
    cl_bb = hdul["CELL_BB"].data["VALUE"].astype(float)
    covmat = hdul["COVMAT"].data.astype(float)
    covmat_cell = hdul["COVMAT_CELL"].data.astype(float)
    tau0p = hdul["TAU_0_PLUS"].data["VALUE"].astype(float)
    tau2p = hdul["TAU_2_PLUS"].data["VALUE"].astype(float)

    np.savetxt(tmp_path / "nz.txt", np.column_stack([z, nz]))
    np.savetxt(tmp_path / "cov.txt", covmat[: 2 * n, : 2 * n])
    tau_cov = np.zeros((3 * n, 3 * n))
    tau_cov[: 2 * n, : 2 * n] = covmat[2 * n : 4 * n, 2 * n : 4 * n]
    np.save(tmp_path / "cov_tau.npy", tau_cov)
    cl_block = np.zeros((5, len(ell)))
    cl_block[0], cl_block[1], cl_block[4] = ell, cl_ee, cl_bb
    np.save(tmp_path / "cl.npy", cl_block)
    np.save(tmp_path / "cl_cov.npy", covmat_cell)

    nz_hdu = cf.nz_to_fits(str(tmp_path / "nz.txt"))
    xip_hdu = cf._create_2pt_hdu(xip, theta, "XI_PLUS", "G+R", "G+R")
    xim_hdu = cf._create_2pt_hdu(xim, theta, "XI_MINUS", "G-R", "G-R")
    cov_hdu = cf.covdat_to_fits(
        str(tmp_path / "cov.txt"), filename_cov_tau=str(tmp_path / "cov_tau.npy")
    )
    ell_r, cl_ee_r, cl_bb_r = cf.load_pseudo_cl(str(tmp_path / "cl.npy"))
    cl_ee_hdu, _ = cf.cl_to_fits(ell_r, cl_ee_r, cl_bb_r)
    cov_cl_hdu = cf.cov_cl_to_fits(str(tmp_path / "cl_cov.npy"), cov_hdu="COVAR_FULL")

    fits.HDUList([fits.PrimaryHDU(), hdul["RHO_STATS"].copy()]).writeto(
        tmp_path / "rho.fits", overwrite=True
    )
    rho_hdu = cf.rho_to_fits(str(tmp_path / "rho.fits"), theta=theta)
    tau_sidecar = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(name="theta", format="D", array=theta),
                fits.Column(name="tau_0_p", format="D", array=tau0p),
                fits.Column(name="tau_2_p", format="D", array=tau2p),
            ]
        )
    )
    fits.HDUList([fits.PrimaryHDU(), tau_sidecar]).writeto(
        tmp_path / "tau.fits", overwrite=True
    )
    tau0_hdu, tau2_hdu = cf.tau_to_fits(str(tmp_path / "tau.fits"), theta=theta)

    out = tmp_path / "reference.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            nz_hdu,
            cov_hdu,
            cov_cl_hdu,
            xip_hdu,
            xim_hdu,
            cl_ee_hdu,
            tau0_hdu,
            tau2_hdu,
            rho_hdu,
        ]
    ).writeto(out, overwrite=True)
    return out


@pytest.mark.parametrize("label", list(_REAL_FILES))
def test_realdata_roundtrip_byte_equal(label, tmp_path):
    """Converter reproduces the current writer byte for byte on a real product."""
    real = _REAL_FILES[label]
    if not real.exists():
        pytest.skip(f"real 2pt-FITS not on disk: {real}")
    cf = _load_cf()

    with fits.open(real) as hdul:
        s, rho_hdu, tau_hdu = _sacc_from_2pt_fits(hdul)
        reference = _current_script_reference(cf, hdul, tmp_path)

    converted = tmp_path / "converted.fits"
    twopoint_convert.sacc_to_twopoint_fits(
        s, str(converted), rho_stats_hdu=rho_hdu, tau_stats_hdu=tau_hdu, n_bins=1
    )

    # The contract: converter == current cosmosis_fitting.py, byte for byte.
    assert converted.read_bytes() == reference.read_bytes()


@pytest.mark.parametrize("label", list(_REAL_FILES))
def test_realdata_ondisk_drift_is_only_cell_bb(label, tmp_path):
    """The stale on-disk file differs from the converter *only* by the CELL_BB HDU.

    Documents (and guards) the known script-version drift: the on-disk products
    were written before CELL_BB was dropped from the assembly, so they carry one
    extra HDU. Every HDU the two share carries the same data to floating-point
    precision — the drift is a whole-HDU addition, never a silent change to a
    shared block. (Bin-edge columns differ by ~1e-17 float noise: the stale file
    stored a clean ``Z_LOW=0``, while ``z_mid - step/2`` rounds to ``-1.7e-18``;
    both are the same number, so the shared-data check is ``allclose``, not
    bitwise — bitwise equality is asserted against the *current* writer above.)
    """
    real = _REAL_FILES[label]
    if not real.exists():
        pytest.skip(f"real 2pt-FITS not on disk: {real}")

    with fits.open(real) as hdul:
        s, rho_hdu, tau_hdu = _sacc_from_2pt_fits(hdul)
        ondisk_names = [h.name for h in hdul]
        converted = tmp_path / "converted.fits"
        twopoint_convert.sacc_to_twopoint_fits(
            s, str(converted), rho_stats_hdu=rho_hdu, tau_stats_hdu=tau_hdu, n_bins=1
        )
        with fits.open(converted) as conv:
            conv_names = [h.name for h in conv]
            # The only HDU the on-disk file has that the converter does not.
            assert set(ondisk_names) - set(conv_names) == {"CELL_BB"}
            # Every shared table HDU carries the same data to float precision.
            for name in conv_names:
                if name == "PRIMARY":
                    continue
                a, b = hdul[name].data, conv[name].data
                if hasattr(a, "names"):
                    assert all(np.allclose(a[c], b[c]) for c in a.names), name
                else:
                    assert np.allclose(a, b), name
