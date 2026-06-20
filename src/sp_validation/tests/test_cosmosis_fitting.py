"""Integration tests for the CosmoSIS data-FITS assembly.

This pins the riskiest, least-tested step in the inference data-product path:
``cosmo_inference/scripts/cosmosis_fitting.py`` turning UNIONS validation
outputs (xi+/-, n(z), a blocked covariance, and -- with ``--use-rho-tau`` --
rho/tau statistics and their covariance) into the single CosmoSIS data FITS that
a chain consumes. We stop short of the chain itself: no cosmosis, no MCMC.

Two danger surfaces drive the design:

1. **The HDU set + ordering.** The assembled ``HDUList`` is built by hand in the
   script's ``__main__`` block (not a reusable function), so the only way it can
   silently change shape is for nobody to be watching it. The CLI tests below
   watch it: they assert the exact extension-name set and the data-vector
   ordering (xi+ then xi-), for both the plain-xi product and the rho/tau one.

2. **The blocked covariance offsets.** ``covdat_to_fits`` lays the xi+ block,
   the xi- block, and -- under ``--use-rho-tau`` -- the tau_0/tau_2 blocks onto
   a single matrix, recording each block's start index in the FITS header
   (``STRT_0..STRT_3``). The tau covariance is *truncated* from 3*nbins to
   2*nbins before blocking; a wrong offset or a wrong truncation lands the data
   on the wrong rows and is invisible until a chain quietly mis-fits. These are
   pinned at the function level with deterministic synthetic inputs whose offsets
   we work out by hand, plus teeth that flip red on a perturbed input or a
   wrong-offset expectation.

The script imports only numpy + astropy.io.fits (no cosmosis), so the whole
assembly path is exercisable in the production container. The end-to-end CLI run
also triggers ``generate_cosmosis_config``; we feed it minimal synthetic template
INIs in ``tmp_path`` so the run completes without touching the repo's real
templates or any cluster data.

:Author: cdaley

"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

# --- load cosmosis_fitting.py as a module (it is a script, not a package) ----

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "cosmo_inference"
    / "scripts"
    / "cosmosis_fitting.py"
)


def _load_module():
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


cf = _load_module()


# --- deterministic synthetic dimensions --------------------------------------
#
# N_ANG angular bins. The xi data vector is xi+ (N_ANG) then xi- (N_ANG); the xi
# covariance is therefore (2*N_ANG) square. The tau covariance is stored as
# 3 statistics x N_ANG bins = (3*N_ANG) square, and covdat_to_fits keeps only
# the first 2*N_ANG rows/cols (tau_0, tau_2) before blocking it onto the xi
# covariance. With N_ANG=3 every offset is small enough to verify by eye.

N_ANG = 3


def _xi_cov_matrix():
    """A (2*N_ANG) covariance with unique, recognizable entries."""
    n = 2 * N_ANG
    return np.arange(1, n * n + 1, dtype=float).reshape(n, n)


def _tau_cov_full():
    """A (3*N_ANG) tau covariance, offset by 1000 so its block is identifiable."""
    n = 3 * N_ANG
    return np.arange(1, n * n + 1, dtype=float).reshape(n, n) + 1000.0


def _write_xi_cov(path):
    np.savetxt(path, _xi_cov_matrix())
    return path


def _write_tau_cov(path):
    np.save(path, _tau_cov_full())
    return path


# =============================================================================
# Function-level: the blocked covariance (covdat_to_fits) -- the danger
# =============================================================================


def test_cov_xi_only_offsets(tmp_path):
    """xi-only covariance: shape (2*N_ANG), XI_MINUS block starts at N_ANG.

    With only xi the covariance is passed through verbatim and the header marks
    the xi+ block at row 0 and the xi- block at row N_ANG (= half the matrix).
    """
    cov_file = _write_xi_cov(tmp_path / "cov_xi.txt")
    hdu = cf.covdat_to_fits(str(cov_file), filename_cov_tau=None)

    assert hdu.data.shape == (2 * N_ANG, 2 * N_ANG)
    assert hdu.header["EXTNAME"] == "COVMAT"
    assert hdu.header["NAME_0"] == "XI_PLUS"
    assert hdu.header["STRT_0"] == 0
    assert hdu.header["NAME_1"] == "XI_MINUS"
    assert hdu.header["STRT_1"] == N_ANG
    # No tau augmentation when filename_cov_tau is None.
    assert "NAME_2" not in hdu.header
    # The matrix is the input, unmodified.
    assert np.array_equal(hdu.data, _xi_cov_matrix())


def test_cov_with_tau_block_layout(tmp_path):
    """xi+tau covariance: the four blocks land at the offsets we predict by hand.

    Layout for N_ANG=3:
        rows  0..2  -> xi+   (STRT_0 = 0)
        rows  3..5  -> xi-   (STRT_1 = 3)
        rows  6..8  -> tau_0 (STRT_2 = 6  = len(covmat_xi))
        rows  9..11 -> tau_2 (STRT_3 = 9  = len(covmat_xi) + len(covmat_tau)/2)
    The tau covariance is truncated from 3*N_ANG to 2*N_ANG before blocking, and
    the cross blocks between xi and tau are zero.
    """
    cov_xi = _write_xi_cov(tmp_path / "cov_xi.txt")
    cov_tau = _write_tau_cov(tmp_path / "cov_tau.npy")

    hdu = cf.covdat_to_fits(str(cov_xi), filename_cov_tau=str(cov_tau))

    # Combined size: xi (2*N_ANG) + truncated tau (2*N_ANG).
    assert hdu.data.shape == (4 * N_ANG, 4 * N_ANG)

    # Block-start offsets, pinned to concrete integers.
    assert hdu.header["STRT_0"] == 0
    assert hdu.header["STRT_1"] == N_ANG
    assert hdu.header["STRT_2"] == 2 * N_ANG
    assert hdu.header["STRT_3"] == 3 * N_ANG
    assert hdu.header["NAME_0"] == "XI_PLUS"
    assert hdu.header["NAME_1"] == "XI_MINUS"
    assert hdu.header["NAME_2"] == "TAU_0_PLUS"
    assert hdu.header["NAME_3"] == "TAU_2_PLUS"

    cov = hdu.data
    xi = _xi_cov_matrix()
    tau_trunc = _tau_cov_full()[: 2 * N_ANG, : 2 * N_ANG]

    # xi block sits top-left; tau block sits bottom-right (truncated).
    assert np.array_equal(cov[: 2 * N_ANG, : 2 * N_ANG], xi)
    assert np.array_equal(cov[2 * N_ANG :, 2 * N_ANG :], tau_trunc)
    # The tau block is the *truncated* matrix: its [0,0] is the full matrix's
    # [0,0], and the dropped tau_5 rows (>= 2*N_ANG) never appear.
    assert cov[2 * N_ANG, 2 * N_ANG] == _tau_cov_full()[0, 0]
    assert _tau_cov_full()[2 * N_ANG, 2 * N_ANG] not in cov

    # Cross blocks between xi and tau are exactly zero.
    assert np.all(cov[: 2 * N_ANG, 2 * N_ANG :] == 0.0)
    assert np.all(cov[2 * N_ANG :, : 2 * N_ANG] == 0.0)


def test_cov_with_tau_offsets_have_teeth(tmp_path):
    """Teeth: the tau block does NOT live at the xi- offset.

    A common wrong expectation is that the tau block starts where xi- starts
    (N_ANG) rather than after all of xi (2*N_ANG). Pin that the truncated tau
    block is genuinely at 2*N_ANG, so an off-by-a-block error would be caught.
    """
    cov_xi = _write_xi_cov(tmp_path / "cov_xi.txt")
    cov_tau = _write_tau_cov(tmp_path / "cov_tau.npy")
    hdu = cf.covdat_to_fits(str(cov_xi), filename_cov_tau=str(cov_tau))

    tau_trunc = _tau_cov_full()[: 2 * N_ANG, : 2 * N_ANG]
    # Correct offset reproduces the tau block...
    assert np.array_equal(hdu.data[2 * N_ANG :, 2 * N_ANG :], tau_trunc)
    # ...and a wrong (xi-) offset does not.
    wrong = hdu.data[N_ANG : N_ANG + 2 * N_ANG, N_ANG : N_ANG + 2 * N_ANG]
    assert not np.array_equal(wrong, tau_trunc)


def test_cov_non_square_raises(tmp_path):
    """A non-square xi covariance must fail loudly, not assemble silently."""
    bad = tmp_path / "cov_bad.txt"
    np.savetxt(bad, np.arange(6, dtype=float).reshape(2, 3))
    with pytest.raises(Exception):
        cf.covdat_to_fits(str(bad), filename_cov_tau=None)


# =============================================================================
# Function-level: the 2pt and n(z) HDUs feeding the data vector
# =============================================================================


def test_2pt_hdu_columns_and_header():
    """A 2pt HDU carries VALUE/ANG and the CosmoSIS 2PTDATA header marker."""
    values = np.array([1.0, 2.0, 3.0])
    theta = np.array([5.0, 50.0, 500.0])
    hdu = cf._create_2pt_hdu(values, theta, "XI_PLUS", "G+R", "G+R")

    assert hdu.name == "XI_PLUS"
    assert set(["BIN1", "BIN2", "ANGBIN", "VALUE", "ANG"]).issubset(set(hdu.data.names))
    assert np.array_equal(hdu.data["VALUE"], values)
    assert np.array_equal(hdu.data["ANG"], theta)
    assert np.array_equal(hdu.data["ANGBIN"], np.arange(1, len(values) + 1))
    assert hdu.header["2PTDATA"] == "T"
    assert hdu.header["QUANT1"] == "G+R"


def test_nz_to_fits(tmp_path):
    """n(z) text -> NZDATA HDU named NZ_SOURCE with correct bin edges/count."""
    z_mid = np.array([0.1, 0.3, 0.5])
    bin1 = np.array([0.2, 0.5, 0.3])
    nz_path = tmp_path / "nz.txt"
    np.savetxt(nz_path, np.column_stack([z_mid, bin1]))

    hdu = cf.nz_to_fits(str(nz_path))

    assert hdu.name == "NZ_SOURCE"
    assert hdu.header["NZDATA"].strip() == "T"
    assert hdu.header["NBIN"] == 1
    assert hdu.header["NZ"] == len(z_mid)
    assert np.array_equal(hdu.data["Z_MID"], z_mid)
    # Edges are midpoints +/- half the (uniform) step.
    step = z_mid[1] - z_mid[0]
    assert np.allclose(hdu.data["Z_LOW"], z_mid - step / 2)
    assert np.allclose(hdu.data["Z_HIGH"], z_mid + step / 2)
    assert np.array_equal(hdu.data["BIN1"], bin1)


# =============================================================================
# End-to-end CLI: the assembled HDU set + ordering (the __main__ assembly)
# =============================================================================
#
# The full HDUList is built in __main__, reachable only through the CLI. We run
# the script as a subprocess against synthetic inputs in tmp_path -- including
# minimal template INIs so generate_cosmosis_config completes -- and inspect the
# written FITS. Nothing here touches the repo's real templates or cluster data.

# Minimal template containing exactly the section headers _generate_ini_file's
# regexes rewrite. Anything else would just be passed through untouched.
_TEMPLATE_INI = "\n".join(
    [
        "[DEFAULT]",
        "[runtime]",
        "[output]",
        "[pipeline]",
        "[2pt_like]",
        "covmat_name=PLACEHOLDER",
        "[test]",
        "[polychord]",
        "",
    ]
)

# Theta grid shared by xi (and forced onto rho/tau for consistency).
_THETA = np.array([5.0, 50.0, 500.0])
_XIP = np.array([10.0, 11.0, 12.0])  # xi+ data vector
_XIM = np.array([20.0, 21.0, 22.0])  # xi- data vector


def _write_treecorr_xi(path, values, extname):
    """A TreeCorr-style xi FITS: data in HDU 1 with ANG + VALUE columns.

    Data mode's ``treecorr_to_fits`` returns ``hdul[1]`` verbatim, so the
    extension name carried by the real TreeCorr file is what lands in the
    assembled product. CosmoSIS looks the blocks up by name
    (``data_sets=XI_PLUS XI_MINUS``), so the inputs must carry those EXTNAMEs;
    we set them here to be faithful to the real files.
    """
    cols = fits.ColDefs(
        [
            fits.Column(name="ANG", format="D", unit="arcmin", array=_THETA),
            fits.Column(name="VALUE", format="D", array=values),
            fits.Column(name="meanr", format="D", array=_THETA),
            fits.Column(name="xip", format="D", array=values),
            fits.Column(name="xim", format="D", array=values),
        ]
    )
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols, name=extname)]
    )
    hdul.writeto(path, overwrite=True)
    return path


def _write_rho_stats(path):
    """rho-stats FITS: HDU 1 carries a theta column (values irrelevant here)."""
    cols = fits.ColDefs(
        [
            fits.Column(name="theta", format="D", array=_THETA),
            fits.Column(name="rho_0", format="D", array=np.ones(N_ANG)),
        ]
    )
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)])
    hdul.writeto(path, overwrite=True)
    return path


def _write_tau_stats(path):
    """tau-stats FITS: theta + tau_0_p/tau_2_p columns (read by tau_to_fits)."""
    cols = fits.ColDefs(
        [
            fits.Column(name="theta", format="D", array=_THETA),
            fits.Column(name="tau_0_p", format="D", array=np.array([1.0, 2.0, 3.0])),
            fits.Column(name="tau_2_p", format="D", array=np.array([4.0, 5.0, 6.0])),
        ]
    )
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)])
    hdul.writeto(path, overwrite=True)
    return path


def _write_nz(path):
    z_mid = np.array([0.1, 0.3, 0.5])
    np.savetxt(path, np.column_stack([z_mid, np.array([0.2, 0.5, 0.3])]))
    return path


def _run_cli(tmp_path, *, use_rho_tau):
    """Run cosmosis_fitting.py against synthetic tmp_path inputs; return FITS path.

    Builds a self-contained output root (with a cosmosis_config holding the
    minimal template + value/prior INIs config generation expects) so the run
    completes end-to-end. Returns the path to the assembled data FITS.
    """
    out_root = tmp_path / "out_root"
    template_dir = out_root / "cosmosis_config"
    template_dir.mkdir(parents=True)

    # Templates + the value/prior INIs generate_cosmosis_config reads/writes.
    for name in (
        "cosmosis_pipeline_A_ia.ini",
        "cosmosis_pipeline_A_psf.ini",
    ):
        (template_dir / name).write_text(_TEMPLATE_INI)
    for name in ("values_ia.ini", "values_psf.ini", "priors.ini", "priors_psf.ini"):
        (template_dir / name).write_text("[parameters]\n")

    nz = _write_nz(tmp_path / "nz.txt")
    xip = _write_treecorr_xi(tmp_path / "xi_plus.fits", _XIP, "XI_PLUS")
    xim = _write_treecorr_xi(tmp_path / "xi_minus.fits", _XIM, "XI_MINUS")
    cov_xi = _write_xi_cov(tmp_path / "cov_xi.txt")

    root = "TESTROOT"
    cmd = [
        sys.executable,
        str(_SCRIPT),
        "--cosmosis-root",
        root,
        "--data-dir",
        str(tmp_path / "chains"),
        "--nz-file",
        str(nz),
        "--output-root",
        str(out_root),
        "--template-dir",
        str(template_dir),
        "--xi",
        str(xip),
        str(xim),
        "--cov-xi",
        str(cov_xi),
    ]
    if use_rho_tau:
        cmd += [
            "--use-rho-tau",
            "--rho-stats",
            str(_write_rho_stats(tmp_path / "rho.fits")),
            "--tau-stats",
            str(_write_tau_stats(tmp_path / "tau.fits")),
            "--cov-tau",
            str(_write_tau_cov(tmp_path / "cov_tau.npy")),
        ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"cosmosis_fitting.py failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )

    out_fits = out_root / "data" / root / f"cosmosis_{root}.fits"
    assert out_fits.exists(), f"expected FITS not written: {out_fits}"
    return out_fits


def test_cli_plain_xi_hdu_set_and_ordering(tmp_path):
    """Plain-xi product: exact HDU set, xi+ -> xi- data ordering, cov shape."""
    out_fits = _run_cli(tmp_path, use_rho_tau=False)

    with fits.open(out_fits) as hdul:
        names = [h.name for h in hdul]
        # Order from __main__: PRIMARY, NZ_SOURCE, COVMAT, XI_PLUS, XI_MINUS.
        assert names == ["PRIMARY", "NZ_SOURCE", "COVMAT", "XI_PLUS", "XI_MINUS"]

        xip = hdul["XI_PLUS"].data["VALUE"]
        xim = hdul["XI_MINUS"].data["VALUE"]
        assert np.array_equal(xip, _XIP)
        assert np.array_equal(xim, _XIM)

        # The data vector is xi+ THEN xi- -- ordering is load-bearing for the
        # covariance blocks below.
        data_vector = np.concatenate([xip, xim])
        assert np.array_equal(data_vector, np.concatenate([_XIP, _XIM]))
        assert len(data_vector) == 2 * N_ANG

        cov = hdul["COVMAT"].data
        assert cov.shape == (2 * N_ANG, 2 * N_ANG)
        assert hdul["COVMAT"].header["STRT_1"] == N_ANG


def test_cli_rho_tau_hdu_set_and_cov_block(tmp_path):
    """--use-rho-tau product: rho/tau HDUs appended, covariance augmented.

    The HDU set gains TAU_0_PLUS, TAU_2_PLUS, RHO_STATS, and the COVMAT grows to
    (4*N_ANG) with the tau blocks at offsets 2*N_ANG and 3*N_ANG.
    """
    out_fits = _run_cli(tmp_path, use_rho_tau=True)

    with fits.open(out_fits) as hdul:
        names = [h.name for h in hdul]
        # Order from __main__: PRIMARY, NZ_SOURCE, COVMAT, XI_PLUS, XI_MINUS,
        # then TAU_0_PLUS, TAU_2_PLUS, RHO_STATS.
        assert names == [
            "PRIMARY",
            "NZ_SOURCE",
            "COVMAT",
            "XI_PLUS",
            "XI_MINUS",
            "TAU_0_PLUS",
            "TAU_2_PLUS",
            "RHO_STATS",
        ]

        # xi data vector unchanged and still xi+ then xi-.
        assert np.array_equal(hdul["XI_PLUS"].data["VALUE"], _XIP)
        assert np.array_equal(hdul["XI_MINUS"].data["VALUE"], _XIM)

        # Covariance now carries the augmented block layout.
        cov_hdu = hdul["COVMAT"]
        assert cov_hdu.data.shape == (4 * N_ANG, 4 * N_ANG)
        assert cov_hdu.header["STRT_0"] == 0
        assert cov_hdu.header["STRT_1"] == N_ANG
        assert cov_hdu.header["STRT_2"] == 2 * N_ANG
        assert cov_hdu.header["STRT_3"] == 3 * N_ANG
        assert cov_hdu.header["NAME_2"] == "TAU_0_PLUS"
        assert cov_hdu.header["NAME_3"] == "TAU_2_PLUS"

        # The tau block is the truncated tau covariance at the bottom-right.
        tau_trunc = _tau_cov_full()[: 2 * N_ANG, : 2 * N_ANG]
        assert np.array_equal(cov_hdu.data[2 * N_ANG :, 2 * N_ANG :], tau_trunc)

        # rho/tau theta is forced onto the xi theta grid (consistency step).
        assert np.allclose(hdul["RHO_STATS"].data["theta"], _THETA)
        assert np.allclose(hdul["TAU_0_PLUS"].data["ANG"], _THETA)


def test_cli_perturbed_xi_changes_data_vector(tmp_path):
    """Teeth: a changed xi+ input must change the written data vector.

    If the assembly silently dropped or reordered the xi+ values, this would not
    move -- so a real input perturbation flowing through to the FITS is the
    end-to-end guarantee.
    """
    baseline = _run_cli(tmp_path, use_rho_tau=False)
    with fits.open(baseline) as hdul:
        base_xip = hdul["XI_PLUS"].data["VALUE"].copy()

    # Perturb xi+ and rerun in a fresh output area.
    tmp2 = tmp_path / "perturbed"
    tmp2.mkdir()
    perturbed_xip = _XIP + 100.0
    # Reuse the runner but with a different xi+ vector by monkey-free rewrite.
    out_root = tmp2 / "out_root"
    template_dir = out_root / "cosmosis_config"
    template_dir.mkdir(parents=True)
    (template_dir / "cosmosis_pipeline_A_ia.ini").write_text(_TEMPLATE_INI)
    (template_dir / "values_ia.ini").write_text("[parameters]\n")
    (template_dir / "priors.ini").write_text("[parameters]\n")

    nz = _write_nz(tmp2 / "nz.txt")
    xip = _write_treecorr_xi(tmp2 / "xi_plus.fits", perturbed_xip, "XI_PLUS")
    xim = _write_treecorr_xi(tmp2 / "xi_minus.fits", _XIM, "XI_MINUS")
    cov_xi = _write_xi_cov(tmp2 / "cov_xi.txt")
    root = "TESTROOT2"
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--cosmosis-root",
            root,
            "--data-dir",
            str(tmp2 / "chains"),
            "--nz-file",
            str(nz),
            "--output-root",
            str(out_root),
            "--template-dir",
            str(template_dir),
            "--xi",
            str(xip),
            str(xim),
            "--cov-xi",
            str(cov_xi),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out_fits = out_root / "data" / root / f"cosmosis_{root}.fits"
    with fits.open(out_fits) as hdul:
        new_xip = hdul["XI_PLUS"].data["VALUE"]

    assert not np.array_equal(base_xip, new_xip)
    assert np.array_equal(new_xip, perturbed_xip)
