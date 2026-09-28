"""Candide-local regression of the calibration path against the v1.4.6.3 release.

Runs ``scripts/calibration/calibrate_comprehensive_cat.py`` with the release's
mask config (``config/calibration/mask_v1.X.6.yaml``) on a row window of the
v1.4.c comprehensive HDF5, and checks that it selects exactly the objects the
released cut catalogue holds from that window, in the same order, with the same
per-object values. The release preserves input order, so the window's objects
form one contiguous run of released rows, found by (RA, Dec).

Globally calibrated columns (``e1``, ``e2``, ``w_des``, the leakage-corrected
ellipticities) depend on the whole-survey response and weights, not on the
window, and are not compared. Everything the release derives object by object
is, including the no-shear reconvolved-PSF size, which the release took from
``NGMIX_Tpsf_1P`` (``sp_validation.grammar``).

Skipped where the release products are absent.
"""

import os
import runpy
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml
from astropy.io import fits

WL = Path("/n17data/UNIONS/WL/v1.4.x")
COMPREHENSIVE = WL / "unions_shapepipe_comprehensive_struc_2024_v1.4.c.hdf5"
RELEASE = WL / "v1.4.6.3" / "unions_shapepipe_cut_struc_2024_v1.4.6.3.fits"
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "calibration" / "calibrate_comprehensive_cat.py"
CONFIG = REPO / "config" / "calibration" / "mask_v1.X.6.yaml"

START, N_ROWS = 200_000_000, 1_000_000

#: Per-object output columns and their release counterparts.
PER_OBJECT = {
    name: name
    for name in (
        "RA",
        "Dec",
        "mag",
        "snr",
        "e1_uncal",
        "e2_uncal",
        "w_iv",
        "FLUX_RADIUS",
        "FWHM_IMAGE",
        "FWHM_WORLD",
        "MAGERR_AUTO",
        "MAG_WIN",
        "MAGERR_WIN",
        "FLUX_AUTO",
        "FLUXERR_AUTO",
        "FLUX_APER",
        "FLUXERR_APER",
        "NGMIX_T_NOSHEAR",
        "fwhm_PSF",
        "R_g11",
        "R_g12",
        "R_g21",
        "R_g22",
        "e1_PSF",
        "e2_PSF",
    )
} | {"NGMIX_T_PSF_RECONV_NOSHEAR": "NGMIX_Tpsf_NOSHEAR"}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (COMPREHENSIVE.exists() and RELEASE.exists()),
        reason="v1.4.c comprehensive catalogue or v1.4.6.3 release absent",
    ),
]


def _window_file(path):
    """Write rows START:START+N_ROWS of the comprehensive HDF5 to ``path``.

    Returns the window's (RA, Dec) as complex keys.
    """
    with h5py.File(COMPREHENSIVE, "r") as src, h5py.File(path, "w") as dst:
        for name in ("data", "data_ext"):
            window = src[name][START : START + N_ROWS]
            dst.create_dataset(name, data=window)
            if name == "data":
                keys = window["RA"] + 1j * window["Dec"]
    return keys


def _run_calibration(run_dir):
    """Run the calibration script in ``run_dir``.

    Returns its cut catalogue and the input window's (RA, Dec) keys.
    """
    comprehensive = run_dir / "unions_shapepipe_comprehensive_window.hdf5"
    window_keys = _window_file(comprehensive)
    config = yaml.safe_load(CONFIG.read_text())
    config["params"]["input_path"] = str(comprehensive)
    (run_dir / "config_mask.yaml").write_text(yaml.safe_dump(config))

    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exit:
        assert exit.code in (0, None), f"calibration script exited with {exit.code}"
    finally:
        os.chdir(cwd)
    return fits.getdata(run_dir / "unions_shapepipe_cut_window.fits", 1), window_keys


def _released_region(ra, dec):
    """Return the released rows from our first object on, with a margin."""
    release = fits.getdata(RELEASE, 1, memmap=True)
    first = np.flatnonzero(
        (np.asarray(release["RA"]) == ra[0]) & (np.asarray(release["Dec"]) == dec[0])
    )
    assert len(first) == 1, "window's first object not found once in the release"
    start = int(first[0])
    margin = 1000
    lo, hi = max(start - margin, 0), start + len(ra) + margin
    return release[lo:hi]


def test_calibration_reproduces_the_release_window(tmp_path):
    ours, window_keys = _run_calibration(tmp_path)
    assert len(ours) > 0
    ra, dec = np.asarray(ours["RA"]), np.asarray(ours["Dec"])

    # The released rows that come from the window, bracketed on both sides
    # by released rows that do not, so none is missed.
    region = _released_region(ra, dec)
    region_keys = np.asarray(region["RA"]) + 1j * np.asarray(region["Dec"])
    in_window = np.isin(region_keys, window_keys)
    assert not in_window[0] and not in_window[-1], "margin too small"
    released = region[in_window]

    # Selection: exactly our objects, in our order.
    np.testing.assert_array_equal(
        np.asarray(released["RA"]) + 1j * np.asarray(released["Dec"]), ra + 1j * dec
    )

    for name, released_name in PER_OBJECT.items():
        np.testing.assert_array_equal(
            np.asarray(ours[name]), np.asarray(released[released_name]), err_msg=name
        )
