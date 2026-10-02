"""Pipeline flags must survive conversion to a projected HEALPix keep-map."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

import sp_validation

REPO = Path(__file__).resolve().parents[4]
# Use the selected package's checkout during cross-branch verification.
PACKAGE_REPO = Path(sp_validation.__file__).resolve().parents[2]
if PACKAGE_REPO != REPO:
    REPO = PACKAGE_REPO
SCRIPT = REPO / "scripts" / "combine_hp_masks.py"


def _load():
    if not SCRIPT.exists():
        pytest.skip(
            "combine_hp_masks.py missing; requires the tomography branch script"
        )
    spec = importlib.util.spec_from_file_location("combine_hp_masks", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_tile(tmp_path, flags):
    w = WCS(naxis=2)
    w.wcs.crpix = [16.0, 16.0]
    w.wcs.cdelt = np.array([-0.05, 0.05])
    w.wcs.crval = [20.0, 5.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.to_header().totextfile(tmp_path / "CFIS_image-233-293.fits", overwrite=True)
    fits.PrimaryHDU(flags.astype(np.int16)).writeto(
        tmp_path / "pipeline_flag-233-293.fits", overwrite=True
    )


@pytest.mark.xfail(
    strict=True,
    reason="#387: pipeline flags are erased before building the keep-map",
)
def test_fully_flagged_tile_projects_to_no_kept_healpix_pixels(tmp_path):
    """read_pixel_mask_files must turn ShapePipe pipeline flags (nonzero = masked)
    into a 0/1 keep-map before projecting it to HEALPix. A 32x32 tile in which
    every image pixel carries a nonzero flag has, by construction, nothing to
    keep, so its projected map must contain no positive pixel. The script sets
    flagged pixels to 0 and then selects ``mask == 0`` on that mutated array,
    so every pixel becomes 1 and the fully flagged tile projects to the same
    footprint as an unflagged one: star and defect holes vanish from the mask.
    """
    mod = _load()
    params = {
        "input_dir_flags": str(tmp_path),
        "input_dir_images": str(tmp_path),
        "nside": 64,
        "verbose": False,
    }

    _write_tile(tmp_path, np.zeros((32, 32)))
    clean = mod.read_pixel_mask_files(["233.293"], params)["233.293"]
    n_clean = int(np.count_nonzero(clean > 0))
    assert n_clean > 0, "fixture tile should cover at least one HEALPix pixel"

    _write_tile(tmp_path, np.ones((32, 32)))
    flagged = mod.read_pixel_mask_files(["233.293"], params)["233.293"]
    n_kept = int(np.count_nonzero(flagged > 0))
    assert n_kept == 0, (
        f"fully flagged tile kept {n_kept} HEALPix pixels "
        f"(sum {np.nansum(flagged)}); unflagged tile kept {n_clean}"
    )
