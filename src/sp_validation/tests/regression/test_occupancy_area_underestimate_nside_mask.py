"""Occupancy-based survey area must recover the footprint at UNIONS density."""

import contextlib
import inspect

import healpy as hp
import numpy as np
import pytest

from sp_validation.cosmo_val.catalog_characterization import (
    CatalogCharacterizationMixin,
)
from sp_validation.cosmo_val.core import CosmologyValidation

COARSE_NSIDE = 256  # Whole coarse pixels give an exactly known footprint area.
N_COARSE = 16  # Approximately 0.84 deg^2.
DENSITY_ARCMIN2 = 5.9  # UNIONS raw density.


def _footprint_and_points(seed=0):
    """Draw uniform points in N_COARSE contiguous NSIDE=256 pixels."""
    rng = np.random.default_rng(seed)
    centre = hp.ang2pix(COARSE_NSIDE, 150.0, 30.0, lonlat=True)
    pixels = {centre}
    while len(pixels) < N_COARSE:
        neighbours = hp.get_all_neighbours(COARSE_NSIDE, list(pixels)).ravel()
        for pixel in neighbours:
            if pixel >= 0 and len(pixels) < N_COARSE:
                pixels.add(int(pixel))
    pixels = np.array(sorted(pixels))
    true_area = pixels.size * hp.nside2pixarea(COARSE_NSIDE, degrees=True)
    n_target = rng.poisson(DENSITY_ARCMIN2 * true_area * 3600.0)

    lon_c, lat_c = hp.pix2ang(COARSE_NSIDE, pixels, lonlat=True)
    lo, hi = lon_c.min() - 2, lon_c.max() + 2
    zlo = np.sin(np.radians(lat_c.min() - 2))
    zhi = np.sin(np.radians(lat_c.max() + 2))
    ra, dec = [], []
    n = 0
    while n < n_target:
        lon = rng.uniform(lo, hi, 400_000)
        lat = np.degrees(np.arcsin(rng.uniform(zlo, zhi, 400_000)))
        keep = np.isin(hp.ang2pix(COARSE_NSIDE, lon, lat, lonlat=True), pixels)
        ra.append(lon[keep])
        dec.append(lat[keep])
        n += keep.sum()
    ra = np.concatenate(ra)[:n_target]
    dec = np.concatenate(dec)[:n_target]
    return ra, dec, true_area


class _Result:
    def __init__(self, ra, dec):
        self.dat_shear = {"RA": ra, "Dec": dec}

    @contextlib.contextmanager
    def temporarily_read_data(self):
        yield


class _Stub(CatalogCharacterizationMixin):
    def __init__(self, ra, dec, nside_mask):
        self.nside_mask = nside_mask
        self.results = {"v": _Result(ra, dec)}


@pytest.mark.xfail(
    strict=True,
    reason="#387: default occupancy drops Poisson-empty footprint pixels",
)
def test_occupancy_area_underestimates_footprint_at_default_nside_mask():
    """Protect the mask-less fallback of CosmologyValidation.area.

    A version with no ``mask`` key gets its area from the number of HEALPix
    pixels at ``nside_mask`` that contain at least one galaxy; that area then
    divides into n_eff_gal and both feed the OneCovariance INI. Here galaxies
    are drawn uniformly at UNIONS raw density (5.9 / arcmin^2) inside a
    footprint made of whole NSIDE=256 pixels, so the true area is known
    exactly and every NSIDE>=256 sub-pixel is genuinely inside it. A correct
    estimator recovers that area to within a few percent; at the default
    nside_mask=8192 (pixel 0.18 arcmin^2, ~1.07 galaxies per pixel) a fraction
    exp(-1.07) ~ 34% of footprint pixels are Poisson-empty and are dropped.
    """
    default_nside = (
        inspect.signature(CosmologyValidation.__init__).parameters["nside_mask"].default
    )
    ra, dec, true_area = _footprint_and_points()
    stub = _Stub(ra, dec, default_nside)
    area = stub.calculate_area_from_binned_catalog("v")
    ratio = area / true_area
    assert abs(ratio - 1) < 0.05, (
        f"occupancy area at nside_mask={default_nside}: {area:.4f} deg^2 vs true "
        f"{true_area:.4f} deg^2 (ratio {ratio:.3f}); n_eff_gal would be "
        f"{1 / ratio:.2f}x too high"
    )
