"""Survey area from a HEALPix mask must ignore hp.UNSEEN (unobserved) pixels."""

import healpy as hp
import numpy as np
import pytest

from sp_validation.cosmo_val.catalog_characterization import (
    CatalogCharacterizationMixin,
)

NSIDE = 64
N_OBSERVED = 10  # Pixels with mask value 1; every other pixel is unobserved.


class _Stub(CatalogCharacterizationMixin):
    def __init__(self, mask_path):
        self.versions = ["v"]
        self.cc = {"v": {"mask": mask_path}}

    def print_start(self, *args, **kwargs):
        pass

    def print_done(self, *args, **kwargs):
        pass

    def print_magenta(self, *args, **kwargs):
        pass


@pytest.fixture
def partial_mask(tmp_path):
    """Store a cut-sky mask with explicit indices, as healpy writes partial maps.

    hp.read_map returns such a map as a full-sky array with hp.UNSEEN
    (-1.6375e30) in every pixel that is not listed in the file.
    """
    mask = np.full(hp.nside2npix(NSIDE), hp.UNSEEN)
    # High indices avoid healpy's unsupported int8 index columns.
    mask[-N_OBSERVED:] = 1.0
    path = tmp_path / "mask_partial.fits"
    hp.write_map(str(path), mask, partial=True, dtype=np.float64)
    expected = N_OBSERVED * hp.nside2pixarea(NSIDE, degrees=True)
    # Guard the fixture: the read-back must really carry the sentinel.
    back = hp.read_map(str(path), dtype=np.float64)
    assert np.sum(back == hp.UNSEEN) == hp.nside2npix(NSIDE) - N_OBSERVED
    return str(path), expected


@pytest.mark.xfail(
    strict=True,
    reason="#387: _area_from_mask counts hp.UNSEEN as negative area",
)
def test_mask_area_excludes_unseen_pixels_in_compute_survey_stats(partial_mask):
    """Protect the area that compute_survey_stats divides into n_eff.

    With overwrite_config=True, it also writes this area to cov_th.A.
    For a mask holding 1 in 10 pixels and hp.UNSEEN everywhere else, the
    footprint is 10 pixels, so the area is 10 * nside2pixarea(64) =
    10 * 0.8393 deg^2 by construction; unobserved pixels must contribute
    zero, not -1.6375e30 each.
    """
    path, expected = partial_mask
    area = _Stub(path)._area_from_mask(path)
    assert area == pytest.approx(expected, rel=1e-12), (
        f"_area_from_mask returned {area:.4g} deg^2, expected {expected:.4f} deg^2"
    )


@pytest.mark.xfail(
    strict=True,
    reason="#387: calculate_area counts hp.UNSEEN as negative area",
)
def test_mask_area_excludes_unseen_pixels_in_calculate_area(partial_mask):
    """Protect CosmologyValidation.area when a version's config names a mask.

    This per-version area determines n_eff_gal and the covariance inputs.
    The mask has 10 observed pixels at NSIDE=64, so the correct area is
    10 * nside2pixarea(64) = 10 * 0.8393 deg^2 by construction.
    """
    path, expected = partial_mask
    stub = _Stub(path)
    stub.calculate_area()
    area = float(stub._area["v"])
    assert area == pytest.approx(expected, rel=1e-12), (
        f"calculate_area returned {area:.4g} deg^2, expected {expected:.4f} deg^2"
    )
