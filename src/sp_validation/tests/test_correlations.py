import healpy as hp
import numpy as np
import pytest

from sp_validation.dashboard import correlations as C
from sp_validation.dashboard import products as P
from sp_validation.dashboard import tangential as TG


@pytest.mark.parametrize(
    "tile_id, name",
    [
        ("51.344", "051.344"),
        ("51.34", "051.340"),
        ("251.3", "251.300"),
        ("7.312", "007.312"),
    ],
)
def test_tile_name_restores_float_formatted_ids(tile_id, name):
    assert C.tile_name(tile_id) == name


def test_tile_centres_follow_cfis_grid():
    names, ra, dec = C.tile_centres(["51.344", "51.344", "180.300"])
    assert names == ["051.344", "180.300"]
    np.testing.assert_allclose(dec, [344 / 2 - 90, 300 / 2 - 90])
    np.testing.assert_allclose(ra[0], 51 / 2 / np.cos(np.deg2rad(82)))


def test_merge_detections_groups_repeat_exposures():
    rng = np.random.default_rng(1)
    ra0 = rng.uniform(150, 151, 50)
    dec0 = rng.uniform(40, 41, 50)
    # Three detections per star, jittered by < 0.1 arcsec.
    ra = np.repeat(ra0, 3) + rng.normal(0, 0.03 / 3600, 150)
    dec = np.repeat(dec0, 3) + rng.normal(0, 0.03 / 3600, 150)
    stars = np.zeros(150, dtype=[("RA", "f8"), ("DEC", "f8"), ("MAG", "f4")])
    stars["RA"], stars["DEC"], stars["MAG"] = ra, dec, np.repeat(np.arange(50), 3)
    ura, udec, umag = C.unique_stars(stars, 0.5)
    assert len(ura) == 50
    order = np.argsort(umag)
    np.testing.assert_allclose(umag[order], np.arange(50))
    np.testing.assert_allclose(udec[order], dec0, atol=1e-4)


def test_quantile_edges_split_equal_counts():
    values = np.arange(300, dtype=float)
    edges = C.quantile_edges(values, 3)
    counts = np.histogram(values, edges)[0]
    np.testing.assert_array_equal(counts, [100, 100, 100])


def test_footprint_randoms_stay_in_footprint():
    nside = 64
    pixels = np.array([hp.ang2pix(nside, 200.0, 45.0, lonlat=True)])
    ra, dec = C.footprint_randoms(5000, nside, pixels, None, np.random.default_rng(0))
    assert len(ra) == 5000
    assert np.all(hp.ang2pix(nside, ra, dec, lonlat=True) == pixels[0])


def test_chi2_null_hartlap_and_guard():
    cov = np.eye(4)
    chi2, nu = TG.chi2_null(np.ones(4), cov, npatch=100)
    assert nu == 4
    np.testing.assert_allclose(chi2, 4 * (100 - 6) / 99)
    assert np.isnan(TG.chi2_null(np.ones(4), cov, npatch=5)[0])


def _result(n=20, amplitude=1e-4):
    theta = np.geomspace(1, 250, n)
    return dict(
        theta=theta,
        xi_t=np.full(n, amplitude),
        xi_x=np.zeros(n),
        error_t=np.full(n, amplitude),
        error_x=np.full(n, amplitude),
        covariance=np.eye(2 * n) * amplitude**2,
        npairs=np.ones(n),
        weight=np.ones(n),
    )


def test_save_load_and_render_roundtrip(tmp_path):
    prov = dict(npatch=100, n_lenses=10, star_mag_edges=[17.0, 19.0, 20.0, 22.0])
    C.save(
        tmp_path,
        "gamma_stars",
        "v",
        {k: _result() for k in ("all", "mag0", "mag1", "mag2")},
        prov,
    )
    product = P.load_correlation(tmp_path / "gamma_stars_v.npz")
    assert product["provenance"]["npatch"] == 100
    np.testing.assert_allclose(product["mag1"]["xi_t"], 1e-4)
    out = TG.gamma_stars(product, tmp_path / "stars.png")
    assert out.exists()
