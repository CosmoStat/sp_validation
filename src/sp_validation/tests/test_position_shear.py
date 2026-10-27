import numpy as np
import pytest
import treecorr

from sp_validation.dashboard.position_shear import measure


def test_injected_tangential_shear_and_joint_covariance():
    rng = np.random.default_rng(42)
    # Four well-separated lens groups, each with sources in a tangential pattern.
    centres = np.array([[0, 0], [4, 0], [0, 4], [4, 4]], dtype=float)
    ra, dec, g1, g2 = [], [], [], []
    for x, y in centres:
        angle = rng.uniform(0, 2 * np.pi, 200)
        radius = rng.uniform(0.03, 0.12, 200)
        ra.extend(x + radius * np.cos(angle) / np.cos(np.deg2rad(y)))
        dec.extend(y + radius * np.sin(angle))
        g1.extend(-0.05 * np.cos(2 * angle))
        g2.extend(0.05 * np.sin(2 * angle))
    lon, lat = np.deg2rad(centres).T
    patch_centers = np.column_stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat))
    )
    lens = treecorr.Catalog(
        ra=centres[:, 0],
        dec=centres[:, 1],
        ra_units="deg",
        dec_units="deg",
        patch_centers=patch_centers,
    )
    source = treecorr.Catalog(
        ra=ra,
        dec=dec,
        g1=g1,
        g2=g2,
        ra_units="deg",
        dec_units="deg",
        patch_centers=patch_centers,
    )
    result = measure(lens, source, min_sep=2, max_sep=7, nbins=3, num_threads=1)
    np.testing.assert_allclose(result["xi_t"], 0.05, atol=2e-4)
    np.testing.assert_allclose(result["xi_x"], 0, atol=1e-4)
    assert result["covariance"].shape == (6, 6)
    assert np.all(result["npairs"] > 0)


def test_rejects_mismatched_patch_centres():
    source = treecorr.Catalog(
        ra=[0, 1, 2],
        dec=[0, 1, 2],
        g1=[0, 0, 0],
        g2=[0, 0, 0],
        ra_units="deg",
        dec_units="deg",
        npatch=2,
    )
    lens = treecorr.Catalog(
        ra=[10, 11, 12],
        dec=[0, 1, 2],
        ra_units="deg",
        dec_units="deg",
        npatch=2,
    )
    with pytest.raises(ValueError, match="patch centres"):
        measure(lens, source, num_threads=1)


def test_random_subtraction_identical_catalogues():
    rng = np.random.default_rng(3)
    ra = rng.uniform(0, 3, 500)
    dec = rng.uniform(0, 3, 500)
    source = treecorr.Catalog(
        ra=ra,
        dec=dec,
        g1=rng.normal(size=500),
        g2=rng.normal(size=500),
        ra_units="deg",
        dec_units="deg",
        npatch=6,
        rng=rng,
    )
    lens = treecorr.Catalog(
        ra=ra[::5],
        dec=dec[::5],
        ra_units="deg",
        dec_units="deg",
        patch_centers=source.patch_centers,
    )
    result = measure(
        lens, source, randoms=lens, min_sep=2, max_sep=60, nbins=4, num_threads=1
    )
    np.testing.assert_allclose(result["xi_t"], 0, atol=1e-14)
    np.testing.assert_allclose(result["xi_x"], 0, atol=1e-14)
    np.testing.assert_allclose(result["covariance"], 0, atol=1e-14)
