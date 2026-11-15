"""Known-answer checks of the catalogue twin of the SOM n(z) (``nz.twin``)."""

import numpy as np
import pytest

from sp_validation.nz import pipeline, twin

CONFIG = dict(
    som=dict(side=12, epochs=15, seed=0),
    groups=100,
    bins=dict(count=3, range=[0.15, 1.15]),
    gold_qc=dict(floor=0.4, nmad_factor=5),
    z2_offset=-0.1,
    grid=dict(min=0, max=3, step=0.02),
)


@pytest.fixture(scope="module")
def sky():
    """Sources whose colours track a hidden redshift, a representative tenth of
    them as calibrators, and the calibrators as the spectral library."""
    rng = np.random.default_rng(0)
    n = 60_000
    z = rng.uniform(0.1, 1.2, n)
    r = rng.uniform(20, 24, n)
    colour = np.column_stack([1.5 * z, z, 0.6 * z, 0.4 * z])
    colour += rng.normal(0, 0.04 * (r[:, None] - 19), colour.shape)
    g = r + colour[:, 1]
    mag = np.column_stack([g + colour[:, 0], g, r, r - colour[:, 2]])
    mag = np.column_stack([mag, mag[:, 3] - colour[:, 3], np.full(n, -99.0)])
    src = dict(
        mag=mag,
        photoz=np.round(z + rng.normal(0, 0.05 * (1 + z)), 2),
        weight=rng.uniform(0.5, 1.5, n),
        object_id=np.arange(n),
    )
    rows = rng.choice(n, 6000, replace=False)
    cal = dict(mag=mag[rows, :5], photoz=src["photoz"][rows], object_id=rows, z=z[rows])
    return src, cal


@pytest.mark.parametrize("kind", twin.SPACES)
def test_representative_twin_is_recovered(sky, kind):
    """Representative calibrators painted by the same model as the sources:
    every bin's Δ⟨z⟩ is consistent with zero (calibrator shot noise at this
    size is ≈ 0.005), and swapping the calibrators' painted redshifts among
    themselves is caught."""
    src, library = sky
    rng = np.random.default_rng(1)
    k = dict(k=16, offset=CONFIG["z2_offset"])
    near = twin.neighbours(
        kind, library, src["mag"], src["photoz"], src["object_id"], **k
    )
    z = twin.paint(kind, library, near, src["photoz"], rng)
    cal = dict(library, z=z[library["object_id"]])
    result = pipeline.estimate(CONFIG, cal, src)
    delta = [b["delta"] for b in twin.score(result, src["weight"], z)]
    assert np.abs(delta).max() < 0.015

    shuffled = dict(cal, z=rng.permutation(cal["z"]))
    wrong = twin.score(
        twin.recalibrate(CONFIG, result, src, shuffled), src["weight"], z
    )
    assert np.abs([b["delta"] for b in wrong]).max() > 0.1


def test_neighbours_skip_the_galaxy_itself(sky):
    src, library = sky
    rows = library["object_id"][:500]
    near = twin.neighbours(
        "colour",
        library,
        src["mag"][rows],
        src["photoz"][rows],
        rows,
        k=8,
        offset=CONFIG["z2_offset"],
    )
    assert near.shape == (500, 8)
    assert not (library["object_id"][near] == rows[:, None]).any()


def test_zero_point_offsets_have_the_requested_scatter_and_patch():
    rng = np.random.default_rng(2)
    n = 200_000
    ra, dec = rng.uniform(100, 250, n), rng.uniform(30, 70, n)
    tile = np.floor(ra) * 1000 + np.floor(dec)
    sigma = [0.02, 0.0, 0.0, 0.0, 0.0]
    patch = dict(band="i", ra=[140, 230], dec=[30, 60], offset=0.02)
    dm = twin.zero_point_offsets(ra, dec, tile, sigma, rng, patch=patch)
    assert dm[:, 0].std() == pytest.approx(0.02, rel=0.1)
    assert (
        dm[:, 1:]
        == 0.02 * ((ra >= 140) & (ra <= 230) & (dec <= 60))[:, None] * [0, 0, 1, 0]
    ).all()


def test_photoz_gradient_recovers_a_linear_photoz():
    rng = np.random.default_rng(3)
    mag = rng.normal(22, 0.5, (50_000, 5))
    slope = np.array([0.1, -0.2, 0.0, 0.3, -0.1])
    photoz = 0.5 + (mag - 22) @ slope
    cell = rng.integers(0, 10, len(mag))
    grad = twin.photoz_gradient(
        np.column_stack([mag, np.full(len(mag), -99.0)]), photoz, cell, 10, offset=0
    )
    assert np.allclose(grad, slope, atol=1e-8)
