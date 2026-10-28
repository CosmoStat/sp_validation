"""Weighted-pair normalization, selection, and shape-free I/O."""

import h5py
import numpy as np
import pytest
from astropy.io import fits

from sp_validation.pair_counts import measure_pair_counts, ordered_effective_pairs


def test_uniform_limit_and_dispersion_convention():
    # Exact expectation for a uniform Poisson field, far from boundaries.
    # All unordered pairs have the same product weight, which must cancel.
    edges = np.geomspace(1, 83, 31)
    area, density, s, pair_weight = 2000 * 3600, 5.0, 0.33, 7.0**2
    unordered = area * density**2 * np.pi * np.diff(edges**2) / 2
    W = unordered * pair_weight
    Q = unordered * pair_weight**2
    ordered = ordered_effective_pairs(W, Q)
    np.testing.assert_allclose(ordered, 2 * unordered)
    np.testing.assert_allclose(4 * s**4 / ordered, 2 * s**4 * Q / W**2)
    sigma_total = np.sqrt(2) * s
    np.testing.assert_allclose(sigma_total**4 / (2 * unordered), 4 * s**4 / ordered)


@pytest.mark.parametrize(
    "W,Q",
    [
        ([0], [1]),
        ([1], [0]),
        ([1], [-1]),
        ([np.nan], [1]),
        ([1], [np.inf]),
        ([1, 2], [1]),
    ],
)
def test_invalid_counts_fail_closed(W, Q):
    with pytest.raises(ValueError):
        ordered_effective_pairs(W, Q)


def test_positions_weights_only_and_unordered_treecorr(tmp_path):
    # There are deliberately no shape or blind columns/records in this FITS.
    ra = np.array([0.0, 0.01, 0.03, 0.07])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    cat = tmp_path / "positions.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="RA", format="D", array=ra),
            fits.Column(name="Dec", format="D", array=np.zeros(4)),
            fits.Column(name="w", format="D", array=weights),
        ]
    ).writeto(cat)
    out = tmp_path / "pairs.npz"
    measure_pair_counts(cat, out, min_sep=0.1, max_sep=10.0, nbins=1)
    products = np.array(
        [weights[i] * weights[j] for i in range(4) for j in range(i + 1, 4)]
    )
    with np.load(out) as z:
        np.testing.assert_allclose(z["weight"], products.sum())
        np.testing.assert_allclose(z["weight_squared"], (products**2).sum())
        np.testing.assert_allclose(z["npairs"], 6)
        np.testing.assert_allclose(z["theta"], [1.0])
        np.testing.assert_allclose(
            np.loadtxt(out.with_suffix(".txt"))[1],
            2 * products.sum() ** 2 / (products**2).sum(),
        )


@pytest.mark.parametrize("layout", ["single", "chunked", "joined"])
def test_hdf5_never_materialises_shapes(tmp_path, monkeypatch, layout):
    cat = tmp_path / "catalogue.fits"  # Detection is by content, not extension.
    data = np.zeros(4, dtype=[(n, "f8") for n in ("RA", "Dec", "w", "e1", "e2")])
    data["RA"] = [0.0, 0.01, 0.03, 0.07]
    data["w"] = [1.0, 2.0, 3.0, 4.0]
    with h5py.File(cat, "w") as f:
        if layout == "single":
            f.create_dataset("data", data=data)
        elif layout == "chunked":
            group = f.create_group("tiles")
            group.create_dataset("0001", data=data[:2])
            group.create_dataset("0002", data=data[2:])
            f.attrs["n_tiles"] = 2
        else:
            f.create_dataset("data", data=data[["RA", "Dec", "e1", "e2"]])
            f.create_dataset("data_ext", data=data[["w"]])
    original = h5py.Dataset.__getitem__
    reads = []

    def only_pair_fields(dataset, key, *args, **kwargs):
        # Grammar may inspect zero-row slices to discover field dtypes.
        if isinstance(key, slice) and key.start == key.stop == 0:
            return original(dataset, key, *args, **kwargs)
        # h5py field access recurses with a projected new_dtype and key ().
        projected = kwargs.get("new_dtype")
        if projected is not None:
            assert set(projected.names) <= {"RA", "Dec", "w"}
            return original(dataset, key, *args, **kwargs)
        # Unprojected whole-dataset or shape-field reads remain forbidden.
        assert isinstance(key, str) and key in ("RA", "Dec", "w"), key
        reads.append(key)
        return original(dataset, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", only_pair_fields)
    out = tmp_path / "pairs.npz"
    measure_pair_counts(cat, out, min_sep=0.1, max_sep=10.0, nbins=1)
    assert set(reads) == {"RA", "Dec", "w"}
    with np.load(out) as z:
        np.testing.assert_allclose(z["weight"], 35.0)
        np.testing.assert_allclose(z["weight_squared"], 273.0)
