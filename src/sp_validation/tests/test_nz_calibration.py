"""Known-answer checks for the provisional SOM n(z) calibration."""

import numpy as np
import pytest

from sp_validation import nz_calibration as nz


def test_deposition_conserves_mass_and_mean():
    grid = np.linspace(0, 5, 501)
    z = np.array([0.0, 0.007, 0.2851, 4.999, 5.0])
    weight = np.array([1.0, 2.0, 3.0, 0.5, 7.0])
    density = nz.deposit(z, weight, grid)
    assert np.trapezoid(density, grid) == pytest.approx(weight.sum(), abs=1e-12)
    assert np.trapezoid(grid * density, grid) == pytest.approx(z @ weight, abs=1e-12)
    with pytest.raises(ValueError, match="outside output grid"):
        nz.deposit(np.array([5.001]), np.ones(1), grid)


def test_z2_only_fills_missing_canonical_z():
    mag = np.array(
        [
            [22, 21, 20, 19, 18, 17],
            [22, 21, 20, 19, 99, 18],
            [22, 21, 20, 19, -99, 99],
            [22, 21, 20, 19, np.nan, 18],
        ],
        dtype="f4",
    )
    mapped, channel = nz.z_magnitudes(mag, -0.1104888916015625)
    np.testing.assert_array_equal(channel, [1, 2, 0, 2])
    assert mapped[0, 4] == 18
    expected = (mag[:, 5] - 0.1104888916015625).astype("f8")
    np.testing.assert_array_equal(mapped[[1, 3], 4], expected[[1, 3]])
    assert np.isnan(mapped[2, 4])
    canonical, channel = nz.z_magnitudes(mag, -0.1104888916015625, use_z2=False)
    np.testing.assert_array_equal(channel, [1, 0, 0, 0])
    assert np.isnan(canonical[1:, 4]).all()


def test_gold_requires_support_and_bpZ_agreement():
    # Supported matching group, unsupported group, discrepant group, empty source.
    mask = nz.gold_mask(
        np.array([10.0, 10.0, 10.0, 0.0]),
        np.array([3.0, 3.0, 3.0, 0.0]),
        np.array([2, 0, 1, 1]),
        np.array([0.6, 0.0, 1.0, 1.0]),
        threshold=0.4,
    )
    np.testing.assert_array_equal(mask, [True, False, False, True])
    assert nz.gold_threshold(np.array([0.2, 0.3]), np.array([0.2, 0.3])) == 0.4


def test_bin_edges_and_gold_ids():
    mag = np.tile([22.0, 21.0, 20.0, 19.0, 18.0, 99.0], (7, 1)).astype("f4")
    model = dict(
        code=nz.features(mag[:1, :5]),
        center=np.zeros(11),
        scale=np.ones(11),
        group_map=np.array([0]),
        edges=np.array([0.1, 0.37, 0.55, 0.76, 1.2]),
    )
    bpz = np.array([0.099, 0.1, 0.37, 0.55, 0.76, 1.2, 0.4], dtype="f8")
    weight = np.ones(7)
    weight[-1] = 0
    prelim, group, _ = nz.source_groups(mag, bpz, weight, model, offset=0, chunk=2)
    np.testing.assert_array_equal(prelim, [-1, 0, 1, 2, 3, -1, -1])
    ids = nz.tom_bin_ids(prelim, group, np.array([[1], [1], [0], [1]], bool))
    np.testing.assert_array_equal(ids, [0, 1, 2, 0, 4, 0, 0])


def test_direct_weights_and_writer(tmp_path):
    grid = np.linspace(0, 5, 501)
    # Bin 1 has group weights 6 and 2, bin 2 has weight 4.
    result = nz.calibrate(
        np.array([0, 0, 1]),
        np.array([0, 1, 1]),
        np.array([6.0, 2.0, 4.0]),
        np.array([0.2, 0.3, 0.8]),
        np.array([0.1, 0.3, 0.5, 0.9]),
        np.array([0.2, 0.2, 0.3, 0.8]),
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.5, 1.2]),
        2,
        grid,
        gold=np.ones((2, 2), bool),
    )
    np.testing.assert_array_equal(result["cal_weights"], [[3, 3, 2, 0], [0, 0, 0, 4]])
    np.testing.assert_allclose([b["mean"] for b in result["bins"]], [0.275, 0.9])
    path = tmp_path / "nz.txt"
    nz.write_nz(path, grid, result["bin_hists"], [8, 4])
    text = np.loadtxt(path)
    assert text.shape == (501, 3)
    np.testing.assert_allclose(
        np.trapezoid(text[:, 1:], text[:, 0], axis=0), [2 / 3, 1 / 3], atol=1e-12
    )
    mixture = text[:, 1:].sum(1)
    assert np.trapezoid(grid * mixture, grid) == pytest.approx(
        (8 * 0.275 + 4 * 0.9) / 12, abs=1e-12
    )


def test_nz_file_cosmo_val_and_sacc_roundtrip(tmp_path):
    import sacc

    from sp_validation.cosmo_val import CosmologyValidation
    from sp_validation.sacc_io import new_sacc

    grid = np.linspace(0, 2, 201)
    hists = np.array(
        [
            nz.deposit(np.array([0.2, 0.4]), np.array([0.5, 0.5]), grid),
            nz.deposit(np.array([0.8, 1.1]), np.array([0.25, 0.75]), grid),
        ]
    )
    path = tmp_path / "nz.txt"
    nz.write_nz(path, grid, hists, [2, 3])
    expected = np.loadtxt(path)
    reader = object.__new__(CosmologyValidation)
    reader.cc = {"test": {"shear": {"redshift_path": str(path)}}}
    z, densities = reader.read_redshift_distribution("test", is_tomography=True)
    np.testing.assert_array_equal(z, expected[:, 0])
    np.testing.assert_array_equal(densities, expected[:, 1:])
    _, summed = reader.read_redshift_distribution("test", is_tomography=False)
    np.testing.assert_array_equal(summed, densities.sum(1))
    tracers = new_sacc([(z, d) for d in densities.T])
    fits_path = tmp_path / "nz.fits"
    tracers.save_fits(str(fits_path))
    loaded = sacc.Sacc.load_fits(str(fits_path))
    for k in range(densities.shape[1]):
        tracer = loaded.get_tracer(f"source_{k}")
        np.testing.assert_array_equal(tracer.z, z)
        np.testing.assert_array_equal(tracer.nz, densities[:, k])
