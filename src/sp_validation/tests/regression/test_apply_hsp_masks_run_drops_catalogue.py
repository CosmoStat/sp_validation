"""Spatial mask application must preserve the catalogue it annotates."""

import h5py
import healpy as hp
import healsparse as hsp
import numpy as np
import pytest

from sp_validation.catalog_builders import ApplyHspMasks


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#387: mask application drops input science columns",
)
def test_apply_hsp_masks_run_preserves_catalogue_columns_and_values(
    tmp_path, monkeypatch
):
    """Protect coordinates, shears and weights when adding spatial mask flags.

    Mask application annotates rows; it must not replace their measurements.
    The three objects occupy distinct HEALPix pixels, and only the first and
    third pixels are marked, so the flags must be [True, False, True].
    Every input column and value must survive unchanged in the output's data
    dataset, with flags alongside them or in row-aligned data_ext, as supported
    by the two-dataset writer. A separate input file must remain unchanged.
    """
    # Header metadata needs a user name even in a minimal container/CI env.
    monkeypatch.setenv("USER", "regression-test")
    original = np.array(
        [
            (10.0, 5.0, 0.12, -0.07, 2.0),
            (100.0, -15.0, -0.31, 0.23, 5.0),
            (210.0, 35.0, 0.08, 0.41, 9.0),
        ],
        dtype=[
            ("RA", "f8"),
            ("Dec", "f8"),
            ("NGMIX_G1_NOSHEAR", "f8"),
            ("NGMIX_G2_NOSHEAR", "f8"),
            ("NGMIX_W", "f8"),
        ],
    )
    input_path = tmp_path / "input.hdf5"
    output_path = tmp_path / "output.hdf5"
    with h5py.File(input_path, "w") as handle:
        handle.create_dataset("data", data=original)

    nside = 32
    pixels = hp.ang2pix(nside, original["RA"], original["Dec"], lonlat=True, nest=True)
    assert len(np.unique(pixels)) == 3, "Fixture requires distinct pixels"
    mask = hsp.HealSparseMap.make_empty(4, nside, dtype=np.bool_)
    mask.update_values_pix(pixels[[0, 2]], True)
    mask.write(str(tmp_path / f"mask_r_nside{nside}_n4.hsp"))

    builder = ApplyHspMasks()
    builder._params.update(
        input_path=str(input_path),
        output_path=str(output_path),
        mask_dir=str(tmp_path),
        nside=nside,
        bits=4,
    )
    # This is the same run method reached by scripts/apply_hsp_masks.py.
    builder.run()

    with h5py.File(output_path, "r") as handle:
        data = handle["data"][:]
        missing = tuple(
            name for name in original.dtype.names if name not in data.dtype.names
        )
        assert not missing, (
            f"Catalogue columns lost: missing={missing}; "
            f"output datasets={list(handle)}; data columns={data.dtype.names}"
        )
        assert len(data) == len(original), "Masking must preserve row count"
        for name in original.dtype.names:
            np.testing.assert_array_equal(
                data[name], original[name], err_msg=f"Changed catalogue column {name}"
            )
        # Develop uses the v2 name; tomography uses the legacy name.
        expected_names = ("MASK_4_Stars", "4_Stars")
        available = set(data.dtype.names)
        if "data_ext" in handle:
            assert len(handle["data_ext"]) == len(original)
            available.update(handle["data_ext"].dtype.names)
        present = [name for name in expected_names if name in available]
        assert len(present) == 1, f"Expected one Stars mask column, got {available}"
        mask_name = present[0]
        flags = data if mask_name in data.dtype.names else handle["data_ext"][:]
        np.testing.assert_array_equal(
            flags[mask_name],
            [True, False, True],
            err_msg="Mask flags must align with catalogue rows",
        )
    with h5py.File(input_path, "r") as handle:
        np.testing.assert_array_equal(handle["data"][:], original)
