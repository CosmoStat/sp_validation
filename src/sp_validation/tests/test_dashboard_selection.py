"""Tests for the cumulative comprehensive-catalogue selection."""

import h5py
import numpy as np
import pytest
import yaml

from sp_validation.dashboard.selection_hist import build


def test_cumulative_selection_and_expected_count(tmp_path):
    names = [
        "mag",
        "FLAGS",
        "NGMIX_FLAGS_NOSHEAR",
        "NGMIX_FLUX_NOSHEAR",
        "NGMIX_FLUX_ERR_NOSHEAR",
        "NGMIX_T_NOSHEAR",
        "NGMIX_T_PSF_RECONV_NOSHEAR",
    ]
    rows = np.zeros(
        4, dtype=[(name, "i4" if name == "FLAGS" else "f8") for name in names]
    )
    rows["mag"] = [20, 21, 22, 23]
    rows["FLAGS"] = [16, 0, 19, 0]
    rows["NGMIX_FLUX_NOSHEAR"] = [20, 5, 20, 20]
    rows["NGMIX_FLUX_ERR_NOSHEAR"] = 1
    rows["NGMIX_T_NOSHEAR"] = 1
    rows["NGMIX_T_PSF_RECONV_NOSHEAR"] = 1
    path = tmp_path / "data.hdf5"
    with h5py.File(path, "w") as f:
        f["data"] = rows
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dat": [
                    {
                        "col_name": "FLAGS",
                        "label": "flags",
                        "kind": "smaller_equal",
                        "value": 2,
                        "ignore_bits": 16,
                    }
                ],
                "metacal": {
                    "gal_snr_min": 10,
                    "gal_snr_max": 500,
                    "gal_rel_size_min": 0.707,
                    "gal_rel_size_max": 3,
                },
            }
        )
    )
    result = build(path, config, expected=2, chunk=2)
    np.testing.assert_array_equal(result["counts"], [4, 3, 2])
    with pytest.raises(ValueError, match="expected 3"):
        build(path, config, expected=3, chunk=2)
