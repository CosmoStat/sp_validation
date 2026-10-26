"""Refitted cell sums versus the retained-row reference, including empty cells."""

import numpy as np
import pytest

from sp_validation import calibration, response
from sp_validation.des_jackknife import accumulate
from sp_validation.tests.test_consistent_response import catalogue


@pytest.mark.parametrize("min_count", [0, 3, 10, 1000])
def test_refit_cell_sums_match_rescans_with_empty_cell(min_count):
    data, _, _ = catalogue()
    # Only group zero trains the upper-right cell. Other artificial branches
    # migrate into it, so its newly pooled weight affects the retained response.
    for suffix in ("NOSHEAR", "1P", "1M", "2P", "2M"):
        data[f"NGMIX_FLUX_{suffix}"] = 30
        data[f"NGMIX_T_{suffix}"] = 0.7
        data[f"NGMIX_FLUX_{suffix}"][:3] = 200
        data[f"NGMIX_T_{suffix}"][:3] = 2
        data[f"NGMIX_G1_{suffix}"][:20] += 0.13
    data["NGMIX_FLUX_1P"][20:24] = 200
    data["NGMIX_T_1P"][20:24] = 2
    data["NGMIX_FLUX_2M"][30:35] = 200
    data["NGMIX_T_2M"][30:35] = 2
    # Sheared-only and no-shear-only objects exercise mask/rg alignment.
    data["NGMIX_FLUX_NOSHEAR"][50] = 9
    data["NGMIX_FLUX_1M"][51] = 9
    gm = calibration.metacal(
        data,
        np.ones(len(data), bool),
        weight_scheme="des",
        size_corr_ell=False,
        des_num_bins=2,
        des_min_count=min_count,
    )
    groups = np.repeat(np.arange(4), 20)
    acc = accumulate(gm, groups, chunk_size=7)
    assert gm.W[3][1, 1] == 3
    assert acc.training[0, :, 3].sum() - acc.training[0, 0, 3] == 0
    for k, label in enumerate(acc.labels):
        keep = groups != label
        idx = gm.mask_dict["ns"]
        retained = keep[idx]
        table = response.fit_w(
            (gm.ns["flux"] / gm.ns["flux_err"])[idx][retained],
            (gm.ns["T"] / gm.ns["Tpsf"])[idx][retained],
            np.array(gm.branches["ns"])[:, idx][:, retained],
            (0.5 * (gm.R11 + gm.R22))[retained],
            gm.W[0],
            gm.W[1],
            min_count=min_count,
        )
        np.testing.assert_allclose(acc.deleted_weight(k), table[2], atol=1e-12, rtol=0)
        if k == 0:
            assert (acc.deleted_weight(k)[1, 1] > 0) == (min_count > 0)
    labels, fast, cov = gm.response_jackknife(groups, refit_w=True)
    ref_labels, ref, ref_cov = gm.response_jackknife(
        groups, refit_w=True, reference=True
    )
    np.testing.assert_array_equal(labels, ref_labels)
    for key in fast:
        np.testing.assert_allclose(fast[key], ref[key], atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            acc.jackknife()[1][key], ref[key], atol=1e-12, rtol=0
        )
    np.testing.assert_allclose(cov, ref_cov, atol=1e-12, rtol=0)


def test_refit_group_validation():
    data, _, _ = catalogue()
    gm = calibration.metacal(
        data,
        np.ones(len(data), bool),
        weight_scheme="des",
        size_corr_ell=False,
    )
    with pytest.raises(ValueError, match="align"):
        gm.response_jackknife(np.arange(10), refit_w=True)
    with pytest.raises(ValueError, match="at least two"):
        gm.response_jackknife(np.zeros(len(data)), refit_w=True)
