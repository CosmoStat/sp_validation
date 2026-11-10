"""Covariance selection at the tomographic COSEBIs boundary."""

from types import SimpleNamespace

import pytest

from sp_validation.cosmo_val.cosebis import CosebisMixin


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"scale_cuts": [(1, 10)]},
        {"evaluate_all_scale_cuts": True},
    ],
)
def test_tomographic_cosebis_rejects_single_covariance(options):
    calls = []
    cv = SimpleNamespace(
        npatch=7,
        print_start=lambda *args: None,
        _binning=lambda *args: {},
        calculate_2pcf_version=lambda *args, **kwargs: calls.append(kwargs) or {},
    )
    with pytest.raises(ValueError, match="cov_path.*single.*tomographic bin pair"):
        CosebisMixin.calculate_cosebis(
            cv, "v", cov_path="xi_cov.txt", compute_tomography=True, **options
        )
    assert calls == []
