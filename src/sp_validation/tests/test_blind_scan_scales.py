"""Scale filtering speeds geometry scans without weakening the leak oracle."""

import numpy as np
import pytest

from sp_validation.tests.blind_scan import assert_no_raw_values


def test_geometry_values_are_filtered_before_pairwise_matching(tmp_path, monkeypatch):
    np.save(tmp_path / "geometry.npy", np.tile([0.0, 1.0, 1000.0], 100_000))
    original = np.isclose
    sizes = []

    def observed(a, b, **kwargs):
        sizes.append(np.asarray(a).size)
        return original(a, b, **kwargs)

    monkeypatch.setattr(np, "isclose", observed)
    assert_no_raw_values(
        tmp_path, {"EE": np.array([2e-11])}, expected={"EE": np.array([2.2e-11])}
    )
    assert max(sizes) == 1  # The expected-twin comparison, no geometry chunks.

    data = np.zeros(100_001)
    data[-1] = 2e-11
    np.save(tmp_path / "single_leak.npy", data)
    with pytest.raises(AssertionError, match="Unblinded EE value"):
        assert_no_raw_values(
            tmp_path,
            {"EE": np.array([2e-11])},
            expected={"EE": np.array([2.2e-11])},
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("fraction", [-1.001, -1.0, -0.999, 0, 0.999, 1.0, 1.001])
def test_filter_preserves_isclose_boundary(tmp_path, dtype, sign, fraction):
    reference = np.array([sign * 2e-11], dtype=dtype)
    raw = float(reference[0])
    tolerance = 1e-25 + 1e-10 * abs(raw)
    candidate = raw + fraction * tolerance
    np.save(tmp_path / "value.npy", [candidate])
    matches = np.isclose(
        np.array([candidate], dtype=float), reference, rtol=1e-10, atol=1e-25
    )[0]
    if matches:
        with pytest.raises(AssertionError, match="Unblinded EE value"):
            assert_no_raw_values(
                tmp_path, {"EE": reference}, expected={"EE": reference * 1.1}
            )
    else:
        assert_no_raw_values(
            tmp_path, {"EE": reference}, expected={"EE": reference * 1.1}
        )
