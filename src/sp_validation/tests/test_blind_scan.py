"""Exercise the leak scanner itself with trusted synthetic artifacts."""

import json
import pickle

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import blinding, sacc_io
from sp_validation.tests.blind_scan import assert_no_raw_values

RAW = np.array([1.23456789e-7, -7.65432198e-8, 4.32198765e-9])


@pytest.mark.parametrize(
    "suffix", [".npz", ".npy", ".txt", ".json", ".pickle", ".fits", ".sacc"]
)
@pytest.mark.parametrize("indices", [[2, 0], [1]])
def test_scanner_catches_subsets_and_reorderings(tmp_path, suffix, indices):
    path = tmp_path / f"accidental{suffix}"
    value = RAW[indices]
    if suffix == ".npz":
        np.savez(path, unrelated=value)
    elif suffix == ".npy":
        np.save(path, value)
    elif suffix == ".txt":
        np.savetxt(path, value, fmt="%.6e")
    elif suffix == ".json":
        path.write_text(json.dumps({"nested": {"unknown": value.tolist()}}))
    elif suffix == ".pickle":
        path.write_bytes(pickle.dumps({"nested": value}))
    elif suffix == ".fits":
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU.from_columns(
                    [fits.Column(name="unexpected", format="D", array=value)]
                ),
            ]
        ).writeto(path)
    else:
        part = sacc_io.new_sacc({0: ([0.1, 1.0], [1.0, 1.0])})
        sacc_io.add_xi(
            part, (0, 0), np.arange(len(value)) + 1, value, value, grid="reporting"
        )
        sacc_io.save(part, path, blind=blinding.NONE)
    with pytest.raises(
        AssertionError, match=f"Unblinded xi value in accidental{suffix}"
    ):
        assert_no_raw_values(tmp_path, {"xi": RAW})


def test_scanner_catches_logs_and_returns_and_accepts_shifted_values(tmp_path):
    np.savez(tmp_path / "sealed.npz", vector=RAW + 1e-6)
    assert_no_raw_values(tmp_path, {"xi": RAW})
    with pytest.raises(AssertionError, match="captured stdout"):
        assert_no_raw_values(tmp_path, {"xi": RAW}, logs=f"xi = {RAW[0]:.6e}")
    with pytest.raises(AssertionError, match="held return"):
        assert_no_raw_values(
            tmp_path, {"xi": RAW}, objects=[("held return", RAW[::-1])]
        )


@pytest.mark.parametrize("persist", [False, True])
def test_scanner_catches_raw_treecorr_objects(tmp_path, persist):
    import treecorr

    gg = treecorr.GGCorrelation(min_sep=1, max_sep=10, nbins=3)
    gg.xip[:] = RAW
    objects = [("raw TreeCorr return", gg)]
    if persist:
        (tmp_path / "raw_correlation.pickle").write_bytes(pickle.dumps(gg))
        objects = []
    with pytest.raises(AssertionError, match="Unblinded xi value"):
        assert_no_raw_values(tmp_path, {"xi": RAW}, objects=objects)


def test_scanner_keeps_weakly_shifted_binary_bins(tmp_path):
    shifted = RAW * (1 + 1e-7)
    expected = {"xi": shifted}
    np.save(tmp_path / "sealed.npy", shifted)
    # At six printed digits these twins cannot be distinguished. This must not
    # hide their distinguishable binary values or falsely accuse safe text.
    np.savetxt(tmp_path / "rounded.txt", shifted, fmt="%.6e")
    assert_no_raw_values(tmp_path, {"xi": RAW}, expected=expected)
    np.save(tmp_path / "single_raw_bin.npy", RAW[1:2])
    with pytest.raises(AssertionError, match="single_raw_bin.npy"):
        assert_no_raw_values(tmp_path, {"xi": RAW}, expected=expected)


def test_scanner_catches_real_cosebis_log_precision(tmp_path):
    references = {"cosebis/(5, 100)/chi2_E": np.array([12.345, 1e-4])}
    expected = {"cosebis/(5, 100)/chi2_E": np.array([14.567, 2e-4])}
    # The second pair both print 0.00: no scalar text oracle can distinguish it.
    assert_no_raw_values(tmp_path, references, expected=expected, logs="chi2(E) = 0.00")
    assert_no_raw_values(
        tmp_path, references, expected=expected, logs="chi2(E) = 14.57"
    )
    with pytest.raises(AssertionError, match="two-decimal chi2_E"):
        assert_no_raw_values(
            tmp_path, references, expected=expected, logs="chi2(E) = 12.35"
        )


def test_scanner_refuses_unknown_binary_format(tmp_path):
    (tmp_path / "unexpected.cache").write_bytes(b"\xff\xfe\x00\x80")
    with pytest.raises(AssertionError, match="No leak decoder"):
        assert_no_raw_values(tmp_path, {"xi": RAW})


def test_scanner_judges_json_numbers_at_their_written_precision(tmp_path):
    near = RAW * (1 + 1e-8)  # a coincidence no binary leak could produce
    (tmp_path / "fit.json").write_text(json.dumps({"residual": near.tolist()}))
    assert_no_raw_values(tmp_path, {"xi": RAW})
    (tmp_path / "rounded.json").write_text(json.dumps([round(float(RAW[0]), 13)]))
    with pytest.raises(AssertionError, match="rounded JSON numbers"):
        assert_no_raw_values(tmp_path, {"xi": RAW})
