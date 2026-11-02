import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sp_validation import nz_photometry


def test_prepare_matches_by_position_and_preserves_cache_layout(tmp_path, monkeypatch):
    selection = tmp_path / "selection.parquet"
    pq.write_table(
        pa.table(
            {
                "TILE_ID": ["001.001", "002.001", "001.001"],
                "NUMBER": [70, 20, 10],
                "RA": [10.0 + 0.2 / 3600, 30.0, 10.0 + 0.7 / 3600],
                "Dec": [2.0, 0.0, 2.0],
                "w_des": [1.1, 1.2, 1.3],
                "FLAGS": [0, 1, 2],
            }
        ),
        selection,
    )
    dr6_root = tmp_path / "dr6"
    dr6_root.mkdir()
    (dr6_root / "UNIONS.001.001_SP_ugriz_photoz_ext.cat").touch()

    def read_tile(path):
        return (
            np.array([10.0]),
            np.array([2.0]),
            np.array([[-99, 20, 21, 22, 23, 99]], dtype="f4"),
            np.array([0.76], dtype="f8"),
            np.array([13], dtype="i1"),
        )

    monkeypatch.setattr(nz_photometry, "_read_dr6_tile", read_tile)
    output = tmp_path / "matched_full.npz"
    nz_photometry.prepare_nz_photometry(selection, dr6_root, output)

    with np.load(output) as cache:
        np.testing.assert_array_equal(cache["tile_names"], ["001.001", "002.001"])
        np.testing.assert_array_equal(cache["number"], [70, 20, 10])
        np.testing.assert_array_equal(cache["match"], [True, False, True])
        np.testing.assert_array_equal(cache["photo_index"], [0, -1, 0])
        expected_sep = np.array([0.2, 0.7]) * np.cos(np.radians(2.0))
        np.testing.assert_allclose(cache["sep"][[0, 2]], expected_sep, atol=1e-5)
        assert cache["zb"].dtype == np.dtype("f8")
        assert cache["mag"].dtype == np.dtype("f4")
        assert cache["zb"][0] == 0.76
        np.testing.assert_array_equal(cache["mag"][0], [-99, 20, 21, 22, 23, 99])
        np.testing.assert_array_equal(cache["tile_has_dr6"], [True, False])
