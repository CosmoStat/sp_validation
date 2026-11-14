import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sp_validation import photopipe_merge as pm


def write_extract(directory):
    table = pa.table(
        dict(
            SeqNr=np.array([1, 2], np.int32),
            ALPHA_J2000=np.array([10.0, 10.01]),
            DELTA_J2000=np.array([20.0, 20.0]),
            MAG_GAAP_r=np.array([21.0, 22.0], np.float32),
            MASK=pa.nulls(2, pa.int32()),
        )
    )
    pq.write_table(table, directory / "ra=024.parquet")
    record = dict(
        status="ok",
        rows=2,
        row_group=0,
        read_at=0.0,
        sources=dict(ext=None, finalmask=None),
    )
    (directory / "ra=024.json").write_text(json.dumps(dict(tiles={"024.237": record})))


def write_shapes(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            dict(
                _healpix_29=np.arange(4),
                XWIN_WORLD=np.array([10.01, 10.0, 10.01, 10.0]),
                YWIN_WORLD=np.array([20.0, 20.0 + 0.2 / 3600, 20.0, 20.0]),
                TILE_ID=np.array([24.237, 24.237, 24.237, 300.25]),
                NUMBER=np.array([2, 1, 9, 1], np.int32),
                _source_row=np.array([3, 2, 1, 0]),
            )
        ),
        path,
    )


def test_merge_pixel_joins_on_id_and_marks_no_match(tmp_path):
    write_extract(tmp_path)
    shape_path = tmp_path / "shapes.parquet"
    write_shapes(shape_path)
    index = pm.load_extract(tmp_path)
    output = tmp_path / "out" / "pixel.parquet"
    counts, _ = pm.merge_pixel(shape_path, output, index, pm._photopipe_schema(index))
    merged = pq.read_table(output)

    assert merged["_source_row"].to_pylist() == [3, 2, 1, 0]
    assert merged["OBJECT_ID"].to_pylist() == [
        24237000002,
        24237000001,
        24237000009,
        300250000001,
    ]
    assert merged["PHOTOPIPE_STATUS"].to_pylist() == [
        pm.MATCHED,
        pm.MATCHED,
        pm.NO_COUNTERPART,
        pm.NO_TILE,
    ]
    assert merged["PHOTOPIPE_SEQNR"].to_pylist() == [2, 1, None, None]
    assert merged["MAG_GAAP_r"].to_pylist() == [22.0, 21.0, None, None]
    assert merged["MATCH_SEP_ARCSEC"].to_pylist()[2:] == [None, None]
    assert abs(merged["MATCH_SEP_ARCSEC"][1].as_py() - 0.2) < 1e-3
    assert "ALPHA_J2000" not in merged.column_names
    assert sorted((c["tile"], c["n_matched"]) for c in counts) == [
        ("024.237", 2),
        ("300.250", 0),
    ]


def test_merge_pixel_counts_pairs_whose_positions_disagree(tmp_path):
    write_extract(tmp_path)
    shape_path = tmp_path / "shapes.parquet"
    write_shapes(shape_path)
    table = pq.read_table(shape_path).drop_columns("XWIN_WORLD")
    table = table.append_column(
        "XWIN_WORLD", pa.array(np.array([10.0, 10.01, 10.0, 10.0]))
    )
    pq.write_table(table, shape_path)
    index = pm.load_extract(tmp_path)
    output = tmp_path / "out" / "pixel.parquet"
    counts, _ = pm.merge_pixel(shape_path, output, index, pm._photopipe_schema(index))
    assert [c["n_far"] for c in counts if c["tile"] == "024.237"] == [2]
