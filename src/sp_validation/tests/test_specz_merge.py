"""The spec-z join finds each matched object in the PhotoPipe extension."""

import json

import fitsio
import lsdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from hats.catalog import PartitionInfo
from hats.io.validation import is_valid_catalog

from sp_validation.assembly import specz_merge as sm
from sp_validation.assembly.specz_merge import MATCH_COLUMNS

N = 600


def write_photopipe(collection, rng):
    """A PhotoPipe extension catalogue; objects with Dec < -30 have no PhotoPipe row."""
    ra = rng.uniform(0, 360, N)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, N)))
    number = np.arange(1, N + 1, dtype=np.float64)
    number[dec < -30] = np.nan
    frame = pd.DataFrame(
        dict(
            OBJECT_ID=24237 * 10**6 + np.arange(1, N + 1),
            XWIN_WORLD=ra,
            YWIN_WORLD=dec,
            PHOTOPIPE_SEQNR=number,
            ALPHA_J2000=np.where(np.isnan(number), np.nan, ra + 1e-5),
            DELTA_J2000=np.where(np.isnan(number), np.nan, dec - 1e-5),
        )
    )
    lsdb.from_dataframe(
        frame, ra_column="XWIN_WORLD", dec_column="YWIN_WORLD", partition_rows=150
    ).write_catalog(
        collection / "photopipe", catalog_name="photopipe", progress_bar=False
    )
    return frame


def write_matches(path, photo, rng):
    """The matched table: one row per (object, spectrum), for objects with Dec > 0.

    ``SeqNr`` is renumbered; the object's number is ``Pair_0_spec``. The last
    rows are a merged pair of objects, a merged pair of spectra, and an object
    with no PhotoPipe row.
    """
    north = photo[(photo["YWIN_WORLD"] > 0) & photo["PHOTOPIPE_SEQNR"].notna()]
    south = photo[photo["PHOTOPIPE_SEQNR"].isna()].iloc[:1]
    rows = pd.concat([north, north.iloc[:2], south])
    n = len(rows)
    table = np.zeros(n, dtype=[(c, "f8") for c in MATCH_COLUMNS] + [("SeqNr", "i4")])
    table["Pair_0_spec"] = np.nan_to_num(rows["PHOTOPIPE_SEQNR"], nan=10**5)
    table["SeqNr"] = np.arange(n)
    table["ALPHA_J2000"] = rows["ALPHA_J2000"].fillna(0)
    table["DELTA_J2000"] = rows["DELTA_J2000"].fillna(0)
    table["N_0"] = table["N_1"] = 1
    table["N_0"][-3], table["N_1"][-2] = 2, 2
    table["ALPHA_J2000_spec"] = rows["XWIN_WORLD"] + rng.uniform(-1e-4, 1e-4, n)
    table["DELTA_J2000_spec"] = rows["YWIN_WORLD"] + rng.uniform(-1e-4, 1e-4, n)
    table["z_spec_spec"] = rng.uniform(0, 2, n)
    table["z_spec_err_spec"] = 1e-4
    dtype = [
        (c, "i4" if c in ("Pair_0_spec", "N_0", "N_1", "SeqNr") else "f8")
        for c in table.dtype.names
    ]
    fitsio.write(str(path), table.astype(dtype), extname="OBJECTS")
    return table


def write_compilation(directory, table):
    """Two tiles' compilations, overlapping; the first spectrum is in neither."""
    directory.mkdir()
    lines = [
        f"{i} {float(ra)!r} {float(dec)!r} {float(z)!r} 0.0001 0.95 survey{i % 2}"
        for i, (ra, dec, z) in enumerate(
            zip(
                table["ALPHA_J2000_spec"],
                table["DELTA_J2000_spec"],
                table["z_spec_spec"],
            )
        )
    ][1:]
    half = len(lines) // 2
    for tile, part in (("024.237", lines[: half + 5]), ("024.238", lines[half:])):
        text = " ra dec zspec zspec_err wspec ref\n" + "\n".join(part) + "\n"
        (directory / f"UNIONS.{tile}_redshifts-20261019.asc").write_text(text)


def test_merge_writes_one_row_per_matched_shape_object(tmp_path):
    rng = np.random.default_rng(5)
    collection = tmp_path / "coll"
    photo = write_photopipe(collection, rng)
    table = write_matches(tmp_path / "specz.cat", photo, rng)
    write_compilation(tmp_path / "asc", table)

    build = sm.merge(collection, tmp_path / "specz.cat", tmp_path / "asc", workers=2)

    catalogue = collection / "specz" / "specz"
    assert is_valid_catalog(catalogue, strict=True)
    out = pq.read_table(catalogue / "dataset").to_pandas()
    # Every single-object, single-spectrum row of a PhotoPipe object, and no other.
    expected = table[:-3]
    assert len(out) == len(expected) == build["n_rows"]
    assert build["n_merged"] == 2 and build["n_matches"] == len(table)
    ids = 24237 * 10**6 + expected["Pair_0_spec"]
    out = out.set_index("OBJECT_ID").loc[ids]
    np.testing.assert_array_equal(out["Z_SPEC"], expected["z_spec_spec"])
    np.testing.assert_array_equal(out["Z_SPEC_RA"], expected["ALPHA_J2000_spec"])
    shapes = photo.set_index("OBJECT_ID").loc[ids]
    np.testing.assert_array_equal(out["XWIN_WORLD"], shapes["XWIN_WORLD"])
    # The survey comes from the compilation; a spectrum it does not list has none.
    survey = out["Z_SPEC_SURVEY"]
    assert survey.isna().tolist() == [True] + [False] * (len(expected) - 1)
    assert survey.iloc[1:].tolist() == [
        f"survey{i % 2}" for i in range(1, len(expected))
    ]
    assert np.isnan(out["Z_SPEC_CONFIDENCE"].iloc[0])
    assert (out["Z_SPEC_CONFIDENCE"].iloc[1:] == np.float32(0.95)).all()
    # Pixels without a spectrum have no file; the rest keep the shapes' row order.
    pixels = PartitionInfo.read_from_dir(catalogue).get_healpix_pixels()
    assert len(pixels) == build["n_pixels"]
    assert len(pixels) < len(
        PartitionInfo.read_from_dir(
            collection / "photopipe" / "photopipe"
        ).get_healpix_pixels()
    )
    photopipe = collection / "photopipe" / "photopipe"
    for file in (catalogue / "dataset").rglob("Npix=*.parquet"):
        order = pq.read_table(photopipe / file.relative_to(catalogue))["OBJECT_ID"]
        rows = pd.Index(order).get_indexer(pq.read_table(file)["OBJECT_ID"])
        assert np.all(np.diff(rows) > 0)
    assert json.loads((collection / "specz" / "BUILD.json").read_text())[
        "n_rows"
    ] == len(expected)
