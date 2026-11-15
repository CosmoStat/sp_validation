"""A recipe builds shapes, PhotoPipe and spec-z into one valid HATS collection."""

import json
import os

import fitsio
import lsdb
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from hats.catalog import CollectionProperties, TableProperties
from hats.io.validation import is_valid_catalog, is_valid_collection

from sp_validation import io
from sp_validation.assembly.hats_collection import build
from sp_validation.assembly.specz_merge import MATCH_COLUMNS

N = 1000


def write_inputs(directory, rng):
    """Write a recipe's inputs: shapes on one tile, PhotoPipe's extract of that
    tile (no row for the last object), and spectra for the objects north of
    Dec 0, all listed in one compilation file."""
    ra = rng.uniform(0, 360, N)
    dec = np.degrees(np.arcsin(rng.uniform(-1, 1, N)))
    number = np.arange(1, N + 1, dtype=np.int32)
    frame = pd.DataFrame(
        dict(
            XWIN_WORLD=ra,
            YWIN_WORLD=dec,
            TILE_ID=np.full(N, 24.237),
            NUMBER=number,
            G1=rng.normal(0, 0.3, N),
        )
    )
    lsdb.from_dataframe(
        frame, ra_column="XWIN_WORLD", dec_column="YWIN_WORLD", partition_rows=300
    ).write_catalog(directory / "source", catalog_name="source", progress_bar=False)

    extract = directory / "extract"
    extract.mkdir()
    photo = pa.table(
        dict(
            SeqNr=number[:-1],
            ALPHA_J2000=ra[:-1] + 1e-5,
            DELTA_J2000=dec[:-1] - 1e-5,
            MAG_GAAP_r=rng.uniform(18, 25, N - 1).astype(np.float32),
            MASK=pa.nulls(N - 1, pa.int32()),
        )
    )
    pq.write_table(photo, extract / "ra=024.parquet")
    record = dict(
        status="ok",
        rows=N - 1,
        row_group=0,
        read_at=0.0,
        sources=dict(ext=None, finalmask=None),
    )
    (extract / "ra=024.json").write_text(json.dumps(dict(tiles={"024.237": record})))

    north = np.flatnonzero(dec[:-1] > 0)
    matches = np.zeros(len(north), dtype=[(c, "f8") for c in MATCH_COLUMNS])
    matches["Pair_0_spec"] = number[north]
    matches["ALPHA_J2000"] = photo["ALPHA_J2000"].to_numpy()[north]
    matches["DELTA_J2000"] = photo["DELTA_J2000"].to_numpy()[north]
    matches["N_0"] = matches["N_1"] = 1
    matches["ALPHA_J2000_spec"] = ra[north] + 1e-4
    matches["DELTA_J2000_spec"] = dec[north]
    matches["z_spec_spec"] = rng.uniform(0, 2, len(north))
    dtype = [
        (c, "i4" if c in ("Pair_0_spec", "N_0", "N_1") else "f8") for c in MATCH_COLUMNS
    ]
    fitsio.write(str(directory / "specz.cat"), matches.astype(dtype))
    (directory / "asc").mkdir()
    lines = [
        f"{i} {r!r} {d!r} {z!r} 0.0001 0.95 deep2_final"
        for i, (r, d, z) in enumerate(
            zip(
                matches["ALPHA_J2000_spec"].tolist(),
                matches["DELTA_J2000_spec"].tolist(),
                matches["z_spec_spec"].tolist(),
            )
        )
    ]
    text = " ra dec zspec zspec_err wspec ref\n" + "\n".join(lines) + "\n"
    (directory / "asc" / "UNIONS.024.237_redshifts-20261019.asc").write_text(text)

    recipe = directory / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                shapes=str(directory / "source" / "source"),
                photopipe_extract=str(extract),
                specz_matches=str(directory / "specz.cat"),
                specz_compilation=str(directory / "asc"),
                collection=str(directory / "coll"),
            )
        )
    )
    return recipe, frame, photo, north, matches


def test_build_writes_the_collection_a_recipe_describes(tmp_path):
    recipe, frame, photo, north, matches = write_inputs(
        tmp_path, np.random.default_rng(4)
    )
    collection = tmp_path / "coll"

    build(recipe, workers=2)

    assert is_valid_collection(collection, strict=True)
    assert is_valid_catalog(collection / "specz" / "specz", strict=True)
    properties = CollectionProperties.read_from_dir(collection)
    assert properties.default_index == "OBJECT_ID"
    # The index names its primary relative to the collection, so a copy still reads.
    index = TableProperties.read_from_dir(collection / "shapes_OBJECT_ID")
    assert index.primary_catalog == "shapes"
    assert properties.all_extensions == {
        "photopipe": "photopipe.properties",
        "specz": "specz.properties",
    }

    out = io.read_catalogue(
        collection,
        columns=["NUMBER", "G1", "PHOTOPIPE_STATUS", "MAG_GAAP_r", "Z_SPEC"],
    )
    row = out["NUMBER"] - 1
    npt.assert_array_equal(out["G1"], frame["G1"].to_numpy()[row])
    expected = np.append(photo["MAG_GAAP_r"].to_numpy(), np.nan)[row]
    npt.assert_array_equal(out["MAG_GAAP_r"], expected)
    npt.assert_array_equal(out["PHOTOPIPE_STATUS"], np.where(row == N - 1, 1, 0))
    z = np.full(N, np.nan)
    z[north] = matches["z_spec_spec"]
    npt.assert_array_equal(out["Z_SPEC"], z[row])

    found = (
        lsdb.open_catalog(collection, columns=["OBJECT_ID", "G1"])
        .id_search(values={"OBJECT_ID": [24237000007]})
        .compute()
    )
    npt.assert_array_equal(found["G1"], frame["G1"].to_numpy()[[6]])
    joined = (
        lsdb.open_catalog(collection, columns=["OBJECT_ID"])
        .load_extension("specz", columns=["Z_SPEC"])
        .compute()
    )
    assert len(joined) == N and joined["Z_SPEC"].notna().sum() == len(north)

    # A second build keeps every part on disk.
    files = sorted(collection.rglob("Npix=*.parquet"))
    times = [os.stat(file).st_mtime_ns for file in files]
    build(recipe)
    assert [os.stat(file).st_mtime_ns for file in files] == times
