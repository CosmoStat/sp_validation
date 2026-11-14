"""Give shape objects their spectroscopic redshift: the spec-z extension of the v2 collection.

The input is PhotoPipe's match of DR6 objects to spectra on every tile, one
FITS table (``UNIONS_DR6_ugriz_specz.cat``): one row per match, holding the
object's PhotoPipe columns and the spectrum's (``ALPHA_J2000_spec``,
``DELTA_J2000_spec``, ``z_spec_spec``, ``z_spec_err_spec``), and the redshift
compilation the match used, one ``UNIONS.<tile>_redshifts-<date>.asc`` per
tile (``ra dec zspec zspec_err wspec ref``), which names each spectrum's survey
(``ref``) and confidence (``wspec``).

The table has no tile column, and its ``SeqNr`` renumbers the rows; the
object's DR6 number in its tile is ``Pair_0_spec``. A row is a PhotoPipe row,
so it is found in the collection's PhotoPipe extension by that number and its
exact position (``PHOTOPIPE_SEQNR``, ``ALPHA_J2000``, ``DELTA_J2000``), which
gives its ``OBJECT_ID`` and pixel. A row that pairs several DR6 objects
(``N_0`` > 1) or several spectra (``N_1`` > 1) averages their positions,
photometry or redshifts, and is left out. Rows whose object is not a shape row
have no place in the collection. A spectrum is found in the compilation by its
exact position; one the compilation does not list has no survey or confidence.

The output, ``COLLECTION/specz/specz``, is a HATS catalogue on the shapes' pixels with
one row for each shape object that has a spectrum, in the shapes' row order:
``_healpix_29``, ``OBJECT_ID``, the shape coordinates, ``Z_SPEC``,
``Z_SPEC_ERR``, ``Z_SPEC_CONFIDENCE`` (``wspec``), ``Z_SPEC_SURVEY`` (``ref``),
and the spectrum's position ``Z_SPEC_RA``, ``Z_SPEC_DEC``, which identifies it:
a spectrum matched to several objects is on each of them. It is the data of the
collection's ``specz`` extension (``scripts/assembly/build_collection.py``),
joined to the shapes on ``OBJECT_ID``; pixels without a spectrum have no file.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import fitsio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hats.catalog import PartitionInfo, TableProperties
from hats.io.parquet_metadata import write_parquet_metadata
from hats.io.paths import get_healpix_from_path

from sp_validation.assembly.photopipe_merge import KEYS
from sp_validation.assembly.photopipe_merge import NAME as PHOTOPIPE

NAME = "specz"
#: Columns of the matched table that identify the object and give the spectrum.
MATCH_COLUMNS = [
    "Pair_0_spec",
    "ALPHA_J2000",
    "DELTA_J2000",
    "N_0",
    "N_1",
    "ALPHA_J2000_spec",
    "DELTA_J2000_spec",
    "z_spec_spec",
    "z_spec_err_spec",
]
PHOTO_COLUMNS = [*KEYS, "PHOTOPIPE_SEQNR", "ALPHA_J2000", "DELTA_J2000"]


def _read_compilation_file(path):
    return pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        header=None,
        usecols=[1, 2, 5, 6],
        names=["ra", "dec", "confidence", "survey"],
    )


def read_spectra(matches, compilation, workers=1):
    """Return the matched table's one-object, one-spectrum rows, and how many it has
    of the others.

    ``matches`` is the FITS table and ``compilation`` the directory of
    ``*_redshifts-*.asc`` files, which gives each spectrum its survey and
    confidence (null where it does not list it). Rows are sorted by object
    right ascension, for ``match_pixel``.
    """
    data = fitsio.read(matches, ext=1, columns=MATCH_COLUMNS)
    spectra = pd.DataFrame(
        {
            name: data[name].astype(data[name].dtype.newbyteorder("="))
            for name in MATCH_COLUMNS
        }
    )
    single = (spectra["N_0"] == 1) & (spectra["N_1"] == 1)
    spectra = spectra[single].drop(columns=["N_0", "N_1"])
    paths = sorted(Path(compilation).glob("*_redshifts-*.asc"))
    with ProcessPoolExecutor(workers) as pool:
        listed = pd.concat(pool.map(_read_compilation_file, paths, chunksize=64))
    listed = listed.drop_duplicates(["ra", "dec"])
    spectra = spectra.merge(
        listed,
        how="left",
        left_on=["ALPHA_J2000_spec", "DELTA_J2000_spec"],
        right_on=["ra", "dec"],
    ).drop(columns=["ra", "dec"])
    spectra["survey"] = (
        spectra["survey"].astype(object).where(spectra["survey"].notna(), None)
    )
    spectra = spectra.sort_values("ALPHA_J2000", kind="stable", ignore_index=True)
    return spectra, int((~single).sum())


def match_pixel(photo, spectra):
    """Return (photo rows, spectrum rows) of the spectra whose object is in ``photo``.

    ``photo`` holds one PhotoPipe pixel file's ``PHOTOPIPE_SEQNR``,
    ``ALPHA_J2000`` and ``DELTA_J2000``; ``spectra`` is ``read_spectra``'s
    table. An object matches when its DR6 number and position are those of the
    spectrum's row exactly.
    """
    ra = photo["ALPHA_J2000"].to_numpy(zero_copy_only=False)
    dec = photo["DELTA_J2000"].to_numpy(zero_copy_only=False)
    number = photo["PHOTOPIPE_SEQNR"].to_numpy(zero_copy_only=False)
    spectrum_ra = spectra["ALPHA_J2000"].to_numpy()
    start = np.searchsorted(spectrum_ra, ra, side="left")
    stop = np.searchsorted(spectrum_ra, ra, side="right")
    rows, others = [], []
    candidates = np.flatnonzero(stop > start)
    for k in range(int((stop - start).max(initial=0))):
        candidates = candidates[start[candidates] + k < stop[candidates]]
        other = start[candidates] + k
        same = (spectra["Pair_0_spec"].to_numpy()[other] == number[candidates]) & (
            spectra["DELTA_J2000"].to_numpy()[other] == dec[candidates]
        )
        rows.append(candidates[same])
        others.append(other[same])
    rows, others = np.concatenate([[], *rows]), np.concatenate([[], *others])
    order = np.argsort(rows, kind="stable")
    return rows[order].astype(np.int64), others[order].astype(np.int64)


_SPECTRA = {}


def merge_pixel(photo_path, output_path):
    """Write one pixel's spec-z rows; return its pixel, their count and spectrum rows."""
    spectra = _SPECTRA["table"]
    photo = pq.read_table(photo_path, columns=PHOTO_COLUMNS)
    rows, matched = match_pixel(photo, spectra)
    if not len(rows):
        return None
    table = photo.select(KEYS).take(rows)
    columns = dict(
        Z_SPEC="z_spec_spec",
        Z_SPEC_ERR="z_spec_err_spec",
        Z_SPEC_CONFIDENCE="confidence",
        Z_SPEC_SURVEY="survey",
        Z_SPEC_RA="ALPHA_J2000_spec",
        Z_SPEC_DEC="DELTA_J2000_spec",
    )
    for name, source in columns.items():
        values = spectra[source].to_numpy()[matched]
        if name == "Z_SPEC_CONFIDENCE":
            values = values.astype(np.float32)
        table = table.append_column(name, pa.array(values))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd", compression_level=3)
    return get_healpix_from_path(str(output_path)), len(rows), matched


def _merge_worker(arguments):
    return merge_pixel(*arguments)


def merge(collection, matches, compilation, workers=1):
    """Build ``COLLECTION/specz``: the catalogue ``specz`` and its ``BUILD.json``.

    Needs the collection's PhotoPipe extension, ``COLLECTION/photopipe``.
    """
    photopipe = Path(collection) / PHOTOPIPE / PHOTOPIPE
    output = Path(collection) / NAME
    catalogue = output / NAME
    spectra, n_merged = read_spectra(matches, compilation, workers)
    _SPECTRA["table"] = spectra
    paths = sorted((photopipe / "dataset").glob("Norder=*/Dir=*/Npix=*.parquet"))
    jobs = [(path, catalogue / path.relative_to(photopipe)) for path in paths]
    with ProcessPoolExecutor(workers, mp_context=get_context("fork")) as pool:
        written = [r for r in pool.map(_merge_worker, jobs, chunksize=4) if r]
    _SPECTRA.clear()

    PartitionInfo.from_healpix([pixel for pixel, _, _ in written]).write_to_file(
        catalogue / "partition_info.csv"
    )
    rows = write_parquet_metadata(catalogue)
    coordinates = TableProperties.read_from_dir(photopipe)
    TableProperties(
        catalog_name=NAME,
        catalog_type="object",
        total_rows=rows,
        ra_column=coordinates.ra_column,
        dec_column=coordinates.dec_column,
        healpix_column=coordinates.healpix_column,
        healpix_order=coordinates.healpix_order,
        hats_max_rows=max(n for _, n, _ in written),
        **TableProperties.new_provenance_dict(
            catalogue, builder="sp_validation.assembly.specz_merge"
        ),
    ).to_properties_file(catalogue)

    used = np.concatenate([m for _, _, m in written])
    surveys = spectra["survey"].iloc[used].fillna("(not in the compilation)")
    build = dict(
        built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        matches=str(matches),
        compilation=str(compilation),
        photopipe=str(photopipe),
        join="(PHOTOPIPE_SEQNR, ALPHA_J2000, DELTA_J2000) = (Pair_0_spec, ALPHA_J2000, DELTA_J2000)",
        n_matches=len(spectra) + n_merged,
        n_merged=n_merged,
        n_rows=rows,
        n_spectra=len(
            spectra.iloc[used][
                ["ALPHA_J2000_spec", "DELTA_J2000_spec"]
            ].drop_duplicates()
        ),
        n_pixels=len(written),
        rows_by_survey=surveys.value_counts().to_dict(),
    )
    (output / "BUILD.json").write_text(json.dumps(build, indent=2) + "\n")
    print(json.dumps(build, indent=2))
    return build
