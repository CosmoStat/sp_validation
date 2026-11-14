"""IO.

:Name: io.py

:Description: Input and output: output directories and statistics files, and
    the catalogue reader (``read_catalogue``, ``Catalogue``), the one place
    that knows how a catalogue's rows are stored on disk.

:Author: Martin Kilbinger


"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pyarrow.parquet as pq
import tqdm
from astropy.io import fits

from sp_validation import grammar
from sp_validation.tables import (
    MAX_READ_THREADS,
    AlignedFile,
    ConcatTable,
    ParquetFile,
    as_table,
    group_dtype,
    parquet_dtype,
)


def make_out_dirs(output_dir, plot_dir, plot_subdirs, verbose=False):
    """Make output directories.

    Create output directories and subdirs

    Parameters
    ----------
    plot_dir: string
        directory name
    plot_subdirs: array of string
        subdirectory names
    verbose: bool, optional, default=False
        verbose output if True
    """
    for d in (output_dir, plot_dir):
        if not os.path.isdir(d):
            if verbose:
                print("Creating dir {}".format(d))
            os.mkdir(d)
    for sd in plot_subdirs:
        dsd = "{}/{}".format(plot_dir, sd)
        if not os.path.isdir(dsd):
            if verbose:
                print("Creating dir {}".format(dsd))
            os.mkdir(dsd)


def open_stats_file(directory, file_name):
    """Open statistics file.

    Open output file for statistics

    Parameters
    ----------
    directory : string
        directory
    file_name : string
        file name
    """
    stats_file = open("{}/{}".format(directory, file_name), "w")

    return stats_file


def print_stats(msg, stats_file, verbose=False):
    """Print stats.

    Print message to stats file.

    Parameters
    ----------
    msg : string
        message
    stats_file : file handler
        statistics output file
    verbose : bool, optional, default=False
        print message to stdout if True
    """
    stats_file.write(msg)
    stats_file.write("\n")
    stats_file.flush()

    if verbose:
        print(msg)


def print_ratio(msg, numerator, denominator, stats_file, verbose=False):
    """Print Ratio.

    pretty-print ratio of two numbers

    msg : string
        message
    numerator : float
        ratio numerator
    denominator : float
        ratio denominator
    stats_file : file handler
        output staistic file
    verbose : bool, optional, default=False
        verbose output if True
    """
    if denominator != 0:
        ratio = numerator / denominator * 100
    else:
        ratio = 0

    print_stats(
        f"{msg} = {numerator}/{denominator}" + f" = {ratio:.1f}%",
        stats_file,
        verbose=verbose,
    )


def write_binned_quantity(quantity, key, bin_edges, extra_key="quantity"):
    shape = quantity.shape
    nx, ny = shape[:2]

    filename = f"{key}_binned.npz"

    combined = {**bin_edges, extra_key: quantity}
    np.savez(filename, **combined)


def read_binned_quantity(filename):
    with np.load(filename) as data:
        return {key: data[key] for key in data.files}


# -- catalogues ---------------------------------------------------------------
#
# A catalogue is one table of objects, stored as row chunks. How the rows are
# stored (the container) is detected here from the path's contents, and each
# chunk handed to ``sp_validation.tables`` as a table; which names its columns
# carry is the column grammar's business (``sp_validation.grammar``). Every
# table this module hands out is presented through ``grammar.adapt``, which
# maps ShapePipe v1 names to v2 and passes any other naming convention through
# untouched, unless a ``column_map`` renames it.

#: Root attributes in which a chunked ShapePipe product declares its chunk
#: count, with the name of one chunk (for error messages).
N_UNITS_ATTRS = {"n_tiles": "tile", "n_exposures": "exposure"}


class Catalogue:
    """An open catalogue, presented as row chunks in the v2 grammar.

    The container is detected from the path's contents (a directory holding
    ``hats.properties``, or ``h5py.is_hdf5``), not from its extension.
    Supported layouts:

    - FITS: one table HDU, ``hdu`` or by default the first table HDU;
    - HDF5 comprehensive catalogue: a root ``data`` dataset, joined
      column-wise with ``data_ext`` when present;
    - HDF5 single dataset at the root, of any name;
    - HDF5 group of per-unit row chunks (one dataset per tile or exposure,
      as ShapePipe writes campaign galaxy and star catalogues), found by
      ``find_dataset_group``, concatenated in dataset-name order, and
      checked against a declared ``n_tiles`` or ``n_exposures`` count;
    - HATS catalogue directory, or a collection holding one: one chunk
      per HEALPix partition (named by its path, ``"Norder=5/Npix=1151"``), in
      ``partition_info.csv`` order. A collection's extensions join its
      primary column-wise, each pixel file read beside the primary's and
      checked against it on the join columns (``tables.AlignedFile``). Rows come in sky order, not the tile
      order of the HDF5 the catalogue was built from; a ``_source_row``
      column, where the builder wrote one, maps them back.

    Nothing but metadata is read on opening: ``chunks`` maps each chunk name
    to a lazy ``grammar.V2View``, and ``whole`` is one lazy view over them
    all (HATS partitions are read in parallel threads). Use as a
    context manager, or ``close`` when done; arrays read stay valid after
    closing.

    Parameters
    ----------
    path : str or os.PathLike
        catalogue file, or HATS catalogue directory
    hdu : int, optional
        FITS HDU to read; ignored for HDF5. Default is ``None``, the first
        table HDU
    column_map : dict, optional
        ``{v2 name: name in the file}`` for a catalogue in another naming
        convention, applied by ``grammar.adapt`` (patterns such as
        ``{"NGMIX_*": "MYFIT_*"}`` allowed). Default is ``None``
    """

    def __init__(self, path, hdu=None, column_map=None):
        self.path = os.fspath(path)
        self.hdu = None
        self._file = None
        if is_hats(self.path):
            parts, self.chunked = _hats_parts(self.path), True
        elif h5py.is_hdf5(self.path):
            self._file = h5py.File(self.path, "r")
            parts, self.chunked = _hdf5_parts(self._file, self.path)
        else:
            self.hdu, parts = _fits_parts(self.path, hdu)
            self.chunked = False
        parts = {key: tuple(map(as_table, tables)) for key, tables in parts.items()}
        self.chunks = {
            key: grammar.adapt(*tables, column_map=column_map)
            for key, tables in parts.items()
        }
        if len(parts) == 1:
            (self.whole,) = self.chunks.values()
        else:
            stacked = [ConcatTable(column) for column in zip(*parts.values())]
            self.whole = grammar.adapt(*stacked, column_map=column_map)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Close the underlying file; a no-op once closed, and for FITS."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __len__(self):
        return len(self.whole)

    def dtype(self, columns=None):
        """Return the structured dtype ``read(columns)`` produces.

        Each column's dtype is promoted across every chunk (``group_dtype``).

        Raises
        ------
        KeyError
            if a chunk lacks one of ``columns``, naming the chunk
        """
        if columns is not None:
            for key, view in self.chunks.items():
                _check_columns(view.dtype, columns, self.path, key)
        return group_dtype(self.chunks, columns)

    def table(self):
        """Return the catalogue as one table.

        That is ``whole``, lazy and valid while the file is open, except for
        a chunked catalogue of ``row_major`` tables (HDF5 tiles or
        exposures), which is read into memory: read a column at a time, it
        would re-read every chunk for each column.
        """
        if self.chunked and self.whole.row_major:
            return self.read()
        return self.whole

    def iter_chunks(self, columns=None, verbose=False):
        """Yield the row chunks one at a time, in memory, restricted to ``columns``.

        A caller filling a preallocated array (``len`` and ``dtype`` give its
        shape) so never holds more than one chunk on top of its output.
        """
        self.dtype(columns)
        for view in tqdm.tqdm(list(self.chunks.values()), disable=not verbose):
            yield view.to_structured(columns)

    def read(self, columns=None, key_column=None, verbose=False):
        """Read the catalogue into one numpy structured array.

        Chunks are read in parallel threads straight into a preallocated
        output, so peak memory is the output plus the chunks in flight.

        Parameters
        ----------
        columns : list of str, optional
            v2-grammar columns to read; default is ``None`` (all)
        key_column : str, optional
            for a chunked catalogue, name of an integer column to add, holding
            for each row the leading integer of its chunk's name. ShapePipe
            star catalogues name each chunk by its exposure number
            (``"2086324"``, or ``"2110000p"`` with the CFIS processed-exposure
            suffix), which this keeps (``key_column="EXPID"``). An unchunked
            catalogue has no chunk names and gets no column. Default is
            ``None``
        verbose : bool, optional
            report progress if ``True``

        Returns
        -------
        numpy.ndarray
            catalogue data in the v2 grammar

        Raises
        ------
        KeyError
            if a chunk lacks one of ``columns``
        ValueError
            if ``key_column`` collides with a column, or a chunk name does
            not begin with an integer
        """
        dtype_out = self.dtype(columns)
        names = list(dtype_out.names)
        add_key = key_column is not None and self.chunked
        if add_key:
            key_values = _chunk_numbers(self.chunks, key_column, dtype_out, self.path)
            dtype_out = np.dtype(dtype_out.descr + [(key_column, "i8")])

        n_rows = len(self)
        if verbose:
            print(
                f"Reading {len(self.chunks)} datasets,"
                + f" estimating {dtype_out.itemsize * n_rows / 1024**3:.1f}"
                + f" Gb memory for the ({len(dtype_out.names)} x {n_rows})"
                + " data array ..."
            )
        out = np.empty(n_rows, dtype=dtype_out)
        self.whole.read(names, out={name: out[name] for name in names})
        if add_key:
            start = 0
            for key, view in self.chunks.items():
                out[key_column][start : start + len(view)] = key_values[key]
                start += len(view)
        return out


def read_catalogue(
    path, columns=None, hdu=None, column_map=None, key_column=None, verbose=False
):
    """Read a catalogue into memory, in the v2 column grammar.

    Any container ``Catalogue`` supports; ``columns`` restricts what is read.
    A catalogue in another naming convention keeps its own column names,
    unless ``column_map`` renames them. See ``Catalogue`` and
    ``Catalogue.read`` for the parameters.

    Returns
    -------
    numpy.ndarray
        catalogue data
    """
    with Catalogue(path, hdu=hdu, column_map=column_map) as catalogue:
        return catalogue.read(columns=columns, key_column=key_column, verbose=verbose)


def open_catalogue(path, hdu=None, column_map=None):
    """Return a catalogue as one table in the v2 grammar (``Catalogue.table``).

    FITS, unchunked HDF5 and HATS catalogues open as a lazy ``grammar.V2View``
    (FITS memory-mapped; an HDF5 file stays open while the view is
    referenced), so a consumer pays only for the columns it indexes, derived
    v2 columns included. HDF5 row chunks are read into a numpy array.
    Parameters as for ``Catalogue``.
    """
    return Catalogue(path, hdu=hdu, column_map=column_map).table()


def validate_calibrated_entry(entry):
    """Validate response/weight identity, also for file-based reader adapters."""
    if "w_col" in entry and not h5py.is_hdf5(entry["path"]):
        with fits.open(entry["path"], memmap=True) as hdus:
            header = hdus[0].header
            if header.get("RAPPLIED", False):
                from sp_validation.response import validate_weight_scheme

                validate_weight_scheme(header["WGHTSCH"], entry["w_col"])
                if (
                    entry.get("R", 1) != 1
                    or entry.get("R11") is not None
                    or entry.get("R22") is not None
                ):
                    raise ValueError(
                        "Response already applied: cat_config R must be 1 "
                        "with no per-object response columns"
                    )


def open_entry(entry):
    """Read a config entry, checking calibrated-response/weight metadata."""
    validate_calibrated_entry(entry)
    return open_catalogue(
        entry["path"], hdu=entry.get("hdu"), column_map=entry.get("column_map")
    )


def _properties(path):
    """Return the ``key=value`` lines of a HATS properties file as a dict."""
    with open(path) as handle:
        return dict(
            line.strip().split("=", 1)
            for line in handle
            if "=" in line and not line.startswith("#")
        )


def hats_catalogue_dir(path):
    """Return the HATS catalogue directory at ``path``, or ``None``.

    ``path`` is a catalogue (holding ``hats.properties``) or a collection
    (holding ``collection.properties``, as ``lsdb`` writes by default), whose
    primary catalogue is returned.
    """
    path = os.fspath(path)
    collection = os.path.join(path, "collection.properties")
    if os.path.isfile(collection):
        path = os.path.join(path, _properties(collection)["hats_primary_table_url"])
    return path if os.path.isfile(os.path.join(path, "hats.properties")) else None


def is_hats(path):
    """Return whether ``path`` is a HATS catalogue or collection directory."""
    return hats_catalogue_dir(path) is not None


def _fits_parts(path, hdu):
    """Return (HDU, {HDU: (its table,)}) of a FITS file, memory-mapped."""
    with fits.open(path, memmap=True) as hdus:
        hdu = _first_table_hdu(hdus, path) if hdu is None else hdu
        return hdu, {str(hdu): (hdus[hdu].data,)}


def _hats_parts(path):
    """Return {partition: (its ParquetFile, ...)} of a HATS catalogue.

    For a collection with extensions (``all_extensions``), each partition's
    tuple holds the primary's file, then for each extension an ``AlignedFile``
    presenting the extension's own columns (``hats_ext_cols``) of that pixel,
    checked against the primary on the join columns.
    """
    primary = hats_catalogue_dir(path)
    pixels = np.loadtxt(
        os.path.join(primary, "partition_info.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.int64,
        ndmin=2,
    )
    keys = [f"Norder={order}/Npix={pixel}" for order, pixel in pixels]
    files = _hats_files(primary, pixels)
    tables = [files]
    collection = os.path.join(path, "collection.properties")
    if os.path.isfile(collection):
        listed = _properties(collection).get("all_extensions", "").split()
        for listing in listed[1::2]:
            listing = os.path.join(path, listing)
            extension = _properties(listing)
            data = hats_catalogue_dir(
                os.path.join(
                    os.path.dirname(listing), extension["hats_assn_join_table_url"]
                )
            )
            join = (extension["hats_col_assn_join"], extension["hats_col_assn_primary"])
            names = extension["hats_ext_cols"].split()
            tables.append(
                [
                    AlignedFile(file, of, join, names)
                    for file, of in zip(_hats_files(data, pixels), files, strict=True)
                ]
            )
    return dict(zip(keys, zip(*tables, strict=True), strict=True))


def _hats_files(path, pixels):
    """Return the ParquetFile of each of ``pixels`` of the HATS catalogue ``path``."""
    properties = _properties(os.path.join(path, "hats.properties"))
    suffix = properties.get("hats_npix_suffix", ".parquet")
    if suffix != ".parquet":
        raise ValueError(
            f"HATS catalogue {path}: partitions with suffix {suffix!r} are not"
            + " supported, only single .parquet files"
        )
    files = [
        os.path.join(
            path,
            "dataset",
            f"Norder={order}",
            f"Dir={pixel // 10000 * 10000}",
            f"Npix={pixel}.parquet",
        )
        for order, pixel in pixels
    ]
    common = os.path.join(path, "dataset", "_common_metadata")
    schema = pq.read_schema(common if os.path.isfile(common) else files[0])

    threads = min(MAX_READ_THREADS, len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(threads) as pool:
        footers = list(pool.map(pq.read_metadata, files))
    dtype = parquet_dtype(schema, footers)
    return [ParquetFile(file, footer, dtype) for file, footer in zip(files, footers)]


def _first_table_hdu(hdus, path):
    for index, hdu in enumerate(hdus):
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            return index
    raise ValueError(f"No table HDU in FITS file {path}")


def _hdf5_parts(hdf5_file, path):
    """Return ({chunk name: tuple of row-aligned datasets}, chunked) of a file."""
    data = hdf5_file.get("data")
    if isinstance(data, h5py.Dataset):
        parts = (data,) + ((hdf5_file["data_ext"],) if "data_ext" in hdf5_file else ())
        return {"data": parts}, False
    group = find_dataset_group(hdf5_file)
    check_n_units(hdf5_file, group, path)
    chunked = group.name != "/" or len(group) > 1
    return {key: (group[key],) for key in sorted(group)}, chunked


def _chunk_numbers(keys, key_column, dtype, path):
    """Return {chunk name: its leading integer}, for ``Catalogue.read``."""
    if key_column in (dtype.names or ()):
        raise ValueError(
            f"Cannot add column {key_column!r} to catalogue {path}:"
            + " a column of that name is already present."
        )
    matches = {key: re.match(r"\d+", key) for key in keys}
    bad = [key for key, match in matches.items() if match is None]
    if bad:
        raise ValueError(
            f"Cannot derive {key_column!r} for catalogue {path}:"
            + f" dataset name(s) {bad} do not begin with an integer."
        )
    return {key: int(match.group()) for key, match in matches.items()}


def _check_columns(dtype, columns, path, chunk=None):
    """Raise a clear error if requested columns are absent from a chunk."""
    missing = [col for col in columns if col not in (dtype.names or ())]
    if missing:
        where = f" (dataset {chunk!r})" if chunk is not None else ""
        raise KeyError(
            f"Column(s) {missing} not found in catalogue {path}{where}."
            + f" Available columns: {sorted(dtype.names or ())}"
        )


def find_dataset_group(hdf5_file):
    """Find Dataset Group.

    Descend from the root of an open HDF5 file to the single group whose
    members are the per-unit datasets (one per tile, or one per exposure).

    This makes the reader independent of how deeply the products nest that
    group: it walks down as long as the current node holds exactly one
    sub-group, and stops as soon as the members are datasets. It therefore
    reads both the nested ``patches/<campaign>/<tile-ID>`` layout (the
    "patches" key is a ShapePipe-side compatibility shim, not a concept) and
    a flat ``tiles/<tile-ID>`` or ``exposures/<exp>`` layout.

    Parameters
    ----------
    hdf5_file : h5py.File or h5py.Group
        open input file

    Returns
    -------
    h5py.Group
        group whose members are the per-unit datasets

    Raises
    ------
    ValueError
        if the file is empty, or a level holds more than one sub-group

    """
    node = hdf5_file
    while True:
        keys = list(node)
        if not keys:
            raise ValueError(
                f"No data found under {node.name!r} in {hdf5_file.file.filename}"
            )
        if all(isinstance(node[key], h5py.Dataset) for key in keys):
            return node
        if len(keys) != 1:
            raise ValueError(
                f"Expected a single container group under {node.name!r} in"
                + f" {hdf5_file.file.filename}, found {len(keys)}: {keys[:5]}"
            )
        node = node[keys[0]]


def check_n_units(hdf5_file, group, file_path):
    """Check Number of Units.

    Compare the number of datasets found against the count a ShapePipe
    product declares in a root attribute (``N_UNITS_ATTRS``: ``n_tiles`` for
    galaxy catalogues, ``n_exposures`` for star catalogues), to catch a
    catalogue truncated by an interrupted merge job or file transfer.

    A missing attribute is not an error: older products carry none.

    Parameters
    ----------
    hdf5_file : h5py.File
        open input file
    group : h5py.Group
        group holding the per-unit datasets
    file_path : str
        input file path, for the error message

    Raises
    ------
    ValueError
        if the number of datasets differs from the declared count

    """
    for attr, unit in N_UNITS_ATTRS.items():
        n_declared = hdf5_file.attrs.get(attr)
        if n_declared is None:
            continue
        n_found = len(group)
        if int(n_declared) != n_found:
            raise ValueError(
                f"Catalogue {file_path} declares {attr} = {int(n_declared)} but"
                + f" holds {n_found} {unit} dataset(s); the file is incomplete."
            )
