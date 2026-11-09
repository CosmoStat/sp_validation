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
# A catalogue is one table of objects. How its rows are stored (the container)
# is detected here from the path's contents; which names its columns carry is
# the column grammar's business (``sp_validation.grammar``). Every table this
# module hands out is presented through ``grammar.adapt``, which maps ShapePipe
# v1 names to v2 and passes any other naming convention through untouched,
# unless a ``column_map`` renames it.

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
    - HATS catalogue directory: one chunk per HEALPix partition (named by
      its path, ``"Norder=5/Npix=1151"``), in ``partition_info.csv`` order,
      each a ``ParquetTable``; ``whole`` spans them all. Rows come in sky
      order, not the tile order of the HDF5 the catalogue was built from;
      a ``_source_row`` column, where the builder wrote one, maps them back.

    Nothing but metadata is read on opening: ``chunks`` maps each chunk name
    to a lazy table, and ``whole`` is one lazy table over the catalogue, or
    ``None`` for HDF5 row chunks, which have no single on-disk table. Use as
    a context manager, or ``close`` when done; tables read into memory stay
    valid after closing.

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
        self.column_map = column_map
        self._file = None
        self.hdu = None
        whole = None
        if is_hats(self.path):
            whole, self._parts = _hats_parts(self.path)
            self.chunked = True
        elif h5py.is_hdf5(self.path):
            self._file = h5py.File(self.path, "r")
            self._parts, self.chunked = _hdf5_parts(self._file, self.path)
        else:
            with fits.open(self.path, memmap=True) as hdus:
                self.hdu = _first_table_hdu(hdus, self.path) if hdu is None else hdu
                self._parts = {str(self.hdu): (hdus[self.hdu].data,)}
            self.chunked = False
        self.chunks = {
            key: grammar.adapt(*parts, column_map=column_map)
            for key, parts in self._parts.items()
        }
        if whole is not None:
            self.whole = grammar.adapt(whole, column_map=column_map)
        elif len(self.chunks) == 1:
            self.whole = next(iter(self.chunks.values()))
        else:
            self.whole = None

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
        return sum(len(table) for table in self.chunks.values())

    def dtype(self, columns=None):
        """Return the structured dtype ``read(columns)`` produces.

        Each column's dtype is promoted across every chunk (``group_dtype``);
        a catalogue with one lazy table (``whole``) has one schema.

        Raises
        ------
        KeyError
            if a chunk lacks one of ``columns``, naming the chunk
        """
        tables = self.chunks if self.whole is None else {"catalogue": self.whole}
        if columns is not None:
            for key, table in tables.items():
                _check_columns(table.dtype, columns, self.path, key)
        return group_dtype(tables, columns)

    def table(self):
        """Return the catalogue as one table, lazily where one exists.

        ``whole`` (a FITS_rec, an h5py Dataset, a ``ParquetTable`` or a
        ``grammar.V2View``) is returned as is, valid while the file is open
        (a ``ParquetTable`` always); HDF5 row chunks are read into memory
        (``read``).
        """
        return self.read() if self.whole is None else self.whole

    def iter_chunks(self, columns=None, verbose=False):
        """Yield the row chunks one at a time, in memory, restricted to ``columns``.

        A caller filling a preallocated array (``len`` and ``dtype`` give its
        shape) so never holds more than one chunk on top of its output.
        """
        self.dtype(columns)
        for key in tqdm.tqdm(list(self.chunks), disable=not verbose):
            yield grammar.materialise(self._load(key), columns)

    def read(self, columns=None, key_column=None, verbose=False):
        """Read the catalogue into one numpy structured array.

        Chunks are copied one at a time into a preallocated output, so peak
        memory is the output plus one chunk.

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
        keys = list(self.chunks)
        add_key = key_column is not None and self.chunked
        if not add_key and self.whole is not None:
            return grammar.materialise(self.whole, columns)

        if add_key:
            key_values = _chunk_numbers(keys, key_column, dtype_out, self.path)
            dtype_out = np.dtype(dtype_out.descr + [(key_column, "i8")])

        n_rows = len(self)
        if verbose:
            print(
                f"Reading {len(keys)} datasets,"
                + f" estimating {dtype_out.itemsize * n_rows / 1024**3:.1f}"
                + f" Gb memory for the ({len(dtype_out.names)} x {n_rows})"
                + " data array ..."
            )

        out = np.empty(n_rows, dtype=dtype_out)
        start = 0
        for key in tqdm.tqdm(keys, disable=not verbose):
            data = self._load(key)
            end = start + len(data)
            for name in dtype_out.names:
                if add_key and name == key_column:
                    out[name][start:end] = key_values[key]
                else:
                    out[name][start:end] = data[name]
            start = end
            del data
        return out

    def _load(self, key):
        """Return chunk ``key`` in memory, or lazily when the file is unchunked.

        An HDF5 chunk (one tile or exposure) is small, so it is read whole,
        which is faster than column by column; any other table (a HATS
        partition included) is read per column.
        """
        if not self.chunked or self._file is None:
            return self.chunks[key]
        parts = (part[()] for part in self._parts[key])
        return grammar.adapt(*parts, column_map=self.column_map)


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
    numpy.ndarray or astropy.io.fits.FITS_rec
        catalogue data
    """
    with Catalogue(path, hdu=hdu, column_map=column_map) as catalogue:
        return catalogue.read(columns=columns, key_column=key_column, verbose=verbose)


def open_catalogue(path, hdu=None, column_map=None):
    """Return a catalogue as one table, reading no data up front.

    A FITS table is memory-mapped, and an unchunked HDF5 table or a HATS
    catalogue read column by column on access (an HDF5 file stays open while
    the table is referenced), so a consumer pays only for the columns it
    indexes, derived v2 columns included. HDF5 row chunks have no single
    on-disk table and are read into memory. Parameters as for ``Catalogue``.

    Returns
    -------
    astropy.io.fits.FITS_rec, h5py.Dataset, ParquetTable, grammar.V2View or
    numpy.ndarray
        catalogue data in the v2 grammar
    """
    catalogue = Catalogue(path, hdu=hdu, column_map=column_map)
    if catalogue.whole is None:
        with catalogue:
            return catalogue.read()
    return catalogue.whole


def open_entry(entry):
    """Return the catalogue a config block names (e.g. a cat_config ``shear``).

    The block's ``path``, optional ``hdu`` and optional ``column_map`` go to
    ``open_catalogue``.
    """
    return open_catalogue(
        entry["path"], hdu=entry.get("hdu"), column_map=entry.get("column_map")
    )


#: Upper bound on the threads reading parquet files at once.
MAX_READ_THREADS = 32


def _read_threads():
    return min(MAX_READ_THREADS, len(os.sched_getaffinity(0)))


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


def _hats_parts(path):
    """Return (whole-catalogue ParquetTable, {partition: (its ParquetTable,)})."""
    path = hats_catalogue_dir(path)
    properties = _properties(os.path.join(path, "hats.properties"))
    suffix = properties.get("hats_npix_suffix", ".parquet")
    if suffix != ".parquet":
        raise ValueError(
            f"HATS catalogue {path}: partitions with suffix {suffix!r} are not"
            + " supported, only single .parquet files"
        )
    pixels = np.loadtxt(
        os.path.join(path, "partition_info.csv"),
        delimiter=",",
        skiprows=1,
        dtype=np.int64,
        ndmin=2,
    )
    keys, files = [], []
    for order, pixel in pixels:
        key = f"Norder={order}/Npix={pixel}"
        keys.append(key)
        files.append(
            os.path.join(
                path,
                "dataset",
                f"Norder={order}",
                f"Dir={pixel // 10000 * 10000}",
                f"Npix={pixel}.parquet",
            )
        )
    common = os.path.join(path, "dataset", "_common_metadata")
    schema = pq.read_schema(common if os.path.isfile(common) else files[0])
    with ThreadPoolExecutor(_read_threads()) as pool:
        parts = list(
            pool.map(lambda file: _ParquetFile(file, pq.read_metadata(file)), files)
        )
    whole = ParquetTable(parts, schema)
    return whole, {
        key: (ParquetTable([part], schema),)
        for key, part in zip(keys, parts, strict=True)
    }


class _ParquetFile:
    """One parquet file, read by row windows of some columns.

    Only the row-group sizes are kept from the footer: a parsed footer holds
    every column's statistics (~2 MB each on run1-snap3-hats, GBs over a
    catalogue), so each read parses it afresh, once for all its columns.
    """

    def __init__(self, path, footer):
        self.path = path
        self.n_rows = footer.num_rows
        sizes = [footer.row_group(i).num_rows for i in range(footer.num_row_groups)]
        self.group_starts = np.concatenate([[0], np.cumsum(sizes)])

    def read(self, names, start, stop):
        """Return {name: rows ``start:stop``}, reading only their row groups."""
        first = int(np.searchsorted(self.group_starts, start, side="right")) - 1
        last = int(np.searchsorted(self.group_starts, stop, side="left"))
        table = pq.ParquetFile(self.path).read_row_groups(
            range(first, last), columns=list(names), use_threads=False
        )
        table = table.slice(start - self.group_starts[first], stop - start)
        out = {}
        for name in names:
            column = table.column(name)
            if column.null_count and not np.issubdtype(
                np.dtype(column.type.to_pandas_dtype()), np.floating
            ):
                raise ValueError(
                    f"Column {name!r} of {self.path} holds nulls, which a"
                    + f" {column.type} numpy column cannot represent"
                )
            out[name] = column.to_numpy()
        return out


class ParquetTable:
    """A lazy table over parquet files read in sequence (a HATS catalogue).

    Quacks like the other tables ``grammar.adapt`` presents: ``dtype`` (from
    the parquet schema, without reading rows), ``len``, ``table[name]`` reads
    one column, a list of names reads a structured array, and any other row
    key (slice, boolean mask, index array) returns a lazy ``grammar.V2View``
    over those rows. ``read_columns`` reads only the files and row groups a
    row range touches, so a slice of a few rows costs a few row groups, and
    reads several columns for the price of one footer per file. Files are
    read in parallel threads (at most ``MAX_READ_THREADS``). Null values read
    as NaN in floating columns and raise in any other.
    """

    def __init__(self, files, schema):
        self._files = list(files)
        self._starts = np.concatenate([[0], np.cumsum([f.n_rows for f in files])])
        self.dtype = np.dtype(
            [(field.name, np.dtype(field.type.to_pandas_dtype())) for field in schema]
        )

    @property
    def names(self):
        return self.dtype.names

    @property
    def shape(self):
        return (len(self),)

    def __len__(self):
        return int(self._starts[-1])

    def __repr__(self):
        return (
            f"ParquetTable({len(self)} rows, {len(self.dtype.names)} columns,"
            + f" {len(self._files)} files)"
        )

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.read_window(key, 0, len(self))
        if isinstance(key, list) and key and all(isinstance(k, str) for k in key):
            return self.to_structured(key)
        return grammar.V2View((self,))[key]

    def __array__(self, dtype=None, copy=None):
        out = self.to_structured()
        return out if dtype is None else out.astype(dtype)

    def to_structured(self, names=None):
        """Read ``names`` (default all) into a numpy structured array."""
        names = self.dtype.names if names is None else list(names)
        out = np.empty(len(self), dtype=[(name, self.dtype[name]) for name in names])
        self.read_columns(names, 0, len(self), out={name: out[name] for name in names})
        return out

    def read_window(self, name, start, stop):
        """Return rows ``start:stop`` of column ``name``, and only those."""
        return self.read_columns([name], start, stop)[name]

    def read_columns(self, names, start, stop, out=None):
        """Return {name: rows ``start:stop``} for each of ``names``.

        ``out``, {name: array of ``stop - start`` rows} for some of ``names``,
        receives their rows in place (e.g. the fields of a structured array).
        """
        missing = [name for name in names if name not in self.dtype.names]
        if missing:
            raise KeyError(f"columns {missing} not in catalogue")
        start, stop, _ = slice(start, stop).indices(len(self))
        size = max(stop - start, 0)
        out = dict(out or {})
        for name in names:
            if name not in out:
                out[name] = np.empty(size, dtype=self.dtype[name])
        jobs = []
        for file, lo, hi in zip(self._files, self._starts, self._starts[1:]):
            a, b = max(start, lo), min(stop, hi)
            if a < b:
                jobs.append((file, a - lo, b - lo, a - start))

        def read(job):
            file, a, b, at = job
            for name, values in file.read(names, a, b).items():
                out[name][at : at + b - a] = values

        if len(jobs) == 1:
            read(jobs[0])
        elif jobs:
            with ThreadPoolExecutor(_read_threads()) as pool:
                list(pool.map(read, jobs))
        return out


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


def promote_dtypes(dtype_a, dtype_b):
    """Promote Dtypes.

    Return a scalar dtype that holds both input dtypes without truncation
    or overflow.

    Parameters
    ----------
    dtype_a : numpy.dtype
        first input dtype
    dtype_b : numpy.dtype
        second input dtype

    Returns
    -------
    numpy.dtype
        promoted dtype

    """
    if dtype_a == dtype_b:
        return dtype_a
    if dtype_a.kind in "SU" and dtype_b.kind in "SU":
        kind = "U" if "U" in (dtype_a.kind, dtype_b.kind) else "S"
        size_a = dtype_a.itemsize // (4 if dtype_a.kind == "U" else 1)
        size_b = dtype_b.itemsize // (4 if dtype_b.kind == "U" else 1)
        return np.dtype(f"{kind}{max(size_a, size_b)}")

    return np.promote_types(dtype_a, dtype_b)


def group_dtype(tables, columns=None):
    """Group Dtype.

    Build the structured output dtype of a group of row-chunk tables,
    promoting each column across *every* table. A campaign that was
    partially reprocessed can carry e.g. ``S7`` tile IDs in one tile and
    ``S12`` in another, or ``f4`` next to ``f8``; taking the dtype of the
    first table alone would silently truncate the others.

    Note that ShapePipe currently writes ``TILE_ID`` as ``f8`` (the tile
    ``183.307`` arrives as the float ``183.307``), not as a string, so the
    string-promotion branch above is for a future string-valued ``TILE_ID``
    and is not exercised by today's products.

    TODO: whether ``TILE_ID`` should be a string is an open schema decision.
    A float cannot represent the ID exactly and cannot be compared for
    equality safely; changing it is a breaking product change, so it is left
    as-is here and this reader deliberately handles both.

    Parameters
    ----------
    tables : list or dict
        row-chunk tables (h5py Datasets, or their ``grammar.adapt`` views),
        optionally keyed by dataset name for error messages
    columns : list of str, optional
        columns to keep; default is ``None`` (all columns of the first table)

    Returns
    -------
    numpy.dtype
        structured output dtype

    Raises
    ------
    KeyError
        if a table lacks one of the columns, naming the tables that do

    """
    if not hasattr(tables, "items"):
        tables = dict(enumerate(tables))
    dtypes = {key: table.dtype for key, table in tables.items()}
    first = next(iter(dtypes.values()))
    names = columns if columns is not None else list(first.names)

    missing = {
        key: [name for name in names if name not in dtype.names]
        for key, dtype in dtypes.items()
    }
    missing = {key: cols for key, cols in missing.items() if cols}
    if missing:
        key, cols = next(iter(missing.items()))
        raise KeyError(
            f"{len(missing)} of {len(tables)} tables lack columns"
            + f" {'requested' if columns is not None else 'of the first table'},"
            + f" e.g. table {key!r} lacks {cols}; tables lacking some:"
            + f" {list(missing)[:20]}. Request columns without them."
        )
    dtypes = list(dtypes.values())

    fields = []
    for name in names:
        promoted = dtypes[0][name]
        for dtype in dtypes[1:]:
            promoted = promote_dtypes(promoted, dtype[name])
        fields.append((name, promoted))

    return np.dtype(fields)


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
