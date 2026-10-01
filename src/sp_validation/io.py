"""IO.

:Name: io.py

:Description: Input and output: output directories and statistics files, and
    the catalogue reader (``read_catalogue``, ``Catalogue``), the one place
    that knows how a catalogue's rows are stored on disk.

:Author: Martin Kilbinger


"""

import os
import re

import h5py
import numpy as np
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
# is detected here from the file's contents; which names its columns carry is
# the column grammar's business (``sp_validation.grammar``). Every table this
# module hands out is presented through ``grammar.adapt``, which maps ShapePipe
# v1 names to v2 and passes any other naming convention through untouched,
# unless a ``column_map`` renames it.

#: Root attributes in which a chunked ShapePipe product declares its chunk
#: count, with the name of one chunk (for error messages).
N_UNITS_ATTRS = {"n_tiles": "tile", "n_exposures": "exposure"}


class Catalogue:
    """An open catalogue file, presented as row chunks in the v2 grammar.

    The container is detected from the file contents (``h5py.is_hdf5``), not
    from its extension. Supported layouts:

    - FITS: one table HDU, ``hdu`` or by default the first table HDU;
    - HDF5 comprehensive catalogue: a root ``data`` dataset, joined
      column-wise with ``data_ext`` when present;
    - HDF5 single dataset at the root, of any name;
    - HDF5 group of per-unit row chunks (one dataset per tile or exposure,
      as ShapePipe writes campaign galaxy and star catalogues), found by
      ``find_dataset_group``, concatenated in dataset-name order, and
      checked against a declared ``n_tiles`` or ``n_exposures`` count.

    Nothing is read on opening: ``chunks`` maps each chunk name to a lazy
    table. Use as a context manager, or ``close`` when done; tables read into
    memory stay valid after closing.

    Parameters
    ----------
    path : str or os.PathLike
        catalogue file
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
        if h5py.is_hdf5(self.path):
            self.hdu = None
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

        Each column's dtype is promoted across every chunk (``group_dtype``).

        Raises
        ------
        KeyError
            if a chunk lacks one of ``columns``, naming the chunk
        """
        if columns is not None:
            for key, table in self.chunks.items():
                _check_columns(table.dtype, columns, self.path, key)
        return group_dtype(self.chunks, columns)

    def table(self):
        """Return the catalogue as one table, lazily when it is not chunked.

        An unchunked catalogue is returned as its lazy table (a FITS_rec, an
        h5py Dataset or a ``grammar.V2View``), valid while the file is open;
        a chunked one is read into memory (``read``).
        """
        if len(self.chunks) == 1:
            return next(iter(self.chunks.values()))
        return self.read()

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
        if not add_key and len(keys) == 1:
            return grammar.materialise(self.chunks[keys[0]], columns)

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

        A chunk (one tile or exposure) is small, so it is read whole, which is
        faster than column by column; an unchunked table is read per column.
        """
        if not self.chunked:
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


def read_catalogue_entry(entry, columns=None):
    """Read the catalogue a config block names, e.g. a cat_config ``shear`` block.

    The block's ``path``, optional ``hdu`` and optional ``column_map`` go to
    ``read_catalogue``; ``columns`` restricts what is read.
    """
    return read_catalogue(
        entry["path"],
        columns=columns,
        hdu=entry.get("hdu"),
        column_map=entry.get("column_map"),
    )


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
