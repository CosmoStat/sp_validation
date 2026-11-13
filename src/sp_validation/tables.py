"""Tables: where a catalogue's rows are stored, behind one interface.

A table is anything with

- ``dtype``: a numpy structured dtype naming its columns, each with the dtype
  (and any subarray shape) it reads as, known without reading rows;
- ``len(table)``: its number of rows;
- ``read(names, rows=None, out=None)``: ``{name: array}`` holding columns
  ``names`` at ``rows``, which is ``None`` (every row), a ``range``, or an
  integer index array in any order, repeats allowed. ``out``,
  ``{name: array}`` for some of ``names``, receives their values in place
  (e.g. the fields of a structured array) and is what is returned for them;
- ``row_major``: whether reading any column reads whole rows from disk, so a
  selection of rows is best read for every column at once;
- ``concurrent``: whether reads of several such tables gain from running in
  parallel threads.

``ArrayTable`` (a numpy structured array or FITS_rec, in memory or
memory-mapped), ``HDF5Table`` (an h5py compound dataset) and ``ParquetFile``
store rows; ``ConcatTable`` stacks tables end to end. ``grammar.V2View``
presents any of them in the v2 column grammar, and is itself a table.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import pyarrow.parquet as pq

#: Upper bound on the threads reading the tables of a ``ConcatTable`` at once.
MAX_READ_THREADS = 32

#: Bytes of whole HDF5 rows read per block when selecting rows of a dataset.
BLOCK_BYTES = 256 * 1024**2


def as_table(obj):
    """Return ``obj`` as a table: a structured array or FITS_rec becomes an
    ``ArrayTable``, an h5py Dataset an ``HDF5Table``; a table is returned as is.
    """
    if isinstance(obj, h5py.Dataset):
        return HDF5Table(obj)
    if isinstance(obj, np.ndarray):
        return ArrayTable(obj)
    return obj


def read_structured(table, names=None, rows=None):
    """Read columns ``names`` (default all) of ``table`` at ``rows`` into a
    numpy structured array, in place."""
    names = table.dtype.names if names is None else list(names)
    n = len(table) if rows is None else len(rows)
    out = np.empty(n, dtype=[(name, table.dtype[name]) for name in names])
    table.read(names, rows, out={name: out[name] for name in names})
    return out


def _outputs(dtype, names, n, out):
    """Return ``out`` with an empty ``n``-row array added for each other name."""
    out = dict(out or {})
    for name in names:
        if name not in out:
            out[name] = np.empty(n, dtype=dtype[name])
    return out


def _deliver(columns, out):
    """Copy ``columns`` into the arrays ``out`` holds for some of them."""
    for name, target in (out or {}).items():
        if columns[name] is not target:
            target[...] = columns[name]
        columns[name] = target
    return columns


def _contiguous(rows, n):
    """Return ``(start, stop)`` if ``rows`` is every row or a unit-step range."""
    if rows is None:
        return 0, n
    if isinstance(rows, range) and rows.step == 1:
        return rows.start, max(rows.start, rows.stop)
    return None


class ArrayTable:
    """A numpy structured array or FITS_rec, in memory or memory-mapped.

    Reading every row of a column returns the array's own column, so a
    memory-mapped FITS table is read only where it is indexed.
    """

    row_major = False
    concurrent = False

    def __init__(self, array):
        self.array = array
        # A FITS_rec column can read as another dtype than it is stored in
        # (logical columns read as bool), so ask a zero-row slice.
        empty = array[:0]
        self.dtype = np.dtype(
            [
                (name, (empty[name].dtype, empty[name].shape[1:]))
                for name in array.dtype.names
            ]
        )

    def __len__(self):
        return len(self.array)

    def read(self, names, rows=None, out=None):
        if rows is None:
            key = slice(None)
        elif isinstance(rows, range):
            key = slice(rows.start, rows.stop if rows.stop >= 0 else None, rows.step)
        else:
            key = rows
        return _deliver({name: self.array[name][key] for name in names}, out)


class HDF5Table:
    """An h5py compound dataset.

    Its rows are stored whole, so a selection is read in one pass for every
    column wanted, in blocks of ``BLOCK_BYTES`` of whole rows over its span
    written straight into the output, skipping blocks that hold no selected
    row: each byte of the file is read at most once. h5py serialises all
    reads on one lock, so datasets are not read concurrently.
    """

    row_major = True
    concurrent = False

    def __init__(self, dataset):
        self.dataset = dataset
        self.dtype = dataset.dtype

    def __len__(self):
        return len(self.dataset)

    def read(self, names, rows=None, out=None):
        names = list(names)
        n = len(self) if rows is None else len(rows)
        out = _outputs(self.dtype, names, n, out)
        source = self.dataset.fields(names)
        block = max(1, BLOCK_BYTES // self.dtype.itemsize)
        span = _contiguous(rows, len(self))
        if span is not None:
            for start in range(span[0], span[1], block):
                end = min(start + block, span[1])
                chunk = source[start:end]
                for name in names:
                    out[name][start - span[0] : end - span[0]] = chunk[name]
            return out
        rows = np.asarray(rows, dtype=np.intp)
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
        done, start, stop = 0, 0, (int(rows[-1]) + 1 if n else 0)
        while done < n:
            start = max(start, int(rows[done]))
            end = min(start + block, stop)
            upto = int(np.searchsorted(rows, end, side="left"))
            chunk = source[start:end][rows[done:upto] - start]
            for name in names:
                out[name][order[done:upto]] = chunk[name]
            done, start = upto, end
        return out


class ParquetFile:
    """One parquet file, read by row windows of some columns.

    Only the row-group sizes are kept from the footer: a parsed footer holds
    every column's statistics (~2 MB each on run1-snap3-hats, GBs over a
    catalogue), so each read parses it afresh, once for all its columns. A
    read touches only the row groups spanning its rows. Null values read as
    NaN in floating columns and raise in any other.
    """

    row_major = False
    concurrent = True

    def __init__(self, path, footer, dtype):
        self.path = path
        self.dtype = dtype
        sizes = [footer.row_group(i).num_rows for i in range(footer.num_row_groups)]
        self.group_starts = np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])

    def __len__(self):
        return int(self.group_starts[-1])

    def read(self, names, rows=None, out=None):
        names = list(names)
        span = _contiguous(rows, len(self))
        pick = None
        if span is None:
            rows = np.asarray(rows, dtype=np.intp)
            span = (int(rows.min()), int(rows.max()) + 1) if len(rows) else (0, 0)
            pick = rows - span[0]
        start, stop = span
        if stop <= start:
            columns = {name: np.empty(0, dtype=self.dtype[name]) for name in names}
        else:
            columns = self._window(names, start, stop)
        if pick is not None:
            columns = {name: column[pick] for name, column in columns.items()}
        return _deliver(columns, out)

    def _window(self, names, start, stop):
        first = int(np.searchsorted(self.group_starts, start, side="right")) - 1
        last = int(np.searchsorted(self.group_starts, stop, side="left"))
        table = pq.ParquetFile(self.path).read_row_groups(
            range(first, last), columns=names, use_threads=False
        )
        table = table.slice(start - self.group_starts[first], stop - start)
        columns = {}
        for name in names:
            column = table.column(name)
            if column.null_count and self.dtype[name].kind != "f":
                raise ValueError(
                    f"Column {name!r} of {self.path} holds nulls, which a"
                    + f" {column.type} numpy column cannot represent"
                )
            columns[name] = column.to_numpy()
        return columns


def parquet_dtype(schema):
    """Return the numpy structured dtype of a parquet schema."""
    return np.dtype(
        [(field.name, np.dtype(field.type.to_pandas_dtype())) for field in schema]
    )


class ConcatTable:
    """Tables stacked end to end.

    Its columns are those every table has, each promoted to hold all of them
    (``group_dtype``). ``concurrent`` tables are read in parallel threads, at
    most ``MAX_READ_THREADS`` at once.
    """

    def __init__(self, tables):
        self.tables = list(tables)
        self.starts = np.concatenate([[0], np.cumsum([len(t) for t in self.tables])])
        self.row_major = any(table.row_major for table in self.tables)
        self.concurrent = all(table.concurrent for table in self.tables)
        common = set.intersection(*(set(t.dtype.names) for t in self.tables))
        self.dtype = group_dtype(
            self.tables, [n for n in self.tables[0].dtype.names if n in common]
        )

    def __len__(self):
        return int(self.starts[-1])

    def read(self, names, rows=None, out=None):
        names = list(names)
        n = len(self) if rows is None else len(rows)
        out = _outputs(self.dtype, names, n, out)
        jobs = []
        span = _contiguous(rows, len(self))
        if span is not None:
            start, stop = span
            for table, lo, hi in zip(self.tables, self.starts, self.starts[1:]):
                a, b = max(start, lo), min(stop, hi)
                if a < b:
                    jobs.append(
                        (table, range(a - lo, b - lo), slice(a - start, b - start))
                    )
        else:
            rows = np.asarray(rows, dtype=np.intp)
            order = np.argsort(rows, kind="stable")
            ordered = rows[order]
            bounds = np.searchsorted(ordered, self.starts)
            for i, table in enumerate(self.tables):
                a, b = bounds[i], bounds[i + 1]
                if a < b:
                    jobs.append((table, ordered[a:b] - self.starts[i], order[a:b]))

        def read(job):
            table, local, at = job
            if isinstance(at, slice):
                table.read(names, local, out={name: out[name][at] for name in names})
            else:
                for name, column in table.read(names, local).items():
                    out[name][at] = column

        if len(jobs) == 1 or not self.concurrent:
            for job in jobs:
                read(job)
        elif jobs:
            threads = min(MAX_READ_THREADS, len(os.sched_getaffinity(0)))
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(read, jobs))
        return out


def group_dtype(tables, columns=None):
    """Return the structured dtype of a group of row-chunk tables.

    Each column is promoted across every table (``np.promote_types``), so a
    partially reprocessed campaign carrying ``S7`` and ``S12`` tile IDs, or
    ``f4`` next to ``f8``, reads without truncation. ``tables`` is a list or a
    dict keyed by name for error messages; ``columns`` defaults to those of the
    first table. Raises ``KeyError`` naming the tables that lack a column.
    """
    if not hasattr(tables, "items"):
        tables = dict(enumerate(tables))
    dtypes = {key: table.dtype for key, table in tables.items()}
    first = next(iter(dtypes.values()))
    names = columns if columns is not None else list(first.names)
    # Chunks of one product usually share a dtype; compare each distinct one.
    distinct = list(dict.fromkeys(dtypes.values()))
    lacking = {dtype: [n for n in names if n not in dtype.names] for dtype in distinct}
    missing = {key: lacking[dtype] for key, dtype in dtypes.items() if lacking[dtype]}
    if missing:
        key, cols = next(iter(missing.items()))
        raise KeyError(
            f"{len(missing)} of {len(tables)} tables lack columns"
            + f" {'requested' if columns is not None else 'of the first table'},"
            + f" e.g. table {key!r} lacks {cols}; tables lacking some:"
            + f" {list(missing)[:20]}. Request columns without them."
        )
    dtypes = distinct

    fields = []
    for name in names:
        promoted = dtypes[0][name]
        for dtype in dtypes[1:]:
            promoted = np.promote_types(promoted, dtype[name])
        fields.append((name, promoted))

    return np.dtype(fields)
