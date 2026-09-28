"""ShapePipe column grammars.

ShapePipe products name their columns in one of two grammars. ``v2`` is the
grammar the rest of sp_validation reads (``HSM_T_PSF``, ``NGMIX_G1_NOSHEAR``,
``MASK_n4``, ...; see ``docs/ngmix_psf_column_migration.md``). ``v1`` is the
grammar of the catalogues released up to and including v1.6.x (``SIGMA_PSF_HSM``,
the 2-vector ``NGMIX_ELL_NOSHEAR``, ...).

This module is the single place that knows the difference. ``adapt`` hands a
table back as a lazy view presenting v2 names and units, so downstream code
reads one grammar and never branches on the generation. Two rule families
apply:

- The v1 shape-measurement columns, applied when the table is v1 (see
  ``detect_generation``):

  - ``E{1,2}_{PSF,STAR}_HSM`` and ``FLAG_{PSF,STAR}_HSM`` are renamed;
  - ``SIGMA_{PSF,STAR}_HSM`` (sigma) is presented as ``HSM_T_{PSF,STAR}``
    holding ``T = 2 sigma^2``, via ``cs_util.size.sigma_to_T``;
  - the 2-vector ngmix columns ``NGMIX_ELL[_ERR|_PSFo]_{shear}`` (or their
    flattened ``_0``/``_1`` forms, as in the comprehensive HDF5) are split
    into ``NGMIX_G{1,2}[_ERR|_PSF_ORIG]_{shear}``;
  - ``NGMIX_T_PSFo_{shear}``, ``NGMIX_Tpsf_{shear}`` and ``NGMIX_MOM_FAIL``
    become ``NGMIX_T_PSF_ORIG_{shear}``, ``NGMIX_T_PSF_RECONV_{shear}`` and
    ``NGMIX_MCAL_TYPES_FAIL``, except that ``NGMIX_T_PSF_RECONV_NOSHEAR`` is
    read from ``NGMIX_Tpsf_1P`` when the table has it (see below).

- The UNIONS mask-bit columns, applied to any table: ``{b}_{label}``
  (``1_Faint_star_halos``, ``4_Stars``, ``2048_z2``, ...), the names under
  which ``catalog_builders.ApplyHspMasks`` wrote the healsparse mask bits into
  the ``data_ext`` dataset of the comprehensive HDF5, become ``MASK_n{b}``,
  the name ShapePipe v2 and ``ApplyHspMasks`` give the same bit of the same
  bitmask (``MASK_LABELS``). These columns belong to the mask product, not to
  a ShapePipe generation, so they neither mark nor require one.

``IMAFLAGS_ISO`` is not mapped: its v1 bits (2 halo, 4 border, 16 Messier,
32 NGC, 128 spike) differ in meaning from the ``MASK_n{b}`` of the same value,
so it passes through under its own name for the v1 mask configs that cut on it.

The mapping is by *name*, deliberately, including where the v1 values mean
something different from their v2 namesakes: v1 ``NGMIX_*_PSFo_*`` hold the
reconvolved-PSF alias rather than a fit to the original PSF, and
``NGMIX_MOM_FAIL`` counts moments-guess failures rather than failed metacal
types. Those values are presented unchanged.

One v1 value is corrected rather than renamed. ShapePipe v1 wrote a wrong
no-shear reconvolved-PSF size: ``NGMIX_Tpsf_NOSHEAR`` differs by ~2% for
nearly every object from the reconvolution kernel metacal applied, which the
sheared types' ``NGMIX_Tpsf_{1P,1M,2P,2M}`` record (they agree to ~1e-5).
The v1.4.6.3 release cut on and wrote the ``1P`` value, so
``NGMIX_T_PSF_RECONV_NOSHEAR`` reads ``NGMIX_Tpsf_1P`` whenever the table has
it, and ``NGMIX_Tpsf_NOSHEAR`` otherwise (a cut catalogue such as v1.4.6.3's,
whose no-shear column already holds the ``1P`` value). ShapePipe v2 reuses one
reconvolved PSF across metacal types, so its columns need no correction.

Every other column passes through under its own name; source names that have a
v2 equivalent are hidden.

``adapt`` also joins row-aligned tables into one view, as the comprehensive
HDF5 splits one catalogue over its ``data`` and ``data_ext`` datasets.
"""

import os
from dataclasses import dataclass

import h5py
import numpy as np
from astropy.io import fits
from cs_util.size import sigma_to_T

#: Metacal shear types carried by the ngmix columns.
SHEARS = ("NOSHEAR", "1P", "1M", "2P", "2M")

#: Bits of the UNIONS healsparse mask product and what each flags. The column
#: for bit ``b`` is ``MASK_n{b}``; the ``data_ext`` dataset of a v1
#: comprehensive HDF5 carries it as ``{b}_{label}``.
MASK_LABELS = {
    1: "Faint_star_halos",
    2: "Bright_star_halos",
    4: "Stars",
    8: "Manual",
    16: "u",
    32: "g",
    64: "r",
    128: "i",
    256: "z",
    512: "Tile_RA_DEC_cut",
    1024: "Maximask",
    2048: "z2",
}


def mask_column(bit):
    """Return the catalogue column holding mask bit ``bit``: ``MASK_n{bit}``.

    Raises
    ------
    KeyError
        if ``bit`` is not a bit of the mask product
    """
    if bit not in MASK_LABELS:
        raise KeyError(f"{bit} is not a UNIONS mask bit; bits: {list(MASK_LABELS)}")
    return f"MASK_n{bit}"


@dataclass(frozen=True)
class Rule:
    """One v2 column presented from a source column.

    ``kind`` is ``"rename"`` (same values), ``"sigma_to_T"`` (T = 2 sigma^2)
    or ``"component"`` (element ``arg`` of a 2-vector, stored either as one
    vector column ``source`` or as the flattened ``{source}_{arg}``).
    ``fallback`` names a column read instead when ``source`` is absent.
    """

    v2: str
    source: str
    kind: str = "rename"
    arg: int | None = None
    fallback: str | None = None

    def sources(self):
        """Return the source names this rule can read from, in preference order."""
        if self.kind == "component":
            return (self.source, f"{self.source}_{self.arg}")
        if self.fallback is not None:
            return (self.source, self.fallback)
        return (self.source,)

    def apply(self, column):
        """Return the v2 column computed from source ``column``."""
        if self.kind == "rename":
            return column
        column = np.asarray(column)
        if self.kind == "sigma_to_T":
            return sigma_to_T(column)
        return column[:, self.arg] if column.ndim == 2 else column


def _v1_rules():
    rules = []
    for obj in ("PSF", "STAR"):
        rules += [
            Rule(f"HSM_G1_{obj}", f"E1_{obj}_HSM"),
            Rule(f"HSM_G2_{obj}", f"E2_{obj}_HSM"),
            Rule(f"HSM_T_{obj}", f"SIGMA_{obj}_HSM", "sigma_to_T"),
            Rule(f"HSM_FLAG_{obj}", f"FLAG_{obj}_HSM"),
        ]
    for shear in SHEARS:
        for i in (0, 1):
            g = f"G{i + 1}"
            rules += [
                Rule(f"NGMIX_{g}_{shear}", f"NGMIX_ELL_{shear}", "component", i),
                Rule(
                    f"NGMIX_{g}_ERR_{shear}", f"NGMIX_ELL_ERR_{shear}", "component", i
                ),
                Rule(
                    f"NGMIX_{g}_PSF_ORIG_{shear}",
                    f"NGMIX_ELL_PSFo_{shear}",
                    "component",
                    i,
                ),
            ]
        rules.append(Rule(f"NGMIX_T_PSF_ORIG_{shear}", f"NGMIX_T_PSFo_{shear}"))
        if shear == "NOSHEAR":
            # v1 wrote a wrong no-shear size; see the module docstring.
            rules.append(
                Rule(
                    "NGMIX_T_PSF_RECONV_NOSHEAR",
                    "NGMIX_Tpsf_1P",
                    fallback="NGMIX_Tpsf_NOSHEAR",
                )
            )
        else:
            rules.append(Rule(f"NGMIX_T_PSF_RECONV_{shear}", f"NGMIX_Tpsf_{shear}"))
    rules.append(Rule("NGMIX_MCAL_TYPES_FAIL", "NGMIX_MOM_FAIL"))
    return tuple(rules)


#: The v1 -> v2 map of the shape-measurement columns, applied to v1 tables.
V1_RULES = _v1_rules()

#: ``{b}_{label}`` -> ``MASK_n{b}``, applied to every table.
MASK_RULES = tuple(
    Rule(mask_column(bit), f"{bit}_{label}") for bit, label in MASK_LABELS.items()
)

#: Names whose presence marks a table as v1 / as v2.
V1_MARKERS = frozenset(name for rule in V1_RULES for name in rule.sources())
V2_MARKERS = frozenset(rule.v2 for rule in V1_RULES)


def column_names(table):
    """Return the column names of a structured array, FITS_rec or h5py Dataset."""
    names = getattr(getattr(table, "dtype", None), "names", None)
    if names is None:
        raise TypeError(f"cannot list the columns of a {type(table).__name__}")
    return tuple(names)


def detect_generation(names):
    """Return ``"v1"``, ``"v2"`` or ``None`` from the column names present.

    ``None`` means grammar-neutral: no shape-measurement column either grammar
    names (e.g. a cut catalogue carrying only ``RA``, ``Dec``, ``e1``, ``e2``,
    ``w``, or a table of mask columns). Mask columns mark no generation.

    Raises
    ------
    ValueError
        if columns of both grammars are present.
    """
    names = set(names)
    v1 = sorted(names & V1_MARKERS)
    v2 = sorted(names & V2_MARKERS)
    if v1 and v2:
        raise ValueError(
            "catalogue mixes ShapePipe column grammars:"
            + f" v1 columns {v1[:5]} alongside v2 columns {v2[:5]}"
        )
    if v1:
        return "v1"
    if v2:
        return "v2"
    return None


def _resolve(names):
    """Return (presented names, {v2 name: (rule, source name)}) for ``names``."""
    rules = MASK_RULES
    if detect_generation(names) == "v1":
        rules = V1_RULES + MASK_RULES
    position = {name: i for i, name in enumerate(names)}
    derived = {}
    consumed = set()
    by_anchor = {}
    for rule in rules:
        sources = [s for s in rule.sources() if s in position]
        if not sources:
            continue
        if rule.v2 in position:
            raise ValueError(
                f"catalogue carries both {sources[0]!r} and {rule.v2!r},"
                + " two names for the same column"
            )
        derived[rule.v2] = (rule, sources[0])
        consumed.update(sources)
        # Presented where the first of its sources sits in the table.
        anchor = min(sources, key=position.get)
        by_anchor.setdefault(anchor, []).append(rule.v2)

    presented = []
    for name in names:
        if name in by_anchor:
            presented += by_anchor[name]
        elif name not in consumed:
            presented.append(name)
    return tuple(presented), derived


def v2_names(names):
    """Return the column names a table with columns ``names`` presents.

    Header-only counterpart of ``adapt``.
    """
    return _resolve(tuple(names))[0]


#: Bytes of whole HDF5 rows read per block when selecting rows of a dataset.
BLOCK_BYTES = 256 * 1024**2


def _read_window(table, name, start, stop):
    """Read rows ``start:stop`` of column ``name``, and only those."""
    if isinstance(table, h5py.Dataset):
        return table.fields(name)[start:stop]
    return table[name][start:stop]


def _take(table, rows, fields=None):
    """Return rows ``rows`` of ``table`` as an in-memory structured array.

    ``rows`` is ``None`` (all rows), a ``range`` or an integer index array;
    ``fields`` restricts the output to those columns. An h5py Dataset is read
    in one pass: whole rows in blocks of ``BLOCK_BYTES`` over the span of the
    selection, skipping blocks that hold no selected row, so each byte of the
    file is read at most once however many columns are wanted.
    """
    if not isinstance(table, h5py.Dataset):
        out = table if rows is None else table[_row_key(rows)]
        if fields is None:
            return np.asarray(out)
        taken = np.empty(len(out), dtype=[(f, out.dtype[f]) for f in fields])
        for f in fields:
            taken[f] = out[f]
        return taken

    if fields is None:
        source, dtype = table, table.dtype
    else:
        fields = list(fields)
        source = table.fields(fields)
        dtype = np.dtype([(f, table.dtype[f]) for f in fields])
    if rows is None:
        rows = range(len(table))
    if isinstance(rows, range) and rows.step == 1:
        return source[rows.start : rows.stop]

    rows = np.asarray(rows, dtype=np.intp)
    out = np.empty(len(rows), dtype=dtype)
    if len(rows) == 0:
        return out
    order = None
    if np.any(rows[1:] < rows[:-1]):
        order = np.argsort(rows, kind="stable")
        rows = rows[order]
    block = max(1, BLOCK_BYTES // table.dtype.itemsize)
    done = 0
    start, stop = int(rows[0]), int(rows[-1]) + 1
    while done < len(rows):
        start = max(start, int(rows[done]))
        end = min(start + block, stop)
        upto = int(np.searchsorted(rows, end, side="left"))
        chunk = source[start:end][rows[done:upto] - start]
        if order is None:
            out[done:upto] = chunk
        else:
            out[order[done:upto]] = chunk
        done, start = upto, end
    return out


def _row_key(rows):
    """Return ``rows`` (a ``range`` or index array) as a numpy row index."""
    if isinstance(rows, range):
        return slice(rows.start, rows.stop if rows.stop >= 0 else None, rows.step)
    return rows


class V2View:
    """Lazy v2-grammar view of one table, or of row-aligned tables joined.

    Wraps numpy structured arrays, FITS_recs or h5py Datasets without reading
    them. ``view[name]`` returns one column as an array (computed on access for
    derived columns); a list of names returns a structured array of those
    columns; ``view[()]``, ``view[...]`` and any other row key (slice, boolean
    mask, index array) select rows and return a view over them.

    Selecting rows of a view over h5py Datasets reads the selected rows of
    every column into memory in one pass per dataset, as indexing the Dataset
    itself would, and wraps them in a view over the in-memory arrays: reading
    many columns of a selection one at a time would otherwise re-read the whole
    span of rows for each. Over in-memory tables, selection stays lazy.

    Column names are case-sensitive, unlike a FITS_rec's.
    """

    def __init__(self, bases, rows=None, dtype=None):
        self._bases = tuple(bases)
        n_rows = {len(base) for base in self._bases}
        if len(n_rows) != 1:
            raise ValueError(f"cannot join tables of lengths {sorted(n_rows)}")
        self._n_base = n_rows.pop()
        self._owner = {}
        for base in self._bases:
            for name in column_names(base):
                if name in self._owner:
                    raise ValueError(f"column {name!r} is in more than one table")
                self._owner[name] = base
        # None (all rows), a ``range`` or an integer index array.
        self._rows = rows
        self._names, self._derived = _resolve(tuple(self._owner))
        self._on_disk = any(isinstance(base, h5py.Dataset) for base in self._bases)
        self._dtype = dtype

    @property
    def names(self):
        """Presented column names."""
        return self._names

    def keys(self):
        """Presented column names."""
        return self._names

    @property
    def dtype(self):
        """Numpy dtype of the presented columns (found without reading rows)."""
        if self._dtype is None:
            self._dtype = np.dtype(
                [(name, self._field_dtype(name)) for name in self._names]
            )
        return self._dtype

    @property
    def shape(self):
        return (len(self),)

    def __len__(self):
        return self._n_base if self._rows is None else len(self._rows)

    def __contains__(self, name):
        return name in self._names

    def __repr__(self):
        return f"V2View({len(self)} rows, {len(self._names)} columns)"

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._column(key)
        if isinstance(key, list) and key and all(isinstance(k, str) for k in key):
            return self.to_structured(key)
        if isinstance(key, (int, np.integer)):
            return self[np.array([key])].to_structured()[0]
        if key is Ellipsis or (isinstance(key, tuple) and not key):
            key = slice(None)
        elif isinstance(key, tuple):
            raise IndexError(f"cannot select rows with the tuple {key!r}")
        rows = self._select(key)
        if self._on_disk:
            bases = [_take(base, rows) for base in self._bases]
            return V2View(bases, dtype=self._dtype)
        return V2View(self._bases, rows=rows, dtype=self._dtype)

    def __array__(self, dtype=None, copy=None):
        out = self.to_structured()
        return out if dtype is None else out.astype(dtype)

    def to_structured(self, names=None):
        """Materialise the presented columns as a numpy structured array.

        Over h5py Datasets, each dataset is read once, for all the columns
        requested of it.
        """
        names = self._names if names is None else list(names)
        missing = [name for name in names if name not in self._names]
        if missing:
            raise KeyError(f"columns {missing} not in catalogue")
        sources = {
            name: self._derived[name] if name in self._derived else (None, name)
            for name in names
        }
        if self._on_disk:
            fields = {}
            for _, source in sources.values():
                fields.setdefault(id(self._owner[source]), {})[source] = None
            taken = {}
            for base in self._bases:
                if id(base) in fields:
                    table = _take(base, self._rows, list(fields[id(base)]))
                    taken.update((f, table[f]) for f in table.dtype.names)
            read = taken.__getitem__
        else:
            read = self._read
        out = np.empty(len(self), dtype=[(n, self.dtype[n]) for n in names])
        for name, (rule, source) in sources.items():
            column = read(source)
            out[name] = column if rule is None else rule.apply(column)
        return out

    def _select(self, key):
        """Base rows of ``key`` applied to this view's rows."""
        n = len(self)
        if isinstance(key, slice):
            local = range(n)[key]
        else:
            key = np.asarray(key)
            if key.dtype == bool:
                if key.shape != (n,):
                    raise IndexError(f"boolean mask of shape {key.shape} for {n} rows")
                local = np.flatnonzero(key)
            elif key.size == 0:
                local = np.empty(0, dtype=np.intp)
            elif key.dtype.kind in "iu" and key.ndim == 1:
                local = np.where(key < 0, key + n, key).astype(np.intp)
                if local.min() < 0 or local.max() >= n:
                    raise IndexError(f"row index out of range for {n} rows")
            else:
                raise IndexError(f"cannot select rows with {key!r}")

        rows = self._rows
        if rows is None:
            return local
        if isinstance(rows, range):
            if isinstance(local, range):
                return range(
                    rows.start + local.start * rows.step,
                    rows.start + local.stop * rows.step,
                    rows.step * local.step,
                )
            return rows.start + local * rows.step
        if isinstance(local, range):
            local = np.arange(local.start, local.stop, local.step, dtype=np.intp)
        return rows[local]

    def _read(self, name):
        base = self._owner[name]
        rows = self._rows
        if rows is None:
            return base[name]
        if len(rows) == 0:
            return _read_window(base, name, 0, 0)
        if isinstance(rows, range):
            lo, hi = min(rows[0], rows[-1]), max(rows[0], rows[-1]) + 1
            window = _read_window(base, name, lo, hi)
            return window[rows.start - lo :: rows.step][: len(rows)]
        lo, hi = int(rows.min()), int(rows.max()) + 1
        return _read_window(base, name, lo, hi)[rows - lo]

    def _column(self, name):
        if name not in self._names:
            raise KeyError(f"column {name!r} not in catalogue")
        if name not in self._derived:
            return self._read(name)
        rule, source = self._derived[name]
        return rule.apply(self._read(source))

    def _field_dtype(self, name):
        """Dtype (with any subarray shape) ``_column(name)`` returns."""
        if name in self._derived:
            rule, source = self._derived[name]
            column = rule.apply(_read_window(self._owner[source], source, 0, 0))
        else:
            column = _read_window(self._owner[name], name, 0, 0)
        return np.dtype((column.dtype, column.shape[1:]))


def adapt(table, *tables):
    """Present ``table`` (joined with any further ``tables``) in the v2 grammar.

    A single table with nothing to rename, or already a ``V2View``, is returned
    unchanged; otherwise the result is a ``V2View``. Each table is anything
    whose ``dtype.names`` lists its columns and whose ``table[name]`` reads
    one: a numpy structured array, FITS_rec or h5py Dataset. Joined tables
    must have equal lengths and disjoint column names.
    """
    if not tables and (
        isinstance(table, V2View) or not _resolve(column_names(table))[1]
    ):
        return table
    return V2View((table,) + tables)


def materialise(table, names=None):
    """Return ``adapt(table)`` as an in-memory numpy structured array.

    ``names`` restricts the output to those v2-grammar columns.
    """
    view = adapt(table)
    if isinstance(view, V2View):
        return view.to_structured(names)
    view = view[()] if isinstance(view, h5py.Dataset) else view
    if names is None:
        return view
    out = np.empty(len(view), dtype=[(name, view.dtype[name]) for name in names])
    for name in names:
        out[name] = view[name]
    return out


def read_catalogue(path, hdu=1):
    """Read a FITS catalogue HDU and present it in the v2 grammar."""
    return adapt(fits.getdata(os.fspath(path), ext=hdu))


def read_column_names(path, hdu=1):
    """Return the v2-grammar column names of a FITS HDU, reading only its header."""
    header = fits.getheader(os.fspath(path), ext=hdu)
    names = [header[f"TTYPE{i}"] for i in range(1, header["TFIELDS"] + 1)]
    return v2_names(names)
