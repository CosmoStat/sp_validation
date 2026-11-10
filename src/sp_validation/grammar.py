"""Column grammars: the v2 schema, and the maps that present a table in it.

ShapePipe products name their columns in one of two grammars. ``v2`` is the
grammar the rest of sp_validation reads (``HSM_T_PSF``, ``NGMIX_G1_NOSHEAR``,
``MASK_4_Stars``, ...; see ``docs/ngmix_psf_column_migration.md``). ``v1`` is the
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

- The UNIONS mask-bit columns, applied to any table. Bit ``b`` of the
  healsparse mask product is the boolean ``MASK_{b}_{label}``
  (``MASK_1_Faint_star_halos``, ``MASK_4_Stars``, ``MASK_2048_z2``, ...;
  ``mask_column``, labels from ``MASK_LABELS``). Two other spellings of the
  same bit are presented under that name: ``{b}_{label}``, the ``data_ext``
  dataset of every released v1 comprehensive HDF5, and ``MASK_n{b}``, written
  by pre-release ShapePipe v2 runs. These columns belong to the mask product,
  not to a ShapePipe generation, so no spelling marks or requires one.

``IMAFLAGS_ISO`` is not mapped: its v1 bits (2 halo, 4 border, 16 Messier,
32 NGC, 128 spike) differ in meaning from the mask bits of the same value,
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

A catalogue in another naming convention can carry a ``column_map``,
``{v2 name: name in the file}``, which ``adapt`` applies as renames ahead of
the rules above: an entry overrides the rule producing the same v2 name and no
other (a column it renames still feeds the other rules that read it), and the
v1 detection runs on the columns the map does not rename. An entry may hold one
``*`` on both sides, matching any stem: ``{"NGMIX_*": "MYFIT_*"}`` presents
every ``MYFIT_<stem>`` column as ``NGMIX_<stem>``. Explicit entries take
precedence over pattern entries.

Every other column passes through under its own name; source names that have a
v2 equivalent are hidden.

``adapt`` also joins row-aligned tables into one view, as the comprehensive
HDF5 splits one catalogue over its ``data`` and ``data_ext`` datasets.
"""

from dataclasses import dataclass

import numpy as np
from cs_util.size import sigma_to_T

from sp_validation import tables

#: Metacal shear types carried by the ngmix columns.
SHEARS = ("NOSHEAR", "1P", "1M", "2P", "2M")

#: Bits of the UNIONS healsparse mask product and what each flags. The column
#: for bit ``b`` is ``MASK_{b}_{label}`` (``mask_column``).
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
    """Return the column holding mask bit ``bit``: ``MASK_{bit}_{label}``.

    Raises
    ------
    KeyError
        if ``bit`` is not a bit of the mask product
    """
    if bit not in MASK_LABELS:
        raise KeyError(f"{bit} is not a UNIONS mask bit; bits: {list(MASK_LABELS)}")
    return f"MASK_{bit}_{MASK_LABELS[bit]}"


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

#: ``{b}_{label}`` (v1 ``data_ext``) or ``MASK_n{b}`` (pre-release v2)
#: -> ``MASK_{b}_{label}``, applied to every table.
MASK_RULES = tuple(
    Rule(mask_column(bit), f"{bit}_{label}", fallback=f"MASK_n{bit}")
    for bit, label in MASK_LABELS.items()
)

#: Names whose presence marks a table as v1 / as v2.
V1_MARKERS = frozenset(name for rule in V1_RULES for name in rule.sources())
V2_MARKERS = frozenset(rule.v2 for rule in V1_RULES)


def column_names(table):
    """Return the column names of a table, structured array, FITS_rec or h5py
    Dataset."""
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


def _map_rules(names, column_map):
    """Return the rename rules ``column_map`` yields for a table's ``names``.

    Entries whose file column is absent yield nothing, so one map serves
    tables that carry only some of its columns.
    """
    if not column_map:
        return ()
    rules = {}
    patterns = []
    for v2, source in column_map.items():
        if "*" not in v2 and "*" not in source:
            if source in names and source != v2:
                rules[v2] = Rule(v2, source)
            continue
        if v2.count("*") != 1 or source.count("*") != 1:
            raise ValueError(
                f"column_map pattern {v2!r}: {source!r} needs exactly one '*'"
                + " on each side"
            )
        patterns.append((v2.split("*"), source.split("*")))
    for (v2_head, v2_tail), (head, tail) in patterns:
        for name in names:
            if (
                len(name) >= len(head) + len(tail)
                and name.startswith(head)
                and name.endswith(tail)
            ):
                stem = name[len(head) : len(name) - len(tail)]
                v2 = f"{v2_head}{stem}{v2_tail}"
                if v2 != name:
                    rules.setdefault(v2, Rule(v2, name))
    return tuple(rules.values())


def _resolve(names, column_map=None):
    """Return (presented names, {v2 name: (rule, source name)}) for ``names``."""
    mapped = _map_rules(names, column_map)
    mapped_sources = {rule.source for rule in mapped}
    rules = MASK_RULES
    if detect_generation([n for n in names if n not in mapped_sources]) == "v1":
        rules = V1_RULES + MASK_RULES
    claimed = {rule.v2 for rule in mapped}
    rules = mapped + tuple(rule for rule in rules if rule.v2 not in claimed)
    position = {name: i for i, name in enumerate(names)}
    derived = {}
    consumed = set()
    by_anchor = {}
    for rule in rules:
        sources = [s for s in rule.sources() if s in position]
        if not sources:
            continue
        if rule.v2 in position and rule.v2 not in mapped_sources:
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


def v2_names(names, column_map=None):
    """Return the column names a table with columns ``names`` presents.

    Header-only counterpart of ``adapt``.
    """
    return _resolve(tuple(names), column_map)[0]


class V2View:
    """Lazy v2-grammar view of one table, or of row-aligned tables joined.

    The tables are anything ``sp_validation.tables`` describes (numpy
    structured arrays, FITS_recs and h5py Datasets are wrapped on the way
    in), read only when indexed. ``view[name]`` returns one column as an
    array (computed on access for derived columns); a list of names returns a
    structured array of those columns; ``view[()]``, ``view[...]`` and any
    other row key (slice, boolean mask, index array) select rows and return a
    view over them.

    Selecting rows of a view over a ``row_major`` table (an HDF5 dataset)
    reads the selected rows of every column in one pass, and returns a view
    over them in memory: reading many columns of a selection one at a time
    would otherwise re-read the whole span of rows for each. Over any other
    table, selection stays lazy.

    A view is itself a table, so a view can be presented again under a
    ``column_map``. Column names are case-sensitive, unlike a FITS_rec's.
    """

    def __init__(self, bases, rows=None, dtype=None, column_map=None):
        self._bases = tuple(tables.as_table(base) for base in bases)
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
        self._column_map = column_map
        self._names, self._derived = _resolve(tuple(self._owner), column_map)
        self._dtype = dtype
        self.row_major = any(base.row_major for base in self._bases)

    @property
    def names(self):
        """Presented column names."""
        return self._names

    def keys(self):
        """Presented column names."""
        return self._names

    @property
    def renames(self):
        """Whether any presented column differs from the stored one of its name."""
        return bool(self._derived)

    @property
    def dtype(self):
        """Numpy dtype of the presented columns (found without reading rows)."""
        if self._dtype is None:
            if not self._derived and len(self._bases) == 1:
                self._dtype = self._bases[0].dtype
            else:
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
            return self.read([key])[key]
        if isinstance(key, list) and key and all(isinstance(k, str) for k in key):
            return self.to_structured(key)
        if isinstance(key, (int, np.integer)):
            return self[np.array([key])].to_structured()[0]
        if key is Ellipsis or (isinstance(key, tuple) and not key):
            key = slice(None)
        elif isinstance(key, tuple):
            raise IndexError(f"cannot select rows with the tuple {key!r}")
        rows = self._base_rows(self._local_rows(key))
        if self.row_major:
            bases = [
                tables.ArrayTable(tables.read_structured(base, rows=rows))
                for base in self._bases
            ]
            return V2View(bases, dtype=self._dtype, column_map=self._column_map)
        return V2View(
            self._bases, rows=rows, dtype=self._dtype, column_map=self._column_map
        )

    def __array__(self, dtype=None, copy=None):
        out = self.to_structured()
        return out if dtype is None else out.astype(dtype)

    def to_structured(self, names=None):
        """Materialise the presented columns as a numpy structured array.

        Each table is read once, for all the columns requested of it.
        """
        names = self._names if names is None else list(names)
        return tables.read_structured(self, names)

    def read(self, names, rows=None, out=None):
        """Return {name: column} for presented ``names`` at ``rows`` of this
        view (the table interface; see ``sp_validation.tables``).

        Each table is read once, for all the columns requested of it; a
        column presented under a new name only is read straight into ``out``.
        """
        names = list(names)
        missing = [name for name in names if name not in self._names]
        if missing:
            raise KeyError(f"columns {missing} not in catalogue")
        rows = self._rows if rows is None else self._base_rows(rows)
        out = out or {}
        sources = {name: self._derived.get(name, (None, name)) for name in names}
        wanted = {}
        for name, (rule, source) in sources.items():
            targets = wanted.setdefault(id(self._owner[source]), {})
            if name in out and (rule is None or rule.kind == "rename"):
                targets.setdefault(source, out[name])
            else:
                targets.setdefault(source, None)
        values = {}
        for base in self._bases:
            if id(base) in wanted:
                targets = wanted[id(base)]
                into = {s: t for s, t in targets.items() if t is not None}
                values.update(base.read(list(targets), rows, out=into))
        columns = {}
        for name, (rule, source) in sources.items():
            column = values[source] if rule is None else rule.apply(values[source])
            if name in out and column is not out[name]:
                out[name][...] = column
                column = out[name]
            columns[name] = column
        return columns

    def _local_rows(self, key):
        """Rows of this view that row key ``key`` selects."""
        n = len(self)
        if isinstance(key, slice):
            return range(n)[key]
        key = np.asarray(key)
        if key.dtype == bool:
            if key.shape != (n,):
                raise IndexError(f"boolean mask of shape {key.shape} for {n} rows")
            return np.flatnonzero(key)
        if key.size == 0:
            return np.empty(0, dtype=np.intp)
        if key.dtype.kind in "iu" and key.ndim == 1:
            local = np.where(key < 0, key + n, key).astype(np.intp)
            if local.min() < 0 or local.max() >= n:
                raise IndexError(f"row index out of range for {n} rows")
            return local
        raise IndexError(f"cannot select rows with {key!r}")

    def _base_rows(self, local):
        """Rows of the tables that this view's rows ``local`` are."""
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
            return rows.start + np.asarray(local, dtype=np.intp) * rows.step
        if isinstance(local, range):
            local = np.arange(local.start, local.stop, local.step, dtype=np.intp)
        return rows[local]

    def _field_dtype(self, name):
        """Dtype (with any subarray shape) a read of column ``name`` returns."""
        rule, source = self._derived.get(name, (None, name))
        dtype = self._owner[source].dtype[source]
        if rule is None:
            return dtype
        column = rule.apply(np.empty(0, dtype=dtype))
        return np.dtype((column.dtype, column.shape[1:]))


def adapt(table, *tables_, column_map=None):
    """Present ``table`` (joined with any further tables) in the v2 grammar.

    Returns a ``V2View``; a ``V2View`` given alone and no ``column_map`` is
    returned unchanged. Each table is a table (``sp_validation.tables``), a
    numpy structured array, FITS_rec or h5py Dataset. Joined tables must have
    equal lengths and disjoint column names. ``column_map`` (``{v2 name:
    name in the table}``, see the module docstring) renames a foreign
    convention's columns; over a ``V2View`` it renames the columns that view
    presents.
    """
    if not tables_ and isinstance(table, V2View) and column_map is None:
        return table
    return V2View((table,) + tables_, column_map=column_map)


def materialise(table, names=None):
    """Return ``adapt(table)`` as an in-memory numpy structured array.

    ``names`` restricts the output to those v2-grammar columns.
    """
    return adapt(table).to_structured(names)
