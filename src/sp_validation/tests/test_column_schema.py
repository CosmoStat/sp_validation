"""The column schema and the code that reads catalogues stay in step.

``config/columns/shapepipe_v2.yaml`` lists every catalogue column
sp_validation reads by fixed name. A static scan of the reading code checks
both directions:

- every column-shaped key the code reads (a subscript or ``get_col`` argument
  that is ShapePipe-prefixed or upper case) is in the schema, or is listed in
  ``NOT_COLUMNS`` as a key of something other than a catalogue;
- every schema entry is read somewhere: it appears in the code as a string, or
  matches an f-string template such as ``f"NGMIX_G{i}_{shear}"``.

``grammar.py`` is left out: the names it holds are ShapePipe v1 source names,
which it maps onto the schema.
"""

import ast
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = ROOT / "config" / "columns" / "shapepipe_v2.yaml"
READING_CODE = [
    path
    for path in (ROOT / "src" / "sp_validation").rglob("*.py")
    if "tests" not in path.parts and path.name != "grammar.py"
] + [
    ROOT / "scripts" / "calibration" / "extract_info.py",
    ROOT / "scripts" / "calibration" / "calibrate_comprehensive_cat.py",
]

#: Column-shaped keys that index something other than a catalogue.
NOT_COLUMNS = {
    # cat_config and other configuration keys
    "A",
    "R",
    "R11",
    "R22",
    # FITS header keywords
    "EXTNAME",
    "R_S11",
    "R_S12",
    "R_S21",
    "R_S22",
    "TTYPE{}",
    # pseudo-Cl, COSEBIs and SACC products
    "BB",
    "COSEBIS",
    "COVAR_BB_BB",
    "COVAR_EE_EE",
    "COVAR_{}_{}",
    "COVDATA",
    "ELL",
    "NAME_{}",
    "STRT_{}",
    # in-memory tables sp_validation builds itself
    "C11",
    "C22",
    "SNR",
    "T",
    # NaMaster field/workspace dicts keyed by bin; cosmology parameter dicts
    "W{}",
    "H0",
    "A_IA",
    "S8",
    # GLASS mock catalogues, which are not ShapePipe products
    "TOM_BIN_ID",
}

#: An f-string template needs this many literal characters to count as
#: reading a schema name, so that e.g. ``f"{a}_{b}"`` matches nothing.
MIN_TEMPLATE_LITERAL = 4

COLUMN_SHAPED = re.compile(r"^(NGMIX|HSM|MASK)_|^[A-Z][A-Z0-9_{}]*$")


def _text(node):
    """Return a string constant, or an f-string with each field as ``{}``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{}"
            for part in node.values
        )
    return None


def _template_regex(text):
    if "{}" not in text:
        return None
    if len(text.replace("{}", "")) < MIN_TEMPLATE_LITERAL:
        return None
    return re.compile(
        "^" + ".+".join(re.escape(part) for part in text.split("{}")) + "$"
    )


def _scan():
    """Return (every string in the code, the column-shaped keys it reads)."""
    strings, keys = set(), set()
    for path in READING_CODE:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            text = _text(node)
            if text is not None:
                strings.add(text)
            if isinstance(node, ast.Subscript):
                candidates = [node.slice]
            elif isinstance(node, ast.Call) and getattr(
                node.func, "attr", getattr(node.func, "id", None)
            ) in ("get_col", "get_maked_col"):
                candidates = node.args[1:2]
            else:
                continue
            for candidate in candidates:
                text = _text(candidate)
                if text is not None and COLUMN_SHAPED.match(text):
                    keys.add(text)
    return strings, keys


@pytest.fixture(scope="module")
def schema():
    return yaml.safe_load(SCHEMA.read_text())


@pytest.fixture(scope="module")
def scan():
    return _scan()


def _names(schema):
    return {name for kind in schema.values() for name in kind["columns"]}


def test_schema_layout(schema):
    for kind, entry in schema.items():
        assert set(entry) == {"about", "columns"}, kind
        for name, meaning in entry["columns"].items():
            assert isinstance(meaning, str) and meaning, (kind, name)


def test_code_reads_only_schema_columns(schema, scan):
    _, keys = scan
    names = _names(schema)
    unknown = set()
    for key in keys - NOT_COLUMNS:
        regex = _template_regex(key)
        if regex is None:
            if key not in names:
                unknown.add(key)
        elif not any(regex.match(name) for name in names):
            unknown.add(key)
    assert not unknown, (
        f"code reads columns missing from {SCHEMA.name}: {sorted(unknown)};"
        + " add them there, or to NOT_COLUMNS if they index no catalogue"
    )


def test_schema_holds_only_columns_the_code_reads(schema, scan):
    from sp_validation.galaxy import DEFAULT_MASK_COLUMNS

    strings, _ = scan
    strings = strings | set(DEFAULT_MASK_COLUMNS)
    templates = [r for r in map(_template_regex, strings) if r is not None]
    unread = sorted(
        name
        for name in _names(schema)
        if name not in strings and not any(r.match(name) for r in templates)
    )
    assert not unread, f"{SCHEMA.name} lists columns no code reads: {unread}"


def test_not_columns_are_still_used(scan):
    _, keys = scan
    assert NOT_COLUMNS <= keys, sorted(NOT_COLUMNS - keys)
