"""Static integrity checks for sp_validation's ASTRA decision record.

Adapted from ShapePipe's tests/helpers/decisions.py, without its tool-specific
readers or program analysis. Public checks return diagnostic lists, not pytest
assertions. No source is imported or executed.

Tags mark implementing sites, never tests (those carry pytest.mark.decision).
Tags require bracketed metadata with keys decision, label and scope; only
decision may repeat. Local ids use letters, digits, underscores, dots and
hyphens, and need not have prose. Paragraphs end at a blank line (not another
tag); overlapping tags count a physical setting once.
Path qualifiers are relative, component-aligned suffixes; # and :: are synonyms.
Absence selects exactly one same-decision tagged file and searches the entire
file (the named INI section, including DEFAULT inheritance). INI section names
are case-sensitive and keys are not, as CosmoSIS lower-cases them and merges
repeated sections; inline comments are stripped and interpolation is off. A YAML or
INI key, or a Python name rebound in the same scope, repeated in the file outside
the governed span fails: the last one wins.
Python settings are defaults, keywords, dict entries and assignments to names,
self attributes and constant string subscripts (``d["key"] = ...``).
Snakemake values use Python literal syntax: assignments, directive scalars and
keyword/dict entries in directive expressions. YAML sequence paths use numeric
indices. YAML aliases are rejected if recursive. Only inline ASTRA analyses are
read; this is an integrity check, not a general ASTRA schema validator.
"""

import ast
import os
import re
import textwrap
import tokenize
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path

import yaml

# Decisions whose every site lives in code the tomography merge rewrites; their
# tags land with that merge.
_MERGE = "sites in code the tomography merge (#374) rewrites"
PENDING_SITES = {
    "real_space.pair_approximation": _MERGE,
    "mocks.mock_cosmology_and_power": _MERGE,
    "mocks.mock_geometry": _MERGE,
    "mocks.mock_source_population": _MERGE,
}
REPO_ROOT = Path(__file__).resolve().parents[3]
_SKIP = {".git", "__pycache__", "scratch"}
_DATA = "src/sp_validation/tests/data"
_ID = r"[a-z][a-z0-9_]*"
_DECISION = re.compile(rf"{_ID}(?:\.{_ID})*\Z")
_TAG = re.compile(r"@sc\s+\[([^\]]*)\](?:\s+([A-Za-z_][\w.-]*))?\s*\Z")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_ABSENT = object()
_KEYS = {"decision", "label", "scope"}
_DECL = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass(frozen=True)
class Site:
    """Inclusive governed line span in a repository-relative file."""

    path: str
    start: int
    end: int


@dataclass(frozen=True)
class Tag:
    """One parsed tag with its declaration, statement or config span."""

    path: str
    line: int
    decisions: tuple[str, ...]
    ident: str | None
    meta: dict
    site: Site


@dataclass(frozen=True)
class Location:
    """A physical setting; error retains a matched but non-literal expression.

    Two settings of one ref and one non-None scope in a file are the same
    setting: the later decides the value read (a config key, or a Python name
    rebound in the same function, class or module body).
    """

    path: str
    line: int
    column: int
    ref: str
    value: object
    error: str = ""
    scope: object = None

    @property
    def identity(self):
        return self.path, self.line, self.column, self.ref

    def describe(self):
        actual = self.error or repr(self.value)
        return f"{self.path}:{self.line}:{self.column + 1} = {actual}"


def load_record(root):
    """Read the root record without constructing arbitrary YAML objects."""
    return yaml.safe_load((Path(root) / "astra.yaml").read_text(encoding="utf-8"))


def decisions(record, prefix=""):
    """Flatten inline sub-analyses into dotted decision ids."""
    result = {}
    for name, definition in (record.get("decisions") or {}).items():
        result[prefix + name] = definition or {}
    for name, analysis in (record.get("analyses") or {}).items():
        result.update(decisions(analysis, prefix + name + "."))
    return result


def source_paths(root):
    """Supported sources in the specified scan scope; never follow symlinks."""
    root = Path(root)
    bases = [
        root / p for p in ("src", "workflow", "cosmo_inference", "cosmo_val", "config")
    ]
    bases.extend(sorted((root / "papers").glob("*/config")))
    for base in bases:
        if base.is_symlink():
            continue
        for directory, dirs, files in os.walk(base, followlinks=False):
            parent = Path(directory)
            dirs[:] = sorted(
                d
                for d in dirs
                if d not in _SKIP
                and not (parent / d).is_symlink()
                and (parent / d).relative_to(root).as_posix() != _DATA
            )
            for name in sorted(files):
                path = parent / name
                if not path.is_symlink() and (
                    path.suffix in {".py", ".smk", ".yaml", ".yml", ".ini"}
                    or name == "Snakefile"
                ):
                    yield path


def _metadata(text):
    match = _TAG.fullmatch(text.strip())
    if not match:
        raise ValueError("malformed @sc tag")
    raw, ident = match.groups()
    meta = {}
    for part in raw.split(","):
        pair = re.fullmatch(r"\s*([A-Za-z_][\w.-]*):([^,\s]+)\s*", part)
        if not pair:
            raise ValueError("malformed @sc metadata")
        key, value = pair.groups()
        if key not in _KEYS:
            raise ValueError(f"unknown @sc metadata key {key}")
        if key in meta and key != "decision":
            raise ValueError(f"duplicate @sc metadata key {key}")
        if key == "decision" and not _DECISION.fullmatch(value):
            raise ValueError(f"malformed decision id {value!r}")
        meta.setdefault(key, []).append(value)
    if "scope" in meta and meta["scope"] != ["file"]:
        raise ValueError("scope must be scope:file")
    return meta, ident


def _comment(line, ini=False):
    stripped = line.lstrip()
    if stripped.startswith(("#", ";") if ini else ("#",)):
        return stripped[1:].lstrip()
    return None


def _following(lines, index, ini=False):
    for j in range(index + 1, len(lines)):
        if not lines[j].strip():
            raise ValueError("blank line between @sc tag and governed site")
        if _comment(lines[j], ini) is None:
            return j
    raise ValueError("@sc tag governs no site")


def _comment_span(path, lines, index, tree, file_scope):
    ini = Path(path).suffix == ".ini"
    start = _following(lines, index, ini)
    if file_scope:
        return Site(path, 1, len(lines))
    if tree is not None:
        # A decorated definition starts at its first decorator.
        statements = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.stmt)
            and min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
            == start + 1
        ]
        if not statements:
            raise ValueError("@sc tag is not immediately above a Python statement")
        node = min(statements, key=lambda n: n.col_offset)
        indent = len(lines[index]) - len(lines[index].lstrip())
        if node.col_offset != indent:
            raise ValueError("@sc tag and Python statement have different indentation")
        return Site(path, start + 1, node.end_lineno)
    rule = re.match(r"^(\s*)rule\s+\w+\s*:", lines[start])
    section = ini and re.match(r"\s*\[[^]]+\]", lines[start])
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if rule:
            if (
                lines[j].strip()
                and _comment(lines[j]) is None
                and len(lines[j]) - len(lines[j].lstrip()) <= len(rule[1])
            ):
                end = j
                break
        elif section:
            if re.match(r"\s*\[[^]]+\]", lines[j]):
                end = j
                break
        elif not lines[j].strip():
            end = j
            break
    return Site(path, start + 1, end)


def scan_tags(root):
    """Collect tags and path:line diagnostics for malformed/unattached tags."""
    tags, errors = [], []
    for path in source_paths(root):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        if "@sc" not in text:
            continue
        lines, tree, raw = text.splitlines(), None, []
        try:
            if path.suffix == ".py":
                tree = ast.parse(text, filename=relative)
                for owner in ast.walk(tree):
                    if not isinstance(owner, (ast.Module, *_DECL)) or (
                        not ast.get_docstring(owner)
                    ):
                        continue
                    doc = owner.body[0].value
                    for offset, line in enumerate(doc.value.splitlines()):
                        if not line.strip().startswith("@sc"):
                            continue
                        if isinstance(owner, ast.Module):
                            errors.append(
                                f"{relative}:{doc.lineno + offset}: @sc in a module "
                                "docstring governs nothing; use a # comment tag"
                            )
                        else:
                            raw.append((doc.lineno + offset, line, owner))
                for token in tokenize.generate_tokens(StringIO(text).readline):
                    if token.type != tokenize.COMMENT:
                        continue
                    body = token.string[1:].lstrip()
                    if body.startswith("@sc"):
                        number, column = token.start
                        if lines[number - 1][:column].strip():
                            errors.append(
                                f"{relative}:{number}: @sc needs its own line"
                            )
                        else:
                            raw.append((number, body, None))
            else:
                for number, line in enumerate(lines, 1):
                    body = _comment(line, path.suffix == ".ini")
                    if body is not None and body.startswith("@sc"):
                        raw.append((number, body, None))
        except (SyntaxError, tokenize.TokenError) as error:
            errors.append(f"{relative}:{getattr(error, 'lineno', 1)}: {error}")
            continue
        for number, body, owner in sorted(raw, key=lambda item: item[0]):
            try:
                meta, ident = _metadata(body)
                if "tests" in Path(relative).parts:
                    raise ValueError(
                        "@sc tags mark implementing sites, not tests; "
                        "mark the test with pytest.mark.decision"
                    )
                file_scope = meta.get("scope") == ["file"]
                if owner is not None:
                    site = Site(
                        relative,
                        1 if file_scope else owner.lineno,
                        len(lines) if file_scope else owner.end_lineno,
                    )
                else:
                    site = _comment_span(relative, lines, number - 1, tree, file_scope)
                tags.append(
                    Tag(
                        relative,
                        number,
                        tuple(meta.get("decision", ())),
                        ident,
                        meta,
                        site,
                    )
                )
            except ValueError as error:
                errors.append(f"{relative}:{number}: {error}")
    return tags, errors


def citation_errors(record, tags):
    """Every decision citation must name a recorded decision."""
    known = decisions(record)
    return [
        f"{t.path}:{t.line}: unknown decision {d!r}"
        for t in tags
        for d in t.decisions
        if d not in known
    ]


def coverage_errors(record, tags, pending=None):
    """Every decision needs a site; pending exceptions must not rot."""
    pending = {} if pending is None else pending
    known = decisions(record)
    cited = {d for t in tags for d in t.decisions}
    errors = [
        f"PENDING_SITES: unknown decision {d!r}" for d in pending if d not in known
    ]
    errors.extend(
        f"PENDING_SITES: {d}: needs a nonempty reason"
        for d, reason in pending.items()
        if not isinstance(reason, str) or not reason.strip()
    )
    errors.extend(
        f"PENDING_SITES: {d}: has a tagged site; remove the exception"
        for d in pending
        if d in cited
    )
    errors.extend(
        f"{d}: decision has no tagged site"
        for d in sorted(known.keys() - cited - pending.keys())
    )
    return errors


def contract_errors(tags):
    """Local-contract ids are globally unique, even on the same site."""
    seen, errors = {}, []
    for tag in tags:
        if tag.ident in seen:
            errors.append(
                f"duplicate local-contract id {tag.ident!r}: "
                f"{seen[tag.ident]} and {tag.path}:{tag.line}"
            )
        if tag.ident:
            seen[tag.ident] = f"{tag.path}:{tag.line}"
    return errors


def marker_errors(root, record):
    """Scan literal pytest.mark.decision calls in test sources, without import."""
    known, errors = decisions(record), []
    for path in source_paths(root):
        relative = path.relative_to(root)
        if path.suffix != ".py" or "tests" not in relative.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as error:
            errors.append(f"{relative}:{error.lineno}: cannot parse test source")
            continue
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or ast.unparse(call.func) != (
                "pytest.mark.decision"
            ):
                continue
            where = f"{relative}:{call.lineno}"
            if (
                len(call.args) != 1
                or call.keywords
                or not isinstance(call.args[0], ast.Constant)
                or not isinstance(call.args[0].value, str)
            ):
                errors.append(f"{where}: decision marker needs one literal string id")
            elif call.args[0].value not in known:
                errors.append(
                    f"{where}: decision marker cites unknown decision "
                    f"{call.args[0].value!r}"
                )
    return errors


def _normal(value):
    if isinstance(value, bool):
        return "bool", value
    if value is None:
        return "none", None
    if isinstance(value, (int, float, Decimal)):
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("numeric values must be finite")
        return "number", number
    if isinstance(value, str):
        return "string", value
    if isinstance(value, list):
        if any(isinstance(v, (list, dict, tuple)) for v in value):
            raise ValueError("only flat lists are supported")
        return "list", tuple(_normal(v) for v in value)
    raise ValueError("expected a scalar or flat list")


def _split(text, delimiter):
    """Split outside quoted strings and brackets, retaining literal punctuation."""
    result, start, quote, depth, escaped = [], 0, "", 0, False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == delimiter and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    if quote or depth:
        raise ValueError("unbalanced Values literal")
    result.append(text[start:].strip())
    return result


def _expected(text):
    text = text.strip()
    if not text:
        raise ValueError("empty Values literal")
    if text == "absent":
        return _ABSENT
    if _NUMBER.fullmatch(text):
        return _normal(Decimal(text))
    if text in {"True", "False", "None"}:
        return _normal(ast.literal_eval(text))
    if text[0] in "\"'":
        return _normal(ast.literal_eval(text))
    if text.startswith("[") and text.endswith("]"):
        items = _split(text[1:-1], ",") if text[1:-1].strip() else []
        values = tuple(_expected(v) for v in items)
        if any(v is _ABSENT or v[0] == "list" for v in values):
            raise ValueError("only flat lists of literals are supported")
        return "list", values
    if any(c in text for c in "[]{}"):
        raise ValueError("expected a scalar or flat list")
    return "string", text


def _python_literal(node, source):
    value = ast.literal_eval(node)
    _normal(value)  # Reject containers outside the scalar/flat-list grammar.
    if isinstance(node, ast.List):
        return "list", tuple(_python_literal(n, source) for n in node.elts)
    if isinstance(value, float):
        # Keep source precision instead of rounding through a binary float.
        literal = ast.get_source_segment(source, node).replace("_", "")
        literal = re.sub(r"\s+", "", literal)
        return _normal(Decimal(literal))
    return _normal(value)


def _python_locations(path, source, tree, offset=0):
    scopes = {}
    if Path(path).suffix == ".py":
        # Name bindings are scoped by their enclosing function, class or module.
        for owner in ast.walk(tree):
            if isinstance(owner, (ast.Module, *_DECL)):
                for statement in ast.walk(owner):
                    if statement is not owner and isinstance(statement, ast.stmt):
                        scopes[statement] = getattr(owner, "lineno", 0)

    def location(ref, key, value, scope=None):
        try:
            actual, error = _python_literal(value, source), ""
        except (ValueError, TypeError, InvalidOperation):
            actual = ast.get_source_segment(source, value)
            error = f"non-literal {actual!r}"
        line = key.lineno + offset
        return Location(path, line, key.col_offset, ref, actual, error, scope)

    for node in ast.walk(tree):
        pairs = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            positional = args.posonlyargs + args.args
            pairs.extend(
                (arg.arg, arg, default)
                for arg, default in zip(
                    positional[-len(args.defaults) :], args.defaults
                )
            )
            pairs.extend(
                (arg.arg, arg, default)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults)
                if default is not None
            )
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                name = (
                    target.id
                    if isinstance(target, ast.Name)
                    else (
                        target.attr
                        if isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                        else None
                    )
                )
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    name = target.slice.value  # d["key"] = ... sets a dict entry.
                if name and node.value is not None:
                    # An augmented assignment isn't a literal setting.
                    value = node if isinstance(node, ast.AugAssign) else node.value
                    scope = scopes.get(node) if isinstance(target, ast.Name) else None
                    pairs.append((name, target, value, scope))
        elif isinstance(node, ast.keyword) and node.arg:
            pairs.append((node.arg, node, node.value))
        elif isinstance(node, ast.Dict):
            pairs.extend(
                (key.value, key, value)
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
        for ref, key, value, *scope in pairs:
            yield location(ref, key, value, *scope)


def _snakemake_locations(path, source):
    """Read Python-like paragraphs and rule directives, never parse a workflow."""
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        directive = re.match(r"^(\s*)(\w+)\s*:\s*(.*)$", line)
        if directive:
            indent, ref, first = directive.groups()
            end = index + 1
            while end < len(lines) and (
                not lines[end].strip()
                or len(lines[end]) - len(lines[end].lstrip()) > len(indent)
            ):
                end += 1
            expression = (
                first + "\n" + textwrap.dedent("\n".join(lines[index + 1 : end]))
            )
            try:
                node = ast.parse(expression.strip(), mode="eval").body
                actual = _python_literal(node, expression.strip())
                yield Location(path, index + 1, len(indent), ref, actual)
            except (SyntaxError, ValueError, TypeError, InvalidOperation):
                if first.strip():
                    yield Location(
                        path,
                        index + 1,
                        len(indent),
                        ref,
                        expression,
                        f"non-literal {expression.strip()!r}",
                    )
            wrapped = "_sc_(" + expression + ")"
            try:
                tree = ast.parse(wrapped)
                yield from _python_locations(path, wrapped, tree, index)
            except SyntaxError:
                if not first.strip():
                    block = textwrap.dedent("\n".join(lines[index + 1 : end]))
                    try:
                        yield from _python_locations(
                            path, block, ast.parse(block), index + 1
                        )
                    except SyntaxError:
                        pass
            index = end
        else:
            end = index + 1
            while end < len(lines) and lines[end].strip():
                if re.match(r"\s*(?:\w+\s*:|rule\s+\w+\s*:)", lines[end]):
                    break
                end += 1
            fragment = textwrap.dedent("\n".join(lines[index:end]))
            try:
                yield from _python_locations(path, fragment, ast.parse(fragment), index)
            except SyntaxError:
                pass
            index = end


def _yaml_locations(path, source):
    loader = yaml.SafeLoader(source)
    try:
        root = loader.get_single_node()

        def walk(node, prefix=(), ancestors=()):
            if node in ancestors:
                raise ValueError("recursive YAML alias")
            if isinstance(node, yaml.MappingNode):
                for key, value in node.value:
                    name = loader.construct_object(key, deep=True)
                    full = (*prefix, str(name))
                    try:
                        actual = _normal(loader.construct_object(value, deep=True))
                        if (
                            isinstance(value, yaml.ScalarNode)
                            and actual[0] == "number"
                            and _NUMBER.fullmatch(value.value.replace("_", ""))
                        ):
                            actual = _normal(Decimal(value.value.replace("_", "")))
                        error = ""
                    except ValueError as problem:
                        actual, error = None, str(problem)
                    yield Location(
                        path,
                        key.start_mark.line + 1,
                        key.start_mark.column,
                        ".".join(full),
                        actual,
                        error,
                        "file",
                    )
                    yield from walk(value, full, (*ancestors, node))
            elif isinstance(node, yaml.SequenceNode):
                for index, value in enumerate(node.value):
                    yield from walk(value, (*prefix, str(index)), (*ancestors, node))

        if root is not None:
            yield from walk(root)
    finally:
        loader.dispose()


def _ini_value(text):
    for convert in (int, float):
        try:
            value = convert(text)
        except ValueError:
            continue
        if _NUMBER.fullmatch(text):
            value = Decimal(text)
        return _normal(value)
    words = {
        "t": True,
        "f": False,
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
    }
    return _normal(words.get(text.lower(), text.strip()))


def _uncomment(line):
    """Drop an inline `;`/`#` comment that follows whitespace, as CosmoSIS does."""
    return re.split(r"(?<=\s)[;#]", line, maxsplit=1)[0]


def _ini_locations(path, source):
    section = "DEFAULT"
    lines = source.splitlines()
    continuation_end = 0
    for number, line in enumerate(lines, 1):
        if number <= continuation_end:
            continue
        if not line.strip() or _comment(line, True) is not None:
            continue
        header = re.fullmatch(r"\s*\[([^]]+)\]\s*(?:[#;].*)?", line)
        if header:
            section = header[1].strip()
            continue
        setting = re.match(r"\s*([^=:\s][^=:]*?)\s*[=:]\s*(.*)$", _uncomment(line))
        if setting:
            key, value = setting.groups()
            indent = len(line) - len(line.lstrip())
            parts = [value.strip()]
            for j in range(number, len(lines)):
                following = lines[j]
                if not following.strip() or _comment(following, True) is not None:
                    continue
                if len(following) - len(following.lstrip()) <= indent:
                    break
                parts.append(_uncomment(following).strip())
                continuation_end = j + 1
            yield Location(
                path,
                number,
                indent,
                section + "." + key.strip(),
                _ini_value("\n".join(parts)),
                scope="file",
            )


def _locations(root, path):
    source = (Path(root) / path).read_text(encoding="utf-8")
    suffix = Path(path).suffix
    if suffix == ".py":
        return list(_python_locations(path, source, ast.parse(source)))
    if suffix in {".yaml", ".yml"}:
        return list(_yaml_locations(path, source))
    if suffix == ".ini":
        return list(_ini_locations(path, source))
    return list(_snakemake_locations(path, source))


def _reference(text):
    parts = re.split(r"#|::", text, maxsplit=1)
    qualifier, ref = parts if len(parts) == 2 else ("", parts[0])
    if not ref or re.search(r"\s|=", ref):
        raise ValueError("invalid Values ref")
    if len(parts) == 2 and (
        not qualifier or Path(qualifier).is_absolute() or ".." in Path(qualifier).parts
    ):
        raise ValueError("qualifier must be a relative path suffix")
    return qualifier.removeprefix("./"), ref


def _same_setting(path, actual, ref):
    """CosmoSIS lower-cases INI keys (not sections); other refs match exactly."""
    if Path(path).suffix != ".ini":
        return actual == ref
    section, _, key = actual.rpartition(".")
    want_section, _, want_key = ref.rpartition(".")
    return section == want_section and key.lower() == want_key.lower()


def value_errors(root, record, tags):
    """Resolve each Values entry once, with exact typed decimal equality."""
    errors, cache = [], {}
    for decision, definition in decisions(record).items():
        rationale = definition.get("rationale", "")
        if not isinstance(rationale, str) or "Values:" not in rationale:
            continue
        try:
            if rationale.count("Values:") != 1 or not rationale.rstrip().endswith("."):
                raise ValueError("rationale needs one terminal Values: sentence")
            entries = _split(rationale.split("Values:", 1)[1].strip()[:-1], ";")
        except ValueError as error:
            errors.append(f"{decision}: {error}")
            continue
        for entry in entries:
            try:
                raw_ref, expected = entry.split("=", 1)
                raw_ref, expected = raw_ref.strip(), expected.strip()
                qualifier, ref = _reference(raw_ref)
                want = _expected(expected)
                sites = {
                    t.site
                    for t in tags
                    if decision in t.decisions
                    and (
                        not qualifier
                        or t.path == qualifier
                        or t.path.endswith("/" + qualifier)
                    )
                }
                paths = {s.path for s in sites}
                found, shadows = {}, {}
                for path in sorted(paths):
                    if path not in cache:
                        cache[path] = _locations(root, path)
                    for loc in cache[path]:
                        matches = _same_setting(path, loc.ref, ref)
                        if want is _ABSENT and Path(path).suffix == ".ini":
                            matches |= _same_setting(
                                path, loc.ref, "DEFAULT." + ref.rsplit(".", 1)[-1]
                            )
                        if matches and (
                            want is _ABSENT
                            or any(
                                s.path == path and s.start <= loc.line <= s.end
                                for s in sites
                            )
                        ):
                            found[loc.identity] = loc
                        elif matches and loc.scope is not None:
                            shadows[loc.identity] = loc
                candidates = list(found.values())
                shadows = [
                    loc
                    for loc in shadows.values()
                    if any(
                        (c.path, c.scope) == (loc.path, loc.scope) for c in candidates
                    )
                ]
                detail = "; ".join(loc.describe() for loc in candidates) or "[]"
                context = (
                    f"{decision}: ref {raw_ref!r}: expected {expected}; "
                    f"candidates found {len(candidates)}: {detail}"
                )
                if want is _ABSENT:
                    if len(paths) != 1:
                        errors.append(
                            context + "; absence requires exactly one "
                            "same-decision tagged file; "
                            f"found {sorted(paths)}"
                        )
                    elif (
                        Path(next(iter(paths))).suffix == ".ini"
                        and ref.rsplit(".", 1)[0] != "DEFAULT"
                        and not re.search(
                            rf"^\s*\[{re.escape(ref.rsplit('.', 1)[0])}\]",
                            (Path(root) / next(iter(paths))).read_text(
                                encoding="utf-8"
                            ),
                            re.MULTILINE,
                        )
                    ):
                        errors.append(
                            context + "; absence requires an existing section"
                        )
                    elif candidates:
                        errors.append(context + "; expected no active setting")
                elif len(candidates) != 1:
                    errors.append(
                        context + "; ref must resolve to exactly one location"
                    )
                elif shadows:
                    errors.append(
                        context + "; the same setting is repeated outside the "
                        "governed span: " + "; ".join(loc.describe() for loc in shadows)
                    )
                elif candidates[0].error:
                    errors.append(
                        context + "; matched value is non-literal or unsupported"
                    )
                elif candidates[0].value != want:
                    errors.append(context + "; value mismatch")
            except (ValueError, TypeError, SyntaxError, yaml.YAMLError) as error:
                errors.append(f"{decision}: Values entry {entry!r}: {error}")
    return errors


def repository_errors(root, pending=None):
    """Run all six integrity checks (missing records are handled by pytest).

    ``pending`` defaults to ``PENDING_SITES`` for sp_validation's own checkout
    and to no exceptions for any other root, such as a test fixture.
    """
    if pending is None:
        pending = PENDING_SITES if Path(root).resolve() == REPO_ROOT else {}
    record = load_record(root)
    tags, errors = scan_tags(root)
    return (
        errors
        + citation_errors(record, tags)
        + coverage_errors(record, tags, pending)
        + value_errors(root, record, tags)
        + contract_errors(tags)
        + marker_errors(root, record)
    )
