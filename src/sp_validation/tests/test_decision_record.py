"""Positive mini-repo and single-fault mutations of decision integrity rules."""

import shutil
from pathlib import Path

import pytest

from sp_validation.tests import decision_record as dr

pytestmark = pytest.mark.fast
FIXTURE = Path(__file__).parent / "data" / "decision_record" / "positive"
ROOT = Path(__file__).resolve().parents[3]
PYTHON = "src/model.py"
YAML = "config/settings.yaml"
INI = "cosmo_inference/settings.ini"
SMK = "workflow/rules.smk"
MARKER = "src/sp_validation/tests/test_link.py"


@pytest.fixture
def mini_repo(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    assert dr.repository_errors(tmp_path) == []
    return tmp_path


def replace(root, path, old, new):
    target = root / path
    text = target.read_text(encoding="utf-8")
    assert text.count(old) == 1, (path, old)
    target.write_text(text.replace(old, new), encoding="utf-8")


def append(root, path, text):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(text)


def snapshot(root):
    record = dr.load_record(root)
    tags, errors = dr.scan_tags(root)
    assert errors == [], errors
    return record, tags


def assert_problem(errors, *fragments):
    assert any(all(f in error for f in fragments) for error in errors), errors


def value_problem(root, *fragments):
    record, tags = snapshot(root)
    assert_problem(dr.value_errors(root, record, tags), *fragments)


def test_positive_fixture():
    assert dr.repository_errors(FIXTURE) == []
    tags, errors = dr.scan_tags(FIXTURE)
    assert not errors
    assert {Path(t.path).suffix for t in tags} == {".py", ".yaml", ".smk", ".ini"}
    assert {t.ident for t in tags if t.ident} == {
        "module-contract",
        "fit-contract",
        "yaml-contract",
        "sampler-contract",
        "rule-contract",
    }
    assert "calibration.validation.noise" in dr.decisions(dr.load_record(FIXTURE))


@pytest.mark.parametrize(
    "path, old",
    [
        (PYTHON, "# @sc [decision:statement,label:selection] module-contract"),
        (YAML, "# @sc [decision:yaml_settings] yaml-contract"),
        (INI, "; @sc [decision:ini_section] sampler-contract"),
        (SMK, "# @sc [decision:workflow_rule] rule-contract"),
    ],
)
def test_malformed_tag_reports_path_and_line(mini_repo, path, old):
    line = (mini_repo / path).read_text().splitlines().index(old) + 1
    replace(mini_repo, path, old, old.replace("decision:", "decision "))
    _, errors = dr.scan_tags(mini_repo)
    assert_problem(errors, f"{path}:{line}", "malformed @sc metadata")


@pytest.mark.parametrize(
    "tag, message",
    [
        ("@sc decision:statement", "malformed @sc tag"),
        ("@sc []", "malformed @sc metadata"),
        ("@sc [decision:BadId]", "malformed decision id"),
        ("@sc [decision:bad-id]", "malformed decision id"),
        ("@sc [decision:statement,scope:paragraph]", "scope must be scope:file"),
        ("@sc [decision:statement,label:a,label:b]", "duplicate @sc metadata key"),
        ("@sc [decision:statement] id extra", "malformed @sc tag"),
    ],
)
def test_bad_tag_grammar(mini_repo, tag, message):
    replace(
        mini_repo,
        PYTHON,
        "@sc [decision:statement,label:selection] module-contract",
        tag,
    )
    _, errors = dr.scan_tags(mini_repo)
    assert_problem(errors, "src/model.py:", message)


@pytest.mark.parametrize(
    "path, anchor",
    [
        (PYTHON, "module_value = 8"),
        (PYTHON, "    local_value = 9"),
        (YAML, "cosmo_val:"),
        (INI, "[sampler]"),
        (INI, "count = 13"),
        (SMK, "rule validate:"),
        (SMK, "SCALE = 5e-4"),
    ],
)
def test_blank_line_detaches_site(mini_repo, path, anchor):
    replace(mini_repo, path, anchor, "\n" + anchor)
    _, errors = dr.scan_tags(mini_repo)
    assert_problem(errors, path + ":", "blank line between @sc tag and governed site")


def test_tag_at_eof_has_no_site(mini_repo):
    append(mini_repo, PYTHON, "\n# @sc [decision:statement]\n")
    _, errors = dr.scan_tags(mini_repo)
    assert_problem(errors, PYTHON, "governs no site")


def test_inline_python_tag_is_rejected(mini_repo):
    replace(
        mini_repo,
        PYTHON,
        "module_value = 8",
        "module_value = 8  # @sc [decision:statement]",
    )
    _, errors = dr.scan_tags(mini_repo)
    assert_problem(errors, PYTHON, "needs its own line")


@pytest.mark.parametrize(
    "decision", ["missing", "calibration.missing", "calibration.validation.missing"]
)
def test_citation_must_exist(mini_repo, decision):
    replace(
        mini_repo,
        PYTHON,
        "decision:statement,label:selection",
        f"decision:{decision},label:selection",
    )
    record, tags = snapshot(mini_repo)
    assert_problem(
        dr.citation_errors(record, tags), PYTHON, "unknown decision", decision
    )


def test_removed_site_orphans_decision(mini_repo):
    replace(
        mini_repo,
        PYTHON,
        "# @sc [decision:statement,label:selection] module-contract\n",
        "",
    )
    record, tags = snapshot(mini_repo)
    assert dr.coverage_errors(record, tags) == [
        "statement: decision has no tagged site"
    ]


def test_pending_site_exception_is_allowed(mini_repo):
    replace(
        mini_repo,
        PYTHON,
        "# @sc [decision:statement,label:selection] module-contract\n",
        "",
    )
    record, tags = snapshot(mini_repo)
    pending = {"statement": "Tomography merge #374"}
    assert dr.coverage_errors(record, tags, pending) == []
    # An exception to coverage does not disable Values checks.
    assert_problem(dr.value_errors(mini_repo, record, tags), "statement", "found 0")


def test_pending_site_allowlist_cannot_rot(mini_repo):
    record, tags = snapshot(mini_repo)
    assert dr.coverage_errors(record, tags, {"missing": "Tomography merge #374"}) == [
        "PENDING_SITES: unknown decision 'missing'"
    ]


def test_pending_site_reason_cannot_be_empty(mini_repo):
    record, tags = snapshot(mini_repo)
    assert_problem(
        dr.coverage_errors(record, tags, {"statement": ""}),
        "PENDING_SITES",
        "nonempty reason",
    )


@pytest.mark.parametrize(
    "path, old, new, decision, ref",
    [
        (PYTHON, "module_value = 8", "renamed = 8", "statement", "module_value"),
        (YAML, "npatch: 100", "renamed: 100", "yaml_settings", "cosmo_val.npatch"),
        (INI, "walkers = 100", "renamed = 100", "ini_section", "sampler.walkers"),
        (SMK, "nmodes=20", "renamed=20", "workflow_rule", "nmodes"),
    ],
)
def test_zero_matches(mini_repo, path, old, new, decision, ref):
    replace(mini_repo, path, old, new)
    value_problem(
        mini_repo,
        decision,
        ref,
        "expected",
        "candidates found 0",
        "[]",
        "exactly one location",
    )


@pytest.mark.parametrize(
    "path, old, new, decision, ref, actual",
    [
        (
            PYTHON,
            "module_value = 8",
            "module_value = 99",
            "statement",
            "module_value",
            "99",
        ),
        (YAML, "npatch: 100", "npatch: 99", "yaml_settings", "cosmo_val.npatch", "99"),
        (INI, "walkers = 100", "walkers = 99", "ini_section", "sampler.walkers", "99"),
        (SMK, "nmodes=20", "nmodes=99", "workflow_rule", "nmodes", "99"),
        (
            PYTHON,
            "threshold=0.0005",
            "threshold=0.00050001",
            "defaults",
            "threshold",
            "0.00050001",
        ),
    ],
)
def test_value_mismatch(mini_repo, path, old, new, decision, ref, actual):
    replace(mini_repo, path, old, new)
    value_problem(
        mini_repo,
        decision,
        ref,
        "expected",
        "candidates found 1",
        path,
        actual,
        "value mismatch",
    )


@pytest.mark.parametrize(
    "path, extra, decision, ref",
    [
        (
            PYTHON,
            "\n# @sc [decision:statement]\nmodule_value = 8\n",
            "statement",
            "module_value",
        ),
        (
            YAML,
            "\n# @sc [decision:yaml_settings]\ncosmo_val:\n  npatch: 100\n",
            "yaml_settings",
            "cosmo_val.npatch",
        ),
        (
            INI,
            "\n# @sc [decision:ini_section]\n[sampler]\nwalkers = 100\n",
            "ini_section",
            "sampler.walkers",
        ),
        (
            SMK,
            "\n# @sc [decision:workflow_rule]\nrule duplicate:\n    params: nmodes=20\n",
            "workflow_rule",
            "nmodes",
        ),
    ],
)
def test_two_equal_matches_are_ambiguous(mini_repo, path, extra, decision, ref):
    append(mini_repo, path, extra)
    value_problem(
        mini_repo, decision, ref, "candidates found 2", path, "exactly one location"
    )


@pytest.mark.parametrize("expression", ["get_value()", "4 + 4", "another_name"])
def test_python_nonliteral_reports_expression(mini_repo, expression):
    replace(mini_repo, PYTHON, "module_value = 8", f"module_value = {expression}")
    value_problem(
        mini_repo,
        "statement",
        "module_value",
        "candidates found 1",
        "non-literal",
        expression,
    )


def test_nonliteral_parameter_default(mini_repo):
    replace(mini_repo, PYTHON, "threshold=0.0005", "threshold=calculate()")
    value_problem(mini_repo, "defaults", "threshold", "non-literal", "calculate()")


def test_nonliteral_keyword(mini_repo):
    replace(mini_repo, PYTHON, "keyword=3", "keyword=calculate()")
    value_problem(mini_repo, "body", "keyword", "non-literal", "calculate()")


def test_multiple_matches_in_one_span_are_ambiguous(mini_repo):
    replace(mini_repo, PYTHON, "    assigned = 1", "    assigned = 1\n    assigned = 1")
    value_problem(mini_repo, "body", "assigned", "candidates found 2")


def test_duplicate_dict_keys_are_distinct_locations(mini_repo):
    replace(
        mini_repo, PYTHON, '{"dict_entry": 4}', '{"dict_entry": 4, "dict_entry": 4}'
    )
    value_problem(mini_repo, "body", "dict_entry", "candidates found 2")


def test_statement_does_not_govern_next_statement(mini_repo):
    replace(mini_repo, PYTHON, "module_value = 8", "other = 8\nmodule_value = 8")
    value_problem(mini_repo, "statement", "module_value", "candidates found 0")


def test_yaml_matches_full_path_not_leaf_name(mini_repo):
    replace(mini_repo, YAML, "  npatch: 100", "  other:\n    npatch: 100")
    value_problem(mini_repo, "yaml_settings", "cosmo_val.npatch", "candidates found 0")


def test_rule_stops_at_next_rule_even_without_blank_line(mini_repo):
    replace(mini_repo, SMK, "\nrule unrelated:", "rule unrelated:")
    assert dr.repository_errors(mini_repo) == []


def test_indented_rule_stops_at_its_indentation(mini_repo):
    target = mini_repo / SMK
    lines = target.read_text().splitlines(keepends=True)
    target.write_text("if True:\n" + "".join("    " + line for line in lines))
    assert dr.repository_errors(mini_repo) == []


@pytest.mark.parametrize("separator", ["#", "::"])
def test_qualifier_disambiguates_across_files(mini_repo, separator):
    append(mini_repo, "src/other.py", "# @sc [decision:statement]\nmodule_value = 8\n")
    value_problem(mini_repo, "statement", "module_value", "candidates found 2")
    replace(
        mini_repo,
        "astra.yaml",
        "Values: module_value = 8.",
        f"Values: model.py{separator}module_value = 8.",
    )
    assert dr.repository_errors(mini_repo) == []


def test_qualifier_uses_path_components(mini_repo):
    replace(
        mini_repo,
        "astra.yaml",
        "Values: module_value = 8.",
        "Values: odel.py#module_value = 8.",
    )
    value_problem(mini_repo, "statement", "odel.py#module_value", "candidates found 0")


def test_overlapping_tags_count_location_once(mini_repo):
    replace(
        mini_repo,
        PYTHON,
        "    assigned = 1",
        "    # @sc [decision:body]\n    assigned = 1",
    )
    assert dr.repository_errors(mini_repo) == []


@pytest.mark.parametrize(
    "path, extra, decision, ref",
    [
        (YAML, "\ntail:\n  missing: 1\n", "yaml_file", "tail.missing"),
        (INI, "\n[sampler]\nmissing = 1\n", "ini_section", "sampler.missing"),
    ],
)
def test_absence_sees_setting_outside_tagged_paragraph(
    mini_repo, path, extra, decision, ref
):
    append(mini_repo, path, extra)
    value_problem(
        mini_repo, decision, ref, "expected absent", "expected no active setting"
    )


def test_absence_sees_default_inheritance(mini_repo):
    append(mini_repo, INI, "\n[DEFAULT]\nmissing = 1\n")
    value_problem(
        mini_repo, "ini_section", "sampler.missing", "expected no active setting"
    )


def test_absence_requires_same_decision_tag_in_file(mini_repo):
    replace(
        mini_repo,
        YAML,
        "decision:yaml_file,scope:file",
        "decision:yaml_settings,scope:file",
    )
    value_problem(
        mini_repo, "yaml_file", "tail.missing", "same-decision tagged file", "found []"
    )


def test_absence_requires_one_file_not_a_vacuous_or_ambiguous_pass(mini_repo):
    append(
        mini_repo,
        "config/other/settings.yaml",
        "# @sc [decision:yaml_file]\nother: 1\n",
    )
    value_problem(
        mini_repo,
        "yaml_file",
        "tail.missing",
        "same-decision tagged file",
        "config/other/settings.yaml",
        "config/settings.yaml",
    )


def test_absence_requires_existing_ini_section(mini_repo):
    replace(mini_repo, INI, "[sampler]", "[renamed]")
    value_problem(mini_repo, "ini_section", "sampler.missing", "existing section")


def test_quoted_absent_is_text(mini_repo):
    replace(
        mini_repo, "astra.yaml", "sampler.method = emcee", 'sampler.method = "absent"'
    )
    replace(mini_repo, INI, "method = emcee", "method = absent")
    assert dr.repository_errors(mini_repo) == []


@pytest.mark.parametrize(
    "actual, expected, passes",
    [
        ("1", "1.0", True),
        ("5e-4", "0.0005", True),
        ("0.123456789012345678901", "0.123456789012345678901", True),
        ("0.123456789012345678902", "0.123456789012345678901", False),
        ("True", "1", False),
        ("1", "True", False),
        ("None", "None", True),
        ("'1'", "1", False),
        ("'False'", "False", False),
        ("[1, 250]", "[1.0, 250.0]", True),
        ("[250, 1]", "[1, 250]", False),
        ("[1]", "1", False),
        ("[1, 250, 250]", "[1, 250]", False),
        ('"semi; = text"', '"semi; = text"', True),
    ],
)
def test_exact_typed_literals(mini_repo, actual, expected, passes):
    replace(mini_repo, PYTHON, "module_value = 8", f"module_value = {actual}")
    # safe_load the record so quoted punctuation doesn't break YAML syntax.
    record, tags = snapshot(mini_repo)
    record["decisions"]["statement"]["rationale"] = (
        f"Values: module_value = {expected}."
    )
    errors = dr.value_errors(mini_repo, record, tags)
    if passes:
        assert errors == []
    else:
        assert_problem(errors, "statement", "module_value", "value mismatch")


@pytest.mark.parametrize(
    "actual, expected",
    [
        ("T", "True"),
        ("f", "False"),
        ("true", "True"),
        ("FALSE", "False"),
        ("yes", "True"),
        ("no", "False"),
        ("1", "1"),
        ("0", "0"),
        ("1.0", "1"),
        ("on", "on"),
        ("off", "off"),
        ("Y", "Y"),
    ],
)
def test_cosmosis_conversion_order(mini_repo, actual, expected):
    replace(mini_repo, INI, "enabled = T", f"enabled = {actual}")
    replace(
        mini_repo,
        "astra.yaml",
        "sampler.enabled = True",
        f"sampler.enabled = {expected}",
    )
    assert dr.repository_errors(mini_repo) == []


@pytest.mark.parametrize(
    "tail, message",
    [
        ("Values: module_value = 8", "terminal Values: sentence"),
        ("Values: module_value = 8. Values: other = 1.", "terminal Values: sentence"),
        ("Values: module_value = 8; .", "Values entry"),
        ("Values: module_value = [1, [2]].", "flat lists"),
        ("Values: module_value = 'unclosed.", "unbalanced"),
    ],
)
def test_invalid_values_syntax(mini_repo, tail, message):
    record, tags = snapshot(mini_repo)
    record["decisions"]["statement"]["rationale"] = tail
    assert_problem(dr.value_errors(mini_repo, record, tags), "statement", message)


def test_duplicate_contract_id(mini_repo):
    replace(mini_repo, YAML, "yaml-contract", "fit-contract")
    _, tags = snapshot(mini_repo)
    assert_problem(
        dr.contract_errors(tags),
        "duplicate local-contract id",
        "fit-contract",
        PYTHON,
        YAML,
    )


def test_unknown_pytest_decision_marker(mini_repo):
    replace(
        mini_repo,
        MARKER,
        'decision("calibration.validation.noise")',
        'decision("missing")',
    )
    assert_problem(
        dr.marker_errors(mini_repo, dr.load_record(mini_repo)),
        MARKER + ":4",
        "decision marker",
        "unknown decision",
        "missing",
    )


def test_nonliteral_pytest_marker_is_rejected(mini_repo):
    replace(
        mini_repo, MARKER, 'decision("calibration.validation.noise")', "decision(NAME)"
    )
    assert_problem(
        dr.marker_errors(mini_repo, dr.load_record(mini_repo)),
        MARKER,
        "one literal string id",
    )


@pytest.mark.parametrize(
    "path",
    [
        "src/extra.py",
        "workflow/Snakefile",
        "cosmo_inference/extra.ini",
        "cosmo_val/extra.yml",
        "config/extra.yaml",
        "papers/paper/config/extra.yaml",
    ],
)
def test_all_scan_roots_are_checked(mini_repo, path):
    append(mini_repo, path, "# @sc [decision:missing]\nvalue = 1\n")
    tags, errors = dr.scan_tags(mini_repo)
    assert errors == []
    assert_problem(dr.citation_errors(dr.load_record(mini_repo), tags), path, "missing")


@pytest.mark.parametrize(
    "path",
    [
        ".git/extra.py",
        "src/__pycache__/extra.py",
        "src/scratch/extra.py",
        "scratch/extra.py",
        "src/sp_validation/tests/data/extra.py",
        "src/sp_validation/tests/data/decision_record/extra.py",
        "outside/extra.py",
        "papers/paper/not_config/extra.py",
    ],
)
def test_out_of_scope_files_are_ignored(mini_repo, path):
    append(mini_repo, path, "# @sc [malformed tag]\nvalue = 1\n")
    assert dr.repository_errors(mini_repo) == []


def test_symlink_sources_are_not_counted_twice(mini_repo):
    (mini_repo / "src" / "alias.py").symlink_to("model.py")
    assert dr.repository_errors(mini_repo) == []


def test_tag_examples_in_strings_are_not_tags(mini_repo):
    append(mini_repo, PYTHON, '\nEXAMPLE = "# @sc [malformed tag]"\n')
    append(mini_repo, MARKER, '\nEXAMPLE = "pytest.mark.decision(\\"missing\\")"\n')
    assert dr.repository_errors(mini_repo) == []


def test_yaml_block_scalar_text_is_not_a_tag(mini_repo):
    append(mini_repo, YAML, "\nexample: |\n  # @sc [malformed tag]\n")
    assert dr.repository_errors(mini_repo) == []


def test_file_scope_tag_needs_no_adjacent_site(mini_repo):
    old = "# @sc [decision:yaml_file,scope:file]\n"
    replace(mini_repo, YAML, old, old + "\n")
    assert dr.repository_errors(mini_repo) == []


def test_python_source_is_never_executed(mini_repo):
    append(mini_repo, PYTHON, "\nraise RuntimeError('must never execute')\n")
    assert dr.repository_errors(mini_repo) == []


def test_deleting_file_fails_its_qualified_values(mini_repo):
    (mini_repo / YAML).unlink()
    record, tags = snapshot(mini_repo)
    assert_problem(dr.coverage_errors(record, tags), "yaml_file", "no tagged site")
    assert_problem(
        dr.value_errors(mini_repo, record, tags),
        "settings.yaml#tail.missing",
        "same-decision tagged file",
    )


def test_snakemake_nonliteral_has_a_candidate(mini_repo):
    replace(mini_repo, SMK, "nmodes=20", "nmodes=calculate()")
    value_problem(
        mini_repo,
        "workflow_rule",
        "nmodes",
        "candidates found 1",
        "non-literal",
        "calculate()",
    )


def test_snakemake_multiline_assignment(mini_repo):
    replace(mini_repo, SMK, "SCALE = 5e-4", "settings = {\n    'SCALE': 5e-4\n}")
    assert dr.repository_errors(mini_repo) == []


def test_ini_multiline_is_a_string(mini_repo):
    replace(mini_repo, INI, "method = emcee", "method = first\n    second")
    record, tags = snapshot(mini_repo)
    record["decisions"]["ini_section"]["rationale"] = (
        'Values: sampler.method = "first\\nsecond".'
    )
    assert dr.value_errors(mini_repo, record, tags) == []


@pytest.mark.parametrize(
    "path, old, new, ref",
    [
        (YAML, "  limits: [1, 250]", "  limits: [250, 1]", "cosmo_val.limits"),
        (SMK, "angular_limits=[1, 250]", "angular_limits=[250, 1]", "angular_limits"),
    ],
)
def test_config_list_order_matters(mini_repo, path, old, new, ref):
    replace(mini_repo, path, old, new)
    value_problem(mini_repo, ref, "value mismatch")


@pytest.mark.parametrize(
    "path, old, ref, decision",
    [
        (YAML, "cosmo_val.optional = None", "cosmo_val.optional", "yaml_settings"),
        (PYTHON, "; optional = None", "optional", "defaults"),
    ],
)
def test_null_is_not_a_missing_location(mini_repo, path, old, ref, decision):
    replace(mini_repo, "astra.yaml", old, old.replace("None", "False"))
    value_problem(mini_repo, decision, ref, "candidates found 1", "value mismatch")


def test_flat_lists_reject_nested_python_lists(mini_repo):
    replace(mini_repo, PYTHON, "module_value = 8", "module_value = [1, [2]]")
    value_problem(mini_repo, "statement", "module_value", "non-literal")


def test_repeated_tag_same_site_is_not_values_ambiguity(mini_repo):
    replace(
        mini_repo,
        PYTHON,
        "module_value = 8",
        "# @sc [decision:statement]\nmodule_value = 8",
    )
    assert dr.repository_errors(mini_repo) == []


def test_comment_prose_can_contain_mentions(mini_repo):
    append(mini_repo, PYTHON, "\n# A prose mention of @sc is not a tag.\n")
    assert dr.repository_errors(mini_repo) == []


def test_pending_dotted_decision_exists(mini_repo):
    record, tags = snapshot(mini_repo)
    assert (
        dr.coverage_errors(
            record,
            tags,
            {"calibration.validation.noise": "This valid id must not be rejected"},
        )
        == []
    )


def test_absence_in_implicit_ini_default_section(mini_repo):
    replace(
        mini_repo,
        "astra.yaml",
        "settings.ini#sampler.missing = absent",
        "settings.ini#DEFAULT.missing = absent",
    )
    assert dr.repository_errors(mini_repo) == []


def test_nonfinite_ini_number_is_not_coerced_to_text(mini_repo):
    replace(mini_repo, INI, "step = 5e-4", "step = inf")
    value_problem(mini_repo, "ini_section", "sampler.step", "finite")


def test_bad_docstring_tag_reports_physical_line(mini_repo):
    text = (mini_repo / PYTHON).read_text()
    line = text[: text.index("    @sc [decision:defaults")].count("\n") + 1
    replace(
        mini_repo,
        PYTHON,
        "decision:defaults,decision:body",
        "decision defaults,decision:body",
    )
    _, errors = dr.scan_tags(mini_repo)
    assert_problem(errors, f"{PYTHON}:{line}", "malformed @sc metadata")


def test_yaml_duplicate_key_outside_span_shadows_pin(mini_repo):
    # PyYAML keeps the last of two equal keys, so the governed one is dead.
    append(mini_repo, YAML, "\ncosmo_val:\n  npatch: 50\n")
    value_problem(
        mini_repo, "yaml_settings", "cosmo_val.npatch", "outside the governed span"
    )


@pytest.fixture
def real_repo():
    if not (ROOT / "astra.yaml").is_file():
        pytest.skip("root astra.yaml is not present yet; fixture checks still run")
    return ROOT


def test_repo_tags_parse(real_repo):
    _, errors = dr.scan_tags(real_repo)
    assert not errors, errors


def test_repo_citations_exist(real_repo):
    tags, _ = dr.scan_tags(real_repo)
    errors = dr.citation_errors(dr.load_record(real_repo), tags)
    assert not errors, errors


def test_repo_decisions_have_sites(real_repo):
    tags, _ = dr.scan_tags(real_repo)
    errors = dr.coverage_errors(dr.load_record(real_repo), tags, dr.PENDING_SITES)
    assert not errors, errors


def test_repo_values_resolve_and_match(real_repo):
    tags, _ = dr.scan_tags(real_repo)
    errors = dr.value_errors(real_repo, dr.load_record(real_repo), tags)
    assert not errors, errors


def test_repo_contract_ids_are_unique(real_repo):
    tags, _ = dr.scan_tags(real_repo)
    errors = dr.contract_errors(tags)
    assert not errors, errors


def test_repo_test_markers_exist(real_repo):
    errors = dr.marker_errors(real_repo, dr.load_record(real_repo))
    assert not errors, errors
