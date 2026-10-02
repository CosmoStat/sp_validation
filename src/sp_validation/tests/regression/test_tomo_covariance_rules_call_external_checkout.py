"""Exercise real covariance rules without catalogues or CosmoCov.

These tests pass on develop: it uses checkout-local scripts and launch-relative
outputs. They guard against the external paths introduced on the tomography
branch. Snakemake >=8 is optional; all generated workflow files stay in tmp_path.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sp_validation


def _run_snakemake(case, *args):
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        XDG_CACHE_HOME=str(case.directory / "cache"),
        MPLCONFIGDIR=str(case.directory / "mplconfig"),
        TMPDIR=str(case.directory / "tmp"),
        # Preserve the dependency location if pytest loaded Snakemake via its
        # pythonpath setting rather than the interpreter's site-packages.
        PYTHONPATH=str(case.snakemake_root) + os.pathsep + env.get("PYTHONPATH", ""),
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            str(case.snakefile),
            "--cores",
            "1",
            "--nolock",
            "--scheduler",
            "greedy",
            *args,
        ],
        cwd=case.directory,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


@pytest.fixture
def covariance_rules(tmp_path):
    snakemake = pytest.importorskip(
        "snakemake",
        minversion="8",
        reason="Real covariance rules require Snakemake >=8",
    )
    # Use the imported package's __file__, not cwd or a hard-coded checkout.
    # This also lets PYTHONPATH select tomography while keeping this test file.
    root = Path(sp_validation.__file__).resolve().parents[2]
    source = root / "workflow/rules/covariance.smk"
    if not source.is_file():
        pytest.skip("Checkout does not contain workflow/rules/covariance.smk")
    (tmp_path / "tmp").mkdir()
    snakefile = tmp_path / "Snakefile"
    case = SimpleNamespace(
        root=root,
        directory=tmp_path,
        snakefile=snakefile,
        snakemake_root=Path(snakemake.__file__).resolve().parents[1],
    )
    # Only globals normally supplied by the workflow are synthetic. Snakemake
    # parses the unmodified rules, resolves entry points and executes the job.
    snakefile.write_text(
        f"""
from pathlib import Path
from snakemake.io import apply_wildcards
import json
COSMO_INFERENCE = Path({str(tmp_path / "cosmo_inference")!r})
COSMO_VAL = Path({str(tmp_path / "cosmo_val")!r})
COSMOLOGY_PARAMS = str(COSMO_INFERENCE / "cosmology.json")
PLANCK18 = dict.fromkeys(
    ["Omega_m", "Omega_v", "sigma_8", "n_s", "h", "Omega_b"], 0.0
)
BLOCK_PAIRS = []
DEFAULT_MASK_SUFFIX = ""
FIDUCIAL = dict.fromkeys([
    "mock_version", "version", "blind", "min_sep", "max_sep", "nbins",
    "min_sep_int", "max_sep_int", "nbins_int",
], "synthetic")
config = {{
    "tools": {{"cosmocov_executable": "unused"}},
    "glass_mocks": {{"seed_range": [1, 1]}},
}}
def fiducial_binning_suffix():
    return "_synthetic"
def covariance_path(*args, **kwargs):
    return "unused-fiducial-input"
def covariance_dir(*args, **kwargs):
    return "unused-covariance-directory"
def redshift_path(*args, **kwargs):
    return "unused-redshift-input"
def build_redshift_path(*args, **kwargs):
    return "unused-redshift-input"
wildcard_constraints:
    mask_suffix="(?:_masked)?"
include: {str(source)!r}
manifest = {{}}
for name in [
    "covariance_process", "generate_glass_mock_rhotau_samples",
    "covariance_glass_mock",
]:
    registered = workflow.get_rule(name)
    manifest[name] = {{
        "script": registered.script, "shellcmd": registered.shellcmd,
        "basedir": str(registered.basedir),
        "output": list(map(str, registered.output)),
    }}
synthetic_wildcards = dict(
    version="synthetic", blind="A", gaussian="ng", min_sep="1", max_sep="2",
    nbins="1", mask_suffix="",
)
manifest["covariance_process"]["synthetic_matrix"] = apply_wildcards(
    workflow.get_rule("covariance_process").output[0], synthetic_wildcards
)
Path("rules.json").write_text(json.dumps(manifest, indent=2))
"""
    )
    # Upper-triangle dump: columns 8 and 9 contain Gaussian and non-Gaussian
    # terms. Their sum is the positive-definite matrix [[3, .25], [.25, 6]].
    raw = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 0.25, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 4, 2],
        ]
    )
    parsed = _run_snakemake(case, "--list-rules")
    assert parsed.returncode == 0, parsed.stdout + parsed.stderr
    case.manifest = json.loads((tmp_path / "rules.json").read_text())
    case.matrix = Path(case.manifest["covariance_process"]["synthetic_matrix"])
    raw_path = Path(str(case.matrix).replace("_processed.txt", ".txt"))
    raw_path.parent.mkdir(parents=True)
    np.savetxt(raw_path, raw)
    return case


@pytest.mark.slow
def test_covariance_postprocessing_runs_from_launched_checkout(covariance_rules):
    """Process a synthetic CosmoCov dump without relying on another checkout.

    The three input rows give G+NG = [[3, .25], [.25, 6]], reflecting the supplied
    off-diagonal element, and G = [[2, .25], [.25, 4]]. Snakemake executes the
    actual rule with existing raw input, so no survey data or expensive
    covariance calculation is needed. A missing external entry point is a
    workflow defect. Develop passes because its processor is checkout-local;
    this guards against the tomography merge reintroducing an external script.
    """
    case = covariance_rules
    result = _run_snakemake(case, "--printshellcmds", str(case.matrix))
    log = result.stdout + result.stderr
    diagnosis = next(
        (line for line in log.splitlines() if "can't open file" in line), log[-2500:]
    )
    assert result.returncode == 0, (
        f"covariance_process returned {result.returncode}; expected exit 0 and "
        f"matrix [[3.0, 0.25], [0.25, 6.0]]; {diagnosis}"
    )
    np.testing.assert_allclose(
        np.loadtxt(case.matrix), [[3, 0.25], [0.25, 6]], rtol=0, atol=0
    )
    gaussian = Path(str(case.matrix).replace("_processed.txt", "_processed_g.txt"))
    np.testing.assert_allclose(
        np.loadtxt(gaussian), [[2, 0.25], [0.25, 4]], rtol=0, atol=0
    )


def test_tau_sampling_selects_script_in_launched_checkout(covariance_rules):
    """The real tau-sampling rule must select code from the launched checkout.

    Identical script contents today cannot guarantee this: edits to an unrelated
    checkout must not change a fixed commit's mock samples. Inspect Snakemake's
    registered entry point instead of inferring provenance from random values.
    Develop passes with its local script directive; the tomography merge must
    not restore an external entry point. No survey inputs are read.
    """
    case = covariance_rules
    rule = case.manifest["generate_glass_mock_rhotau_samples"]
    if rule["script"]:
        selected = (Path(rule["basedir"]) / rule["script"]).resolve()
    else:
        tokens = shlex.split(rule["shellcmd"])
        assert tokens[0] == "python", f"Unexpected tau command: {rule['shellcmd']}"
        selected = (case.directory / tokens[1]).resolve()
    expected = (
        case.root / "workflow/scripts/generate_glass_mock_rhotau_samples.py"
    ).resolve()
    assert selected == expected, (
        f"Tau sampling selects {selected}; expected launched-checkout script {expected}"
    )
    assert selected.is_file(), f"Selected tau script is missing: {selected}"


def test_glass_covariance_outputs_stay_in_launch_directory(covariance_rules):
    """All seven GLASS outputs must live in the launch directory's results tree.

    Resolve the real rule's output objects relative to the synthetic launch
    directory, the correct base for portable products and independent launches.
    No GLASS catalogues are read or external products written. Develop passes
    with relative outputs; this guards against the tomography merge restoring
    absolute paths that could overwrite another analysis.
    """
    case = covariance_rules
    expected_dir = case.directory / "results/covariance/glass_mock_v1.4.6"
    outputs = case.manifest["covariance_glass_mock"]["output"]
    assert len(outputs) == 7
    wrong = []
    for output in outputs:
        actual = (case.directory / output).resolve()
        if actual.parent != expected_dir:
            wrong.append(str(actual))
    assert not wrong, (
        f"GLASS covariance outputs escape launch directory: {wrong}; "
        f"expected directory {expected_dir}"
    )
