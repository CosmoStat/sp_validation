"""Host-side harness: the workflow as a person launches it.

These tests run under the host launcher, never inside the image::

    uv run --isolated --no-project --python 3.12 --with snakemake \\
        --with snakemake-executor-plugin-slurm --with pytest pytest workflow/tests

``sp_validation`` is absent from that environment, so every Snakefile has to
parse with the standard library and Snakemake alone -- the condition a host
Snakemake is in. The ``candide`` tests need candide itself; CI deselects them.

The ``toy`` fixture is a disposable checkout: copies of ``workflow/`` and
``papers/cosmo_val/``, this checkout's ``src/`` symlinked in, a one-catalogue
``cosmo_val/cat_config.yaml``, a touched catalogue file, the processed CosmoCov
covariances already in place (their inputs live on candide), and both output
roots in tmp.
"""

import dataclasses
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]

# The toy catalogue and its leakage-corrected variant.
VERSIONS = ("SP_v0.1", "SP_v0.1_leak_corr")


@dataclasses.dataclass
class Job:
    rule: str
    input: list
    output: list
    wildcards: dict


_RULE = re.compile(r"^(?:local)?rule (\w+):$")
_FIELD = re.compile(r"^    (\w+): (.*)$")


def parse_jobs(text):
    """The jobs a ``snakemake -n`` listing schedules."""
    jobs, fields = [], None
    for line in text.splitlines():
        if match := _RULE.match(line):
            fields = {"rule": match[1]}
            jobs.append(fields)
        elif fields is not None and (match := _FIELD.match(line)):
            fields[match[1]] = match[2]
        else:
            fields = None
    return [
        Job(
            rule=f["rule"],
            input=f["input"].split(", ") if "input" in f else [],
            output=f["output"].split(", ") if "output" in f else [],
            wildcards=dict(
                pair.split("=", 1)
                for pair in f.get("wildcards", "").split(", ")
                if pair
            ),
        )
        for f in jobs
    ]


def _load_module(path, name, env):
    """Import a workflow module by path under ``env`` (it reads env at import)."""
    saved = os.environ.copy()
    os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return module


# The package's image model, loaded by path: sp_validation is absent here.
container = _load_module(
    REPO / "src" / "sp_validation" / "container.py", "spv_container", {}
)


@dataclasses.dataclass
class Toy:
    root: Path
    rundir: Path
    cosmo_val: Path
    cosmo_inference: Path
    env: dict
    config: dict
    common: object
    covariances: dict  # (version, "g" | "ng") -> the processed CosmoCov file

    def snakemake(self, *args, cwd=None, env=None, timeout=300):
        """Run the host Snakemake in the toy's paper directory, or in ``cwd``.

        ``env`` replaces the toy's environment.
        """
        return subprocess.run(
            [sys.executable, "-m", "snakemake", "--cores", "1", *args],
            cwd=cwd or self.rundir,
            env=env or self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )


def _cat_config(catalogue):
    entry = {
        "subdir": str(catalogue.parent),
        "pipeline": "SP",
        "colour": "orange",
        "marker": "^",
        "cov_th": {"A": 100.0, "n_e": 5.0, "sigma_e": 0.3},
        "shear": {
            "path": str(catalogue),
            "redshift_path": str(catalogue.parent / "nz_SP_v0.1_A.txt"),
            "w_col": "w",
            "e1_col": "e1",
            "e2_col": "e2",
            "e1_col_corrected": "e1_leak_corrected",
            "e2_col_corrected": "e2_leak_corrected",
        },
    }
    return {VERSIONS[0]: entry, "paths": {"output": "./output"}}


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    skip = shutil.ignore_patterns(".snakemake", "__pycache__", "tests")
    shutil.copytree(REPO / "workflow", root / "workflow", ignore=skip)
    shutil.copytree(
        REPO / "papers" / "cosmo_val", root / "papers" / "cosmo_val", ignore=skip
    )
    (root / "src").symlink_to(REPO / "src")

    catalogue = root / "data" / "toy_shear.fits"
    catalogue.parent.mkdir()
    catalogue.touch()
    (root / "cosmo_val").mkdir()
    (root / "cosmo_val" / "cat_config.yaml").write_text(
        yaml.safe_dump(_cat_config(catalogue))
    )

    rundir = root / "papers" / "cosmo_val"
    config_path = rundir / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["versions"] = list(VERSIONS)
    config["fiducial"]["version"] = VERSIONS[1]
    config["fiducial"]["mock_version"] = VERSIONS[0]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SNAKEMAKE_PROFILE", "APPTAINERENV_PYTHONPATH")
    }
    env.update(
        COSMO_VAL=str(root / "out" / "cosmo_val"),
        COSMO_INFERENCE=str(root / "out" / "cosmo_inference"),
        XDG_CACHE_HOME=str(root / "cache"),
        # No local image: launches resolve the registry tag.
        SPV_CONTAINER=str(root / "cache" / "absent.sif"),
        SPV_SANDBOX=str(root / "cache" / "absent-sandbox"),
        TMPDIR=str(tmp_path_factory.getbasetemp()),
        PYTHONUNBUFFERED="1",
        PYTHONNOUSERSITE="1",
    )
    common = _load_module(root / "workflow" / "common.py", "toy_common", env)

    # The processed CosmoCov covariances the cosmo_val rules read, in place.
    grids = common.xi_grids(config, config["fiducial"])
    mask = "_masked" if config["covariance"].get("default_masked") else ""
    covariances = {}
    for version in VERSIONS:
        for gaussian, grid in (("ng", grids["reporting"]), ("g", grids["integration"])):
            path = covariances[version, gaussian] = Path(
                common.covariance_path(
                    version,
                    gaussian,
                    grid["min_sep"],
                    grid["max_sep"],
                    grid["nbins"],
                    mask,
                )
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    return Toy(
        root=root,
        rundir=rundir,
        cosmo_val=Path(env["COSMO_VAL"]),
        cosmo_inference=Path(env["COSMO_INFERENCE"]),
        env=env,
        config=config,
        common=common,
        covariances=covariances,
    )


on_candide = pytest.mark.skipif(
    not Path("/n17data/cdaley/unions").exists() or shutil.which("apptainer") is None,
    reason="needs candide: /n17data and apptainer",
)
