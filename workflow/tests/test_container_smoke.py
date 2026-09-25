"""P5: one real SLURM job through the committed candide profile.

The executor, the apptainer deployment method and the bind mounts come from
that profile; the image is the module-level ``container:`` in the test
Snakefile, exactly as real workflows declare it. That contract is what's under
test, so it runs only on a candide submit host.

The job writes a YAML report (see data/container_smoke/container_smoke.py); the
assertions below check what it reports.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import REPO, on_candide

SMOKE = Path(__file__).resolve().parent / "data" / "container_smoke"


def _reference_eigenvalues() -> np.ndarray:
    """The same deterministic computation the job runs inside the container."""
    rng = np.random.default_rng(seed=42)
    a = rng.standard_normal((8, 8))
    return np.linalg.eigh(a + a.T)[0]


def test_smoke_snakefile_names_the_workflow_image():
    """The test Snakefile's literal image must track the package's CONTAINER_URI."""
    uri = re.search(
        r'^CONTAINER_URI = "(.+)"$',
        (REPO / "src/sp_validation/container.py").read_text(),
        re.MULTILINE,
    ).group(1)
    assert f'"{uri}"' in (SMOKE / "Snakefile").read_text(), uri


@pytest.mark.candide
@on_candide
@pytest.mark.skipif(shutil.which("sbatch") is None, reason="needs a SLURM submit host")
def test_container_smoke():
    # Not pytest's tmp_path: that lives in the submit host's /tmp, which the
    # compute node cannot see, so the job's output would "go missing". The
    # workdir must be on a shared filesystem.
    workdir = Path(tempfile.mkdtemp(prefix="container_smoke_", dir=Path.home()))

    env = os.environ | {"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--profile",
            str(REPO / "workflow/profiles/candide"),
            "-s",
            str(SMOKE / "Snakefile"),
            "--directory",
            str(workdir),
            "--jobs",
            "1",
            "container_smoke",
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout

    report = yaml.safe_load((workdir / "results/container_smoke.yaml").read_text())

    # The job ran inside the image, not on the bare host. Everything below would
    # pass on the host too, so this is the assertion that makes them mean
    # something: apptainer sets APPTAINER_CONTAINER in every process it starts.
    assert report["container"]["apptainer_container"] != "unset", report["container"]

    # The install must resolve to an editable src/ checkout, not a site-packages
    # copy. It need not be *this* checkout: the container's editable install
    # points at the shared /n17data working tree, while the Snakefile under test
    # is read from wherever the test runs.
    module_file = Path(report["sp_validation"]["file"])
    assert module_file.parts[-3:] == ("src", "sp_validation", "__init__.py"), (
        module_file
    )
    assert "site-packages" not in module_file.parts, module_file

    # The numeric stack agrees with the same computation run here.
    np.testing.assert_allclose(
        report["numeric"]["eigenvalues"],
        _reference_eigenvalues(),
        rtol=1e-10,
        atol=1e-12,
    )

    # numeric.omp_num_threads is recorded but deliberately NOT asserted: the
    # profile leaves OMP_NUM_THREADS unset by design, and rules that need it
    # pinned set it themselves, so "unset" here is correct rather than a gap.

    # git worked inside the container, so /home is bound and usable.
    assert re.fullmatch(r"[0-9a-f]{40}", report["provenance"]["commit"]), report[
        "provenance"
    ]

    shutil.rmtree(workdir)  # keep only on failure, for post-mortem
