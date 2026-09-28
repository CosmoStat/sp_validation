"""Rule container_smoke: exercise the containerized-SLURM path end to end.

Cheap sanity check of the profile-driven container path -- same executor
(slurm), same software-deployment-method (apptainer), same apptainer-args
binds, same container image every real rule uses.  Five things it proves, each
written to the output YAML:

  * the job really ran inside the image (``APPTAINER_CONTAINER``, set by
    apptainer itself -- without it the rest could all pass on the bare host);
  * which ``sp_validation`` the job imports (file + version, not just import
    success);
  * which ``snakemake`` package unpickled the injected ``snakemake`` object
    (file + version);
  * the numeric stack works (numpy eigh on a small fixed matrix).
    ``OMP_NUM_THREADS`` is recorded but not asserted -- see the assertions;
  * which commit of this checkout is running (git rev-parse from inside the
    container -- proves /home is bound and usable, not just readable).

Driven by the co-located Snakefile; the assertions on the output YAML live in
workflow/tests/test_container_smoke.py (candide only).
"""

import os
import platform
import subprocess
import sys

import numpy as np
import yaml

# --- the job is actually inside the image ---------------------------------
container_info = {
    "apptainer_container": os.environ.get("APPTAINER_CONTAINER", "unset"),
}

# --- the sp_validation the job imports -------------------------------------
import sp_validation  # noqa: E402

sp_validation_info = {
    "version": getattr(sp_validation, "__version__", "unknown"),
    "file": sp_validation.__file__,
}

# --- the snakemake package the preamble unpickled with ---------------------
# Read from sys.modules: importing it here would rebind the injected global.
snakemake_package = sys.modules["snakemake"]
snakemake_info = {
    "version": snakemake_package.__version__,
    "file": snakemake_package.__file__,
}

# --- numeric stack + threading -----------------------------------------
rng = np.random.default_rng(seed=42)
a = rng.standard_normal((8, 8))
symmetric = a + a.T
eigenvalues = np.linalg.eigh(symmetric)[0]

numeric_info = {
    "numpy_version": np.__version__,
    "eigenvalues": [float(v) for v in eigenvalues],
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
}

# --- provenance: what commit is actually running in the container ---------
# workflow/tests/data/container_smoke/ -> repo root, four levels up: the
# checkout the Snakefile came from.
repo_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), *([os.pardir] * 4))
)
try:
    commit = subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
except (subprocess.CalledProcessError, FileNotFoundError) as exc:
    commit = f"unavailable ({exc})"

provenance = {
    "repo_dir": repo_dir,
    "commit": commit,
    "hostname": platform.node(),
    "python": platform.python_version(),
}

with open(snakemake.output[0], "w") as f:
    yaml.safe_dump(
        {
            "container": container_info,
            "sp_validation": sp_validation_info,
            "snakemake": snakemake_info,
            "numeric": numeric_info,
            "provenance": provenance,
        },
        f,
        sort_keys=False,
    )
