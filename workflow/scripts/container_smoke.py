"""Rule container_smoke: exercise the containerized-SLURM path end to end.

Cheap sanity check for the profile-driven-container pivot -- same executor
(slurm), same software-deployment-method (apptainer), same apptainer-args
binds, same container image every real rule uses. No rule-level `container:`
or `apptainer exec` anywhere here; Snakemake wraps the job entirely from the
profile.  Three things it proves, each written to the output YAML:

  * the editable ``sp_validation`` install resolves on the container's
    PYTHONPATH (import provenance: file + version, not just import success);
  * the numeric stack works and honours threading env (numpy eigh on a small
    fixed matrix, plus OMP_NUM_THREADS as seen inside the job);
  * which commit of this checkout is running (git rev-parse from inside the
    container -- proves /home is bound and usable, not just readable).

Run it directly with `snakemake ... container_smoke` before trusting the
pivot on real compute.
"""

import os
import platform
import subprocess

import numpy as np
import yaml
from snakemake.script import snakemake

# --- editable install resolves inside the container ------------------------
import sp_validation

sp_validation_info = {
    "version": getattr(sp_validation, "__version__", "unknown"),
    "file": sp_validation.__file__,
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
repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            "sp_validation": sp_validation_info,
            "numeric": numeric_info,
            "provenance": provenance,
        },
        f,
        sort_keys=False,
    )
