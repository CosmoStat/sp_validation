"""Shared harness for the cosmo_val Snakemake rule scripts.

Every diagnostic rule in ``workflow/rules/cosmo_val.smk`` runs a thin script
that (1) instantiates one ``CosmologyValidation`` object from the parameters
Snakemake passes in, (2) calls the slice of the diagnostic suite that rule
owns, and (3) verifies the declared outputs landed. This module carries the
common machinery so each rule script stays a few readable lines.

The in-memory ``cv`` object is re-instantiated per rule. The DAG links rules
through the *real* data products each method writes (rho/tau FITS, xi text,
pseudo-Cl FITS, E/B npz, leakage pkl); the lazy ``cv`` properties that aren't
persisted (``c1``/``c2``, ``xi_psf_sys``, ``rho_tau_fits``) are recomputed in
whichever rule needs them, which is cheap next to the science compute they
depend on. See common.cv_init_params for the constructor contract.
"""

import os
import sys
from pathlib import Path


def _unbuffer_streams():
    """Line-buffer stdout/stderr so SLURM logs stream live (matches run_2pcf)."""
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


def make_cv(snakemake):
    """Build a CosmologyValidation from a rule's ``snakemake.params``.

    ``params["cv_init"]`` is the kwargs dict assembled by common.cv_init_params.
    The object is created with the run directory as cwd so it finds
    ``cat_config.yaml`` and writes under ``output/`` exactly as interactive runs
    do.
    """
    from sp_validation.cosmo_val import CosmologyValidation

    rundir = snakemake.params["rundir"]
    os.chdir(rundir)
    return CosmologyValidation(**dict(snakemake.params["cv_init"]))


def verify_outputs(snakemake):
    """Fail loudly if any declared output is missing after the method ran."""
    missing = [str(o) for o in snakemake.output if not Path(o).exists()]
    if missing:
        raise FileNotFoundError(
            "cosmo_val rule finished but these declared outputs are absent:\n  "
            + "\n  ".join(missing)
        )


def touch_sentinels(snakemake):
    """Create any sentinel outputs (pure-plot rules with no data product)."""
    for out in snakemake.output:
        path = Path(out)
        if path.suffix == ".done":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
