"""Shared harness for the cosmo_val Snakemake rule scripts.

Every diagnostic rule in ``workflow/rules/cosmo_val.smk`` runs a thin script
that (1) instantiates one ``CosmologyValidation`` object from the parameters
Snakemake passes in, (2) calls the slice of the diagnostic suite that rule
owns, and (3) verifies the declared outputs landed. This module carries the
common machinery so each rule script stays a few readable lines.

Compute rules instantiate ``cv`` independently and link through explicit
cached data products. Plot rules read those products directly; they don't
instantiate ``cv`` or enter its lazy catalogue/measurement paths.
See common.cv_init_params for the compute constructor contract.
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

    ``params["cv_init"]`` is the kwargs dict assembled by common.cv_init_params,
    which names the catalogue config and the output directory explicitly.
    """
    from sp_validation.cosmo_val import CosmologyValidation

    return CosmologyValidation(**dict(snakemake.params["cv_init"]))


def prepare_cached_psf(cv):
    """Attach FITS-only handlers; never enter the catalogue/pair-count path."""
    from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat

    output = str(Path(cv.cc["paths"]["output"]) / "rho_tau_stats")
    cv._rho_stat_handler = RhoStat(
        output=output, treecorr_config=cv.treecorr_config, verbose=True
    )
    cv._tau_stat_handler = TauStat(
        catalogs=cv._rho_stat_handler.catalogs,
        output=output,
        treecorr_config=cv.treecorr_config,
        verbose=True,
    )


def take_reporting_parts(cv, paths):
    """Hand ``cv`` the reporting ξ± parts at ``paths``, one per version, in order."""
    from sp_validation import sacc_io

    for version, path in zip(cv.versions, paths, strict=True):
        part = sacc_io.load(path, blind=cv.blind(version))
        cv.xi_parts[version, "reporting"] = part


def verify_outputs(snakemake):
    """Fail loudly if any declared output is missing after the method ran."""
    missing = [str(o) for o in snakemake.output if not Path(o).exists()]
    if missing:
        raise FileNotFoundError(
            "cosmo_val rule finished but these declared outputs are absent:\n  "
            + "\n  ".join(missing)
        )
