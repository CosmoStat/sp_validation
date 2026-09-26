"""Assemble the terminal ``{version}.sacc`` analysis file from per-statistic parts.

Dual-mode: under Snakemake (``script:``) the injected ``snakemake`` object
supplies the inputs; as a standalone CLI the same assembly runs from flags.

Each part is a single-statistic SACC; they load in CANONICAL order and are
rebuilt into one Sacc with a single ``BlockDiagonalCovariance``.

Every part must carry a covariance block. ξ± reporting and pseudo-Cℓ take
theirs from the analytic inputs — the CosmoCov ``.txt`` (``--xi-cov``) and the
NaMaster covariance FITS (``--pseudo-cl-cov``) — which replace any estimate the
part was born with; the pseudo-Cℓ cross-spectrum blocks (EE↔BB, …) are dropped,
matching what the B-mode PTE reads today.
"""

import argparse

import numpy as np

from sp_validation import sacc_io
from sp_validation.blinding import declared_custody
from sp_validation.cosmo_val.sacc_writers import assemble_analysis_sacc
from sp_validation.custody import confirm

# NaMaster iNKA covariance FITS: per-spectrum HDU names, in SACC insertion order.
_CL_HDUS = ("COVAR_EE_EE", "COVAR_BB_BB", "COVAR_EB_EB")

# Canonical part order — the order points are inserted in, which must match the
# covariance block order. Missing parts are simply skipped.
CANONICAL = ("xi_reporting", "pseudo_cl", "cosebis", "pure_eb", "rho_tau")


def _pseudo_cl_cov_block(cov_fits):
    """Block-diagonal ``[EE; BB; EB]`` from the NaMaster iNKA covariance FITS."""
    from astropy.io import fits

    with fits.open(cov_fits) as hdul:
        missing = [name for name in _CL_HDUS if name not in {h.name for h in hdul}]
        if missing:
            raise ValueError(f"{cov_fits} lacks the pseudo-Cℓ cov HDUs {missing}")
        blocks = [np.asarray(hdul[name].data, float) for name in _CL_HDUS]
    n = blocks[0].shape[0]
    full = np.zeros((3 * n, 3 * n))
    for i, block in enumerate(blocks):
        full[i * n : (i + 1) * n, i * n : (i + 1) * n] = block
    return full


# The statistics whose analysis covariance is external, and the input each one
# takes it from. A part of one of these types may be born with an estimate of
# its own — the ξ± reporting part carries the jackknife it was measured with —
# but the analysis file takes the external one, always.
_INJECTED = {"xi_reporting": "xi_cov", "pseudo_cl": "pseudo_cl_cov"}


def _attach_cov(part, name, xi_cov, pseudo_cl_cov):
    """Give ``part`` (mutated in place) the covariance the analysis file uses.

    For the two statistics with an external covariance the supplied block
    replaces whatever the part was born with, loudly; every other part keeps
    its own. Raises if the block a part needs was not supplied.
    """
    if name not in _INJECTED:
        if part.covariance is None:
            raise ValueError(
                f"the {name!r} part carries no covariance and none is injected "
                "for it; its writer must attach one"
            )
        return part

    supplied = xi_cov if name == "xi_reporting" else pseudo_cl_cov
    if supplied is None:
        raise ValueError(
            f"the {name!r} part takes its analysis covariance from "
            f"--{_INJECTED[name].replace('_', '-')}, which was not supplied"
        )
    block = (
        np.loadtxt(supplied)
        if name == "xi_reporting"
        else _pseudo_cl_cov_block(supplied)
    )
    if part.covariance is not None:
        print(
            f"{name}: replacing the part's own covariance with {supplied} "
            "(the analysis covariance)"
        )
    part.add_covariance(block, overwrite=True)
    return part


def assemble_sacc(
    version,
    part_paths,
    out_path,
    *,
    custody,
    expected=None,
    xi_cov=None,
    pseudo_cl_cov=None,
):
    """Assemble ``{version}.sacc`` from the per-statistic ``part_paths`` mapping.

    @sc one-custody-per-assembly
    Every part, signal-bearing or not, must carry one stamp, and it must be the
    custody ``version`` is declared under: a stale concealed part after a
    reveal, a part under another blind or catalogue, a mock in data and
    unblinded parts in a blinded file are all refused.

    Parameters
    ----------
    version : str
        Catalogue version, for error messages.
    part_paths : dict
        ``{statistic: path}`` with statistic in :data:`CANONICAL`. Only present
        statistics are assembled; order is forced to canonical.
    expected : sequence of str, optional
        Statistics that must be present, from the caller's config toggles. A
        typo'd input keyword would otherwise silently drop a statistic.
    custody : sp_validation.custody.Custody
        The custody ``version`` is declared under.
    xi_cov, pseudo_cl_cov
        Covariance sourcing — see the module docstring.
    """
    if expected is not None:
        unknown = [name for name in expected if name not in CANONICAL]
        if unknown:
            raise ValueError(
                f"expected parts {unknown} are not assemblable statistics; "
                f"valid names are {CANONICAL}"
            )
        missing = [name for name in expected if not part_paths.get(name)]
        if missing:
            raise ValueError(
                f"expected parts {missing} missing from part_paths for {version} "
                f"(got {sorted(part_paths)}); a required statistic would be "
                "silently dropped from the terminal analysis file"
            )
    parts = []
    for name in CANONICAL:
        path = part_paths.get(name)
        if path is None:
            continue
        parts.append(_attach_cov(sacc_io.load(path), name, xi_cov, pseudo_cl_cov))
    if not parts:
        raise ValueError(f"no parts found for {version}: {part_paths}")
    s = sacc_io.save(
        assemble_analysis_sacc(parts), out_path, derived_from=parts, custody=custody
    )
    print(f"Assembled {len(parts)} parts -> {out_path}")
    return s


def _from_snakemake(smk):
    p = smk.params
    inp = smk.input
    part_paths = {
        name: getattr(inp, name)
        for name in CANONICAL
        if hasattr(inp, name) and getattr(inp, name)
    }
    assemble_sacc(
        version=p["version"],
        part_paths=part_paths,
        out_path=str(smk.output[0]),
        custody=confirm(declared_custody(p["cat_config"], p["version"]), p["custody"]),
        expected=list(p["expected"]),
        xi_cov=getattr(inp, "xi_cov", None),
        pseudo_cl_cov=getattr(inp, "pseudo_cl_cov", None),
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Assemble the terminal {version}.sacc from per-statistic parts."
    )
    ap.add_argument("--version", required=True, help="Catalogue version")
    ap.add_argument("--out", required=True, help="Output {version}.sacc path")
    ap.add_argument(
        "--cat-config",
        required=True,
        help="cat_config.yaml declaring the catalogue's custody",
    )
    for name in CANONICAL:
        ap.add_argument(
            f"--{name.replace('_', '-')}", default=None, help=f"{name} part"
        )
    ap.add_argument("--xi-cov", default=None, help="CosmoCov ξ covariance .txt")
    ap.add_argument(
        "--pseudo-cl-cov", default=None, help="NaMaster pseudo-Cℓ covariance FITS"
    )
    a = ap.parse_args(argv)
    part_paths = {name: getattr(a, name) for name in CANONICAL if getattr(a, name)}
    assemble_sacc(
        version=a.version,
        part_paths=part_paths,
        out_path=a.out,
        custody=declared_custody(a.cat_config, a.version),
        xi_cov=a.xi_cov,
        pseudo_cl_cov=a.pseudo_cl_cov,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
