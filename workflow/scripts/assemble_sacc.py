"""Assemble the terminal ``{version}.sacc`` analysis file from per-statistic parts.

Dual-mode: under Snakemake (``script:``) the injected ``snakemake`` object
supplies the inputs; as a standalone CLI the same assembly runs from flags.

Each part is a single-statistic SACC; they load in CANONICAL order and are
rebuilt into one Sacc with a single ``BlockDiagonalCovariance``.

Every part must carry a covariance block. ξ± reporting and pseudo-Cℓ are born
without one, so theirs are injected here from the CosmoCov ``.txt``
(``--xi-cov``) and the NaMaster covariance FITS (``--pseudo-cl-cov``); the
pseudo-Cℓ cross-spectrum blocks (EE↔BB, …) are dropped, matching what the
B-mode PTE reads today.
"""

import argparse

import numpy as np

from sp_validation import sacc_io
from sp_validation.cosmo_val.sacc_writers import assemble_analysis_sacc

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


def _attach_cov(part, name, xi_cov, pseudo_cl_cov):
    """Ensure ``part`` (mutated in place) carries a covariance block.

    Raises if a required ξ± / pseudo-Cℓ block was not supplied.
    """
    if part.covariance is not None:
        return part
    if name == "xi_reporting" and xi_cov is not None:
        part.add_covariance(np.loadtxt(xi_cov))
        return part
    if name == "pseudo_cl" and pseudo_cl_cov is not None:
        part.add_covariance(_pseudo_cl_cov_block(pseudo_cl_cov))
        return part
    raise ValueError(
        f"the {name!r} part carries no covariance and no covariance input was "
        "given (--xi-cov / --pseudo-cl-cov); supply the block"
    )


def assemble_sacc(
    version,
    part_paths,
    out_path,
    *,
    expected=None,
    xi_cov=None,
    pseudo_cl_cov=None,
    allow_unblinded=False,
):
    """Assemble ``{version}.sacc`` from the per-statistic ``part_paths`` mapping.

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
    xi_cov, pseudo_cl_cov
        Covariance sourcing — see the module docstring.
    allow_unblinded : bool, optional
        Passed to :func:`sacc_io.load` for every part; ``True`` only for mocks.
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
        part = sacc_io.load(path, allow_unblinded=allow_unblinded)
        parts.append(_attach_cov(part, name, xi_cov, pseudo_cl_cov))
    if not parts:
        raise ValueError(f"no parts found for {version}: {part_paths}")
    # Through sacc_io.gather, the one terminal seam: it fails closed unless every
    # blindable part shares one blind, and stamps that blind on the result.
    s = sacc_io.gather(parts, assemble=assemble_analysis_sacc)
    sacc_io.save(s, out_path, type=s.metadata["type"])
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
        expected=list(p["expected"]),
        xi_cov=getattr(inp, "xi_cov", None),
        pseudo_cl_cov=getattr(inp, "pseudo_cl_cov", None),
        allow_unblinded=(p.get("type", "data") == "mock"),
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Assemble the terminal {version}.sacc from per-statistic parts."
    )
    ap.add_argument("--version", required=True, help="Catalogue version")
    ap.add_argument("--out", required=True, help="Output {version}.sacc path")
    ap.add_argument(
        "--type",
        choices=("data", "mock"),
        default="data",
        help="Run type. 'mock' reads parts freely; 'data' fails closed on "
        "unblinded parts (only concealed/blinded parts load).",
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
        xi_cov=a.xi_cov,
        pseudo_cl_cov=a.pseudo_cl_cov,
        allow_unblinded=(a.type == "mock"),
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
