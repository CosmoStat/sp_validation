"""Assemble the terminal ``{version}.sacc`` analysis file from per-statistic parts.

Dual-mode. Under Snakemake (``script:`` directive) the injected ``snakemake``
object supplies the parts + covariance inputs; as a standalone CLI (argparse)
the same assembly runs from explicit flags (the lightcone/ASTRA path).

Each per-statistic ``*.sacc`` *part* (written born-as-SACC by the mixins and the
run_2pcf / generate_pseudo_cl scripts) holds one statistic. The assembler loads
them in canonical order — ξ± coarse, pseudo-Cℓ, COSEBIs, pure-E/B, ρ/τ — and
calls :func:`sacc_writers.assemble_analysis_sacc`, which rebuilds one Sacc with a
single block-diagonal ``FullCovariance`` (point-insertion order = block order,
validated by ``sacc_io.assemble_covariance``).

Covariance sourcing (the part-by-part decision)
-----------------------------------------------
``assemble_analysis_sacc`` REQUIRES every part to carry its own covariance block.
The COSEBIs, pure-E/B and ρ/τ parts already do (their writers attach it). The
ξ± coarse and pseudo-Cℓ parts are born cov-less by design; this script injects
their blocks before assembly:

* **ξ± coarse** — the CosmoCov theory covariance ``.txt`` (``--xi-cov``). For the
  single-bin round it is already ``[ξ+; ξ−]``-ordered (CosmoCov / covdat_to_fits:
  ``STRT_0=0`` XI_PLUS, ``STRT_1=len/2`` XI_MINUS), which is exactly the SACC
  ξ insertion order, so ``np.loadtxt`` → ``add_covariance`` needs no permutation.
* **pseudo-Cℓ** — the NaMaster iNKA / OneCovariance covariance FITS
  (``--pseudo-cl-cov`` + ``--pseudo-cl-cov-hdu``). The FITS carries the 16
  EE/EB/BE/BB cross-blocks (each ``nbp × nbp``); SACC stores EE, BB, EB (in that
  order), so we assemble the block-diagonal ``[EE_EE; BB_BB; EB_EB]``. The
  cross-spectrum blocks (EE↔BB, …) are dropped — matching how the B-mode PTE
  today reads only ``COVAR_BB_BB``. **TODO(PR-cov):** carry the full dense
  EE/BB/EB cross-covariance once the analysis needs cross-spectrum correlations.

When a cov input is absent the assembly cannot proceed on a real product; pass
``--allow-placeholder`` to attach a documented diagonal placeholder
(``placeholder_var`` on every point of the cov-less parts) so the DAG dry-run and
the fast test can still produce a structurally-valid ``FullCovariance``. The
placeholder is a flagged stand-in, never a science covariance.
"""

import argparse

import numpy as np

from sp_validation import sacc_io
from sp_validation.cosmo_val.sacc_writers import assemble_analysis_sacc

# NaMaster iNKA covariance FITS: per-spectrum HDU names. SACC insertion order is
# EE, BB, EB, so the block-diagonal is assembled in that order.
_CL_HDU = {"EE": "COVAR_EE_EE", "BB": "COVAR_BB_BB", "EB": "COVAR_EB_EB"}
_CL_ORDER = ("EE", "BB", "EB")

# Canonical part order — the order assemble_analysis_sacc inserts points in, which
# must match the covariance block order. Missing parts are simply skipped.
CANONICAL = ("xi_coarse", "pseudo_cl", "cosebis", "pure_eb", "rho_tau")


def _pseudo_cl_cov_block(cov_fits, hdu):
    """Block-diagonal ``[EE_EE; BB_BB; EB_EB]`` from the NaMaster iNKA cov FITS.

    ``hdu`` selects the file flavor: for the per-spectrum iNKA file we read the
    three named diagonal HDUs; for a single dense HDU (OneCovariance / g+ng
    ``COVAR_FULL``) that already spans EE/BB/EB we return it as-is.
    """
    from astropy.io import fits

    with fits.open(cov_fits) as hdul:
        names = {h.name for h in hdul}
        if all(_CL_HDU[s] in names for s in _CL_ORDER):
            blocks = [np.asarray(hdul[_CL_HDU[s]].data, float) for s in _CL_ORDER]
            n = blocks[0].shape[0]
            full = np.zeros((3 * n, 3 * n))
            for i, block in enumerate(blocks):
                full[i * n : (i + 1) * n, i * n : (i + 1) * n] = block
            return full
        return np.asarray(hdul[hdu].data, float)


def _attach_cov(part, name, xi_cov, pseudo_cl_cov, pseudo_cl_cov_hdu, placeholder_var):
    """Ensure ``part`` carries a covariance, injecting the xi/pseudo-Cℓ block.

    ``part`` is mutated in place. cosebis/pure_eb/rho_tau parts already carry
    their covariance and pass straight through. Raises loudly if a required xi /
    pseudo-Cℓ block is missing and no placeholder was requested.
    """
    if part.covariance is not None:
        return part
    if name == "xi_coarse":
        if xi_cov is not None:
            part.add_covariance(np.loadtxt(xi_cov))
            return part
    elif name == "pseudo_cl":
        if pseudo_cl_cov is not None:
            part.add_covariance(_pseudo_cl_cov_block(pseudo_cl_cov, pseudo_cl_cov_hdu))
            return part
    if placeholder_var is None:
        raise ValueError(
            f"the {name!r} part carries no covariance and no covariance input was "
            f"given (--xi-cov / --pseudo-cl-cov). Supply the block, or pass "
            "--allow-placeholder to attach a documented diagonal placeholder."
        )
    part.add_covariance(np.full(len(part.mean), float(placeholder_var)))
    return part


def assemble_sacc(
    version,
    part_paths,
    out_path,
    *,
    expected=None,
    xi_cov=None,
    pseudo_cl_cov=None,
    pseudo_cl_cov_hdu="COVAR_FULL",
    placeholder_var=None,
):
    """Assemble ``{version}.sacc`` from the per-statistic ``part_paths`` mapping.

    Parameters
    ----------
    version : str
        Catalogue version (stored in the assembled file's metadata).
    part_paths : dict
        ``{statistic: path}`` with statistic in :data:`CANONICAL`. Only the
        present statistics are assembled; order is forced to canonical.
    out_path : str
        Destination ``{version}.sacc``.
    expected : sequence of str, optional
        Statistics that MUST be present in ``part_paths`` (from the caller's
        config toggles). Raises loudly if any is missing or has no path — so a
        typo'd input keyword (``cosebi`` for ``cosebis``) can't silently drop a
        statistic from the terminal file. Names not in :data:`CANONICAL` are
        rejected too (catches a typo in the expected list itself).
    xi_cov, pseudo_cl_cov, pseudo_cl_cov_hdu, placeholder_var
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
    nz = metadata = None
    for name in CANONICAL:
        path = part_paths.get(name)
        if path is None:
            continue
        part = sacc_io.load(path)
        if nz is None:
            # The nz tracers + metadata are identical across parts (same version);
            # take them from the first loaded part for the assembled file.
            nz = {i: sacc_io.get_nz(part, i) for i in range(_n_source_bins(part))}
            metadata = dict(part.metadata)
        parts.append(
            _attach_cov(
                part, name, xi_cov, pseudo_cl_cov, pseudo_cl_cov_hdu, placeholder_var
            )
        )
    if not parts:
        raise ValueError(f"no parts found for {version}: {part_paths}")
    s = assemble_analysis_sacc(nz, metadata, parts)
    sacc_io.save(s, out_path)
    print(f"Assembled {len(parts)} parts -> {out_path}")
    return s


def _n_source_bins(part):
    """Count the ``source_{i}`` NZ tracers on a part (single-bin round -> 1)."""
    i = 0
    while sacc_io.source_name(i) in part.tracers:
        i += 1
    return i


def _from_snakemake(smk):
    p = smk.params
    inp = smk.input
    part_paths = {
        name: getattr(inp, name)
        for name in CANONICAL
        if hasattr(inp, name) and getattr(inp, name)
    }
    # The rule declares which statistics it wired (from its config toggles); a
    # typo in an input keyword drops the part from part_paths above, so validate
    # against this expected list rather than trusting the hasattr filter.
    expected = list(p["expected"])
    assemble_sacc(
        version=p["version"],
        part_paths=part_paths,
        out_path=str(smk.output[0]),
        expected=expected,
        xi_cov=getattr(inp, "xi_cov", None),
        pseudo_cl_cov=getattr(inp, "pseudo_cl_cov", None),
        pseudo_cl_cov_hdu=p.get("pseudo_cl_cov_hdu", "COVAR_FULL"),
        placeholder_var=p.get("placeholder_var", None),
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Assemble the terminal {version}.sacc from per-statistic parts."
    )
    ap.add_argument("--version", required=True, help="Catalogue version")
    ap.add_argument("--out", required=True, help="Output {version}.sacc path")
    for name in CANONICAL:
        ap.add_argument(
            f"--{name.replace('_', '-')}", default=None, help=f"{name} part"
        )
    ap.add_argument("--xi-cov", default=None, help="CosmoCov ξ covariance .txt")
    ap.add_argument(
        "--pseudo-cl-cov",
        default=None,
        help="NaMaster/OneCovariance pseudo-Cℓ cov FITS",
    )
    ap.add_argument(
        "--pseudo-cl-cov-hdu",
        default="COVAR_FULL",
        help="HDU name for a single dense pseudo-Cℓ cov (EE/BB/EB-spanning)",
    )
    ap.add_argument(
        "--allow-placeholder",
        type=float,
        default=None,
        metavar="VAR",
        help="Attach a diagonal placeholder (variance VAR) to cov-less parts",
    )
    a = ap.parse_args(argv)
    part_paths = {name: getattr(a, name) for name in CANONICAL if getattr(a, name)}
    assemble_sacc(
        version=a.version,
        part_paths=part_paths,
        out_path=a.out,
        xi_cov=a.xi_cov,
        pseudo_cl_cov=a.pseudo_cl_cov,
        pseudo_cl_cov_hdu=a.pseudo_cl_cov_hdu,
        placeholder_var=a.allow_placeholder,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
