"""Give a ξ± SACC part the OneCovariance matrix of its binning as its covariance.

Dual-mode. Under Snakemake (``script:`` directive) the injected ``snakemake``
object supplies the paths; as a standalone CLI the same step runs from flags:

    python xi_onecov_sacc.py \
        --xi <version>_xi_..._npatch=1_tomo.sacc \
        --cov <covariance_dir>/<base>.dat \
        --blind none \
        --out <version>_xi_..._npatch=1_tomo_onecov.sacc

OneCovariance writes ξ± type-major; the part is pair-major, so the matrix is
reordered to the part's points
(:func:`sp_validation.cosmo_val.sacc_writers.onecov_xi_covariance`) before it
replaces the part's own estimate. The output is a derivation of the part and
carries its blind stamp, which must be the catalogue's declared blind.
"""

import argparse

from sp_validation import sacc_io
from sp_validation.blinding import Blind
from sp_validation.cosmo_val.sacc_writers import onecov_xi_covariance


def xi_onecov_sacc(xi_path, cov_path, out_path, *, blind):
    """Write the part at ``xi_path`` with the matrix at ``cov_path`` to ``out_path``.

    Returns
    -------
    sacc.Sacc
        The data set as written.
    """
    part = sacc_io.load(xi_path)
    s = part.copy()
    s.add_covariance(
        onecov_xi_covariance(part, sacc_io.read_onecov_matrix(cov_path)),
        overwrite=True,
    )
    out = sacc_io.save(s, out_path, derived_from=[part], blind=Blind(blind))
    print(f"Wrote ξ± with the OneCovariance covariance {cov_path}: {out_path}")
    return out


def _from_snakemake(smk):
    xi_onecov_sacc(
        smk.input["xi"],
        smk.input["cov"],
        smk.output["sacc"],
        blind=smk.params["blind"],
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Attach a OneCovariance ξ± matrix to a ξ± SACC part."
    )
    ap.add_argument("--xi", required=True, help="ξ± SACC part")
    ap.add_argument("--cov", required=True, help="OneCovariance matrix (.dat)")
    ap.add_argument(
        "--blind", required=True, help="The blind the catalogue is declared under"
    )
    ap.add_argument("--out", required=True, help="Output SACC path")
    a = ap.parse_args(argv)
    xi_onecov_sacc(a.xi, a.cov, a.out, blind=a.blind)


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
