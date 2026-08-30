"""TreeCorr ξ±(θ) two-point correlation function for one catalog version.

Dual-mode. Under Snakemake (``script:`` directive) the injected ``snakemake``
object supplies the parameters; as a standalone CLI (argparse) the same compute
runs from explicit flags. The CLI form is what the lightcone/ASTRA recipe calls,
so the measurement is driven directly (no nested Snakemake) with lc handling
orchestration:

    python run_2pcf.py \
        --ver SP_v1.4.6.3_leak_corr \
        --min-sep 1.0 --max-sep 250.0 --nbins 20 --npatch 1 \
        --cat-config /path/to/cosmo_val/cat_config.yaml \
        --out <output_dir>

The measurement is binning-agnostic: the reporting and the fine integration
grids are the same compute with different ``--min-sep/--max-sep/--nbins``.
``CosmologyValidation.calculate_2pcf`` writes the ``.txt`` dump (a raw
byproduct); the ξ± data product is born as SACC here, a *part* named by its
binning and tagged with its ``--grid``. The reporting part carries no covariance
(its block is supplied at assembly from CosmoCov); the integration part carries
a ``DiagonalCovariance`` from TreeCorr ``varxip``/``varxim``, the only estimate
available at npatch=1, which is what COSEBIs and pure-E/B consume.

``output_dir`` is passed explicitly (rather than via the ``COSMO_VAL`` env hook)
so lc can point each run at its own ``{output}`` tree.
"""

import argparse
import os

import numpy as np

from sp_validation import sacc_io
from sp_validation.cosmo_val import CosmologyValidation
from sp_validation.cosmo_val.sacc_writers import xi_to_sacc


def run_2pcf(
    ver,
    min_sep,
    max_sep,
    nbins,
    npatch,
    cat_config,
    output_dir,
    sacc_out=None,
    grid="reporting",
):
    """Measure ξ±(θ) for ``ver`` and write its reporting SACC part.

    Parameters mirror the TreeCorr reporting/integration grids: ``min_sep`` /
    ``max_sep`` in arcmin, ``nbins`` logarithmic bins, ``npatch`` spatial
    patches (1 for the paper fiducial). ``cat_config`` is an absolute path to
    the catalog configuration; ``output_dir`` overrides
    ``cat_config['paths']['output']`` so the ``.txt`` byproduct lands where lc
    expects. ``sacc_out`` is the exact destination for the reporting ξ± SACC part
    (the Snakemake-declared output); it defaults to ``{ver}_xi_reporting.sacc``
    under the resolved output directory for the CLI path.

    Returns
    -------
    treecorr.GGCorrelation
        The measured correlation object (also the source of the SACC part).
    """
    cv = CosmologyValidation(
        versions=[ver],
        catalog_config=cat_config,
        output_dir=output_dir,
        # so the SACC provenance metadata stamps the npatch actually measured
        npatch=npatch,
    )
    gg = cv.calculate_2pcf(
        ver=ver,
        npatch=npatch,
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
    )

    # Born-as-SACC ξ± part. theta = meanr; theta_nom = rnom.
    s = xi_to_sacc(
        cv.sacc_nz(ver),
        cv.sacc_metadata(ver),
        gg.meanr,
        gg.xip,
        gg.xim,
        grid=grid,
        theta_nom=gg.rnom,
        npairs=gg.npairs,
        weight=gg.weight,
        variances=(
            np.concatenate([gg.varxip, gg.varxim]) if grid == "integration" else None
        ),
    )
    out_path = sacc_out or os.path.join(
        output_dir or cv.cc["paths"]["output"],
        f"{ver}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc",
    )
    sacc_io.save(s, out_path, type="data")
    print(f"Wrote {grid} ξ± SACC part: {out_path}")
    return gg


def _from_snakemake(smk):
    p = smk.params
    run_2pcf(
        ver=p["ver"],
        min_sep=float(p["min_sep"]),
        max_sep=float(p["max_sep"]),
        nbins=int(p["nbins"]),
        npatch=int(p["npatch"]),
        # cat_config / output_dir were previously resolved via an os.chdir into
        # the cosmo_val dir + the COSMO_VAL env var; expose them as optional
        # params so the rule can pass them explicitly, falling back to the
        # class defaults (./cat_config.yaml, COSMO_VAL env) otherwise.
        cat_config=p.get("cat_config", "./cat_config.yaml"),
        output_dir=p.get("output_dir", None),
        # Grid label is resolved by the rule from the binning wildcards
        # (workflow/rules/twopoint.smk XI_GRIDS).
        grid=p.get("grid", "reporting"),
        # Write the SACC part exactly where the rule declares it (the .txt
        # byproduct still lands under the resolved output dir via _output_path).
        sacc_out=smk.output["sacc"],
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="TreeCorr ξ± 2PCF for one catalog version."
    )
    ap.add_argument(
        "--ver",
        required=True,
        help="Catalog version key in cat_config, e.g. SP_v1.4.6.3_leak_corr",
    )
    ap.add_argument(
        "--min-sep", type=float, required=True, help="Min separation [arcmin]"
    )
    ap.add_argument(
        "--max-sep", type=float, required=True, help="Max separation [arcmin]"
    )
    ap.add_argument("--nbins", type=int, required=True, help="Number of log bins")
    ap.add_argument(
        "--npatch", type=int, default=1, help="TreeCorr patch count (paper fiducial: 1)"
    )
    ap.add_argument(
        "--cat-config", required=True, help="Absolute path to cat_config.yaml"
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    ap.add_argument(
        "--grid",
        default="reporting",
        choices=["reporting", "integration"],
        help="SACC grid tag; 'integration' also attaches the varxip/varxim "
        "DiagonalCovariance",
    )
    a = ap.parse_args(argv)
    run_2pcf(
        ver=a.ver,
        min_sep=a.min_sep,
        max_sep=a.max_sep,
        nbins=a.nbins,
        npatch=a.npatch,
        cat_config=a.cat_config,
        output_dir=a.out,
        grid=a.grid,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
