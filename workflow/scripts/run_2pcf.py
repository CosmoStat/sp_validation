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
grids are the same compute with different ``--min-sep/--max-sep/--nbins``. The
ξ± is born as a SACC part, named by its binning, tagged with its ``--grid`` and
sealed under the catalogue's blind by ``CosmologyValidation.calculate_2pcf``.

``output_dir`` is passed explicitly so lc can point each run at its own
``{output}`` tree.
"""

import argparse
import os

from sp_validation.cosmo_val import CosmologyValidation


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
    """Measure ξ±(θ) for ``ver`` and write its sealed SACC part.

    Parameters mirror the TreeCorr reporting/integration grids: ``min_sep`` /
    ``max_sep`` in arcmin, ``nbins`` logarithmic bins, ``npatch`` spatial
    patches (1 for the paper fiducial). ``cat_config`` is an absolute path to
    the catalog configuration; ``output_dir`` overrides
    ``cat_config['paths']['output']``. ``sacc_out`` is the exact destination for
    the part (the Snakemake-declared output); it defaults to a binning-derived
    name under the resolved output directory for the CLI path.

    Returns
    -------
    sacc.Sacc
        The part as written.
    """
    cv = CosmologyValidation(
        versions=[ver],
        catalog_config=cat_config,
        output_dir=output_dir,
    )
    out_path = sacc_out or os.path.join(
        output_dir or cv.cc["paths"]["output"],
        f"{ver}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc",
    )
    part = cv.calculate_2pcf(
        ver,
        grid=grid,
        npatch=npatch,
        out=out_path,
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
    )
    print(f"Wrote {grid} ξ± SACC part: {out_path}")
    return part


def _from_snakemake(smk):
    p = smk.params
    run_2pcf(
        ver=p["ver"],
        min_sep=float(p["min_sep"]),
        max_sep=float(p["max_sep"]),
        nbins=int(p["nbins"]),
        npatch=int(p["npatch"]),
        cat_config=p["cat_config"],
        output_dir=p["output_dir"],
        grid=p.get("grid", "reporting"),
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
        "--grid", default="reporting", help="SACC grid tag for the measured points"
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
