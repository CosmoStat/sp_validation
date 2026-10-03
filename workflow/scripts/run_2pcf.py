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

``CosmologyValidation.calculate_2pcf`` does the TreeCorr work and returns the
ξ± ``GGCorrelation`` objects in memory; it no longer writes the ``.txt`` dump
or the ξ+/ξ- FITS files. ``output_dir`` is still passed explicitly (rather than
via the ``COSMO_VAL`` env hook) so the patch-centre file lands in each run's own
``{output}`` tree.
"""

import argparse

from sp_validation.cosmo_val import CosmologyValidation


def run_2pcf(
    ver,
    npatch,
    compute_tomography,
    min_sep,
    max_sep,
    nbins,
    cat_config,
    output_dir,
):
    """Measure ξ±(θ) for ``ver``.

    Parameters mirror the TreeCorr reporting/integration grids: ``min_sep`` /
    ``max_sep`` in arcmin, ``nbins`` logarithmic bins, ``npatch`` spatial
    patches (1 for the paper fiducial). ``cat_config`` is an absolute path to
    the catalog configuration; ``output_dir`` overrides
    ``cat_config['paths']['output']`` so products land where lc expects.
    """
    cv = CosmologyValidation(
        versions=[ver],
        catalog_config=cat_config,
        output_dir=output_dir,
    )
    return cv.calculate_2pcf(
        npatch=npatch,
        compute_tomography=compute_tomography,
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
    )


def _from_snakemake(smk):
    p = smk.params
    run_2pcf(
        ver=p["ver"],
        npatch=int(p["npatch"]),
        compute_tomography=p.get("compute_tomography", False),
        min_sep=float(p["min_sep"]),
        max_sep=float(p["max_sep"]),
        nbins=int(p["nbins"]),
        # cat_config / output_dir were previously resolved via an os.chdir into
        # the cosmo_val dir + the COSMO_VAL env var; expose them as optional
        # params so the rule can pass them explicitly, falling back to the
        # class defaults (./cat_config.yaml, COSMO_VAL env) otherwise.
        cat_config=p.get("cat_config", "./cat_config.yaml"),
        output_dir=p.get("output_dir", None),
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
        "--tomo",
        action="store_true",
        help="Compute tomographic ξ±(θ) measurement. True or False.",
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
    a = ap.parse_args(argv)
    run_2pcf(
        ver=a.ver,
        npatch=a.npatch,
        compute_tomography=a.tomo,
        min_sep=a.min_sep,
        max_sep=a.max_sep,
        nbins=a.nbins,
        cat_config=a.cat_config,
        output_dir=a.out,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
