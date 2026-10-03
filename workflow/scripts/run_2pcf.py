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
        --out <output_dir> --type mock

``CosmologyValidation.calculate_2pcf`` does the TreeCorr work and
``save_2pcf_sacc`` writes the result as a SACC file. Both live on the class:
serialising needs the version's n(z) and its tomographic bin map, so this
script only names the file — from the ``sacc_name`` param under Snakemake, from
``--sacc`` on the CLI (defaulting to the untagged ``xi_{ver}.sacc``, since each
lc recipe gets its own output tree). The *directory* is not this script's to
choose: ``save_2pcf_sacc`` joins the name onto the version's output directory,
so the SACC lands beside every other product of the run. Under Snakemake that
directory comes from the catalog config; on the CLI ``--out`` overrides it, so
lc can point each run at its own ``{output}`` tree.
"""

import argparse

from sp_validation.cosmo_val import CosmologyValidation


def run_2pcf(
    ver,
    compute_tomography,
    npatch,
    min_sep,
    max_sep,
    nbins,
    cat_config,
    output_dir,
    sacc_path,
    data_type="data",
):
    """Measure ξ±(θ) for ``ver`` and write it as the SACC file ``sacc_name``.

    Parameters mirror the TreeCorr reporting/integration grids: ``min_sep`` /
    ``max_sep`` in arcmin, ``nbins`` logarithmic bins, ``npatch`` spatial
    patches (1 for the paper fiducial). ``cat_config`` is an absolute path to
    the catalog configuration; ``output_dir`` overrides
    ``cat_config['paths']['output']`` so products land where lc expects, and
    is left ``None`` under Snakemake so the directory comes from the catalog
    config. ``sacc_name`` is a bare filename — ``save_2pcf_sacc`` joins it onto
    whichever of the two that resolves to.

    ``data_type`` is the ``'data'``/``'mock'`` provenance stamp SACC requires.
    It defaults to ``'data'`` because that is the fail-safe direction: a mock
    mislabelled as data is merely refused by ``sacc_io.load`` until it carries
    a blinding stamp, whereas real data mislabelled as a mock would load
    unblinded. Callers running on mocks pass ``'mock'`` explicitly.
    """
    cv = CosmologyValidation(
        versions=[ver],
        catalog_config=cat_config,
        output_dir=output_dir,
    )
    ggs = cv.calculate_2pcf(
        compute_tomography=compute_tomography,
        npatch=npatch,
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
    )
    # calculate_2pcf keys its result by version; save_2pcf_sacc takes the
    # per-version {pair: GGCorrelation} dict.
    cv.save_2pcf_sacc(ver, sacc_path, ggs[ver], type=data_type)


def _from_snakemake(smk):
    p = smk.params
    run_2pcf(
        ver=p["ver"],
        compute_tomography=str(p["compute_tomography"]).lower() == "true",
        npatch=int(p["npatch"]),
        min_sep=float(p["min_sep"]),
        max_sep=float(p["max_sep"]),
        nbins=int(p["nbins"]),
        cat_config=p.get("cat_config", "./cat_config.yaml"),
        output_dir=None,
        sacc_path=p["sacc_path"],
        data_type=p.get("data_type", "data"),
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
        "--compute-tomography",
        action="store_true",
        help="Compute tomographic 2PCF (default: non-tomographic)",
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
        "--sacc",
        required=True,
        help="Path to SACC file, written under --out (default: xi_{ver}.sacc)",
    )
    ap.add_argument(
        "--type",
        choices=("data", "mock"),
        default="data",
        help="Catalog provenance stamped on the SACC file, default=%(default)s",
    )
    a = ap.parse_args(argv)

    run_2pcf(
        ver=a.ver,
        compute_tomography=a.compute_tomography,
        min_sep=a.min_sep,
        max_sep=a.max_sep,
        nbins=a.nbins,
        npatch=a.npatch,
        cat_config=a.cat_config,
        output_dir=a.out,
        sacc_path=a.sacc,
        data_type=a.type,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
