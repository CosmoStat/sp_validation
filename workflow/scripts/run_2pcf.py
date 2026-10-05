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

The integration job measures unpatched ξ± once. A configured reporting job
consumes that SACC part, averages its means with pair weights on exactly nested
edges, and measures the reporting covariance. Without a fine input, a standalone
job uses unpatched reporting means.
``CosmologyValidation.calculate_2pcf_version`` measures the non-tomographic
``("all", "all")`` pair and writes its ``xi_{basename}.txt`` dump (a raw
byproduct); the ξ± data product is born as SACC here, a *part* named by its
binning and tagged with its ``--grid``. The part carries the covariance the
measurement estimated: the dense jackknife covariance when it had patches, the
shot-noise ``varxip``/``varxim`` diagonal when it had none.

``output_dir`` is passed explicitly so lc can point each run at its own
``{output}`` tree.
"""

import argparse
import os

import numpy as np
import sacc
import treecorr

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
    b_target=0.01,
    integration=None,
    fine_xi=None,
):
    """Measure ξ±(θ) for ``ver`` and write its reporting SACC part.

    Parameters mirror the TreeCorr reporting/integration grids: ``min_sep`` /
    ``max_sep`` in arcmin, ``nbins`` logarithmic bins, ``npatch`` spatial
    patches (1 for the paper fiducial). ``cat_config`` is an absolute path to
    the catalog configuration; ``output_dir`` overrides
    ``cat_config['paths']['output']`` so the ``.txt`` byproduct lands where lc
    expects. ``sacc_out`` is the exact destination for the SACC part (the
    Snakemake-declared output); it defaults to a binning-derived name under
    the resolved output directory for the CLI path.

    ``integration`` and ``fine_xi`` together select the fine-grid mean source;
    this measurement producer consumes the unblinded intermediate before the
    data products enter blinding.

    Returns
    -------
    treecorr.GGCorrelation
        The measured correlation object (also the source of the SACC part).
    """
    if (integration is None) != (fine_xi is None):
        raise ValueError(
            "supply the configured integration grid and its fine ξ± part together"
        )
    cv = CosmologyValidation(
        versions=[ver],
        catalog_config=cat_config,
        output_dir=output_dir,
        # so the SACC provenance metadata stamps the npatch actually measured
        npatch=npatch,
        b_target=b_target,
        theta_min=min_sep,
        theta_max=max_sep,
        nbins=nbins,
        integration=integration,
    )
    fine_correlations = None
    if fine_xi is not None:
        # This is a raw-catalogue measurement producer, before blinding.
        # A concealed input cannot be mixed with its raw-catalogue covariance.
        part = sacc.Sacc.load_fits(os.fspath(fine_xi))
        if part.metadata.get("concealed", False):
            raise ValueError(
                "reporting measurement requires a pre-blinding fine ξ± product"
            )
        if (
            part.metadata.get("catalogue_version") != ver
            or part.metadata.get("npatch") != 1
        ):
            raise ValueError(
                "fine ξ± must be the same catalogue version measured with npatch=1"
            )
        fine = treecorr.GGCorrelation(cv._binning(**cv.integration, var_method="shot"))
        theta, xip, xim = sacc_io.get_xi(part, (0, 0), grid="integration")
        fields = sacc_io.get_xi_aux(part, (0, 0), grid="integration")
        if fields["theta_nom"].shape != fine.rnom.shape or not np.allclose(
            fields["theta_nom"], fine.rnom, rtol=1e-12, atol=0
        ):
            raise ValueError(
                "fine ξ± product does not match the configured integration grid"
            )
        for name, values in {"meanr": theta, "xip": xip, "xim": xim, **fields}.items():
            if name != "theta_nom":
                getattr(fine, name)[:] = values
        fine_correlations = {"tomo_bin_all_tomo_bin_all": fine}
    gg = cv.calculate_2pcf_version(
        ver,
        npatch=npatch,
        compute_tomography=False,
        fine_correlations=fine_correlations,
        min_sep=min_sep,
        max_sep=max_sep,
        nbins=nbins,
    )["tomo_bin_all_tomo_bin_all"]

    # Born-as-SACC ξ± part. theta = meanr; theta_nom = rnom.
    jackknife = gg.var_method == "jackknife"
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
        meanlogr=gg.meanlogr,
        xip_im=gg.xip_im,
        xim_im=gg.xim_im,
        covariance=gg.cov if jackknife else None,
        variances=None if jackknife else np.concatenate([gg.varxip, gg.varxim]),
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
        cat_config=p["cat_config"],
        output_dir=p["output_dir"],
        grid=p.get("grid", "reporting"),
        b_target=p.get("b_target", 0.01),
        integration=p.get("integration"),
        fine_xi=smk.input.fine_xi[0] if smk.input.fine_xi else None,
        # The SACC part goes exactly where the rule declares it; the .txt
        # byproduct still lands under the resolved output dir.
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
    ap.add_argument("--b-target", type=float, default=0.01)
    ap.add_argument("--fine-xi", help="Unpatched integration-grid SACC part")
    ap.add_argument("--min-sep-int", type=float)
    ap.add_argument("--max-sep-int", type=float)
    ap.add_argument("--nbins-int", type=int)
    a = ap.parse_args(argv)
    integration_args = (a.min_sep_int, a.max_sep_int, a.nbins_int)
    if a.fine_xi is not None and any(value is None for value in integration_args):
        ap.error("--fine-xi requires --min-sep-int, --max-sep-int and --nbins-int")
    if a.fine_xi is None and any(value is not None for value in integration_args):
        ap.error("integration-grid arguments require --fine-xi")
    integration = (
        dict(zip(("min_sep", "max_sep", "nbins"), integration_args))
        if a.fine_xi
        else None
    )
    run_2pcf(
        ver=a.ver,
        min_sep=a.min_sep,
        max_sep=a.max_sep,
        nbins=a.nbins,
        npatch=a.npatch,
        cat_config=a.cat_config,
        output_dir=a.out,
        grid=a.grid,
        b_target=a.b_target,
        integration=integration,
        fine_xi=a.fine_xi,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
