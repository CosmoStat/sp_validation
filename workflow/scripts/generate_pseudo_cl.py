"""Generate pseudo-Cls (data vector only, no covariance).

Dual-mode. Under Snakemake (``script:`` directive) the injected ``snakemake``
object supplies the parameters and the native product is renamed to the tagged
output filename the rule declares; as a standalone CLI (argparse) the same
compute runs from explicit flags and the primitive's native
``pseudo_cl_{ver}.sacc`` is left in place under ``--out`` (no rename — each
lc/ASTRA recipe gets its own output directory, so the untagged native name is
unambiguous and the primitives' skip-if-exists never collides across nbins
runs). The C_ell data vector is born as SACC (EE/BB/EB with a shared bandpower
window) — see ``sp_validation.cosmo_val.sacc_writers.pseudo_cl_to_sacc``. The
CLI form is what the lightcone/ASTRA recipe calls, so the measurement is driven
directly (no nested Snakemake) with lc handling orchestration:

    python generate_pseudo_cl.py \
        --ver SP_v1.4.6.3_leak_corr \
        --cat-config /path/to/cosmo_val/cat_config.yaml \
        --out <output_dir> \
        --binning powspace --nbins 32 --power 0.5

Supports two binning modes:
- Linear binning with configurable nbins for COSEBIS
- Power-space binning with configurable nbins for standard C_ell analysis

See generate_pseudo_cl_cov.py for covariance generation.
"""

import argparse
import json
import os

from sp_validation import sacc_io
from sp_validation.cosmo_val import CosmologyValidation


def generate_pseudo_cl(
    version: str,
    out_path: str,
    cat_config: str,
    nside: int = 1024,
    npatch: int = 1,
    blind: str = None,
    cosmo_params: dict = None,
    binning: str = "linear",
    nbins: int = None,
    power: float = 0.5,
):
    """Generate a pseudo-Cl data vector, born as a SACC part at ``out_path``.

    Parameters
    ----------
    version : str
        Catalog version (e.g., "SP_v1.4.6_leak_corr")
    out_path : str
        Exact destination the SACC part is *born at* — its final (possibly
        tagged) name. No native-basename + rename step, so this producer's
        skip-if-exists never collides with the untagged cv_pseudo_cl diagnostic.
    cat_config : str
        Path to catalog configuration YAML
    nside : int
        HEALPix nside for map-based estimation
    npatch : int
        Number of jackknife patches
    blind : str, optional
        Blind identifier (A, B, or C) to override n(z) path
    cosmo_params : dict, optional
        Cosmological parameters. Keys: Omega_m, sigma_8, n_s, h, Omega_b.
        If None, uses Planck 2018 defaults.
    binning : str
        Binning mode: "linear", "logspace", or "powspace"
    nbins : int
        Number of ell bins (required)
    power : float
        Power for powspace binning (0.5 = sqrt spacing)

    Returns
    -------
    str
        ``out_path`` (the SACC part written).
    """
    output_dir = os.path.dirname(out_path)
    os.makedirs(output_dir, exist_ok=True)

    blind_str = f" blind={blind}" if blind else ""
    if binning == "linear":
        # For linear binning, nbins determines ell_step such that we cover 2-2048
        ell_step = max(1, (2048 - 2) // nbins)
        bin_str = f"nbins={nbins} (ell_step={ell_step})"
    elif binning == "logspace":
        bin_str = f"nbins={nbins} (geomspace)"
    else:
        bin_str = f"nbins={nbins}, power={power}"

    print(f"\n{'=' * 60}")
    print(f"Generating pseudo-Cl for {version}{blind_str}")
    print(f"Binning: {binning} ({bin_str})")
    if cosmo_params:
        print(
            f"Cosmology: Om={cosmo_params.get('Omega_m')}, s8={cosmo_params.get('sigma_8')}"
        )
    else:
        print("Cosmology: Planck 2018 defaults")
    print(f"{'=' * 60}\n")

    # Remap config cosmology param names to get_cosmo() expected names
    if cosmo_params:
        cosmo_params = {
            "Omega_m": cosmo_params.get("Omega_m"),
            "Omega_b": cosmo_params.get("Omega_b"),
            "h": cosmo_params.get("h"),
            "sig8": cosmo_params.get("sigma_8"),
            "ns": cosmo_params.get("n_s"),
        }

    # Build CV kwargs based on binning mode
    cv_kwargs = dict(
        versions=[version],
        catalog_config=cat_config,
        output_dir=output_dir,
        binning=binning,
        nside=nside,
        cell_method="catalog",
        nrandom_cell=100,
        blind=blind,
        cosmo_params=cosmo_params,
        npatch=npatch,
        theta_min=1.0,
        theta_max=250.0,
        nbins=20,
    )
    if binning == "linear":
        cv_kwargs["ell_step"] = ell_step
    else:
        cv_kwargs["n_ell_bins"] = nbins
        cv_kwargs["power"] = power

    cv = CosmologyValidation(**cv_kwargs)

    # Calculate pseudo-Cls only (no covariance). The data vector is born as a
    # SACC part directly at out_path (its final, possibly-tagged name) — no
    # shared native basename, no rename, so this producer's skip-if-exists never
    # collides with the untagged cv_pseudo_cl diagnostic (which would otherwise
    # let one rule adopt + delete the other's differently-blinded file).
    cv.calculate_pseudo_cl(out_path=out_path)

    if os.path.exists(out_path):
        # Pipeline-internal readback of the unblinded data part just written
        # (blinding is a downstream Smokescreen step).
        s = sacc_io.load(out_path, allow_unblinded=True)
        ell = sacc_io.get_pseudo_cl(s, (0, 0))[0]
        print(f"Generated pseudo-Cl with {len(ell)} ell bins")
        print(f"ell range: [{ell.min():.1f}, {ell.max():.1f}]")
    return out_path


def _from_snakemake(smk):
    p = smk.params
    # Born directly at the rule's declared (tagged) output — no rename step.
    generate_pseudo_cl(
        version=p["version"],
        out_path=smk.output.pseudo_cl,
        cat_config=p["cat_config"],
        nside=int(p["nside"]),
        npatch=int(p["npatch"]),
        blind=p.get("blind", None),
        cosmo_params=p.get("cosmo_params", None),
        binning=p["binning"],
        nbins=int(p["nbins"]),
        power=float(p.get("power", 0.5)),
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Pseudo-Cl data vector for one catalog version."
    )
    ap.add_argument(
        "--ver",
        required=True,
        help="Catalog version key in cat_config, e.g. SP_v1.4.6.3_leak_corr",
    )
    ap.add_argument(
        "--cat-config", required=True, help="Absolute path to cat_config.yaml"
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    ap.add_argument(
        "--nside", type=int, default=1024, help="HEALPix nside for map estimation"
    )
    ap.add_argument(
        "--npatch",
        type=int,
        default=1,
        help="Jackknife patch count (paper fiducial: 1)",
    )
    ap.add_argument(
        "--binning",
        choices=["linear", "logspace", "powspace"],
        default="powspace",
        help="Ell binning mode",
    )
    ap.add_argument("--nbins", type=int, required=True, help="Number of ell bins")
    ap.add_argument(
        "--power",
        type=float,
        default=0.5,
        help="Power for powspace binning (0.5 = sqrt spacing)",
    )
    ap.add_argument(
        "--blind", choices=["A", "B", "C"], default=None, help="Blind identifier"
    )
    ap.add_argument(
        "--cosmo-json",
        default=None,
        help="Path to a Planck18-style cosmology JSON; omit for Planck18 defaults",
    )
    a = ap.parse_args(argv)

    cosmo_params = None
    if a.cosmo_json:
        with open(a.cosmo_json) as f:
            cosmo_params = json.load(f)

    # lc/ASTRA path: --out is a per-recipe directory; the untagged native name
    # is unambiguous there (each recipe gets its own tree, so no cross-nbins or
    # cross-blind collision).
    out_path = os.path.join(a.out, f"pseudo_cl_{a.ver}.sacc")
    generate_pseudo_cl(
        version=a.ver,
        out_path=out_path,
        cat_config=a.cat_config,
        nside=a.nside,
        npatch=a.npatch,
        blind=a.blind,
        cosmo_params=cosmo_params,
        binning=a.binning,
        nbins=a.nbins,
        power=a.power,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
