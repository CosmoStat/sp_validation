"""Gather MC sample chunks and compute the final pure E/B covariance.

CLI refactor of the former Snakemake ``script:`` gather rule. Reads the actual
ξ± data vectors (reporting + integration grids), computes the pure E/B/ambiguous
decomposition (Schneider 2022), stacks the per-chunk MC sample blocks, and
forms the empirical 6-block covariance. Writes the per-(version, blind)
``<version>_<blind>_pure_eb_semianalytic.npz`` consumed by every downstream
pure-mode plot / PTE.

    python gather_pure_eb_chunks.py \
        --version SP_v1.4.6.3_leak_corr --blind A \
        --xi-reporting  <xi 20-bin .txt> \
        --xi-integration <xi 1000-bin .txt> \
        --chunks-dir <dir with pure_eb_chunk_*.npz> \
        --min-sep 1.0 --max-sep 250.0 --nbins 20 \
        --min-sep-int 0.5 --max-sep-int 300.0 --nbins-int 1000 \
        --out <output_dir>
"""

import argparse
import glob
import os

import numpy as np


def _load_xi(path, nbins):
    """Load ξ± from a TreeCorr text dump."""
    data = np.loadtxt(path, comments="#", max_rows=nbins)
    return {"meanr": data[:, 1], "xip": data[:, 3], "xim": data[:, 4]}


def gather(
    version,
    blind,
    xi_reporting,
    xi_integration,
    chunk_files,
    min_sep,
    max_sep,
    nbins,
    nbins_int,
    output_dir,
):
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    print(f"Gathering pure E/B for blind {blind}")

    gg = _load_xi(xi_reporting, nbins)
    gg_int = _load_xi(xi_integration, nbins_int)

    eb_results = get_pure_EB_modes(
        theta=gg["meanr"],
        xip=gg["xip"],
        xim=gg["xim"],
        theta_int=gg_int["meanr"],
        xip_int=gg_int["xip"],
        xim_int=gg_int["xim"],
        tmin=min_sep,
        tmax=max_sep,
    )
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = eb_results

    chunk_files = sorted(chunk_files)
    all_samples = []
    for chunk_file in chunk_files:
        data = np.load(chunk_file)
        all_samples.append(data["eb_samples"])
        print(f"Loaded {len(data['eb_samples'])} samples from {chunk_file}")

    eb_samples = np.vstack(all_samples)
    print(f"Total samples: {len(eb_samples)}")

    cov_pure_eb = np.cov(eb_samples.T)

    package = {
        "theta": gg["meanr"],
        "theta_int": gg_int["meanr"],
        "xip_total": gg["xip"],
        "xim_total": gg["xim"],
        "xip_E": xip_E,
        "xim_E": xim_E,
        "xip_B": xip_B,
        "xim_B": xim_B,
        "xip_amb": xip_amb,
        "xim_amb": xim_amb,
        "cov_pure_eb": cov_pure_eb,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{version}_{blind}_pure_eb_semianalytic.npz")
    np.savez(out_path, **package)
    print(f"Saved to {out_path}")
    return out_path


def _resolve_chunks(args):
    if args.chunks:
        files = args.chunks
    else:
        files = glob.glob(os.path.join(args.chunks_dir, "pure_eb_chunk_*.npz"))
    if not files:
        raise SystemExit(f"No chunk .npz found (chunks-dir={args.chunks_dir})")
    return files


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--blind", default="A")
    ap.add_argument("--xi-reporting", required=True)
    ap.add_argument("--xi-integration", required=True)
    ap.add_argument("--chunks-dir", help="Directory holding pure_eb_chunk_*.npz")
    ap.add_argument("--chunks", nargs="+", help="Explicit chunk .npz paths")
    ap.add_argument("--min-sep", type=float, default=1.0)
    ap.add_argument("--max-sep", type=float, default=250.0)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--min-sep-int", type=float, default=0.5)
    ap.add_argument("--max-sep-int", type=float, default=300.0)
    ap.add_argument("--nbins-int", type=int, default=1000)
    ap.add_argument("--npatch", type=int, default=1)
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    a = ap.parse_args(argv)
    gather(
        version=a.version,
        blind=a.blind,
        xi_reporting=a.xi_reporting,
        xi_integration=a.xi_integration,
        chunk_files=_resolve_chunks(a),
        min_sep=a.min_sep,
        max_sep=a.max_sep,
        nbins=a.nbins,
        nbins_int=a.nbins_int,
        output_dir=a.out,
    )


if __name__ == "__main__":
    _from_cli()
