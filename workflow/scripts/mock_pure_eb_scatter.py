"""Compute the pure E/B configuration-space data vector for GLASS mocks.

The reporting grid is the external 20-bin GLASS ``xi`` product (1--250
arcmin).  The integration grid is the locally-produced 1000-bin ``gg``
product (0.5--500 arcmin).  This is deliberately only the data-vector part of
``sp_validation.b_modes.calculate_pure_eb_correlation``: no covariance or
jackknife machinery is run here.

The script is dual-mode.  Snakemake supplies one mock through ``snakemake``;
standalone use accepts either explicit mock IDs or a glob of fine-``xi``
files.  The latter is useful while the 350-mock production is still filling
in.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import treecorr
from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

_EB_KEYS = ("xip_E", "xim_E", "xip_B", "xim_B", "xip_amb", "xim_amb")
_ID_RE = re.compile(r"(?:glass_mock_|mock_)(\d{5})")


def _native(values):
    """Return a native-byte-order float array for numba inputs."""

    return np.asarray(values, dtype=float)


def _mock_id_from_path(path: Path) -> str:
    match = _ID_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not recover a five-digit mock ID from {path}")
    return match.group(1)


def _reporting_path(reporting_dir: Path, mock_id: str) -> Path:
    """External GLASS products use `_4096_nbins=20`; locally produced
    reporting ξ (v2/masked rules) use `_nbins=20`.  Prefer the external
    name, fall back to the local one."""
    external = reporting_dir / f"xi_glass_mock_{mock_id}_4096_nbins=20.fits"
    if external.is_file():
        return external
    return reporting_dir / f"xi_glass_mock_{mock_id}_nbins=20.fits"


def _output_path(output_dir: Path, mock_id: str) -> Path:
    return output_dir / f"pure_eb_glass_mock_{mock_id}.npz"


def compute_one(
    mock_id: str,
    fine_xi_path: str | Path,
    reporting_xi_path: str | Path,
    output_path: str | Path,
    *,
    fine_min_sep: float = 0.5,
    fine_max_sep: float = 500.0,
    fine_nbins: int = 1000,
    reporting_min_sep: float = 1.0,
    reporting_max_sep: float = 250.0,
    reporting_nbins: int = 20,
    integration_max_sep: float | None = None,
) -> Path:
    """Transform one fine/reporting pair and write its six pure-mode arrays."""

    fine_xi_path = Path(fine_xi_path)
    reporting_xi_path = Path(reporting_xi_path)
    output_path = Path(output_path)
    if not fine_xi_path.is_file():
        raise FileNotFoundError(fine_xi_path)
    if not reporting_xi_path.is_file():
        raise FileNotFoundError(reporting_xi_path)

    gg_int = treecorr.GGCorrelation(
        min_sep=fine_min_sep,
        max_sep=fine_max_sep,
        nbins=fine_nbins,
        sep_units="arcmin",
    )
    gg_int.read(str(fine_xi_path))

    gg = treecorr.GGCorrelation(
        min_sep=reporting_min_sep,
        max_sep=reporting_max_sep,
        nbins=reporting_nbins,
        sep_units="arcmin",
    )
    gg.read(str(reporting_xi_path))

    # Match calculate_pure_eb_correlation: tmin/tmax are the actual reporting
    # bin edges, not the nominal constructor arguments.  Explicit casts also
    # handle FITS big-endian arrays before they enter numba.
    tmin = float(gg.left_edges[0])
    tmax = float(gg.right_edges[-1])
    theta_int = _native(gg_int.meanr)
    xip_int = _native(gg_int.xip)
    xim_int = _native(gg_int.xim)
    if integration_max_sep is not None:
        # The paper covariance input ends at 300 arcmin.  The production GLASS
        # file ends at 500 arcmin; retaining the default full file is required
        # for the published data-vector product, while this switch makes an
        # apples-to-apples covariance diagnostic possible.  It matters for the
        # padded xi- branch of cosmo_numba (whose effective integration extent
        # follows the supplied integration grid).
        keep = theta_int <= float(integration_max_sep)
        if np.count_nonzero(keep) < 10:
            raise ValueError(
                "integration_max_sep leaves too few fine xi bins: "
                f"{integration_max_sep}"
            )
        theta_int, xip_int, xim_int = (
            values[keep] for values in (theta_int, xip_int, xim_int)
        )
    pure_modes = get_pure_EB_modes(
        theta=_native(gg.meanr),
        xip=_native(gg.xip),
        xim=_native(gg.xim),
        theta_int=theta_int,
        xip_int=xip_int,
        xim_int=xim_int,
        tmin=tmin,
        tmax=tmax,
        parallel=True,
    )
    arrays = dict(zip(_EB_KEYS, (_native(values) for values in pure_modes)))

    if not all(np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError(f"Non-finite pure E/B output for mock {mock_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        mock_id=np.asarray(mock_id),
        theta=_native(gg.meanr),
        reporting_left_edges=_native(gg.left_edges),
        reporting_right_edges=_native(gg.right_edges),
        tmin=tmin,
        tmax=tmax,
        integration_max_sep=(
            float(integration_max_sep)
            if integration_max_sep is not None
            else float(gg_int.meanr[-1])
        ),
        **arrays,
    )
    print(
        f"Saved {output_path}: mock={mock_id}, ntheta={len(gg.meanr)}, "
        f"theta=[{tmin:.6g}, {tmax:.6g}] arcmin",
        flush=True,
    )
    return output_path


def _standalone_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock-ids",
        nargs="+",
        help="Five-digit mock IDs; if omitted, IDs are recovered from --fine-glob.",
    )
    parser.add_argument(
        "--fine-glob",
        default="results/glass_mock/gg_glass_mock_*_nbins=1000.fits",
        help="Glob for fine mock ξ files (quoted so the script expands it).",
    )
    parser.add_argument(
        "--reporting-dir",
        default="/n09data/guerrini/glass_mock_v1.4.6/results",
        help="Directory containing external 20-bin GLASS ξ FITS files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/glass_mock",
        help="Directory for pure_eb_glass_mock_<id>.npz files.",
    )
    parser.add_argument(
        "--integration-max-sep",
        type=float,
        default=None,
        help=(
            "Optional upper cut on the fine grid; use 300 to match the "
            "paper CosmoCov integration input."
        ),
    )
    return parser.parse_args(argv)


def _standalone_main(argv=None):
    args = _standalone_args(argv)
    fine_paths = {}
    if args.mock_ids:
        for raw_id in args.mock_ids:
            mock_id = str(raw_id).zfill(5)
            fine_path = Path(args.fine_glob.format(mock_id=mock_id))
            # An explicit ID may use a literal path pattern, or the default
            # wildcard.  Resolve both forms without importing glob globally.
            if any(char in args.fine_glob for char in "*?["):
                matches = sorted(Path(path) for path in glob.glob(args.fine_glob))
                matches = [p for p in matches if _mock_id_from_path(p) == mock_id]
                if matches:
                    fine_path = matches[0]
            if fine_path.is_file():
                fine_paths[mock_id] = fine_path
            else:
                print(f"Skipping {mock_id}: missing fine ξ {fine_path}")
    else:
        fine_paths = {
            _mock_id_from_path(path): path
            for path in sorted(Path(path) for path in glob.glob(args.fine_glob))
        }

    if not fine_paths:
        raise FileNotFoundError(f"No fine ξ files matched {args.fine_glob!r}")

    for mock_id, fine_path in sorted(fine_paths.items()):
        reporting_path = _reporting_path(Path(args.reporting_dir), mock_id)
        output_path = _output_path(Path(args.output_dir), mock_id)
        # A concurrently-produced mock is allowed to be absent on the
        # external reporting side; leave it for a later invocation.
        if not reporting_path.is_file():
            print(f"Skipping {mock_id}: missing reporting ξ {reporting_path}")
            continue
        if output_path.is_file():
            continue
        compute_one(
            mock_id,
            fine_path,
            reporting_path,
            output_path,
            integration_max_sep=args.integration_max_sep,
        )


def _snakemake_main(smk):
    compute_one(
        str(smk.wildcards.mock_id),
        smk.input.fine,
        smk.input.reporting,
        smk.output.pure_eb,
    )


if "snakemake" in globals():
    _snakemake_main(snakemake)
else:
    _standalone_main()
