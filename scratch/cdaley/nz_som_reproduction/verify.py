"""Check the frozen full-area delivery against a calibration output directory."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


def verify(config, output):
    """Compare delivered full-area n(z) and every sidecar bin ID."""
    output = Path(output)
    reference = config["reference"]
    actual = np.loadtxt(output / "nz_tomo.txt")
    expected = np.loadtxt(reference["nz"])
    z = actual[:, 0]
    means = np.trapezoid(z[:, None] * actual[:, 1:], z, axis=0)
    means /= np.trapezoid(actual[:, 1:], z, axis=0)
    ref_mean = np.trapezoid(expected[:, :1] * expected[:, 1:], expected[:, 0], axis=0)
    ref_mean /= np.trapezoid(expected[:, 1:], expected[:, 0], axis=0)
    error = abs(means - ref_mean)
    assert np.all(error < 1e-4), error
    mismatches, count = 0, 0
    # Both sidecars preserve the selected-catalogue row order.
    a = pq.ParquetFile(output / "tom_bin.parquet")
    b = pq.ParquetFile(reference["tom_bins"])
    assert a.metadata.num_rows == b.metadata.num_rows
    for aa, bb in zip(
        a.iter_batches(columns=["TILE_ID", "NUMBER", "TOM_BIN_ID"]),
        b.iter_batches(columns=["TILE_ID", "NUMBER", "TOM_BIN_ID"]),
    ):
        assert aa.column(0).equals(bb.column(0))
        assert aa.column(1).equals(bb.column(1))
        mismatches += np.count_nonzero(
            aa.column(2).to_numpy() != bb.column(2).to_numpy()
        )
        count += len(aa)
    assert count == a.metadata.num_rows, (count, a.metadata.num_rows)
    assert mismatches == 0, mismatches
    result = dict(
        means=means.tolist(),
        reference_means=ref_mean.tolist(),
        absolute_mean_errors=error.tolist(),
        rows=count,
        tom_bin_mismatches=int(mismatches),
        max_density_difference=float(abs(actual - expected).max()),
    )
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("output")
    args = parser.parse_args()
    verify(yaml.safe_load(Path(args.reference).read_text()), args.output)
