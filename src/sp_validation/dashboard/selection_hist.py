"""Stream cumulative selection histograms from a comprehensive catalogue."""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml

from sp_validation.masks import apply_condition


def build(path, config, expected, chunk=500_000):
    """Apply recorded data cuts and the noshear metacal cuts cumulatively."""
    cfg = yaml.safe_load(Path(config).read_text())
    cuts = cfg["dat"]
    cols = list(
        dict.fromkeys(
            [x["col_name"] for x in cuts]
            + [
                "NGMIX_FLAGS_NOSHEAR",
                "NGMIX_FLUX_NOSHEAR",
                "NGMIX_FLUX_ERR_NOSHEAR",
                "NGMIX_T_NOSHEAR",
                "NGMIX_T_PSF_RECONV_NOSHEAR",
                "mag",
            ]
        )
    )
    edges = np.linspace(15, 30, 151)
    hist = np.zeros((len(cuts) + 2, len(edges) - 1), dtype=np.int64)
    counts = np.zeros(len(cuts) + 2, dtype=np.int64)
    with h5py.File(path) as f:
        data = f["data"]
        for start in range(0, len(data), chunk):
            rows = data.fields(cols)[start : start + chunk]
            mask = np.ones(len(rows), dtype=bool)
            stages = [mask.copy()]
            for cut in cuts:
                mask &= apply_condition(
                    rows[cut["col_name"]],
                    cut["kind"],
                    cut["value"],
                    ignore_bits=cut.get("ignore_bits", 0),
                )
                stages.append(mask.copy())
            mc = cfg["metacal"]
            with np.errstate(divide="ignore", invalid="ignore"):
                snr = rows["NGMIX_FLUX_NOSHEAR"] / rows["NGMIX_FLUX_ERR_NOSHEAR"]
                size = rows["NGMIX_T_NOSHEAR"] / rows["NGMIX_T_PSF_RECONV_NOSHEAR"]
            mask &= (
                (rows["NGMIX_FLAGS_NOSHEAR"] == 0)
                & (snr > mc["gal_snr_min"])
                & (snr < mc["gal_snr_max"])
                & (size > mc["gal_rel_size_min"])
                & (size < mc["gal_rel_size_max"])
            )
            stages.append(mask)
            for i, stage in enumerate(stages):
                counts[i] += stage.sum()
                hist[i] += np.histogram(rows["mag"][stage], bins=edges)[0]
    if counts[-1] != expected:
        raise ValueError(f"Selection retained {counts[-1]} rows; expected {expected}")
    return {
        "edges": edges,
        "hist": hist,
        "counts": counts,
        "labels": np.array(
            ["All detections"] + [c["label"] for c in cuts] + ["Metacal SNR and size"]
        ),
    }


def plot(data, out):
    """Plot cumulative histogram steps from the validated product."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from . import style as S

    with S.figure_style():
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        n = len(data["labels"])
        for i, (hist, label) in enumerate(zip(data["hist"], data["labels"])):
            label = str(label)
            if label.startswith('r"') and label.endswith('"'):
                label = label[2:-1]
            color = S.INK if i == n - 1 else mpl.colormaps["tab20"](i / max(n - 2, 1))
            ax.stairs(
                hist,
                data["edges"],
                label=label,
                color=color,
                linewidth=1.5 if i == n - 1 else 0.8,
            )
        ax.set_yscale("log")
        ax.set_xlabel("r magnitude")
        ax.set_ylabel("Detections")
        ax.legend(ncol=2, fontsize=6)
        return S.save(fig, out, "full")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalogue", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--expected", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", required=True)
    args = p.parse_args()
    np.savez_compressed(
        args.out,
        version=args.version,
        selection="cumulative config_mask cuts and noshear metacal cuts",
        units="r-band magnitude, AB",
        **build(args.catalogue, args.config, args.expected),
    )


if __name__ == "__main__":
    main()
