"""Run the SOM n(z) calibration on twins of the v2 catalogue (``nz.twin``).

The twin keeps the recipe's sources and EGS calibrators as they are and
paints their redshifts with a model (``colour``: a DEEP2/3 neighbour's
redshift in the SOM's colour space; ``bpz``: BPZ plus a neighbour's
spec-z − BPZ in (BPZ, r), from the spectra of lensing sources). Rungs add
one ingredient each; realisation s paints with seed s on every rung, so
rungs pair by seed. Δ⟨z⟩ is the calibrated minus the true mean of a bin.

Rungs, with what was predicted before the first run:

- ``representative``: calibrators are as many sources drawn at random, any
  BPZ. Prediction: |mean Δ| ≤ 0.003 in every bin under both models (gold
  selection and the shear weight varying inside a group, as on the toy
  ladder), scatter 0.001–0.003.
- ``egs``: the real EGS calibrators, with their selection, depth and field.
  Prediction: |Δ| ≤ 0.01 under both models, largest in bins 3–4, where EGS's
  deeper g and z put calibrators in different cells from sources of the same
  redshift.
- ``success``: ``egs`` without calibrators painted above z = 1.4, where DEEP2
  measures almost nothing. Prediction: no change under ``colour`` (its
  library has almost no such spectra); under ``bpz`` bin 4 low by 0.01–0.03.
- ``zp_patch``: ``success`` with the i-band patch of Romain Paviot's note
  (+20 mmag over RA 140–230, Dec 30–60) on sources and calibrators, taken up
  by galaxies at ``UPTAKE``; BPZ moves along its gradient in each SOM cell.
  Prediction: Δ moves by ≤ 0.005 per bin.
- ``zp_map``: ``success`` with a new zero-point map per realisation (a
  coherent field and per-tile scatter at the note's amplitudes, plus the
  patch). Prediction: Δ moves by 0.002–0.005 rms over maps; a bin's true
  mean varies across the sky by 0.005–0.01 rms between 54 deg² regions.

Stages write under ``--output`` and are skipped when their products exist:
``inputs.npz`` (sources with complete photometry, calibrators, libraries),
``reference.npz`` (the recipe on the real catalogue; its means must match
``--reference``), ``near_<model>.npz``, and one JSON line per (rung, model,
seed) in ``records.jsonl``.
"""

import argparse
import json
from pathlib import Path

import healpy as hp
import numpy as np

from sp_validation.data_roots import load_yaml
from sp_validation.io import read_catalogue
from sp_validation.nz import calibration as nz
from sp_validation.nz import inputs, pipeline, twin
from sp_validation.object_id import NUMBER_LIMIT

RECIPE = Path(__file__).parents[2] / "config/calibration/nz/v2.yaml"
K = 16
SKY = {"ra": [0.0, 360.0], "dec": [-90.0, 90.0]}
#: The ``colour`` model's library: every DEEP2/3 spectrum, both fields.
COLOUR_SURVEYS = ["deep2_final", "deep3_final"]
#: The ``bpz`` model's library: galaxy redshift surveys of the compilation
#: (no QSO or AGN catalogue, no literature aggregator that may carry
#: photometric redshifts), restricted to objects that are lensing sources.
BPZ_SURVEYS = [
    "DESI_dr1_iron", "rcsed_final", "SDSS_SPECOBJ_None", "saga_dr3", "hecate",
    "HETDEX_PSC_PDR1_V1.5", "SPITZER_DATAFUSION", "herschel_help", "LAMOST_LRS_dr12",
    "uzc_final", "gogreen_dr2", "candels_v2", "mosdef_final", "c3r2_dr2", "c3r2_dr3",
    "MMT_NEPW", "shels_f2_final", "JADES_DR4", "3dhst", "THRILS",
    *COLOUR_SURVEYS,
]  # fmt: skip
Z_SUCCESS = 1.4
UPTAKE = 0.75  # galaxies follow 0.6-0.9 of the stellar zero-point offset
#: Stellar per-tile scatter in u g r i z, mag (r fixed by the note's
#: regression; u from our own EGS-against-wide u-g term).
SIGMA = [0.020, 0.009, 0.0, 0.015, 0.019]
PATCH = dict(band="i", ra=[140.0, 230.0], dec=[30.0, 60.0], offset=0.020)
REGION_NSIDE = 8


def load_inputs(config, path, workers):
    if path.exists():
        return dict(np.load(path))
    rc = config["inputs"]["calibrators"]
    offset = config["z2_offset"]
    cal = inputs.calibrators(**rc, offset=offset, workers=workers)
    src = inputs.sources(**config["inputs"]["sources"], workers=workers)
    pos = read_catalogue(
        config["inputs"]["sources"]["catalogue"], columns=["RA", "Dec"]
    )
    m, _ = nz.z_magnitudes(src["mag"], offset)
    rows = np.flatnonzero(
        (src["status"] == 0)
        & nz.detected(m).all(1)
        & np.isfinite(src["photoz"])
        & (src["weight"] > 0)
    )
    libs = {}
    for name, surveys in (("colour", COLOUR_SURVEYS), ("bpz", BPZ_SURVEYS)):
        lib = inputs.calibrators(
            rc["photometry"], surveys, SKY, rc["z_min"], offset, workers=workers
        )
        if name == "bpz":
            keep = np.isin(lib["object_id"], src["object_id"][rows])
            lib = {k: v[keep] for k, v in lib.items()}
        libs.update(
            {f"lib_{name}_{k}": lib[k] for k in ("z", "photoz", "mag", "object_id")}
        )
    out = dict(
        **{
            f"cal_{k}": cal[k] for k in ("z", "photoz", "mag", "object_id", "ra", "dec")
        },
        **{f"src_{k}": src[k][rows] for k in ("mag", "photoz", "weight", "object_id")},
        src_ra=np.asarray(pos["RA"])[rows],
        src_dec=np.asarray(pos["Dec"])[rows],
        **libs,
    )
    np.savez(path, **out)
    return out


def part(data, prefix):
    n = len(prefix)
    return {k[n:]: v for k, v in data.items() if k.startswith(prefix)}


def summarise(result):
    """What a later stage needs of an :func:`pipeline.estimate` result."""
    keep = ("edges", "cal_group", "prelim", "group", "tom")
    return dict(
        **{k: result[k] for k in keep},
        **{f"model_{k}": v for k, v in result["model"].items()},
        means=[b["mean"] for b in result["est"]["bins"]],
    )


def restore(saved, config, src, cal):
    """An :func:`pipeline.estimate` result from :func:`summarise`'s arrays."""
    result = dict(
        {k: saved[k] for k in ("edges", "cal_group", "prelim", "group", "tom")},
        model={k[6:]: v for k, v in saved.items() if k.startswith("model_")},
    )
    return twin.recalibrate(config, result, src, cal)


def tile_of(object_id):
    return object_id // NUMBER_LIMIT


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path, help="summary.json of the recipe's run")
    p.add_argument("--realisations", type=int, default=20)
    p.add_argument(
        "--rungs",
        nargs="+",
        default=["representative", "egs", "success", "zp_patch", "zp_map"],
    )
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    config = load_yaml(RECIPE)
    offset, w = config["z2_offset"], a.workers
    data = load_inputs(config, a.output / "inputs.npz", w)
    src, cal = part(data, "src_"), part(data, "cal_")
    libs = {m: part(data, f"lib_{m}_") for m in twin.SPACES}

    path = a.output / "reference.npz"
    if not path.exists():
        result = pipeline.estimate(config, cal, src, workers=w)
        model = result["model"]
        _, cell_all = nz.assign(
            nz.z_magnitudes(src["mag"], offset)[0], model, workers=w
        )
        grad = twin.photoz_gradient(
            src["mag"], src["photoz"], cell_all, len(model["code"]), offset=offset
        )
        np.savez(path, **summarise(result), cell_all=cell_all, grad=grad)
        print("reference means", [round(m, 6) for m in summarise(result)["means"]])
    ref = dict(np.load(path))
    if a.reference:
        want = [b["mean"] for b in json.loads(a.reference.read_text())["bins"]]
        print("reference", ref["means"], "recipe run", want)
    base = restore(ref, config, src, cal)
    grad, cell_all = ref["grad"], ref["cell_all"]
    cal_cell = nz.assign(cal["mag"], base["model"])[1]

    near = {}
    for m in twin.SPACES:
        path = a.output / f"near_{m}.npz"
        if not path.exists():
            np.savez(
                path,
                src=twin.neighbours(
                    m,
                    libs[m],
                    src["mag"],
                    src["photoz"],
                    src["object_id"],
                    k=K,
                    offset=offset,
                    workers=w,
                ),
                cal=twin.neighbours(
                    m,
                    libs[m],
                    cal["mag"],
                    cal["photoz"],
                    cal["object_id"],
                    k=K,
                    offset=offset,
                    workers=w,
                ),
            )
        near[m] = dict(np.load(path))

    def painted(m, seed):
        rng = np.random.default_rng(seed)
        zs = twin.paint(m, libs[m], near[m]["src"], src["photoz"], rng)
        zc = twin.paint(m, libs[m], near[m]["cal"], cal["photoz"], rng)
        return zs, zc

    def perturbed(dm_src, dm_cal):
        """Sources and calibrators seen through zero-point offsets."""
        s = dict(src, mag=twin.offset_magnitudes(src["mag"], dm_src))
        s["photoz"] = src["photoz"] + (grad[cell_all] * dm_src).sum(1)
        c = dict(cal, mag=cal["mag"] + dm_cal)
        c["photoz"] = cal["photoz"] + (grad[cal_cell] * dm_cal).sum(1)
        return s, c

    region = hp.ang2pix(REGION_NSIDE, src["ra"], src["dec"], lonlat=True)

    def regional(result, z):
        """Shear-weighted true mean of each bin per sky region, (n_bin, n_pix)."""
        out = []
        for b in range(1, len(result["edges"])):
            sel = result["tom"] == b
            wz = np.bincount(
                region[sel], src["weight"][sel] * z[sel], hp.nside2npix(REGION_NSIDE)
            )
            ww = np.bincount(
                region[sel], src["weight"][sel], hp.nside2npix(REGION_NSIDE)
            )
            out.append((wz, ww))
        return out

    records = a.output / "records.jsonl"
    done = set()
    if records.exists():
        done = {
            (r["rung"], r["model"], r["seed"])
            for r in map(json.loads, records.read_text().splitlines())
        }

    def write(rung, m, seed, result, z, **extra):
        rec = dict(
            rung=rung,
            model=m,
            seed=seed,
            bins=twin.score(result, src["weight"], z),
            **extra,
        )
        with records.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(rung, m, seed, [round(b["delta"], 4) for b in rec["bins"]], flush=True)

    zp_patch = None
    for seed in range(a.realisations):
        paints = {m: painted(m, seed) for m in twin.SPACES}
        for rung in a.rungs:
            todo = [m for m in twin.SPACES if (rung, m, seed) not in done]
            if not todo:
                continue
            if rung == "representative":
                rows = np.random.default_rng(10_000 + seed).choice(
                    len(src["photoz"]), len(cal["z"]), replace=False
                )
                rep = dict(
                    mag=nz.z_magnitudes(src["mag"][rows], offset)[0],
                    photoz=src["photoz"][rows],
                    z=paints["colour"][0][rows],
                )
                result = pipeline.estimate(config, rep, src, workers=w)
                for m in todo:
                    write(
                        rung,
                        m,
                        seed,
                        twin.recalibrate(
                            config, result, src, dict(rep, z=paints[m][0][rows])
                        ),
                        paints[m][0],
                    )
                continue
            for m in todo:
                zs, zc = paints[m]
                ok = np.ones(len(zc), bool) if rung == "egs" else zc <= Z_SUCCESS
                if rung in ("egs", "success"):
                    c = {k: v[ok] for k, v in dict(cal, z=zc).items()}
                    write(
                        rung,
                        m,
                        seed,
                        twin.recalibrate(config, base, src, c),
                        zs,
                        n_cal=int(ok.sum()),
                    )
                    continue
                if rung == "zp_patch":
                    if zp_patch is None:
                        dm_s = (
                            twin.zero_point_offsets(
                                src["ra"],
                                src["dec"],
                                tile_of(src["object_id"]),
                                [0] * 5,
                                None,
                                patch=PATCH,
                            )
                            * UPTAKE
                        )
                        dm_c = (
                            twin.zero_point_offsets(
                                cal["ra"],
                                cal["dec"],
                                tile_of(cal["object_id"]),
                                [0] * 5,
                                None,
                                patch=PATCH,
                            )
                            * UPTAKE
                        )
                        s, c = perturbed(dm_s, dm_c)
                        zp_patch = (s, c, pipeline.estimate(config, c, s, workers=w))
                    s, c, result = zp_patch
                    keep = {k: v[ok] for k, v in dict(c, z=zc).items()}
                    write(rung, m, seed, twin.recalibrate(config, result, s, keep), zs)
                    continue
                # zp_map: one map per seed, shared by the models
                if m == todo[0]:
                    rng = np.random.default_rng(20_000 + seed)
                    ra = np.r_[src["ra"], cal["ra"]]
                    dec = np.r_[src["dec"], cal["dec"]]
                    tiles = tile_of(np.r_[src["object_id"], cal["object_id"]])
                    dm = (
                        twin.zero_point_offsets(ra, dec, tiles, SIGMA, rng, patch=PATCH)
                        * UPTAKE
                    )
                    n = len(src["ra"])
                    s, c = perturbed(dm[:n], dm[n:])
                    result_map = pipeline.estimate(config, c, s, workers=w)
                    egs_offset = dm[n:].mean(0).tolist()
                keep = {k: v[ok] for k, v in dict(c, z=zc).items()}
                result = twin.recalibrate(config, result_map, s, keep)
                c0 = {k: v[ok] for k, v in dict(cal, z=zc).items()}
                r0 = twin.recalibrate(config, base, src, c0)
                spread = []
                for (wz1, ww1), (wz0, ww0) in zip(
                    regional(result, zs), regional(r0, zs)
                ):
                    both = (ww1 > 0) & (ww0 > 0)
                    d = wz1[both] / ww1[both] - wz0[both] / ww0[both]
                    spread.append(float(np.sqrt(np.average(d**2, weights=ww0[both]))))
                write(
                    rung, m, seed, result, zs, egs_offset=egs_offset, region_rms=spread
                )


if __name__ == "__main__":
    main()
