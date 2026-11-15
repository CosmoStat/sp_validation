"""Run the SOM n(z) calibration on twins of the v2 catalogue (``nz.twin``).

The twin keeps the recipe's sources and EGS calibrators as they are and
paints their redshifts with a model (:data:`MODELS`), a feature space of
``nz.twin`` and a library of spectra:

- ``colour``: a DEEP2/3 neighbour's redshift in the SOM's colour space;
- ``bpz``: BPZ plus a DEEP2/3 neighbour's spec-z − BPZ in (BPZ, r);
- ``bpz_desi``: the same, with the neighbours from every galaxy redshift
  survey matched to a lensing source, which at r > 22 is almost all DESI.
  DESI targets galaxies by colour for z ≈ 0.6–1.6, so at faint r and low BPZ
  this library carries 8–20% of redshifts above 1.4: a stress model for
  catastrophic outliers the magnitude-limited DEEP2/3 cannot show. It was
  added after its first two realisations, which ran as ``bpz``.

Rungs add one ingredient each; realisation s paints with seed s on every
rung, so rungs pair by seed. Δ⟨z⟩ is the calibrated minus the true mean of a
bin.

Rungs, with what was predicted before the first run, for ``colour`` and
the model now called ``bpz_desi``:

- ``representative``: calibrators are as many sources drawn at random, any
  BPZ. Prediction: |mean Δ| ≤ 0.003 in every bin under both models (gold
  selection and the shear weight varying inside a group, as on the toy
  ladder), scatter 0.001–0.003.
- ``egs``: the real EGS calibrators, with their selection, depth and field.
  Prediction: |Δ| ≤ 0.01 under both models, largest in bins 3–4, where EGS's
  deeper g and z put calibrators in different cells from sources of the same
  redshift.
- ``gold_independent``: ``egs`` (``success`` for ``bpz_desi``) with each
  bin's gold groups decided by an independent painting of the calibrators
  (seed + 500), not by the calibrators the bins are weighted with. Not a
  rung of the data's ladder but a diagnosis of ``egs``: written after
  ``egs`` came out low in bin 1. Prediction: bin 1 rises by 0.005–0.02.
- ``success``: ``egs`` without calibrators painted above z = 1.4, where DEEP2
  measures almost nothing; only for ``bpz_desi``, whose library reaches
  beyond DEEP2 (the DEEP2/3 library already has DEEP2's reach: 1.9% of it,
  and 1.8% of the real calibrators, lie above 1.4). Prediction (for
  ``bpz_desi``): bin 4 low by 0.01–0.03.
- ``zp_patch``: ``egs`` (``success`` for ``bpz_desi``) with the i-band patch
  of Romain Paviot's note (+20 mmag over RA 140–230, Dec 30–60) on sources
  and calibrators, taken up by galaxies at ``UPTAKE``; BPZ moves along its
  gradient in each SOM cell, and the SOM is retrained. Prediction: Δ moves
  by ≤ 0.005 per bin.
- ``zp_map``: the same with a new zero-point map per realisation (a coherent
  field and per-tile scatter at the note's amplitudes, plus the patch).
  Prediction: Δ moves by 0.002–0.005 rms over maps; a bin's true mean varies
  across the sky by 0.005–0.01 rms between 54 deg² regions.
- ``zp_ensemble``: the same with the offset the recipe's ensemble draws for
  the EGS against the wide field (``ensemble.zero_points``: correlated colour
  offsets and a grey term, on the calibrators only, no uptake). Its rms over
  realisations is what the ensemble's zero-point term should be. Prediction,
  written after a reviewer's linear fit to ``zp_map``: 1.0–1.4 times the
  ensemble's term (0.009, 0.0045, 0.009, 0.011), the larger factor in bins
  2–3.

Every zero-point record carries the calibrators' mean offset and, per bin,
the rms over sky regions (nside 8) of the change in a region's true mean,
about the bin's change over the whole sky.

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
from sp_validation.nz import ensemble, inputs, pipeline, twin
from sp_validation.object_id import NUMBER_LIMIT

RECIPE = Path(__file__).parents[2] / "config/calibration/nz/v2.yaml"
K = 16
SKY = {"ra": [0.0, 360.0], "dec": [-90.0, 90.0]}
DEEP2 = ["deep2_final", "deep3_final"]
#: Galaxy redshift surveys of the compilation: no QSO or AGN catalogue, no
#: literature aggregator that may carry photometric redshifts.
GALAXY_SURVEYS = [
    "DESI_dr1_iron", "rcsed_final", "SDSS_SPECOBJ_None", "saga_dr3", "hecate",
    "HETDEX_PSC_PDR1_V1.5", "SPITZER_DATAFUSION", "herschel_help", "LAMOST_LRS_dr12",
    "uzc_final", "gogreen_dr2", "candels_v2", "mosdef_final", "c3r2_dr2", "c3r2_dr3",
    "MMT_NEPW", "shels_f2_final", "JADES_DR4", "3dhst", "THRILS",
    *DEEP2,
]  # fmt: skip
#: Libraries: every DEEP2/3 spectrum (both fields), and the galaxy surveys'
#: spectra of objects that are themselves lensing sources.
LIBRARIES = {"deep2": DEEP2, "lensing": GALAXY_SURVEYS}
Z_SUCCESS = 1.4
#: Redshift models: (feature space of ``nz.twin``, library, the redshift
#: above which DEEP2 measures nothing, for a library that reaches beyond it).
MODELS = {
    "colour": ("colour", "deep2", None),
    "bpz": ("bpz", "deep2", None),
    "bpz_desi": ("bpz", "lensing", Z_SUCCESS),
}
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
    for name, surveys in LIBRARIES.items():
        lib = inputs.calibrators(
            rc["photometry"], surveys, SKY, rc["z_min"], offset, workers=workers
        )
        if name == "lensing":
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
        default=[
            "representative",
            "egs",
            "gold_independent",
            "success",
            "zp_patch",
            "zp_map",
            "zp_ensemble",
        ],
    )
    p.add_argument("--workers", type=int, default=1)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=True)
    config = load_yaml(RECIPE)
    offset, w = config["z2_offset"], a.workers
    zmax = config["grid"]["max"]
    data = load_inputs(config, a.output / "inputs.npz", w)
    src, cal = part(data, "src_"), part(data, "cal_")
    libs = {name: part(data, f"lib_{name}_") for name in LIBRARIES}

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
    for m, (kind, lib, _) in MODELS.items():
        path = a.output / f"near_{m}.npz"
        if not path.exists():
            np.savez(
                path,
                src=twin.neighbours(
                    kind,
                    libs[lib],
                    src["mag"],
                    src["photoz"],
                    src["object_id"],
                    k=K,
                    offset=offset,
                    workers=w,
                ),
                cal=twin.neighbours(
                    kind,
                    libs[lib],
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
        kind, lib, _ = MODELS[m]
        rng = np.random.default_rng(seed)
        zs = twin.paint(kind, libs[lib], near[m]["src"], src["photoz"], rng, zmax)
        zc = twin.paint(kind, libs[lib], near[m]["cal"], cal["photoz"], rng, zmax)
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

    zp = config["ensemble"]["zero_points"]
    colour_cov = np.outer(zp["colour_sigma"], zp["colour_sigma"]) * np.asarray(
        zp["colour_corr"]
    )
    n = len(src["ra"])
    ra, dec = np.r_[src["ra"], cal["ra"]], np.r_[src["dec"], cal["dec"]]
    tiles = tile_of(np.r_[src["object_id"], cal["object_id"]])

    def offsets(rung, seed):
        """Zero-point offsets (sources, calibrators) of a zero-point rung."""
        if rung == "zp_patch":
            dm = twin.zero_point_offsets(ra, dec, tiles, [0] * 5, None, patch=PATCH)
            return dm[:n] * UPTAKE, dm[n:] * UPTAKE
        rng = np.random.default_rng(20_000 + seed)
        if rung == "zp_map":
            dm = twin.zero_point_offsets(ra, dec, tiles, SIGMA, rng, patch=PATCH)
            return dm[:n] * UPTAKE, dm[n:] * UPTAKE
        colours = rng.multivariate_normal(np.zeros(4), colour_cov)
        band = ensemble.band_offsets(colours, rng.normal(0, zp["grey"]))
        return np.zeros((n, 5)), np.tile(band, (len(cal["z"]), 1))

    def shift_rms(result, r0, z):
        """Per bin, the weighted rms over sky regions of the change in the
        region's true mean, about the bin's change over the whole sky."""
        out = []
        for (wz1, ww1), (wz0, ww0) in zip(regional(result, z), regional(r0, z)):
            both = (ww1 > 0) & (ww0 > 0)
            d = wz1[both] / ww1[both] - wz0[both] / ww0[both]
            d -= np.average(d, weights=ww0[both])
            out.append(float(np.sqrt(np.average(d**2, weights=ww0[both]))))
        return out

    perturbed_runs = {}
    for seed in range(a.realisations):
        paints = {m: painted(m, seed) for m in MODELS}
        for rung in a.rungs:
            todo = [m for m in MODELS if (rung, m, seed) not in done]
            if rung == "success":
                todo = [m for m in todo if MODELS[m][2]]
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
                    rep_m = dict(rep, z=paints[m][0][rows])
                    r = twin.recalibrate(config, result, src, rep_m)
                    write(rung, m, seed, r, paints[m][0])
                continue
            if rung.startswith("zp_"):
                key = (rung, 0 if rung == "zp_patch" else seed)
                if key not in perturbed_runs:
                    for k in [k for k in perturbed_runs if k[0] != "zp_patch"]:
                        del perturbed_runs[k]
                    dm_s, dm_c = offsets(rung, seed)
                    s, c = perturbed(dm_s, dm_c)
                    perturbed_runs[key] = (
                        s,
                        c,
                        pipeline.estimate(config, c, s, workers=w),
                        dm_c.mean(0).tolist(),
                    )
                s, c, result_zp, cal_offset = perturbed_runs[key]
            for m in todo:
                zs, zc = paints[m]
                reach = MODELS[m][2]
                ok = zc <= reach if reach and rung != "egs" else np.ones(len(zc), bool)
                c0 = {k: v[ok] for k, v in dict(cal, z=zc).items()}
                r0 = twin.recalibrate(config, base, src, c0)
                if rung == "gold_independent":
                    zc2 = painted(m, 500 + seed)[1]
                    c2 = {k: v[ok] for k, v in dict(cal, z=zc2).items()}
                    gold = twin.recalibrate(config, base, src, c2)["est"]["gold"]
                    r0 = twin.recalibrate(config, base, src, c0, gold=gold)
                if not rung.startswith("zp_"):
                    write(rung, m, seed, r0, zs, n_cal=int(ok.sum()))
                    continue
                keep = {k: v[ok] for k, v in dict(c, z=zc).items()}
                r = twin.recalibrate(config, result_zp, s, keep)
                write(
                    rung,
                    m,
                    seed,
                    r,
                    zs,
                    cal_offset=cal_offset,
                    region_rms=shift_rms(r, r0, zs),
                )


if __name__ == "__main__":
    main()
