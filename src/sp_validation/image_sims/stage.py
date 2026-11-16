"""Stage a real campaign from CADC as the field of a twin simulation.

A field directory holds what the generator reads of the real data:

    tiles.txt                         the tiles, ``XXX.YYY``
    tiles/CFIS.XXX.YYY.r.fits         tile images (header: WCS, size, and the
                                      HISTORY cards naming the exposures)
    tiles/CFIS.XXX.YYY.r.weight.fits.fz
    exposures.txt                     the exposures the tiles coadd
    exposures/NNNNNNNp{,.weight,.flag}.fits.fz
    READY.json                        tile -> exposures, from the HISTORY cards

Run it on a login node, where one CADC stream reaches ~55 MB/s; transfers from
compute nodes collapse under load. ``vcp`` needs a CADC proxy certificate.

    python -m sp_validation.image_sims.stage nearest --seed 253.246 --n 150 \\
        --manifest .../v2.x/run1-snap3/MANIFEST.json FIELD
    python -m sp_validation.image_sims.stage fetch FIELD

``nearest`` writes ``tiles.txt``: the *n* v2 tiles nearest the seed tile.
``fetch`` stages the tiles, reads their exposures from the HISTORY cards,
then stages the exposures; files already present are kept.
"""

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits

TILES = "vos:cfis/tiles_DR6"
EXPOSURES = {"": "vos:cfis/pitcairn", ".weight": "vos:cfis/weights", ".flag": "vos:cfis/flags"}


def tile_centre(tile):
    """(RA, Dec) in degrees of CFIS tile ``XXX.YYY``."""
    x, y = (int(v) for v in tile.split("."))
    dec = y / 2 - 90
    return x / 2 / np.cos(np.radians(dec)), dec


def nearest(seed, n, tiles):
    """The *n* tiles of *tiles* nearest *seed* on the sky, seed included."""
    tiles = sorted(tiles)
    radec = np.radians([tile_centre(t) for t in tiles])
    xyz = np.stack([np.cos(radec[:, 1]) * np.cos(radec[:, 0]),
                    np.cos(radec[:, 1]) * np.sin(radec[:, 0]),
                    np.sin(radec[:, 1])], 1)
    order = np.argsort(-(xyz @ xyz[tiles.index(seed)]), kind="stable")
    return sorted(tiles[i] for i in order[:n])


def copy(source, dest, certfile):
    """``vcp`` *source* to *dest* unless it is already there."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size:
        return
    part = dest.with_name(dest.name + ".part")
    subprocess.run(["vcp", f"--certfile={certfile}", source, str(part)], check=True)
    part.rename(dest)
    print(dest, flush=True)


def exposures_of(tile_image):
    """The exposures a tile coadds, from its HISTORY cards."""
    history = fits.getheader(tile_image)["HISTORY"]
    return sorted({re.search(r"(\d{7})p\.fits", str(h)).group(1) for h in history})


def fetch(field, certfile, streams):
    field = Path(field)
    tiles = field.joinpath("tiles.txt").read_text().split()
    (field / "tiles").mkdir(exist_ok=True)
    (field / "exposures").mkdir(exist_ok=True)

    def run(jobs):
        with ThreadPoolExecutor(streams) as pool:
            list(pool.map(lambda j: copy(*j, certfile), jobs))

    run([(f"{TILES}/CFIS.{t}.r.fits", field / "tiles" / f"CFIS.{t}.r.fits") for t in tiles])
    ready = {t: exposures_of(field / "tiles" / f"CFIS.{t}.r.fits") for t in tiles}
    exps = sorted({e for es in ready.values() for e in es})
    field.joinpath("exposures.txt").write_text("".join(f"{e}\n" for e in exps))
    field.joinpath("READY.json").write_text(json.dumps({"tiles": ready}, indent=1))
    run([(f"{root}/{e}p{kind}.fits.fz", field / "exposures" / f"{e}p{kind}.fits.fz")
         for e in exps for kind, root in EXPOSURES.items()])
    run([(f"{TILES}/CFIS.{t}.r.weight.fits.fz", field / "tiles" / f"CFIS.{t}.r.weight.fits.fz")
         for t in tiles])
    print(f"{len(tiles)} tiles, {len(exps)} exposures staged in {field}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    near = sub.add_parser("nearest", help="choose the field's tiles")
    near.add_argument("field")
    near.add_argument("--seed", required=True, help="seed tile, XXX.YYY")
    near.add_argument("--n", type=int, required=True)
    near.add_argument("--manifest", required=True,
                      help="a v2 run manifest (JSON with a `tiles` list)")
    get = sub.add_parser("fetch", help="stage the field's tiles and exposures")
    get.add_argument("field")
    get.add_argument("--certfile", default=str(Path.home() / ".ssl" / "cadcproxy.pem"))
    get.add_argument("--streams", type=int, default=4)
    args = p.parse_args(argv)

    if args.step == "nearest":
        with open(args.manifest) as f:
            tiles = json.load(f)["tiles"]
        Path(args.field).mkdir(parents=True, exist_ok=True)
        chosen = nearest(args.seed, args.n, tiles)
        Path(args.field, "tiles.txt").write_text("".join(f"{t}\n" for t in chosen))
        print(f"{len(chosen)} tiles around {args.seed} in {args.field}/tiles.txt")
    else:
        fetch(args.field, args.certfile, args.streams)


if __name__ == "__main__":
    main()
