"""Twin image simulations of a real campaign, processed by ShapePipe.

Two steps, each a command a Lightcone recipe runs:

``truth``
    The input galaxy catalogue of a simulation (positions, magnitudes, sizes,
    intrinsic shapes), as the generator draws it.
``catalogue``
    One shear branch: the generator draws its exposures and coadds its tiles,
    then ShapePipe's workflow processes them (``sp run --here``) into the
    merged shape catalogue, which is the output. Images and ShapePipe's run
    tree live in a temporary directory and go with it.

The generator is the ``unions`` module of UNIONS-WL/MultiBand_ImSim and the
processing is CosmoStat/shapepipe's workflow, each checked out at the commit
the analysis pins. Everything else a simulation is made of comes from the
selected options (:mod:`sp_validation.image_sims.options`) and the field: a
staged real campaign, ``tiles/CFIS.XXX.YYY.r.fits`` and
``exposures/NNNNNNNp{,.weight,.flag}.fits.fz``.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from sp_validation.image_sims import options

SIM_NUM = 1


def checkout(repository, commit, dest):
    """Check *repository* out at *commit* into *dest*."""
    subprocess.run(
        ["git", "clone", "-q", "--filter=blob:none", "--no-checkout", repository, str(dest)],
        check=True,
    )
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)
    return dest


def generator_config(settings, field, stars, skills, swarp, branches, root):
    """The generator's run config (``unions/config.yaml`` of MultiBand_ImSim)."""
    return {
        "campaign_dir": str(field),
        "stars": str(stars),
        "skills": str(Path(skills) / "*.feather"),
        "mag_column": settings["mag_column"],
        "mag_range": list(settings["mag_range"]),
        "layout": settings["layout"],
        "grid_spacing_arcsec": settings["grid_spacing_arcsec"],
        "shear": settings["shear"],
        "branches": list(branches),
        "seed": settings["seed"],
        "output_root": str(root / "sims"),
        "sim_num": SIM_NUM,
        "psf_dict": str(root / "sims" / "psf_dict.pickle"),
        "psf_stamp": settings["psf_stamp"],
        "swarp": str(swarp),
    }


def shapepipe_config(settings, sim, root, run):
    """ShapePipe's run config for one branch under ``sp run --here``."""
    return {
        "machine": "here",
        "input_type": "image_sims",
        "run": run,
        "ngmix_chunks": settings["ngmix_chunks"],
        "clean": False,
        "clean_tiles": False,
        "tile_list": str(sim / "tiles.txt"),
        "retrieve": "symlink",
        "inputs": {
            "tiles": str(sim / "images" / "SP_tiles"),
            "exposures": str(sim / "images" / "SP_exp"),
        },
        "psf_dict": str(root / "sims" / "psf_dict.pickle"),
        "outputs": {
            "run_dir": str(root / "run"),
            "products_dir": str(root / "products"),
            "index_db": str(root / "products" / "index" / "run_index.sqlite"),
        },
    }


def write_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return path


def truth(settings, field, stars, skills, swarp, output, root):
    """Write the simulation's input galaxy catalogue to *output*."""
    gen = checkout(settings["generator_repository"], settings["generator_commit"],
                   root / "generator")
    sys.path[:0] = [str(gen)]
    from unions import twin

    cfg = generator_config(settings, field, stars, skills, swarp, [], root)
    twin.make_truth(twin.Run(cfg)).write(output, format="fits")


def catalogue(settings, branch, field, stars, skills, swarp, cores, memory_mb,
              output, root):
    """Simulate one branch and write ShapePipe's merged catalogue to *output*."""
    gen = checkout(settings["generator_repository"], settings["generator_commit"],
                   root / "generator")
    sp = checkout(settings["shapepipe_repository"], settings["shapepipe_commit"],
                  root / "shapepipe")
    env = dict(os.environ, PYTHONPATH=str(gen),
               PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}")

    cfg = generator_config(settings, field, stars, skills, swarp, [branch], root)
    subprocess.run(
        [sys.executable, "-m", "unions.twin",
         write_yaml(cfg, root / "generator.yaml"), "-j", str(cores)],
        cwd=gen, env=env, check=True,
    )

    sim = root / "sims" / f"{branch}_{settings['layout']}_{SIM_NUM}"
    run = write_yaml(shapepipe_config(settings, sim, root, branch), root / "shapepipe.yaml")
    env.pop("PYTHONPATH")
    subprocess.run(
        [str(sp / "workflow" / "bin" / "sp"), "run", "--here", "-c", str(run),
         "--cores", str(cores), "--resources", f"mem_mb={memory_mb}"],
        env=env, check=True,
    )
    shutil.copyfile(root / "products" / f"final_cat_{branch}.hdf5", output)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["truth", "catalogue"])
    p.add_argument("--options", required=True, help="options table (YAML)")
    p.add_argument("--select", nargs="+", required=True, metavar="DECISION=OPTION")
    p.add_argument("--field", required=True, help="staged real campaign")
    p.add_argument("--stars", required=True, help="v2 star catalogue (per-CCD PSF)")
    p.add_argument("--skills", required=True, help="SKiLLS input galaxies (*.feather)")
    p.add_argument("--swarp", required=True, help="SWarp executable")
    p.add_argument("--branch", help="shear branch, 1{z,p,m}2{z,p,m} (catalogue)")
    p.add_argument("--cores", type=int, default=1)
    p.add_argument("--memory-mb", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    settings = options.settings(args.options, options.parse(args.select))
    root = Path(tempfile.mkdtemp(prefix="image-sim-"))
    inputs = dict(field=args.field, stars=args.stars, skills=args.skills, swarp=args.swarp)
    if args.step == "truth":
        truth(settings, output=args.output, root=root, **inputs)
    else:
        catalogue(settings, args.branch, cores=args.cores, memory_mb=args.memory_mb,
                  output=args.output, root=root, **inputs)


if __name__ == "__main__":
    main()
