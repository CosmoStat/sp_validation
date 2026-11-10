"""Write a sampled-mock run root: registry, run config and GLASS config, mirroring a blinded v2 tomographic data run.

Run in the container from this checkout's root:

    python prepare.py --root <run root> --data-run <data run dir> --nz <n(z) file> --catalogue <dir with survey_tomo.json and the tomographic footprint>

Each MOCK_{seed:05d} entry is the data's SP_v2.0 entry with only the catalogue
swapped for a sampled GLASS mock, so masks, n(z), bin/weight columns and response
are identical, and every mock reads the data run's own analytic tomographic
covariances through `reference_roots` (<data run>/cosmo_val, cosmo_inference).
The GLASS config takes each bin's n_eff and per-component σ_e from survey_tomo.json,
so the mock shape-noise power σ_e²/n_eff equals the analytic covariance's.
"""
import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--data-run", type=Path, required=True)
p.add_argument("--nz", type=Path, required=True)
p.add_argument("--catalogue", type=Path, required=True)
p.add_argument("--seeds", type=int, default=100)
a = p.parse_args()

UNIONS = Path("/project/rrg-kyi/cdaley/unions").resolve()
ROOT, DATA, nz, CAT = (x.resolve() for x in (a.root, a.data_run, a.nz, a.catalogue))
# Whole-sample footprint, stars and PSF catalogue: identical for every v2 tomographic run.
INPUTS = UNIONS / "analyses/mock-ensemble-tomo/inputs"
REPO = Path(__file__).resolve().parents[3]

source = yaml.safe_load((DATA / "cat_config.yaml").read_text())["SP_v2.0"]
config = yaml.safe_load((DATA / "nibi_tomo.yaml").read_text())

# Relocate the data entry's Nibi paths to Fir.
source["subdir"] = str(INPUTS / "catalogue")
source["mask"] = str(INPUTS / "catalogue/footprint-nside4096-ring.fits")
source["cov_th"]["tomo"]["mask"] = str(CAT / "footprint-tomo-nside4096-ring.fits")
source["shear"]["redshift_path"] = str(nz)
source["psf"]["mask_config"] = str(REPO / "config/calibration/mask_v2.0.yaml")

survey = json.loads((CAT / "survey_tomo.json").read_text())
bins = [survey["samples"][f"bin{b}"] for b in (1, 2, 3, 4)]
glass = dict(
    nside=4096, dx=120.0, zmax=3.0, nbins=4,
    n_arcmin2=[b["n_eff"] for b in bins],
    sigma_e=[b["sigma_e_per_component"] for b in bins],
    bias=1.2, phz_sigma_0=0.03, limber=True,
    mask_path=str(CAT / "footprint-tomo-nside4096-ring.fits"),
    nz_path=str(nz), output_path=str(ROOT / "gen"), output_prefix="unions_v2_tomo",
)
for b, (n, s) in enumerate(zip(glass["n_arcmin2"], glass["sigma_e"])):
    assert abs(n - source["cov_th"]["tomo"]["n_e"][b]) < 1e-9 and abs(s - source["cov_th"]["tomo"]["sigma_e"][b]) < 1e-9, b
ROOT.mkdir(parents=True, exist_ok=True)
(ROOT / "glass_v2_tomo.yaml").write_text(yaml.safe_dump(glass, sort_keys=False))

entries = {
    "SP_v2.0": source,
    "nz": {"subdir": str(nz.parent), "dndz": {"path": str(nz)}},
    "paths": {"blinds": str(UNIONS / "analyses/mock-ensemble-tomo/blinds"),
              "output": str(ROOT / "cosmo_val")},
}
versions = [f"MOCK_{s:05d}" for s in range(1, a.seeds + 1)]
for version in versions:
    entry = deepcopy(source)
    entry["subdir"] = str(ROOT / "catalogues")
    entry["shear"]["path"] = f"{version}.fits"
    entry["blind"] = "mock-ensemble-tomo"
    for kind in ("psf", "star"):
        entry[kind]["path"] = str(INPUTS / "catalogue" / source[kind]["path"])
    entries[version] = entry

config["catalog_config"] = str(ROOT / "cat_config.yaml")
config["container"] = str(INPUTS / "containers/sp_validation-ddd58fe0.sif")
config["versions"] = versions
config["cosmo_val"]["include_rho_tau"] = False
# process_gg publishes unpatched means; skip the discarded jackknife pass.
config["cosmo_val"]["npatch"] = 1
config["fiducial"]["npatch"] = 1
config["fiducial"]["version"] = config["fiducial"]["mock_version"] = versions[0]
config["cosmo_val"]["path_onecovariance"] = str(ROOT / "unused-onecovariance")
config["covariance"]["tomographic_reference"] = {v: "SP_v2.0" for v in versions}
config["covariance"]["reference_roots"] = {
    "SP_v2.0": {"cosmo_val": str(DATA / "cosmo_val"),
                "cosmo_inference": str(DATA / "cosmo_inference")}}
config["sampled"] = {
    "root": str(ROOT),
    "generator_checkout": str(UNIONS / "code/wt-mockens-sampled"),
    "glass_config": str(ROOT / "glass_v2_tomo.yaml"),
    "seed_offset": 1000,
}

planck = REPO / "papers/cosmo_val/results/cosmology/planck18.json"
(ROOT / "results/cosmology").mkdir(parents=True, exist_ok=True)
(ROOT / "results/cosmology/planck18.json").write_bytes(planck.read_bytes())
(ROOT / "cat_config.yaml").write_text(yaml.safe_dump(entries, sort_keys=False))
(ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
for name in ("tmp", "cache", "logs", "cosmo_val", "cosmo_inference", "catalogues", "benchmarks", "gen/results"):
    (ROOT / name).mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rev(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


(ROOT / "provenance.json").write_text(json.dumps(dict(
    cosmo_val_checkout=str(REPO), cosmo_val_sha=rev(REPO),
    generator_checkout=config["sampled"]["generator_checkout"],
    generator_sha=rev(config["sampled"]["generator_checkout"]),
    data_run=str(DATA), data_config_sha256=sha(DATA / "cat_config.yaml"),
    nz_sha256=sha(nz), glass_config_sha256=sha(ROOT / "glass_v2_tomo.yaml"),
    glass_seed="1000 + mock number", mock_blind="mock-ensemble-tomo",
), indent=2))
