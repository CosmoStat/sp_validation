"""Build an isolated four-entry registry for known-truth painted mocks."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path("/n17data/cdaley/scratch/glass-mock-cv/cvpaint")
REPO = Path("/n17data/cdaley/scratch/glass-mock-cv/run-int")
MOCK = ROOT.parent / "mocks/v1.4.6.3/seed_00001"
source = yaml.safe_load((REPO / "cosmo_val/cat_config.yaml").read_text())
nz = json.loads((MOCK / "maps/maps.json").read_text())["config"]["nz_path"]
assert nz.endswith("_B.txt")
copy = ROOT / "catalogues/painted_blinded.h5"
if not copy.exists() or copy.stat().st_size != (MOCK / "painted.h5").stat().st_size:
    shutil.copyfile(MOCK / "painted.h5", copy)
assert copy.resolve() != (MOCK / "painted.h5").resolve()
cc = {}
for i, (name, path) in enumerate(
    [
        ("GLASS1_clean", MOCK / "painted.h5"),
        ("GLASS1_blinded", copy),
        ("GLASS1_psfleak", MOCK / "painted_psfleak.h5"),
        ("GLASS1_addB", MOCK / "painted_addB.h5"),
    ]
):
    entry = deepcopy(source["SP_v1.4.6.3"])
    entry.update(
        blind="glassmock-test" if name.endswith("blinded") else "none",
        subdir=str(MOCK),
        label=name,
        colour=f"C{i}",
        marker="o",
    )
    entry["shear"].update(
        path=str(path),
        ra_col="RA",
        dec_col="Dec",
        redshift_path=nz,
        R=1.0,
        w_col="w_des",
    )
    entry["patch_number"] = 100
    cc[name] = entry
cc["nz"] = dict(subdir=str(MOCK), dndz=dict(path=nz))
cc["paths"] = dict(
    output=str(ROOT / "output"), blinds=str(ROOT.parent / "cvmock/blinds")
)
(ROOT / "cat_config.yaml").write_text(yaml.safe_dump(cc, sort_keys=False))
config = yaml.safe_load((REPO / "papers/cosmo_val/config/config.yaml").read_text())
config["versions"] = list(cc)[:4]
config["catalog_config"] = str(ROOT / "cat_config.yaml")
config["container"] = str(Path.home() / ".cache/sp_validation/sp_validation.sif")
config["fiducial"].update(version="GLASS1_clean", mock_version="GLASS1_clean")
config["tools"]["python_executable"] = str(ROOT / "onecov-python")
(ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print("Prepared four mock entries; confirmed maps use B n(z).")
