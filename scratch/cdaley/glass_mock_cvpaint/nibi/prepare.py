"""Four-entry registry and run config for the painted seed-1 mocks on the v2 footprint (Nibi).

Mock shears from the painter; PSF/star catalogues, mask and cov_th from the real-data
SP_v2.0 entry of run-int1028, with relative paths made absolute. The blinded entry is a
byte copy under the mock-only blind `glassmock-test` in the run's own registry.
"""

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(os.environ["CVPAINT_ROOT"])
MOCK = Path(os.environ["CVPAINT_MOCK"])
BLINDS = Path(os.environ["CVPAINT_BLINDS"])
REPO = Path(os.environ["REPO"])
SOURCE = Path(os.environ["SOURCE_CAT_CONFIG"])

nz = json.loads((MOCK / "maps/maps.json").read_text())["config"]["nz_path"]
assert nz.endswith("_B.txt")
nz = str(ROOT.parent / "nz" / Path(nz).name)  # the same file, sha256-checked on copy
assert Path(nz).is_file()
assert (BLINDS / "glassmock-test.blind.json").is_file()

src = yaml.safe_load(SOURCE.read_text())["SP_v2.0"]
base = Path(src["subdir"])
for block in ("psf", "star"):
    src[block]["path"] = str(base / src[block]["path"])
    assert Path(src[block]["path"]).exists(), src[block]["path"]
assert Path(src["mask"]).is_file()

copy = ROOT / "catalogues/painted_blinded.h5"
copy.parent.mkdir(parents=True, exist_ok=True)
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
    entry = deepcopy(src)
    entry.update(
        blind="glassmock-test" if name.endswith("blinded") else "none",
        subdir=str(MOCK),
        label=name,
        colour=f"C{i}",
        marker="o",
        ls="dotted",
        patch_number=100,
    )
    entry["shear"].update(path=str(path), redshift_path=nz, R=1.0, w_col="w")
    cc[name] = entry
cc["nz"] = dict(subdir=str(MOCK), dndz=dict(path=nz))
cc["paths"] = dict(output=str(ROOT / "output"), blinds=str(BLINDS))
(ROOT / "cat_config.yaml").write_text(yaml.safe_dump(cc, sort_keys=False))

# The candide run's config, with Nibi's tool paths.
config = yaml.safe_load((REPO / "scratch/cdaley/glass_mock_cvpaint/config.yaml").read_text())
onecov = "/scratch/cdaley/cosmo-val-v2/run-int1028/software/OneCovariance"
config["catalog_config"] = str(ROOT / "cat_config.yaml")
config["container"] = os.environ["SIF"]
config["cosmo_val"]["path_onecovariance"] = onecov
config["tools"] = dict(onecov_executable=f"{onecov}/covariance.py", python_executable="python")
(ROOT / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
print("Prepared four v2 mock entries; maps use the v1.4.6.3 B n(z).")
