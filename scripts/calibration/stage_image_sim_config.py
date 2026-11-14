"""Stage a simulation mask config with the independent DES-table policy."""

import argparse
from pathlib import Path

import yaml

from sp_validation.image_sim_config import calibration_config

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--w-cols", nargs="+", required=True)
p.add_argument("--source", default="external")
p.add_argument("--product-name", default="")
p.add_argument("--product-path", default="")
a = p.parse_args()
config = calibration_config(
    yaml.safe_load(a.input.read_text()),
    a.w_cols,
    a.source,
    dict(name=a.product_name, path=a.product_path)
    if a.product_name not in ("", "-") and a.product_path not in ("", "-")
    else None,
)
# An earlier workflow may have staged a symlink to a shared mask config.
# Replace that symlink rather than overwriting its target.
a.output.unlink(missing_ok=True)
a.output.write_text(yaml.safe_dump(config, sort_keys=False))
