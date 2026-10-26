"""Write an m-bias config for one grid and response weighting, from the
workflow's own config with only the paths changed."""

import sys

import yaml

grid, scheme, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
base = "/n17data/mkilbing/astro/Runs/shapepipe/CFIS/v2.0/image_sims/new_workflow"
with open(f"{base}/grid_{grid}/results/m_bias_config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["grids_dir"] = f"{out_root}/{scheme}"
cfg["results_dir"] = f"{out_root}/{scheme}/results"
cfg["output_path"] = f"{out_root}/{scheme}/results/m_bias_results.yaml"
cfg["provenance"]["response_weighting"] = f"{scheme} (wdes_response.py)"
with open(f"{out_root}/{scheme}/m_bias_config.yaml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
