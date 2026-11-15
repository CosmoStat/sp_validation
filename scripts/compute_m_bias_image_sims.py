#!/usr/bin/env python
"""Multiplicative and additive shear bias from image simulations.

Usage:
    compute_m_bias_image_sims.py -c config.yaml [-v]

Each simulation's calibration step (``calibrate_comprehensive_cat.py``) leaves
its per-tile response sums beside its cut catalogue
(``<catalogue stem>_sums.npz``). For every shear component the ``+g`` and
``-g`` simulations give ``m`` and ``c`` (``sp_validation.response.shear_bias``),
and deleting each tile from all simulations at once gives their covariance
(``shear_bias_jackknife``).

Config keys:
    grids_dir, num, sims_type   where the simulations live:
                                ``{grids_dir}/{branch}_grid_{num}`` (grid) or
                                ``{grids_dir}/{branch}_{num}``
    sums_name                   the sums file in each simulation directory
    shear_amplitude             injected |g|
    pairs                       list of {plus, minus, component}
    weight_table                path of a DES weight-table product, ``self``
                                (each simulation's own table, refitted per
                                deletion) or ``unit`` (unit weights, or IV
                                weights for IV sums)
    des_min_count               pooling threshold of ``self`` tables
    output_path                 results YAML
    provenance                  copied into the results
"""

import argparse
import sys

import numpy as np
import yaml

from sp_validation import response


def table_rule(config, edges):
    """The table every simulation is weighted with."""
    choice = config["weight_table"]
    if choice == "self":
        return response.refit(config["des_min_count"])
    if choice == "unit":
        return response.unit_table(edges)
    return response.read_table(choice)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="config YAML file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    num = config["num"]
    grid = config.get("sims_type", "grid") == "grid"
    suffix = f"_grid_{num}" if grid else f"_{num}"

    def sums(branch):
        path = f"{config['grids_dir']}/{branch}{suffix}/{config['sums_name']}"
        if args.verbose:
            print(f"Loading {path}")
        return response.Sums.load(path)

    pairs = {
        p["component"]: (sums(p["plus"]), sums(p["minus"])) for p in config["pairs"]
    }
    g_in = config["shear_amplitude"]
    table = table_rule(config, next(iter(pairs.values()))[0].edges)
    values = response.shear_bias(pairs, g_in, table)
    labels, _, covariance = response.shear_bias_jackknife(pairs, g_in, table)
    order = list(values)

    results = {key: float(values[key]) for key in order}
    for key, error in zip(order, np.sqrt(np.diag(covariance)), strict=True):
        results[f"{key}_err"] = float(error)
    results.update(
        mc_covariance=covariance.tolist(),
        covariance_order=order,
        jackknife_tiles=[str(t) for t in labels],
        weight_table=config["weight_table"],
        shear_amplitude=g_in,
    )
    if "provenance" in config:
        results["provenance"] = config["provenance"]
    with open(config["output_path"], "w") as f:
        yaml.safe_dump(results, f, default_flow_style=False)

    print(f"g_in = ±{g_in}, {len(labels)} tiles, table: {config['weight_table']}")
    for key in order:
        print(f"  {key} = {results[key]:+.6f} ± {results[key + '_err']:.6f}")
    print(f"Results written to {config['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
