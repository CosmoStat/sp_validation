"""Resolve the non-fiducial catalog-version list for the Paper II version sweep.

The Design-B version sweep recomputes every non-fiducial catalog variant through
lc so the version-comparison figures (paper Figure 5) are reproduced end-to-end
rather than read from the canonical CosmoStat tree. The variant set is the config
``versions:`` list minus the |e|<0.7 ``_ecut`` cross-checks (out of scope for
every comparison filter) and minus the fiducial version (``config['fiducial']
['version']``), which is produced by the single-output recipes and handed to the
comparison plotters through their ``--fiducial-*`` overrides.

The remaining 7 are the union per-version product set: the 3 non-fiducial
leak-corrected catalogs feeding the Tier-1 overlay figures, plus the 4
uncorrected catalogs feeding the Tier-2 appendix PTE matrices. Both the sweep
drivers (as an import) and the bash drivers (as a ``--config`` CLI printing one
version per line) resolve the list here, so there is a single source of truth.
"""

import argparse

import yaml


def nonfiducial_versions(config):
    """Non-fiducial, non-ecut catalog versions from a loaded bmodes config."""
    fiducial = config["fiducial"]["version"]
    return [v for v in config["versions"] if "_ecut" not in v and v != fiducial]


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Print the non-fiducial sweep version list, one per line."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    a = ap.parse_args(argv)
    with open(a.config) as f:
        config = yaml.safe_load(f)
    for v in nonfiducial_versions(config):
        print(v)


if __name__ == "__main__":
    _from_cli()
