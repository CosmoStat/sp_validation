"""The blinding theory's nonlinear recipe is the inference pipeline's.

The blinding shift is a difference of CCL theory vectors; inference runs CAMB
through CosmoSIS. The shift means what it is meant to only if both use one
recipe, so the blinding reads it from the CosmoSIS config it must match.
"""

import pathlib
import re

from sp_validation import blinding_theory as bt

INI = (
    pathlib.Path(__file__).resolve().parents[3]
    / "cosmo_inference/cosmosis_config/templates/cosmosis_pipeline_A_ia_cell.ini"
)


def test_halofit_recipe_matches_inference_config():
    match = re.search(r"^halofit_version\s*=\s*(\S+)", INI.read_text(), re.MULTILINE)
    assert match, f"no halofit_version in {INI}"
    cfg = bt.TheoryConfig()
    assert cfg.halofit_version == match.group(1)
    assert cfg.transfer_function == "boltzmann_camb"
