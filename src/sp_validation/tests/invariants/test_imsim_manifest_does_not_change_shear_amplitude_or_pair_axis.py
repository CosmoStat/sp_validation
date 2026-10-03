"""The manifest preserves the recorded shear amplitude and injection axes.
Its pairs retain positive/negative branch identity for grid and non-grid runs.
"""

import random
import types
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.fast,
    pytest.mark.decision("shear_bias_simulations.mbias_estimator"),
]

SEED = 7
RUN = 7
BRANCHES = ["1z2z", "1p2z", "1m2z", "1z2p", "1z2m"]
INJECTIONS = {
    "1z2z": (0.0, 0.0),
    "1p2z": (0.02, 0.0),
    "1m2z": (-0.02, 0.0),
    "1z2p": (0.0, 0.02),
    "1z2m": (0.0, -0.02),
}


@pytest.fixture(scope="module")
def manifest_module():
    """Load the workflow script without writing bytecode into its read-only tree."""
    script = Path.cwd() / "workflow" / "scripts" / "im_build_manifest.py"
    module = types.ModuleType("im_build_manifest_under_test")
    module.__file__ = str(script)
    exec(
        compile(script.read_text(encoding="utf-8"), str(script), "exec"),
        module.__dict__,
    )
    return module


def _write_campaign(module, base, sims_type, injections):
    """Write tiny basic_info records at the script's production campaign paths."""
    suffix = f"_{sims_type}_{RUN}" if sims_type == "grid" else f"_{RUN}"
    rng = random.Random(SEED)
    for branch, (g1, g2) in injections.items():
        path = Path(module.basic_info_path(str(base), f"{branch}{suffix}"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"run_marker = {rng.randrange(1 << 32)}\ng_cosmic = {g1} {g2}\n",
            encoding="utf-8",
        )


@pytest.mark.parametrize("sims_type", ["grid", "random"])
def test_manifest_uses_shared_amplitude_and_axis_correct_pairs(
    manifest_module, tmp_path, sims_type
):
    """Preserve the injected amplitude and component pairing.

    Exact fixture literals and dictionary equality give a statistical false-alarm
    probability of 0; this identity check allows no estimator uncertainty.
    """
    _write_campaign(manifest_module, tmp_path, sims_type, INJECTIONS)

    manifest = manifest_module.build_manifest(str(tmp_path), sims_type, RUN, BRANCHES)

    assert manifest == {
        "input_sims_base": str(tmp_path),
        "sims_type": sims_type,
        "num": RUN,
        "shear_amplitude": 0.02,
        "reference": "1z2z",
        "branches": {
            "1z2z": {"g1": 0.0, "g2": 0.0},
            "1p2z": {"g1": 0.02, "g2": 0.0},
            "1m2z": {"g1": -0.02, "g2": 0.0},
            "1z2p": {"g1": 0.0, "g2": 0.02},
            "1z2m": {"g1": 0.0, "g2": -0.02},
        },
        "pairs": [
            {"plus": "1p2z", "minus": "1m2z", "component": 0},
            {"plus": "1z2p", "minus": "1z2m", "component": 1},
        ],
    }


def test_manifest_rejects_mixed_amplitudes_instead_of_redefining_g_in(
    manifest_module, tmp_path
):
    """Reject an inconsistent branch magnitude rather than infer a new denominator.

    The literal mismatch is deterministic, with a statistical false-alarm
    probability of 0 and no numerical tolerance.
    """
    injections = dict(INJECTIONS)
    injections["1p2z"] = (0.03, 0.0)
    _write_campaign(manifest_module, tmp_path, "grid", injections)

    with pytest.raises(
        SystemExit, match=r"injected \|g\| differs across sheared branches"
    ):
        manifest_module.build_manifest(str(tmp_path), "grid", RUN, BRANCHES)


def test_manifest_rejects_signs_that_disagree_with_branch_names(
    manifest_module, tmp_path
):
    """Reject an injection whose sign contradicts its branch code.

    This exact sign check has a statistical false-alarm probability of 0.
    """
    injections = dict(INJECTIONS)
    injections["1m2z"] = (0.02, 0.0)
    _write_campaign(manifest_module, tmp_path, "grid", injections)

    with pytest.raises(SystemExit, match=r"branch '1m2z' names g1 sign -"):
        manifest_module.build_manifest(str(tmp_path), "grid", RUN, BRANCHES)
