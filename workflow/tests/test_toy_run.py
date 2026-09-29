"""Rule xi, run for real through apptainer, before and after a blind is drawn.

The launch is the README's with the machine-independent default profile: the
host Snakemake, the image `spv-container` manages, jobs on this node. A toy
checkout under your home directory (which the profile binds) carries copies of
workflow/, papers/cosmo_val/ and src/, and one synthetic catalogue declared
public. The first launch makes its reporting parts and their figure; then a
blind is drawn and declared on the catalogue, and the same launch re-measures
the parts concealed.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from conftest import REPO, _load_module, container, on_candide

custody = _load_module(REPO / "src" / "sp_validation" / "custody.py", "custody", {})

VERSIONS = ("SP_v0.1", "SP_v0.1_leak_corr")
REPORTING = {"theta_min": 5.0, "theta_max": 60.0, "nbins": 6, "npatch": 4}
BINNING = "minsep=5.0_maxsep=60.0_nbins=6_npatch=4"


def _in_image(image, root, *command):
    """``command``'s stdout, run in ``image`` on the toy checkout's src/."""
    return subprocess.run(
        [
            "apptainer",
            "exec",
            "--cleanenv",
            "--env",
            f"PYTHONPATH={root / 'src'}",
            str(image),
            *command,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout


def _stamps(image, root, parts):
    """The custody stamp of each SACC part, read in the image."""
    script = (
        "import json, sys\n"
        "from sp_validation import custody, sacc_io\n"
        "print(json.dumps([custody.read_stamp(sacc_io.load(p).metadata).stamp"
        " for p in sys.argv[1:]]))"
    )
    return json.loads(_in_image(image, root, "python", "-c", script, *map(str, parts)))


def _toy_checkout(root, image):
    """A checkout with one synthetic catalogue, SP_v0.1, declared public."""
    skip = shutil.ignore_patterns(".snakemake", "__pycache__", "tests")
    shutil.copytree(REPO / "workflow", root / "workflow", ignore=skip)
    shutil.copytree(
        REPO / "papers" / "cosmo_val", root / "papers" / "cosmo_val", ignore=skip
    )
    shutil.copytree(
        REPO / "src", root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    (root / "cosmo_val").mkdir()
    _in_image(
        image,
        root,
        "python",
        "-c",
        "import sys\n"
        f"sys.path.insert(0, {str(root / 'src/sp_validation/tests')!r})\n"
        "from pathlib import Path\n"
        "from _synthetic import write_synthetic_catalogs\n"
        f"write_synthetic_catalogs(Path({str(root / 'cosmo_val')!r}),"
        f" catalogues={{{VERSIONS[0]!r}: 'none'}})",
    )

    config_path = root / "papers" / "cosmo_val" / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["versions"] = list(VERSIONS)
    config["fiducial"]["version"] = VERSIONS[1]
    config["fiducial"]["mock_version"] = VERSIONS[0]
    config["cosmo_val"].update(REPORTING)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))


@pytest.mark.candide
@on_candide
def test_xi_before_and_after_its_catalogue_is_blinded():
    image, kind = container.resolve_image()
    assert kind != "tag", "no local image; run `spv-container pull`"
    root = Path(tempfile.mkdtemp(prefix="toy_run_", dir=Path.home()))
    _toy_checkout(root, image)
    out = root / "out"
    cat_config = root / "cosmo_val" / "cat_config.yaml"
    parts = [out / f"{v}_xi_{BINNING}.sacc" for v in VERSIONS]
    target = str(out / "snakemake_sentinels" / "plot_2pcf.done")

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SNAKEMAKE_PROFILE", "APPTAINERENV_PYTHONPATH")
    }
    env.update(
        COSMO_VAL=str(out),
        COSMO_INFERENCE=str(root / "inference"),
        XDG_CACHE_HOME=str(root / "cache"),
        PYTHONNOUSERSITE="1",
        PYTHONUNBUFFERED="1",
    )

    def launch(*args):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "snakemake",
                "--profile",
                str(root / "workflow" / "profiles" / "default"),
                "--cores",
                "4",
                *args,
                "--config",
                f"container={image}",
                "--",
                target,
            ],
            cwd=root / "papers" / "cosmo_val",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )

    # Declared public: parts stamped public, and their figure drawn.
    result = launch()
    assert result.returncode == 0, result.stdout
    assert [s["blind"] for s in _stamps(image, root, parts)] == ["none"] * 2
    assert (out / "xi_p.png").is_file()

    # A blind drawn and declared: the parts' params changed.
    _in_image(
        image,
        root,
        "python",
        "-c",
        "import dataclasses, sys, yaml\n"
        "from sp_validation import blinding\n"
        "from sp_validation.blinding_theory import TheoryConfig\n"
        "fast = TheoryConfig(transfer_function='eisenstein_hu')\n"
        "blinding.init('toy', yaml.safe_load(open(sys.argv[1])),"
        " fiducial=dataclasses.asdict(fast))",
        str(cat_config),
    )
    catalogues = yaml.safe_load(cat_config.read_text())
    catalogues[VERSIONS[0]]["blind"] = "toy"
    cat_config.write_text(yaml.safe_dump(catalogues, sort_keys=False))
    dry = launch("-n")
    assert dry.returncode == 0, dry.stdout
    assert "Params have changed" in dry.stdout, dry.stdout

    result = launch()
    assert result.returncode == 0, result.stdout
    record = json.loads((root / "cosmo_val/blinds/toy.blind.json").read_text())
    for stamp in _stamps(image, root, parts):
        assert stamp == {
            "blind": "toy",
            "blind_commitment": custody.commitment(record),
        }
    assert not list(out.rglob("*_xi_*.txt"))
    assert not list((root / "cosmo_val" / "output").iterdir())

    shutil.rmtree(root)  # kept on failure, for post-mortem
