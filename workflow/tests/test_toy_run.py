"""Rule xi, run for real through apptainer, before and after a blind is drawn.

The launch is the README's with the machine-independent default profile: the
host Snakemake, the image `spv-container` manages, jobs on this node. A toy
checkout under your home directory (which the profile binds) carries copies of
workflow/, papers/cosmo_val/ and src/, and one synthetic catalogue declared
unblinded. The first launch stops for want of its patch centres and names the
command that draws them; after it has run, its reporting parts and their figure
are made; then the catalogue is declared blinded, a blind is drawn for it, and
the same launch re-measures the parts concealed. The jobs run under a
matplotlibrc asking for a LaTeX package no image has, as a user's own may.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from conftest import REPO, container, on_candide

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
    """A checkout with one synthetic catalogue, SP_v0.1, declared unblinded."""
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
        f" catalogues={{{VERSIONS[0]!r}: 'unblinded'}})",
    )

    config_path = root / "papers" / "cosmo_val" / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["versions"] = list(VERSIONS)
    config["fiducial"]["version"] = VERSIONS[1]
    config["fiducial"]["mock_version"] = VERSIONS[0]
    config["cosmo_val"].update(REPORTING)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    # A user's matplotlibrc typesetting with a package the image lacks.
    rc = root / "xdg" / "matplotlib" / "matplotlibrc"
    rc.parent.mkdir(parents=True)
    rc.write_text(
        "text.usetex: True\n"
        "text.latex.preamble: \\usepackage{spvalidationabsentpackage}\n"
    )
    (root / "fast.json").write_text(
        json.dumps({"theory": {"transfer_function": "eisenstein_hu"}})
    )


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
        APPTAINERENV_XDG_CONFIG_HOME=str(root / "xdg"),
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

    # No centres: the launch names the command that draws them, which works.
    result = launch()
    assert result.returncode != 0, result.stdout
    (command,) = [
        line.split("spv-container exec ", 1)[1]
        for line in result.stdout.splitlines()
        if "sp_validation.cosmo_val.patch_centers" in line
    ]
    _in_image(image, root, *shlex.split(command))
    assert (out / "patches" / "SP_v0.1_npatch=4.dat").is_file()

    # Declared unblinded: parts stamped unblinded, and their figure drawn.
    result = launch()
    assert result.returncode == 0, result.stdout
    assert [s["blinding"] for s in _stamps(image, root, parts)] == ["unblinded"] * 2
    assert (out / "xi_p.png").is_file()

    # Declared blinded, and its blind drawn: the parts' params changed.
    catalogues = yaml.safe_load(cat_config.read_text())
    catalogues[VERSIONS[0]]["blinding"] = "blinded"
    cat_config.write_text(yaml.safe_dump(catalogues, sort_keys=False))
    _in_image(
        image,
        root,
        "python",
        "-m",
        "sp_validation.blinding",
        "init",
        "toy",
        VERSIONS[0],
        "--cat-config",
        str(cat_config),
        "--config",
        str(root / "fast.json"),
    )
    dry = launch("-n")
    assert dry.returncode == 0, dry.stdout
    assert "Params have changed" in dry.stdout, dry.stdout

    result = launch()
    assert result.returncode == 0, result.stdout
    record = json.loads((root / "cosmo_val/blinds/toy/commitment.json").read_text())
    for stamp in _stamps(image, root, parts):
        assert stamp["blinding"] == "blinded", stamp
        assert stamp["blinding_commitment"] == record["seed_commitment"], stamp
    assert not list(out.rglob("*_xi_*.txt"))
    assert not list((root / "cosmo_val" / "output").iterdir())

    shutil.rmtree(root)  # kept on failure, for post-mortem
