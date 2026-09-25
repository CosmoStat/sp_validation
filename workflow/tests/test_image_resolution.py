"""Every launcher runs the image ``sp_validation/container.py`` resolves."""

import os
import subprocess

import pytest
from conftest import REPO, container

SWEEP_ENV = REPO / "papers" / "bmodes" / "scripts" / "container_env.sh"


@pytest.mark.parametrize("sandbox", [False, True], ids=["sif", "sandbox"])
def test_sweep_drivers_run_the_resolved_image(tmp_path, monkeypatch, sandbox):
    """The Paper II sweep drivers' CONTAINER is the image container.py resolves.

    Their shell copy of the resolution reads the same cache, whatever
    XDG_CACHE_HOME says, and prefers the sandbox the same way.
    """
    cache = tmp_path / "home" / ".cache" / "sp_validation"
    cache.mkdir(parents=True)
    (cache / "sp_validation.sif").touch()
    if sandbox:
        (cache / "sandbox").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "node-local"))
    for var in ("SPV_CONTAINER", "SPV_SANDBOX"):
        monkeypatch.delenv(var, raising=False)

    shell = subprocess.run(
        ["bash", "-c", f'. "{SWEEP_ENV}" && printf %s "$CONTAINER"'],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=True,
    )
    assert shell.stdout == container.resolve_image()[0]
