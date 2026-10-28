"""Check parts against the declared blind before derivations or reporting."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sp_validation import blinding, sacc_io

SCRIPTS = Path(__file__).resolve().parents[3] / "workflow" / "scripts"


@pytest.mark.parametrize("loader", ["load_pure_eb_results", "load_cosebis_results"])
@pytest.mark.parametrize("stamp", [None, "wrong"])
@pytest.mark.parametrize("expected", ["none", "expected"])
def test_npz_consumers_refuse_missing_or_wrong_blind(tmp_path, loader, stamp, expected):
    import numpy as np

    from sp_validation import b_modes

    path = tmp_path / "cosmological.npz"
    data = dict(scale_cuts=np.empty((0, 2)), left_edges=[], right_edges=[])
    if stamp is not None:
        data["blind"] = stamp
    np.savez(path, **data)
    with pytest.raises(ValueError, match="stamp"):
        getattr(b_modes, loader)(path, blind=expected)


@pytest.fixture
def part_path(tmp_path):
    part = sacc_io.new_sacc({0: ([0.1, 1.0], [1.0, 1.0])})
    sacc_io.add_xi(part, (0, 0), [10.0], [1.0], [0.5], grid="integration")
    path = tmp_path / "xi.sacc"
    sacc_io.save(part, path, blind=blinding.NONE)
    return path


@pytest.mark.parametrize(
    "script, input_key, reader",
    [
        ("cv_cosebis.py", "xi", "get_xi"),
        ("cv_pure_eb.py", "xi_integration", "xi_correlation"),
    ],
)
@pytest.mark.parametrize("expected", ["none", "toy"])
def test_derived_script_checks_before_reading_signal(
    part_path, monkeypatch, script, input_key, reader, expected
):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import cv_runner

    monkeypatch.setattr(cv_runner, "_unbuffer_streams", lambda: None)

    class ReachedSignal(Exception):
        pass

    def read_signal(*args, **kwargs):
        raise ReachedSignal

    monkeypatch.setattr(sacc_io, reader, read_signal)
    smk = SimpleNamespace(
        params={"version": "v", "blind": expected, "fiducial_scale_cut": [1, 100]},
        input={input_key: part_path},
    )
    error = ReachedSignal if expected == "none" else ValueError
    with pytest.raises(error):
        runpy.run_path(str(SCRIPTS / script), init_globals={"snakemake": smk})
    assert not list(part_path.parent.glob("*.npz"))
    assert not list(part_path.parent.glob("*.png"))


@pytest.mark.parametrize("as_name", [False, True])
@pytest.mark.parametrize("stamp", ["none", "toy"])
def test_load_checks_expected_blind(part_path, as_name, stamp):
    part = sacc_io.load(part_path)
    part.metadata["blind"] = stamp
    part.save_fits(str(part_path), overwrite=True)
    expected = "toy" if as_name else blinding.Blind("toy")
    if stamp == "none":
        with pytest.raises(ValueError, match="stamped none.*toy"):
            sacc_io.load(part_path, blind=expected)
    else:
        assert sacc_io.stamp(sacc_io.load(part_path, blind=expected)) == "toy"
