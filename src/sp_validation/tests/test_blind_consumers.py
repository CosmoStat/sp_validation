"""Check parts against the declared blind before derivations or reporting."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sp_validation import blinding, sacc_io

SCRIPTS = Path(__file__).resolve().parents[3] / "workflow" / "scripts"


@pytest.fixture
def part_path(tmp_path):
    part = sacc_io.new_sacc({0: ([0.1, 1.0], [1.0, 1.0])})
    sacc_io.add_xi(part, (0, 0), [10.0], [1.0], [0.5], grid="integration")
    path = tmp_path / "xi.sacc"
    sacc_io.save(part, path, blind=blinding.NONE)
    return path


@pytest.mark.parametrize("expected", ["none", "toy"])
def test_reporting_parts(part_path, monkeypatch, expected):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from cv_runner import take_reporting_parts

    cv = SimpleNamespace(
        versions=["v"], xi_parts={}, blind=lambda v: blinding.Blind(expected)
    )
    if expected == "toy":
        with pytest.raises(ValueError, match="stamped none.*toy"):
            take_reporting_parts(cv, [part_path])
        assert not cv.xi_parts
    else:
        take_reporting_parts(cv, [part_path])
        assert sacc_io.stamp(cv.xi_parts["v", "reporting"]) == "none"


@pytest.mark.parametrize("script, input_key, reader", [
    ("cv_cosebis.py", "xi", "get_xi"),
    ("cv_pure_eb.py", "xi_integration", "xi_correlation"),
])
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


def test_expect_names_both_stamps(part_path):
    with pytest.raises(ValueError, match="stamped none.*toy"):
        sacc_io.expect(sacc_io.load(part_path), blinding.Blind("toy"))
