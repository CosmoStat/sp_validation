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


def _cat_config(tmp_path, blind):
    import yaml

    path = tmp_path / "cat_config.yaml"
    path.write_text(yaml.safe_dump({"SP_v": {"blind": blind}, "paths": {}}))
    return path


@pytest.mark.parametrize("version", ["SP_v", "SP_v_leak_corr"])
def test_dashboard_takes_blind_from_declaration(tmp_path, version):
    from sp_validation.dashboard import products

    assert products.declared_blind(_cat_config(tmp_path, "toy"), version) == "toy"


@pytest.mark.parametrize("declared", ["none", "toy"])
def test_dashboard_refuses_stale_public_part(
    part_path, tmp_path, monkeypatch, declared
):
    """A product saved public, of a catalogue since declared under a blind, is
    refused before its signal is read."""
    from sp_validation.dashboard import products

    class ReachedSignal(Exception):
        pass

    def read_signal(*args, **kwargs):
        raise ReachedSignal

    monkeypatch.setattr(sacc_io, "xi_correlation", read_signal)
    blind = products.declared_blind(_cat_config(tmp_path, declared), "SP_v")
    error = ReachedSignal if declared == "none" else ValueError
    with pytest.raises(error):
        products.load_xi(part_path, blind)


@pytest.mark.parametrize("declared", ["none", "toy"])
def test_cosebis_grid_refuses_stale_public_xi(
    part_path, tmp_path, monkeypatch, declared
):
    """The scale-cut grid derives from ξ under the declared blind, not ξ's stamp."""
    import numpy as np

    from sp_validation.dashboard import cosebis_grid

    class ReachedSignal(Exception):
        pass

    def read_signal(*args, **kwargs):
        raise ReachedSignal

    monkeypatch.setattr(sacc_io, "get_xi", read_signal)
    monkeypatch.setattr(cosebis_grid, "_paths", lambda r, v: (part_path, part_path))
    monkeypatch.setattr(
        cosebis_grid, "_grid", lambda r: (np.geomspace(1, 100, 3), (1, 100, 10), 6)
    )
    error = ReachedSignal if declared == "none" else ValueError
    with pytest.raises(error):
        cosebis_grid.compute(tmp_path, "SP_v", _cat_config(tmp_path, declared))
