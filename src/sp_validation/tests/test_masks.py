"""TESTS FOR THE SHARED MASK-CONDITION GRAMMAR.

Covers ``sp_validation.masks.apply_condition`` — the single ``kind``/``value``
evaluator shared by the object-selection ``Mask`` class and the
cosmo_inference footprint builder (issue #181) — plus an equivalence check
against ``Mask.apply()`` itself, and the one catalogue-masking path
(``catalogue_selection``) that galaxies and PSF stars share, with mask
columns a catalogue lacks read from the config's sky ``maps``.

:Author: cdaley

"""

import importlib.util
from pathlib import Path

import healpy as hp
import healsparse as hsp
import numpy as np
import numpy.testing as npt
import pytest
import yaml
from astropy.table import Table

from sp_validation import masks
from sp_validation.masks import (
    Mask,
    apply_condition,
    catalogue_cuts,
    catalogue_selection,
    get_masks_from_config,
    map_path,
)

pytestmark = pytest.mark.fast


_ARRAY = np.array([1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "kind, value, expected",
    [
        ("equal", 3, [False, False, True, False, False]),
        ("not_equal", 3, [True, True, False, True, True]),
        ("greater", 3, [False, False, False, True, True]),
        ("greater_equal", 3, [False, False, True, True, True]),
        ("less", 3, [True, True, False, False, False]),
        ("less_equal", 3, [True, True, True, False, False]),
        ("range", [2, 4], [False, True, True, True, False]),
    ],
)
def test_apply_condition_kinds(kind, value, expected):
    npt.assert_array_equal(apply_condition(_ARRAY, kind, value), expected)


def test_smaller_equal_alias_matches_less_equal():
    npt.assert_array_equal(
        apply_condition(_ARRAY, "smaller_equal", 3),
        apply_condition(_ARRAY, "less_equal", 3),
    )


@pytest.mark.parametrize("dtype", ["i4", "u4"])
def test_ignore_bits_and_config_mask(dtype):
    flags = np.array([0, 1, 2, 3, 16, 17, 18, 19, 20, 32], dtype=dtype)
    expected = [True, True, True, False, True, True, True, False, False, False]
    npt.assert_array_equal(apply_condition(flags, "less_equal", 2, 16), expected)
    config = {
        "dat": [
            {
                "col_name": "FLAGS",
                "label": "flags",
                "kind": "less_equal",
                "value": 2,
                "ignore_bits": 16,
            }
        ]
    }
    masks, _ = get_masks_from_config(config, {"FLAGS": flags})
    npt.assert_array_equal(masks[0]._mask, expected)
    masks[0].create_descr()
    assert masks[0]._descr == "(FLAGS & ~16) <= 2"


def test_ignore_bits_rejects_float():
    with pytest.raises(ValueError, match="integer"):
        apply_condition(np.array([1.0]), "less_equal", 2, 16)


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        apply_condition(_ARRAY, "not_a_real_kind", 3)


def test_mask_apply_matches_apply_condition():
    dat = np.array([(1,), (2,), (3,), (4,), (5,)], dtype=[("col", "i8")])

    my_mask = Mask("col", "test_mask", kind="greater_equal", value=3, dat=dat)

    npt.assert_array_equal(
        my_mask._mask, apply_condition(dat["col"], "greater_equal", 3)
    )


def test_not_equal_2bands_is_two_column_or():
    # Keep an object if EITHER band column differs from the sentinel
    dat = np.array(
        [(-99, -99), (-99, 20.0), (21.0, -99), (21.0, 20.0)],
        dtype=[("mag_z", "f8"), ("mag_z2", "f8")],
    )

    my_mask = Mask(
        "mag_z",
        "zband",
        kind="not_equal_2bands",
        value=-99,
        col_name2="mag_z2",
        dat=dat,
    )

    npt.assert_array_equal(my_mask._mask, [False, True, True, True])


def test_not_equal_2bands_requires_col_name2():
    with pytest.raises(ValueError, match="col_name2"):
        Mask("mag_z", "zband", kind="not_equal_2bands", value=-99)


def test_not_equal_2bands_descr_names_both_columns():
    my_mask = Mask(
        "mag_z", "zband", kind="not_equal_2bands", value=-99, col_name2="mag_z2"
    )
    my_mask.create_descr()

    assert my_mask._descr == "!=-99 in mag_z or mag_z2"


def test_dat_ext_cut_list_is_refused_not_dropped():
    """A config's ``dat_ext`` cuts would be silently ignored; refuse them."""
    config = {
        "dat": [{"col_name": "FLAGS", "label": "SE", "kind": "equal", "value": 0}],
        "dat_ext": [{"col_name": "4_Stars", "kind": "equal", "value": False}],
    }
    dat = np.zeros(3, dtype=[("FLAGS", "i2"), ("MASK_4_Stars", "?")])
    with pytest.raises(ValueError, match="dat_ext.*4_Stars.*MASK_4_Stars"):
        get_masks_from_config(config, dat)
    del config["dat_ext"]
    assert catalogue_cuts(config) == config["dat"]
    masks, _ = get_masks_from_config(config, dat)
    assert len(masks) == 1


def test_masking_script_refuses_dat_ext():
    spec = importlib.util.spec_from_file_location(
        "masking_script", Path(__file__).resolve().parents[3] / "scripts/masking.py"
    )
    masking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(masking)
    config = {"dat": [], "dat_ext": [{"col_name": "8_Manual"}]}
    dat = np.zeros(3, dtype=[("MASK_8_Manual", "?")])
    with pytest.raises(ValueError, match="dat_ext"):
        masking.apply_masks(dat, config, footprint_only=True)
    assert "IMAFLAGS_ISO" in masking.SPATIAL_CUTS


_MANUAL_CUT = {
    "col_name": "MASK_8_Manual",
    "label": "manual mask",
    "kind": "equal",
    "value": False,
}


def _manual_map(path, *positions):
    """Write a bool healsparse map, True at ``positions`` (ra, dec in deg)."""
    nside = 1024
    sky_map = hsp.HealSparseMap.make_empty(32, nside, dtype=np.bool_)
    for ra, dec in positions:
        sky_map[hp.ang2pix(nside, ra, dec, lonlat=True, nest=True)] = True
    sky_map.write(str(path))
    return str(path)


def _mask_config(tmp_path):
    """A galaxy-style mask config: a FLAGS cut, the manual cut, a bit-8 map."""
    config = {
        "maps": {"MASK_8_Manual": _manual_map(tmp_path / "manual.hsp", (10.1, 20.1))},
        "dat": [
            {"col_name": "FLAGS", "label": "SE", "kind": "less_equal", "value": 2},
            _MANUAL_CUT,
        ],
    }
    (tmp_path / "mask.yaml").write_text(yaml.safe_dump(config))
    return config


def _star_entry(tmp_path, **mask):
    stars = Table(
        dict(
            RA=[10.0, 10.1, 10.2, 10.3],
            DEC=[20.0, 20.1, 20.2, 20.3],
            HSM_FLAG_PSF=[0, 0, 0, 1],
            HSM_FLAG_STAR=[0, 0, 0, 0],
        )
    )
    tmp_path.mkdir(exist_ok=True)
    stars.write(tmp_path / "stars.fits")
    return dict(
        path=str(tmp_path / "stars.fits"),
        ra_col="RA",
        dec_col="DEC",
        PSF_flag="HSM_FLAG_PSF",
        star_flag="HSM_FLAG_STAR",
        **mask,
    )


def _star_mask(entry):
    from sp_validation.cosmo_val.psf_systematics import PSFSystematicsMixin

    stub = PSFSystematicsMixin()
    stub.cc = {"v": {"psf": entry}}
    return stub._get_star_mask("v")


def test_star_inside_the_manual_map_is_dropped(tmp_path):
    """A star catalogue has no MASK_8_Manual; the config's map supplies it."""
    _mask_config(tmp_path)
    unmasked = _star_mask(_star_entry(tmp_path / "a"))
    masked = _star_mask(
        _star_entry(
            tmp_path / "b",
            mask_config=str(tmp_path / "mask.yaml"),
            mask_cuts=["MASK_8_Manual"],
        )
    )
    np.testing.assert_array_equal(unmasked, [True, True, True, False])
    np.testing.assert_array_equal(masked, [True, False, True, False])


def test_galaxy_column_wins_over_the_map(tmp_path):
    """A catalogue that has the mask column is cut on it, map or no map."""
    config = _mask_config(tmp_path)
    galaxies = np.zeros(
        4, dtype=[("RA", "f8"), ("Dec", "f8"), ("FLAGS", "i2"), ("MASK_8_Manual", "?")]
    )
    galaxies["RA"], galaxies["Dec"] = [10.0, 10.1, 10.2, 10.3], [20.0, 20.1, 20.2, 20.3]
    galaxies["FLAGS"] = [0, 0, 4, 0]
    galaxies["MASK_8_Manual"] = [False, False, False, True]
    expected = [True, True, False, False]
    np.testing.assert_array_equal(catalogue_selection(config, galaxies), expected)
    del config["maps"]
    np.testing.assert_array_equal(catalogue_selection(config, galaxies), expected)


def test_galaxy_selection_unchanged_on_the_v2_config(tmp_path):
    """mask_v2.0.yaml's cuts select the same galaxies as evaluating each cut
    directly, its bit-8 map unread because the catalogue has the column."""
    config_path = Path(__file__).resolve().parents[3] / "config/calibration"
    config = yaml.safe_load((config_path / "mask_v2.0.yaml").read_text())
    rng = np.random.default_rng(1)
    galaxies = {
        cut["col_name"]: np.where(rng.random(200) < 0.9, cut["value"], not cut["value"])
        for cut in config["dat"]
        if isinstance(cut["value"], bool)
    }
    galaxies.update(
        FLAGS=rng.choice([0, 1, 16, 18, 4], 200),
        N_EPOCH=rng.choice([1, 2, 3, 3], 200),
        mag=rng.uniform(10, 35, 200),
        NGMIX_MCAL_TYPES_FAIL=rng.choice([0, 0, 0, 1], 200),
        NGMIX_N_EPOCH=rng.choice([0, 1, 2, 2], 200),
        NGMIX_G1_PSF_ORIG_NOSHEAR=rng.choice([-10.0, 0.1, 0.1, 0.1], 200),
        NGMIX_G2_PSF_ORIG_NOSHEAR=rng.choice([-10.0, 0.1, 0.1, 0.1], 200),
        RA=rng.uniform(0, 1, 200),
        Dec=rng.uniform(0, 1, 200),
    )
    galaxies = Table(galaxies)
    direct = np.ones(200, dtype=bool)
    for cut in config["dat"]:
        direct &= apply_condition(
            galaxies[cut["col_name"]],
            cut["kind"],
            cut["value"],
            cut.get("ignore_bits", 0),
        )
    assert "MASK_8_Manual" in config["maps"]
    assert 0 < direct.sum() < 200
    np.testing.assert_array_equal(catalogue_selection(config, galaxies), direct)


def test_remote_map_is_fetched_once_into_the_cache(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        Path(cmd[2]).write_text("map")

    monkeypatch.setenv("SP_VALIDATION_MASK_CACHE", str(tmp_path))
    monkeypatch.setattr(masks.subprocess, "run", fake_run)
    url = "arc:home/someone/masks/m.hsp"
    local = tmp_path / "home/someone/masks/m.hsp"
    assert map_path(url) == str(local) == map_path(url)
    assert local.read_text() == "map"
    assert len(calls) == 1 and calls[0][:2] == ["vcp", url]
    assert map_path("/a/local/map.hsp") == "/a/local/map.hsp"
