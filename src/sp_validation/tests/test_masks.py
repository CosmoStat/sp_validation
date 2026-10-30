"""TESTS FOR THE SHARED MASK-CONDITION GRAMMAR.

Covers ``sp_validation.masks.apply_condition`` — the single ``kind``/``value``
evaluator shared by the object-selection ``Mask`` class and the
cosmo_inference footprint builder (issue #181) — plus an equivalence check
against ``Mask.apply()`` itself, and the ``mask_veto`` that drops PSF stars
from rho/tau (``star_veto``).

:Author: cdaley

"""

import hashlib
import importlib.util
from pathlib import Path

import healpy as hp
import healsparse as hsp
import numpy as np
import numpy.testing as npt
import pytest
from astropy.table import Table

from sp_validation.masks import (
    Mask,
    apply_condition,
    catalogue_cuts,
    get_masks_from_config,
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


def _star_entry(tmp_path, **veto):
    stars = Table(
        dict(
            RA=[10.0, 10.1, 10.2, 10.3],
            DEC=[20.0, 20.1, 20.2, 20.3],
            HSM_FLAG_PSF=[0, 0, 0, 1],
            HSM_FLAG_STAR=[0, 0, 0, 0],
            MASK_8_Manual=[False, True, False, False],
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
        **veto,
    )


def _star_mask(entry):
    from sp_validation.cosmo_val.psf_systematics import PSFSystematicsMixin

    stub = PSFSystematicsMixin()
    stub.cc = {"v": {"psf": entry}}
    return stub._get_star_mask("v")


def test_mask_veto_column_drops_manually_masked_stars(tmp_path):
    unvetoed = _star_mask(_star_entry(tmp_path / "a"))
    vetoed = _star_mask(_star_entry(tmp_path / "b", mask_veto="MASK_8_Manual"))
    np.testing.assert_array_equal(unvetoed, [True, True, True, False])
    np.testing.assert_array_equal(vetoed, [True, False, True, False])
    assert hashlib.sha1(unvetoed).digest() != hashlib.sha1(vetoed).digest()


def test_mask_veto_map_drops_stars_inside_it(tmp_path):
    nside = 1024
    sky_map = hsp.HealSparseMap.make_empty(32, nside, dtype=np.bool_)
    sky_map[hp.ang2pix(nside, 10.1, 20.1, lonlat=True, nest=True)] = True
    sky_map.write(str(tmp_path / "manual.hsp"))
    entry = _star_entry(tmp_path, mask_veto=[str(tmp_path / "manual.hsp")])
    np.testing.assert_array_equal(_star_mask(entry), [True, False, True, False])
