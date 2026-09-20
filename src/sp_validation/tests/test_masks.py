"""TESTS FOR THE SHARED MASK-CONDITION GRAMMAR.

Covers ``sp_validation.masks.apply_condition`` — the single ``kind``/``value``
evaluator shared by the object-selection ``Mask`` class and the
cosmo_inference footprint builder (issue #181) — plus an equivalence check
against ``Mask.apply()`` itself.

:Author: cdaley

"""

import numpy as np
import numpy.testing as npt
import pytest

from sp_validation.masks import Mask, apply_condition

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


def test_covered_2bands_pairs_each_bit_with_its_own_band():

    # Footprint bits (set = no imaging) in one group, magnitudes in the other
    bits = np.array(
        [(False, True), (False, True), (True, False), (True, True), (False, False)],
        dtype=[("256_z", "?"), ("2048_z2", "?")],
    )
    mags = np.array(
        [(21.0, 20.0), (-99, 20.0), (21.0, 20.0), (21.0, 20.0), (-99, -99)],
        dtype=[("mag_z", "f8"), ("mag_z2", "f8")],
    )

    my_mask = Mask(
        "256_z",
        "zband",
        kind="covered_2bands",
        value=-99,
        col_name2="2048_z2",
        mag_col="mag_z",
        mag_col2="mag_z2",
        dat=bits,
        dat_other=mags,
    )

    # row 0: z measured and covered                      -> keep
    # row 1: z absent, z2 measured but not covered       -> drop
    # row 2: z not covered, z2 measured and covered      -> keep
    # row 3: neither band covered                        -> drop
    # row 4: both covered but neither measured           -> drop
    npt.assert_array_equal(my_mask._mask, [True, False, True, False, False])


def test_covered_2bands_requires_its_columns_and_other_group():

    with pytest.raises(ValueError, match="mag_col"):
        Mask("256_z", "zband", kind="covered_2bands", value=-99, col_name2="2048_z2")

    with pytest.raises(ValueError, match="dat_other"):
        Mask(
            "256_z",
            "zband",
            kind="covered_2bands",
            value=-99,
            col_name2="2048_z2",
            mag_col="mag_z",
            mag_col2="mag_z2",
        )
