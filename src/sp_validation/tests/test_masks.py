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
