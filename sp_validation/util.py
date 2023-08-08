"""UTIL.

:Description: This script contains utility methods.

:Author: Martin Kilbinger <martin.kilblinger@cea.fr>

"""


import sys
import os

import math
import numpy as np


def millify(n):
    """Millify.

    Return human-readible names of large numbers.

    Parameters
    ----------
    n : int
        input number

    Returns
    -------
    str
        output name

    """
    millnames = ['', ' Thousand', ' Million', ' Billion', ' Trillion']

    n = float(n)
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        )
    )

    return '{n / 10**(3 * millidx):.0f}{millnames[millidx])}'


def transform_nan(value):
    """Transform Nan.

    Transform a ``nan`` to a very large number.

    Parameters
    ----------
    value : float
        input value

    Returns
    -------
    float
        output value

    """
    large = 1e30

    if np.isnan(value) or np.isinf(value):
        res = 1e30
    else:
        res = value

    return res
