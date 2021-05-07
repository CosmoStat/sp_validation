"""
  
:Name: util.py

:Description: This script contains utility methods.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import math


def millify(n):

    millnames = ['',' Thousand',' Million',' Billion',' Trillion']

    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])
