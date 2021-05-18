"""
  
:Name: util.py

:Description: This script contains utility methods.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import math
import io
import numpy as np
from nbformat import current


def millify(n):

    millnames = ['',' Thousand',' Million',' Billion',' Trillion']

    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


# Define function to read and run external notebook

# See https://nbviewer.jupyter.org/gist/minrk/5491090/analysis.ipynb
def execute_notebook(nbfile):
    
    with io.open(nbfile) as f:
        nb = current.read(f, 'json')
    
    ip = get_ipython()
    
    res = -1
    for cell in nb.worksheets[0].cells:
        if cell.cell_type == 'code':
            print('#'*80)
            res = ip.run_cell(cell.input)
            print()
        elif cell.cell_type == 'markdown':
            print(cell.source)
            print()
        else:
            continue

    if res.error_in_exec:
        print('Error of last cell call = {}'.format(res.error_in_exec))

    return res


def transform_nan(value):
    """Transform Nan

    Transform a nan to a very large number.

    Parameters
    ----------
    value : float
        input value

    Returns
    -------
    res : float
        output value
    """

    large = 1e30

    if np.isnan(value) or np.isinf(value):
        res = 1e30
    else:
        res = value

    return res
