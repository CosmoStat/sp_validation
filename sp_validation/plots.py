"""
  
:Name: plots.py

:Description: This script contains methods for plots.

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import numpy as np
import matplotlib.pylab as plt

from sp_validation import util
from sp_validation import io
from sp_validation import basic
from sp_validation import plot_style


def plot_spatial_density(ra, dec, title, x_label, y_label, cbar_label, out_path, n_grid=1000, verbose=False):

    plt.figure(figsize=(30, 30))

    plt.hexbin(ra, dec, gridsize=n_grid)

    cbar = plt.colorbar()
    cbar.set_label(cbar_label, rotation=270, labelpad=40)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(out_path)
    plt.show()
