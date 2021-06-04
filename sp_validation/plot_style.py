"""
:Name: plot_style.py

:Description: Commands to set plot styles for matplotlib.

:Author: Axel Guinot

:Date: 01/2021

:Package: sp_validation

"""

import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 10

mpl.rcParams['font.size'] = 20

mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['ytick.minor.size'] = 5

mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['ytick.major.size'] = 7

mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

mpl.rcParams['boxplot.boxprops.linewidth'] = 2.
mpl.rcParams['boxplot.medianprops.linewidth'] = 2.
mpl.rcParams['boxplot.flierprops.markersize'] = 12
mpl.rcParams['boxplot.whiskerprops.linewidth'] = 2.
mpl.rcParams['boxplot.capprops.linewidth'] = 2.


mpl.rcParams['axes.xmargin'] = mpl.rcParamsDefault['axes.xmargin']
