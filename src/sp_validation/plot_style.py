"""PLOT STYLE.

:Name: plot_style.py

:Description: Commands to set plot styles for matplotlib.

:Author: Axel Guinot


"""

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["lines.markersize"] = 10

font_size = 16
mpl.rcParams["font.size"] = font_size
mpl.rcParams["xtick.labelsize"] = font_size
mpl.rcParams["ytick.labelsize"] = font_size
mpl.rcParams["axes.labelsize"] = font_size
mpl.rcParams["legend.fontsize"] = 12
mpl.rcParams["axes.titlesize"] = 20

mpl.rcParams["xtick.minor.size"] = 5
mpl.rcParams["ytick.minor.size"] = 5

mpl.rcParams["xtick.major.size"] = 7
mpl.rcParams["ytick.major.size"] = 7

mpl.rcParams["xtick.major.width"] = 2
mpl.rcParams["ytick.major.width"] = 2

mpl.rcParams["boxplot.boxprops.linewidth"] = 2
mpl.rcParams["boxplot.medianprops.linewidth"] = 2
mpl.rcParams["boxplot.flierprops.markersize"] = 12
mpl.rcParams["boxplot.whiskerprops.linewidth"] = 2
mpl.rcParams["boxplot.capprops.linewidth"] = 2

mpl.rcParams["axes.xmargin"] = mpl.rcParamsDefault["axes.xmargin"]
