"""Plot an analytic covariance without invoking its producer."""

import numpy as np

from sp_validation.cosmo_val.diagnostic_plots import plot_covariance

plot_covariance(np.loadtxt(snakemake.input.matrix), snakemake.output.figure)
