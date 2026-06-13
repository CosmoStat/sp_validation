"""Rule cv_rho_tau_fits: PSF-error model fit and xi_psf_sys contours.

Conditional in the original driver on rho_tau_method != "none". Reads
rho_stats/tau_stats FITS (declared inputs), runs the MCMC fit
(calculate_rho_tau_fits via plot_rho_tau_fits), and writes contour /
xi_psf_sys figures into the leakage output dir. Sentinel-tracked: figure paths
derive from internal handler state. The fitted xi_psf_sys it produces is
recomputed (not persisted) by the cv_ratio_xi_sys_xi rule.
"""

from snakemake.script import snakemake

from cv_runner import make_cv, touch_sentinels, _unbuffer_streams

_unbuffer_streams()
cv = make_cv(snakemake)
if cv.rho_tau_method != "none":
    cv.plot_rho_tau_fits()
touch_sentinels(snakemake)
