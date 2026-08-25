"""Rule cv_rho_tau_fits: PSF-error model fit and xi_psf_sys contours.

Conditional in the original driver on rho_tau_method != "none". Reads
rho_stats/tau_stats FITS (declared inputs), runs the MCMC fit
(calculate_rho_tau_fits via plot_rho_tau_fits), and writes contour /
xi_psf_sys figures into the leakage output dir. Sentinel-tracked: figure paths
derive from internal handler state. The fitted xi_psf_sys it produces is
recomputed (not persisted) by the cv_ratio_xi_sys_xi rule.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels

# `snakemake` is injected as a module global by Snakemake's `script:` preamble
# before this file runs; no import is needed (and `from snakemake.script
# import snakemake` is IDE-hint-only -- snakemake.script has no such runtime
# attribute and raises ImportError if actually executed).
_unbuffer_streams()
cv = make_cv(snakemake)
if cv.rho_tau_method != "none":
    cv.plot_rho_tau_fits()
touch_sentinels(snakemake)
