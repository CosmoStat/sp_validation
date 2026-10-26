"""Rule cv_plot_tau_stats: tau-statistics overlay across versions.

Reads tau_stats_{base}.fits for every version (declared inputs), with error
bars from the version's tau covariance of the configured type; writes
tau_stats.png into the leakage output dir. Sentinel-tracked (see rho_stats).
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_tau_stats(
    tomography=False,
    cov_type=cv.cov_estimate_method,
    savefig="tau_stats.png",
    show=False,
)
touch_sentinels(snakemake)
