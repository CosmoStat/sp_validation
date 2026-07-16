"""Rule cv_plot_tau_stats: tau-statistics overlay across versions.

Reads tau_stats_{base}.fits for every version (declared inputs); writes
tau_stats.png into the leakage output dir. Sentinel-tracked (see rho_stats).
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_tau_stats()
touch_sentinels(snakemake)
