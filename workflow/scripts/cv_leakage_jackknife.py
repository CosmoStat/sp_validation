"""Rule cv_leakage_jackknife: the joint ρ/τ jackknife measurement per version."""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs

_unbuffer_streams()
cv = make_cv(snakemake)
for ver in cv.versions:
    cv.leakage_jackknife(ver)
verify_outputs(snakemake)
