"""Rule cv_summarize_bmodes: collect B-mode PTEs across all statistics.

The terminal diagnostic. summarize_bmodes reads the in-memory
_pure_eb_results / _cosebis_results / _pseudo_cls dicts, which are populated by
plot_pure_eb / plot_cosebis / plot_pseudo_cl. The per-version E/B and COSEBIs
npz products and the pseudo-Cl FITS are declared as inputs (so the DAG forces
those rules first), but the summary still needs the live result objects (it
reads each version's TreeCorr `gg`, which the npz cannot hold). So this rule
re-runs the three B-mode methods in-process: they reload the existing 2pcf /
data-vector files via their skip-if-exists paths and recompute only the cheap
PTE statistics, exactly as the original linear driver did on its shared cv.

Writes the summary table to bmode_summary.txt (declared output) — the original
driver only printed it.
"""

import json

from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
p = snakemake.params
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])

# Repopulate the in-memory B-mode result dicts from existing data products.
cv.plot_pure_eb(
    min_sep_int=p["pure_eb_min_sep_int"],
    max_sep_int=p["pure_eb_max_sep_int"],
    nbins_int=p["pure_eb_nbins_int"],
    fiducial_xip_scale_cut=fiducial_scale_cut,
    fiducial_xim_scale_cut=fiducial_scale_cut,
)
for version in cv.versions:
    cv.plot_cosebis(
        version=version,
        min_sep_int=p["cosebis_min_sep_int"],
        max_sep_int=p["cosebis_max_sep_int"],
        nbins_int=p["cosebis_nbins_int"],
        npatch=p["cosebis_npatch"],
        nmodes=p["cosebis_nmodes"],
        scale_cuts=[tuple(sc) for sc in p["cosebis_scale_cuts"]],
        fiducial_scale_cut=fiducial_scale_cut,
    )
if p.get("include_pseudo_cl", False):
    cv.plot_pseudo_cl()

summary = cv.summarize_bmodes(fiducial_scale_cut=fiducial_scale_cut)

# summarize_bmodes prints its table and returns {version: {stat: pte}}. Persist
# the returned dict (the table itself is reproducible from it) so downstream
# tooling and the all-rule have a real, machine-readable artifact to depend on.
with open(snakemake.output["summary_json"], "w") as f:
    json.dump(summary, f, indent=2, default=str)

verify_outputs(snakemake)
