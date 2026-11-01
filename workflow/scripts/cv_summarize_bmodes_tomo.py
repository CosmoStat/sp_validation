"""Tomographic null PTEs, including the full cross-pair covariance.

Uses exactly the selection in the bound-9 report: snapped reporting edges
for pure E/B, fiducial-cut COSEBIs, and all pseudo-Cl bands.
"""
import json

import numpy as np
from scipy.stats import chi2

from sp_validation import sacc_io
from sp_validation.b_modes import bins_from_scale_cut

p = snakemake.params
s = sacc_io.load(snakemake.input.sacc, blind=p.blind)
eb = sacc_io.load_npz(snakemake.input.pure_eb, blind=p.blind)
a, b = bins_from_scale_cut(eb["left_edges"], eb["right_edges"], tuple(p.scale_cut))
pairs = sorted({dp.tracers for dp in s.data})
cov = np.asarray(s.covariance.dense)


def test(rows):
    rows = np.asarray(rows, int)
    d = s.mean[rows]
    value = float(d @ np.linalg.solve(cov[np.ix_(rows, rows)], d))
    return dict(chi2=value, dof=len(rows), pte=float(chi2.sf(value, len(rows))))


selected = {}
for name, dtype in (("xip_B", sacc_io.PURE_TYPES["xip_B"]),
                    ("xim_B", sacc_io.PURE_TYPES["xim_B"]),
                    ("cosebis", sacc_io.COSEBI_BB),
                    ("cl_bb", sacc_io.CL_BB), ("cl_eb", sacc_io.CL_EB)):
    if dtype not in s.get_data_types():
        continue
    selected[name] = [np.asarray(s.indices(dtype, pair), int) for pair in pairs]
    if name in ("xip_B", "xim_B"):
        selected[name] = [rows[a:b] for rows in selected[name]]
selected["combined"] = [np.concatenate([rp, rm]) for rp, rm in
                         zip(selected["xip_B"], selected["xim_B"])]
summary = dict(scale_cut=list(p.scale_cut),
               pairs={" x ".join(pair): {key: test(rows[i]) for key, rows in selected.items()}
                      for i, pair in enumerate(pairs)},
               joint={key: test(np.concatenate(rows)) for key, rows in selected.items()})
with open(snakemake.output.summary, "w") as f:
    json.dump(summary, f, indent=2)
