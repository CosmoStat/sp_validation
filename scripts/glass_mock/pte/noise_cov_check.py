from pathlib import Path

import numpy as np

from sp_validation import sacc_io

r = Path("/n17data/cdaley/scratch/glass-mock-cv/pte")
base = "covariance_GLASSpte_s01_ng_minsep=1.0_maxsep=250.0_nbins=20_masked"
c = np.loadtxt(r / "inference/data/covariance" / base / (base + ".dat"))
for i in range(1, 11):
    s = sacc_io.load(
        r
        / f"output/GLASSpte_s{i:02d}_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=1.sacc",
        blind="none",
    )
    x = sacc_io.xi_correlation(s, grid="reporting")
    ratios = np.r_[x.varxip, x.varxim] / np.diag(c)
    print(
        i,
        "shot/total diag min max",
        float(ratios.min()),
        float(ratios.max()),
        "first",
        float(ratios[0]),
        "last",
        float(ratios[19]),
    )
