"""Scatter: compute COSEBIS E_n/B_n for one GLASS mock."""

import numpy as np
import treecorr

from cosmo_numba.B_modes.cosebis import COSEBIS
from sp_validation.b_modes import scale_cut_to_bins

xi_path = snakemake.input.xi
nmodes = snakemake.params.nmodes
theta_min = snakemake.params.theta_min
theta_max = snakemake.params.theta_max
out_path = snakemake.output.cosebis

gg = treecorr.GGCorrelation(min_sep=0.5, max_sep=500, nbins=1000, sep_units="arcmin")
gg.read(xi_path)

start, stop = scale_cut_to_bins(gg, theta_min, theta_max)
inds = np.arange(start, stop)
# .astype(float) ensures native byte order (FITS big-endian → numba needs native)
theta_cut = gg.meanr[inds].astype(float)
xip_cut = gg.xip[inds].astype(float)
xim_cut = gg.xim[inds].astype(float)

cosebis_obj = COSEBIS(np.min(theta_cut), np.max(theta_cut), nmodes, precision=120)
En, Bn = cosebis_obj.cosebis_from_xipm(theta_cut, xip_cut, xim_cut, parallel=True)

np.savez(
    out_path,
    En=En,
    Bn=Bn,
    theta_min_actual=np.min(theta_cut),
    theta_max_actual=np.max(theta_cut),
)
print(
    f"Saved {out_path}: {nmodes} modes, θ=[{np.min(theta_cut):.2f}, {np.max(theta_cut):.2f}]"
)
