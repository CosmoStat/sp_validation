"""UNIONS v2.0.0: the blinded tomographic pseudo-Cℓ, from the release.

The catalogue config's ``SP_v2.0.0`` entry is the one description of the
release: calibrated catalogue, n(z), masks and blind. Without a local copy,
sp_validation fetches what the run reads from the CADC vault into
``~/.cache/sp_validation/vos/`` (about 15 GB, once). Cℓ_EE is written concealed
under the UNIONS blind; Cℓ_BB and Cℓ_EB are not blinded.

Run inside the sp_validation container (``run_demo.sh``). The SACC lands in
``output/pseudo_cl_SP_v2.0.0_tomo.sacc``.
"""

from sp_validation.cosmo_val import CosmologyValidation

cv = CosmologyValidation(["SP_v2.0.0"], compute_tomography=True)
cv.calculate_pseudo_cl()
