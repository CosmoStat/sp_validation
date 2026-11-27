"""UNIONS likelihoods on CLOE's theory in JAX: differentiable, batched, fast.

- :mod:`.theory`: the vendored CLOE JAX theory (cloelib traced by JAX, with the
  separable :mod:`.theory.cloelib_patches`).
- :mod:`.vector`: data vectors as linear maps of C_ell (SACC bandpower windows,
  xi+-, gamma_t, w, COSEBIs), :class:`~.vector.Theory`, scale cuts.
- :mod:`.probe`: the likelihood of a SACC as a cloerun probe (nested sampling).
- :mod:`.survey`, :mod:`.covariance`, :mod:`.forecast`: a harmonic 3x2pt survey,
  its coupled-bandpower covariance, and the forecast steps
  (``workflow/cloe/Snakefile``).
- :mod:`.networks`: the CosmoPower-JAX networks, fetched by checksum.

GLASS 3x2pt mocks of a survey are :mod:`sp_validation.mocks.glass_3x2pt`.
"""
