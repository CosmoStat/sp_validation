"""The CLOE JAX theory: angular power spectra from cloelib, under jit, vmap and grad.

Vendored, byte for byte, from ``likelihood/euclid_spt_like/`` of the Euclid x SPT
6x2pt likelihood (``cailmdaley/euclid-spt-6x2pt``, private) at commit 2fc432e3
(25 Nov 2026). Every commit touching these files there is by Cail Daley (four,
from 16 Nov 2026, each co-authored by Claude Opus); the JAX adapter was carried
into that repository from the Euclid cmbx pipeline (``dr1_cmbx``'s
``eDR1like``) with its numerics unchanged. Only theory is vendored: no Euclid or
SPT data, data-vector reader or configuration.

- ``cloe``: cloelib's grids, engines and CAMB background (:class:`~.cloe.Grid`).
- ``jax_vector``: :class:`~.jax_vector.JaxVector`, the theory of a data vector
  (background, CosmoPower-JAX HMcode-2020 + feedback P(k), NLA, n(z) shifts,
  Limber, bands, chi^2) as pure JAX functions.
- ``cloe_jax``: cloelib's tracers and ``AngularTwoPoint`` Limber sum, traced.
- ``jax_hmcode``: HMcode-2020's feedback response in JAX (the ``camb_mead2020``
  engine only).
- ``vector``: :func:`~.vector.sparse_ells` and :func:`~.vector.interpolation_matrix`,
  the multipole nodes and their spline onto every ell. Its ``EuclidVector`` (the
  source's release-pickle reader) is not used here: :mod:`sp_validation.cloe.vector`
  reads SACC.
- ``cloelib_patches``: the separable layer. Each patch (P1, P3, P4, P5, P6, P8) is
  a mixin on one cloelib class that makes it fast or traceable under JAX, written
  against cloelib 0.11 alone so that it can go upstream into CLOE as a merge
  request as it stands. When cloelib carries a patch, delete its layer from that
  module and its name from ``cloe_jax.PATCHES``; when it carries all of them, the
  module goes.

The files are kept identical to their source, so ``diff`` against it shows any
local change, and ruff skips this directory (``pyproject.toml``). Importing
``jax_vector`` enables JAX's 64-bit mode process-wide. The CosmoPower-JAX networks
are fetched by :mod:`sp_validation.cloe.networks`.
"""
