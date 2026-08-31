"""SACC likelihood shim for CosmoSIS — the sp_validation-owned native path.

This module is a thin subclass of CosmoSIS's ``SaccClLikelihood`` (from the
CosmoSIS Standard Library, ``likelihood/sacc/sacc_like.py``) that fixes two
upstream defects so the native SACC likelihood matches the converter →
``2pt_like`` path bit for bit on our real-space ξ± analysis file. It exists as a
CosmoSIS *module file* (``setup``/``execute``/``cleanup`` at module scope), loaded
via ``file = .../sacc_like_unions.py`` in an ini, and is NOT imported by
``sp_validation/__init__`` (cosmosis is an optional dependency).

Why the shim exists
-------------------
Two things in upstream ``sacc_like`` break a real-space ξ likelihood; both are
documented and empirically verified (probe: Δχ²=3184 raw vs Δχ²=0 shimmed on the
single-bin analysis file), and both are fixed here by overriding ``build_data``:

1. **No arcmin→radian conversion (the killer).**
   ``sacc_likelihoods/twopoint.py`` (L74-90 at CSL commit 4fd2f1c) builds a
   ``SpectrumInterp`` over ``block[section, "theta"]`` — theory θ in *radians*
   (CosmoSIS convention) — and evaluates it at each data point's raw ``theta``
   tag, which our SACC files store in *arcmin*. ``2pt_like.py``
   (L179-183) converts its real-space data to radians for exactly this reason;
   ``sacc_like`` never does, so the spline is evaluated ~3437× outside its grid,
   ``SpectrumInterp`` returns 0 there, and χ² silently collapses to dᵀC⁻¹d. The
   only prior use was the ℓ-space (unit-free) Cℓ path, which is why it was never
   caught. We evaluate the theory against a radian-θ *copy* of the loaded SACC,
   keeping ``self.sacc_data`` in arcmin so upstream's save_theory /
   save_realization paths (which copy it and overwrite only values) never write
   radian tags into a file downstream consumers read as arcmin.

2. **Theory↔data ordering is assumed, never enforced.**
   The data vector is ``sacc.get_mean()`` (insertion order); the theory vector is
   built by looping ``get_data_types() × get_tracer_combinations() × points``.
   A comment in ``twopoint.py`` (L35) claims ``to_canonical_order`` was called on
   load — it is not. The two orders agree only when the file is grouped
   type-major (all of one data type, then the next). Our single-pair
   ``[ξ+; ξ−]`` files satisfy this; a tomographic *pair-major* file (ξ+/ξ− per
   pair) would silently misalign theory against data. We reconstruct the
   theory-loop index order and require it equals ``arange`` — a hard ValueError
   otherwise (see ``test_ordering_guard_raises_pair_major_tomographic``).

When upstream fixes the units, this shim dies: the tripwire test
``test_upstream_unit_gap_tripwire`` in ``tests/test_sacc_like.py`` fails the day
raw ``sacc_like`` stops producing a wildly different χ², signalling the shim can
be retired.

Requires a CSL checkout: ``setup`` reads ``csl_dir`` from the module options and
imports the upstream ``sacc_like`` from ``<csl_dir>/likelihood/sacc``. The tests
locate it via the ``CSL_DIR`` environment variable (see the test module docstring
for the checkout recipe).
"""

import os
import sys

import numpy as np

# cosmosis is an OPTIONAL dependency: this module is a CosmoSIS module file, but
# importing it must succeed without cosmosis installed (the CI image has the
# science stack but no cosmosis, and test_imports.py bare-imports every module).
# So every cosmosis touch is deferred into setup() — nothing at top level imports
# it. numpy is fine at top level (always present).

# arcmin → radian: the conversion 2pt_like applies to real-space data and that
# sacc_like omits. Applied only to `theta` tags of `real`-category data types.
ARCMIN_TO_RAD = np.pi / (180.0 * 60.0)


def _import_upstream_sacc_like(csl_dir):
    """Import the upstream ``sacc_like`` module from a CSL checkout.

    ``<csl_dir>/likelihood/sacc`` is prepended to ``sys.path`` so both
    ``sacc_like`` and its sibling ``sacc_likelihoods`` package (imported by
    ``sacc_like`` for the theory-extraction functions) resolve. Idempotent: the
    path is only inserted once.
    """
    sacc_dir = os.path.join(csl_dir, "likelihood", "sacc")
    if not os.path.isdir(sacc_dir):
        raise ValueError(
            f"csl_dir={csl_dir!r} has no likelihood/sacc directory; point csl_dir "
            "at a CosmoSIS Standard Library checkout (see module docstring)"
        )
    if sacc_dir not in sys.path:
        sys.path.insert(0, sacc_dir)
    import sacc_like  # noqa: E402 — resolved from the sys.path insertion above

    return sacc_like


def _make_subclass(sacc_like):
    """Build ``SaccLikeUnions`` as a subclass of the upstream ``SaccClLikelihood``.

    A factory (not an import-time ``class ... :`` statement) so importing this
    module never requires the upstream class — that dependency is deferred to
    ``setup``, which has ``csl_dir`` in hand. Only ``build_data`` is overridden;
    scale cuts, covariance handling (Sellentin/Hartlap), theory extraction and
    ``save_theory`` all ride upstream unmodified.
    """

    class SaccLikeUnions(sacc_like.SaccClLikelihood):
        """CSL ``SaccClLikelihood`` with the arcmin→rad + ordering-guard fixes."""

        def build_data(self):
            # Run the upstream build FIRST: it loads the SACC, applies data_sets
            # selection and the arcmin-grammar scale cuts (matching 2pt_like's
            # angle_range convention and the ini ergonomics), populates
            # self.sacc_data / self.sections_for_names, and returns the
            # (unit-independent) data vector we pass straight through.
            x, data_vector = super().build_data()

            # self.sacc_data STAYS in arcmin — save_theory / save_realization copy
            # it and only overwrite point values, so its θ tags must remain the
            # units the file was written in (any consumer, incl. re-ingesting the
            # saved SACC as a data_file, assumes arcmin). The radian conversion the
            # theory spline needs lives on a separate copy, swapped in only for the
            # extraction (see extract_theory_points).
            self._sacc_data_rad = self._radian_theta_copy(self.sacc_data)
            self._assert_theory_order_matches_data()

            return x, data_vector

        def _radian_theta_copy(self, sacc_data):
            """A copy of ``sacc_data`` with ``real``-category θ tags in radians.

            The theory spline is built on ``block[section, "theta"]`` in radians,
            so the ``theta`` tag each point is evaluated against must be radians
            too. Scoped to data types whose category
            (``sections_for_names[dt][0]``) is ``real``: cosebis ``n`` tags and
            spectrum ``ell`` tags are unit-free and left untouched, and only the
            ``theta`` tag is scaled (``theta_nom`` etc. are metadata the likelihood
            never evaluates). Operates on a ``.copy()`` so the original stays
            arcmin for the save paths.
            """
            real_types = {
                dt
                for dt, (category, _section) in self.sections_for_names.items()
                if category == "real"
            }
            converted = sacc_data.copy()
            for point in converted.data:
                if point.data_type in real_types and "theta" in point.tags:
                    point.tags["theta"] = point.tags["theta"] * ARCMIN_TO_RAD
            return converted

        def extract_theory_points(self, block):
            """Extract theory against the radian-θ copy, then restore the original.

            Upstream ``extract_theory_points`` reads ``self.sacc_data`` (the θ tag
            per point) to evaluate the theory spline; that read needs radians.
            Swap in ``self._sacc_data_rad`` for the duration of the upstream call
            and restore in ``finally`` so everything else — including the
            save_theory / save_realization copies that run afterward in
            ``do_likelihood`` — sees the untouched arcmin ``self.sacc_data``.
            """
            original = self.sacc_data
            self.sacc_data = self._sacc_data_rad
            try:
                return super().extract_theory_points(block)
            finally:
                self.sacc_data = original

        def _assert_theory_order_matches_data(self):
            """Require the theory-loop order to equal the data-vector order.

            The data vector is ``sacc.get_mean()`` (insertion order); upstream
            builds theory by looping data types, then tracer combinations, then
            points, and concatenating — assuming (never enforcing) that this
            reproduces insertion order. It does only for type-major files. We
            reconstruct that loop's index order and require it be ``arange``;
            otherwise theory and data would be silently misaligned (the same bug
            class the PR-2/PR-3 reviews caught for the converter).
            """
            order = [
                int(i)
                for dt in self.sacc_data.get_data_types()
                for tracers in self.sacc_data.get_tracer_combinations(dt)
                for i in self.sacc_data.indices(dt, tracers)
            ]
            expected = np.arange(len(self.sacc_data.mean))
            if not np.array_equal(order, expected):
                raise ValueError(
                    "SACC data/theory ordering mismatch: the theory loop "
                    "(get_data_types × get_tracer_combinations × points) does not "
                    "reproduce the get_mean() insertion order, so sacc_like would "
                    "compare theory against data point-by-point in the WRONG order "
                    "and return a silently wrong χ². This happens when the file is "
                    "grouped pair-major (ξ+/ξ− interleaved per tracer pair) rather "
                    "than type-major (all ξ+, then all ξ−). Upstream assumes "
                    "to_canonical_order() was applied on load but never calls it; "
                    "write the SACC type-major, or call to_canonical_order() before "
                    "saving."
                )

    return SaccLikeUnions


def setup(options):
    """CosmoSIS ``setup`` — build and instantiate the shimmed likelihood.

    Mirrors ``GaussianLikelihood.build_module``'s setup: wrap the raw options in
    ``SectionOptions`` and instantiate the likelihood (whose ``__init__`` calls
    ``build_data``). The one addition is reading ``csl_dir`` from the module
    options to locate and import the upstream class before subclassing it. The
    cosmosis import is deferred to here (call time) so importing this module never
    requires cosmosis.
    """
    from cosmosis.datablock import SectionOptions, option_section

    csl_dir = options.get_string(option_section, "csl_dir")
    sacc_like = _import_upstream_sacc_like(csl_dir)
    likelihood_class = _make_subclass(sacc_like)
    return likelihood_class(SectionOptions(options))


def execute(block, config):
    """CosmoSIS ``execute`` — run the likelihood (mirrors ``build_module``)."""
    likelihood_calculator = config
    likelihood_calculator.do_likelihood(block)
    return 0


def cleanup(config):
    """CosmoSIS ``cleanup`` — mirror of ``build_module``'s cleanup."""
    likelihood_calculator = config
    likelihood_calculator.cleanup()
