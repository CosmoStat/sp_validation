Repository structure
====================

The repository is laid out so that a single, generic analysis can feed many
papers, and so that several people can run it side by side without colliding.

.. code-block:: text

   src/sp_validation/    library code (incl. glass_mock core)
   cosmo_val/            validation: code + config        (promoted from notebooks/)
   cosmo_inference/      inference: code + config         (cosmosis / cosmocov)
   workflow/             ALL analysis — modular Snakemake, multi-person → results/
   papers/               final-figure assembly only (PDF, colour, layout)
   scripts/              real reduction + runner scripts (catalog builders, masking, glass-mock runners)
   scratch/              per-person — ad hoc work + personal workflows (tracked)
   results/              analysis products + diagnostic plots (contents gitignored, dir kept)
   docs/  tests/  config/

Analysis versus presentation
-----------------------------

The dividing line that organizes everything is the **inputs to a paper figure**.

Everything up to that point is *analysis*. It lives in ``workflow/``: generic,
reusable, modular Snakemake rules, organized for several people, producing both
the data products and the diagnostic plots that vet them.
All of it lands in a single top-level ``results/``.

The figure itself is *presentation*. It lives in ``papers/<paper>/``:
final-figure assembly only — PDF, colour, layout — tied to one manuscript, and
free to never touch Snakemake at all.
A paper directory reads finished products out of ``results/`` and emits
publication-ready figures; it does not write analysis products back in.

Where each thing lives
----------------------

- ``src/sp_validation/`` — the importable library, including the glass-mock
  core. Reusable functions belong here, not copied into scripts or notebooks.
- ``cosmo_val/`` and ``cosmo_inference/`` — the side-by-side code-and-config
  homes for the two later stages: validation diagnostics (rho/tau statistics,
  E-/B-mode decomposition, PSF-leakage tests) and CosmoSIS / CosmoCov inference.
- ``workflow/`` — the shared analysis: all of it, as Snakemake rules and their
  scripts.
- ``papers/`` — final-figure assembly, one directory per paper.
- ``scripts/`` — real reduction and runner scripts (catalogue builders, masking,
  glass-mock runners) run from the command line rather than imported.
- ``scratch/<person>/`` — personal, ad hoc work and custom workflows. Tracked on
  purpose, because sharing scratch is useful; promote anything generic into
  ``workflow/``.
- ``results/`` — the single sink for analysis products and diagnostic plots. Its
  contents are gitignored and the directory is kept.

Scaling by modularity
----------------------

The workflow scales by being modular, not monolithic.
A paper or run composes the shared rules with Snakemake's ``module`` directive
under its own config and an output ``prefix``, so each run namespaces cleanly
under ``results/<name>/`` without clobbering another.
The shared rules live in ``workflow/`` exactly once; runs differ only in the
config they bring.
