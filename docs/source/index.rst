sp_validation
=============

.. Include table of contents (defines the left-sidebar navigation)
.. include:: toc.rst

``sp_validation`` validates the weak-lensing catalogues — galaxy and star
shapes, and the many quantities derived from them — produced by the
`ShapePipe <https://github.com/CosmoStat/shapepipe>`_ pipeline. It is developed
by the `CosmoStat <https://www.cosmostat.org>`_ lab at CEA Paris-Saclay and is
used to prepare and vet the shear catalogues behind the `UNIONS
<https://www.skysurvey.cc>`_ cosmic-shear analyses.

The package bundles a Python library together with the scripts and notebooks
that drive a validation campaign — taking a raw ShapePipe catalogue through to
the calibrated, science-ready shear catalogue and the diagnostics a cosmology
analysis depends on.

What it does
------------

``sp_validation`` performs four main tasks, typically run in sequence:

#. **Shear validation.** Take a shear catalogue carrying metacalibration
   information, apply the shear calibration, and run diagnostic tests such as
   PSF leakage. A calibrated shear catalogue is produced on output. See
   :doc:`run_validation`.
#. **Post-processing.** Further process the calibrated catalogue into
   science-ready catalogues — masking, galaxy-sample selection, and merging of
   per-patch catalogues. See :doc:`post_processing`.
#. **Cosmology validation.** Run detailed diagnostics on the calibrated
   catalogue: rho- and tau-statistics, E-/B-mode decomposition, and comparison
   plots across catalogue versions.
#. **Cosmology inference.** Measure the two-point shear correlation functions
   and feed a CosmoSIS / CosmoCov pipeline to derive cosmological constraints.

The flow chart below illustrates the path from ShapePipe output products to
calibrated, well-selected galaxy catalogues.

.. figure:: ../images/flow_chart.png
   :alt: sp_validation processing flow chart
   :width: 100%

   From ShapePipe output products to calibrated and selected galaxy catalogues.

Where to go next
----------------

- :doc:`installation` — install via container (recommended) or a development
  checkout.
- :doc:`quickstart` — an orientation to the four stages of a validation run.
- :doc:`about` — the package in its CosmoStat / UNIONS context, with authors and
  contact.
- The **User Guide** in the sidebar walks through each stage: running shear
  validation, post-processing, and the PSF-leakage diagnostics.
- The **API Reference** documents every module of the ``sp_validation`` library.
