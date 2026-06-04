Quickstart
==========

This page orients a first run; each stage links to its full guide. It assumes
``sp_validation`` is installed (see :doc:`installation`) and that you have a
ShapePipe shear catalogue to validate.

A validation campaign runs in four stages:

#. **Configure.** Set the inputs for a run — catalogue paths, field name, and
   optional masks — in ``notebooks/params.py``.
#. **Shear validation.** Extract shape information and run the basic diagnostics
   to produce the pre-calibration catalogues. See :doc:`run_validation`.
#. **Post-processing.** Mask, select, and calibrate into a science-ready shear
   catalogue. See :doc:`post_processing`.
#. **Cosmology inference.** Measure the two-point correlation functions and run
   the CosmoSIS / CosmoCov pipeline in ``cosmo_inference/`` to derive
   cosmological constraints.

The **User Guide** in the sidebar covers each stage, and the **API Reference**
documents the library modules you call from your own scripts and notebooks.
