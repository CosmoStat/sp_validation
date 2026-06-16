Quickstart
==========

This page outlines a first run; each stage links to its full guide.
It assumes ``sp_validation`` is installed (see :doc:`installation`) and that you have a
ShapePipe shear catalogue to validate.

Validation proceeds in four stages.

#. **Configure.** Set the inputs for a run (catalogue paths, field name, optional masks)
   in ``scripts/examples/params.py``.
#. **Shear validation.** Extract shape information and run the basic diagnostics,
   producing the pre-calibration catalogues. See :doc:`run_validation`.
#. **Post-processing.** Mask, select, and calibrate into a science-ready shear catalogue.
   See :doc:`post_processing`.
#. **Cosmology inference.** Measure the two-point correlation functions, then run the
   CosmoSIS / CosmoCov pipeline in ``cosmo_inference/`` to derive cosmological constraints.

The **User Guide** in the sidebar expands each stage, and the **API Reference** documents
the library modules you call from your own scripts and notebooks.
