About
=====

``sp_validation`` is developed by the `CosmoStat <https://www.cosmostat.org>`_ lab at CEA Paris-Saclay.
It sits between `ShapePipe <https://github.com/CosmoStat/shapepipe>`_, the weak-lensing measurement pipeline, and the cosmology analyses of the `UNIONS <https://www.skysurvey.cc>`_ survey.

What it does
------------

Raw shape measurements carry calibration biases, residual PSF leakage, and selection effects.
``sp_validation`` characterises and corrects these before the catalogue reaches a cosmology analysis.
It calibrates the shear estimates, runs the diagnostic null tests (rho/tau statistics, E-/B-mode decomposition, PSF-leakage fits), and produces the science-ready catalogues and two-point statistics that feed cosmological inference.

Authors and contributors
-------------------------

**Authors**
  Martin Kilbinger, Cail Daley, Sacha Guerrini.

**Contributors**
  Emma Ayçoberry, Lucie Baumont, Clara Bonini, Samuel Farrens, Lisa Goh,
  Axel Guinot, Fabian Hervas Peters.

The cosmology-inference pipeline (``cosmo_inference/``) is developed by Lisa Goh and Sacha Guerrini.

Contact
-------

For questions about the package, contact Martin Kilbinger (`martin.kilbinger@cea.fr <mailto:martin.kilbinger@cea.fr>`_).
Bug reports and contributions are welcome through the `GitHub repository <https://github.com/CosmoStat/sp_validation>`_; see :doc:`contributing` for guidelines.

License
-------

``sp_validation`` is released under the MIT License.
