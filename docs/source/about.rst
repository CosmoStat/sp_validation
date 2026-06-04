About
=====

``sp_validation`` is developed by the `CosmoStat <https://www.cosmostat.org>`_
lab at CEA Paris-Saclay. It is the validation layer between `ShapePipe
<https://github.com/CosmoStat/shapepipe>`_ — the weak-lensing measurement
pipeline — and the cosmology analyses of the `UNIONS
<https://www.skysurvey.cc>`_ survey.

The problem it solves
---------------------

A weak-lensing shear catalogue is only as trustworthy as the validation behind
it. Raw shape measurements carry calibration biases, residual PSF leakage, and
selection effects that must be characterised and corrected before the catalogue
can be used to constrain cosmology. ``sp_validation`` provides that
characterisation end to end: it calibrates the shear estimates, runs the
diagnostic null tests (rho/tau statistics, E-/B-mode decomposition, PSF-leakage
fits), and produces the science-ready catalogues and two-point statistics that
feed cosmological inference. The result is a shear catalogue whose systematics
budget has been measured rather than assumed.

Authors and contributors
-------------------------

**Authors**
  Martin Kilbinger, Cail Daley, Sacha Guerrini.

**Contributors**
  Emma Ayçoberry, Lucie Baumont, Clara Bonini, Samuel Farrens, Lisa Goh,
  Axel Guinot, Fabian Hervas Peters.

The cosmology-inference pipeline (``cosmo_inference/``) is developed by Lisa Goh
and Sacha Guerrini.

Contact
-------

For questions about the package, contact Martin Kilbinger
(`martin.kilbinger@cea.fr <mailto:martin.kilbinger@cea.fr>`_). Bug reports and
contributions are welcome through the
`GitHub repository <https://github.com/CosmoStat/sp_validation>`_; see
:doc:`contributing` for guidelines.

License
-------

``sp_validation`` is released under the MIT License.
