"""The calibrated shear sample: selection, metacalibration and weights of a
ShapePipe snapshot's shapes, written as the catalogue cosmo_val measures.

- :mod:`.snapshot`: the snapshot a sample is built from, the record of it the
  catalogue carries, and the check of a ``cat_config`` entry against it;
- :mod:`.pipeline`: the build, run from a recipe
  (``config/calibration/sample/``, ``scripts/calibration/calibrate_sample.py``);
- :mod:`.bins`: the response of each tomographic bin, written into the
  sample's ``catalogs/response.json`` (``scripts/calibration/bin_responses.py``);
- :mod:`.compare`: two samples compared object by object.
"""
