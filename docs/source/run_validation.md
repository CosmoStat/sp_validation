# Running shear validation

## Extract shear information

(<= SP v1.4.1: Formerly known as shapepipe validation.)

This step extracts the relevant information from the merged final_cat hdf5 catalogue (<= v1.4.1: fits)
including the sheared values for metacalibration.

### Set up

All inputs and settings are contained in the python configuration script
`scripts/calibration/params.py`, that needs to be edited accordingly.
The main parameters are:
- `campaign`: campaign name (the ShapePipe tile list), can be any string. 
- `data_dir`: input directory for data. Set to `.` for validation run in
  current directory.
- `galaxy_cat_path`: path to galaxy catalogue, FITS or HDF5 (the container is
  detected from the file).
- `star_cat_path`: path to star catalogue, FITS or HDF5.

Optional parameters are:
- `path_tile_ID`: path to ascii file containing lines of tile IDs. Used to
  identify missing tiles.  
- `param_list_path`: path to ascii (SExtractor) parameter file; used to avoid
  hdf5 data read errors if input parameters vary from tile to tile. Set to
  `None` if not required.  
- `mask_external_path`: path to external mask file, format `.reg`. Set to
  `None` if not required.  
- `galaxy_column_map`, `star_column_map`: `{canonical name: name in the file}`
  for a catalogue whose columns are not named as ShapePipe v2 names them; see
  [Catalogue columns](catalogue_columns.md) for the columns read and the map.
  Set to `None` if not required.

Link or copy the campaign's merged products -- `final_cat_<campaign>.hdf5`
and `full_starcat_<campaign>.hdf5` -- into the directory where the validation
is to be run.


### Run

Run the python script `scripts/calibration/extract_info.py`.