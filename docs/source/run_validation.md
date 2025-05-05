## Extract shear information

(<= SP v1.4.1: Formerly known as shapepipe validation.)

This step extracts the relevant information from the merged final_cat hdf5 catalogue (<= v1.4.1: fits)
including the sheared values for metacalibration.

### Set up

All inputs and settings are contained in the python configuration script
`notebooks/params.py`, that needs to be edited accordingly.
The main parameters are:
- `name`: field or patch name, can be any string. E.g. `P3` for patch 3. 
- `data_dir`: input directory for data. Set to `.` for validation run in
  current directory.
- `galaxy_cat_path`: path to galaxy catalogue, format `.fits`. or `.hdf5`.
- `star_cat_path`: path to star catalogue, format `.fits`.

Optional parameters are:
- `path_tile_ID`: path to ascii file containing lines of tile IDs. Used to
  identify missing tiles.  
- `param_list_path`: path to ascii (SExtractor) parameter file; used to avoid
  hdf5 data read errors if input parameters vary from tile to tile. Set to
  `None` if not required.  
- `mask_external_path`: path to external mask file, format `.reg`. Set to
  `None` if not required.  

See the script `prepare_patch_for_spval.sh` for an example of copying
the required input files to where the validation is to be run.


### Run

Run the jupyter notebook `extract_information.ipynb` or python script `extract_information.py`. 