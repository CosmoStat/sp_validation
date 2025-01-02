## Shear validation

### Set up a shear validation run

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


### Run shear validation
                                                                                
There are two possibilities to carry out the shear validation, by running the jupyter notebooks
in `jupyter-lab`, or by running a python script in the command line via `ipython`.  
                                                                                
#### Running the jupyter notebooks                                              
                                                                                
1. In the run directory start JupyterLab:                                       
  ```bash                                                                       
  jupyer-lab                                                                    
  ```                                                                           
                                                                                
2. Load and run first notebook `main_set_up.ipynb`:                             
   1. Double-click in file manager tab on the left.                             
   2. Run notebook.                                                             
                                                                                
3. Run all further notebooks:                                                   
   1. Double-click as above.                                                    
   2. Change kernel to `main_set_up.ipynb`:                                     
      Either via the menu `Kernel -> Change Kernel` or                          
      by clicking on `Python 3` on the top left of the notebook.                
   3. Run notebook.                                                             
                                                                                
Run the notebooks in the following order:                                       
   1. `main_set_up.ipynb` (main notebook)                                       
   2. `metacal_global.ipynb`                                                    
   3. [`metacal_local.ipynb`] optional                                          
   4. `psf_leakage.ipynb`                                                       
   5. `write_cat.ipynb`                                                         
   6. `maps.ipynb`                                                              
   7. `cosmology.ipynb`                                                         
                                                                                
#### Running the python script                                                  
                                                                                
1. Create the python script from the jupyter notbooks. In `notebooks`:          
  ```bash                                                                       
  python config_convert.py                                                      
  ```                                                                           
                                                                                
2. Run python script. In run directory:                                         
  ```bash                                                                       
   ipython /path/to/sp_validation/notebooks/validation.py                       
   ```
