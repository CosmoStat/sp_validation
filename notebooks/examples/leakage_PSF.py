# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # PSF leakage
# SP validation

# +
import os
import matplotlib.pylab as plt

from sp_validation import cat
from sp_validation import io
from sp_validation import run
# -

# ## Set input parameters

params_in = {}

# ### Input catalogues

# +
# Data directory
data_dir = f"{os.environ['HOME']}/astro/data/CFIS/v1.0/ShapePipe"
params_in['data_dir'] = data_dir

# Input galaxy shear catalogue
params_in['input_path_shear'] = f"{data_dir}/unions_shapepipe_extended_2022_v1.0.fits"

# Input star/PSF catalogue
params_in['input_path_PSF'] = f"{data_dir}/unions_shapepipe_star_2022_v1.0.3.fits"

# Input galaxy redshift distribution (for theoretical model of xi_pm)
params_in['dndz_path'] = f"{data_dir}/../nz/dndz_SP_A.txt"
# -

# ### Other parameters

# +
# PSF ellipticty column names
params_in['e1_PSF_star_col'] =  'e1'
params_in['e2_PSF_star_col'] =  'e2'

# Set verbose output
params_in['verbose'] = True
# -

# ## Compute leakage

# Create leakage instance
obj = run.LeakageScale()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]

# ### Run
# There are two options to run the leakage compuations:
# 1. Run all at once with single class routine.
# 2. Execute individual steps.

# ### Option 1. Run all at once

# +
#obj.run()
# -

# ### Option 2. Execute individual steps

# +
# Check parameter validity                                              
obj.check_params() 

# Get all parameters defined in instance    
params = obj._params
# -

# Prepare output
if not os.path.exists(params["output_dir"]):
    os.mkdir(params["output_dir"])
obj._stats_file = io.open_stats_file(params["output_dir"], "stats_file_leakage.txt") 

# #### Prepare input

# Read input shear                                                      
dat_shear = obj.read_shear_cat()

# Apply cuts to galaxy catalogue if required                            
dat_shear = cat.cut_data(dat_shear, params["cut"], params["verbose"])

# Read star catalogue                                                   
dat_PSF = io.open_fits_or_npy(                                          
    params["input_path_PSF"],                                           
    hdu_no=params["hdu_psf"],                                           
) 

# Deal with close objects in PSF catalogue (= stars on same position    
        # from different exposures)                                             
dat_PSF = obj.handle_close_objects(dat_PSF)

# Copy variables to instance
obj.dat_shear = dat_shear
obj.dat_PSF = dat_PSF

# #### Compute correlation functions
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.

obj.compute_corr_gp_pp_alpha()

# #### Scale-dependent alpha function

# Average over scales
obj.compute_alpha_mean()

# Plot
obj.plot_alpha_leakage()

# #### xi_sys cross-correlation function

# Compute
obj.compute_xi_sys()

# Plot
obj.plot_xi_sys()

params["dndz_path"] = '/home/mkilbing/astro/data/CFIS/v1.0/ShapePipe/../nz/dndz_SP_A.txt'

# Theoretical model for xi_pm
xi_p_theo, xi_m_theo = run.get_theo_xi_planck(
    obj.r_corr_gp.meanr,
    params["dndz_path"],
)

#obj Plot ratio
obj.plot_xi_sys_ratio(xi_p_theo, xi_m_theo)


