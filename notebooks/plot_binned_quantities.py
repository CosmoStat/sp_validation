# %%
# Plot binned quantites, which are the outputs of leakage_minima.py

# %%                                                                             
from IPython import get_ipython                                                  

ipython = get_ipython()                                                          

# enable autoreload for interactive sessions                                     
if ipython is not None:                                                          
    ipython.run_line_magic("load_ext", "autoreload")                             
    ipython.run_line_magic("autoreload", "2")   

# %%                          
import matplotlib.pyplot as plt                                                  
import numpy as np  
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cs_util import plots as cs_plots

from sp_validation import run_joint_cat as sp_joint
from sp_validation import cat as sp_cat
from sp_validation.basic import metacal
from sp_validation import calibration                                                             
from sp_validation import io           
from sp_validation import plots          


# %%
# enable inline plotting for interactive sessions                                
# (must be done *after* importing package that sets agg backend)                 
if ipython is not None:                                                          
    ipython.run_line_magic("matplotlib", "inline")    

# %%
bin_edges = {}
quantities = {}

# %%
keys = ["number", "response", "leakage"]

for key in keys:
    fname = f"{key}_binned.npz"
    result = io.read_binned_quantity(fname)
    for xy in result:
        if xy  != "quantity":
            bin_edges[xy] = result[xy]
    quantities[key] = result["quantity"]

# %%
vmin = {"diag": -0.2, "offdiag": -0.03}
vmax =  {"diag": 1.2, "offdiag": 0.03}

plots.plot_binned(quantities, "response", bin_edges["snr"], bin_edges["size_ratio"], "R", vmin=vmin, vmax=vmax, xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")

# %%
plots.plot_binned(quantities, "number", bin_edges["snr"], bin_edges["size_ratio"], "R", vmin=1, vmax=np.nanmax(quantities["number"]), xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")

# %%
vmin = {"diag": -0.2, "offdiag": -0.2}
vmax = {"diag": 0.1, "offdiag": 0.1}

plots.plot_binned(quantities, "leakage", bin_edges["snr"], bin_edges["size_ratio"], r"\alpha", vmin=vmin, vmax=vmax, xlabel="SNR", ylabel=r"$r / r_{\rm psf}$")


# %%
