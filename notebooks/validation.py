
# coding: utf-8

# # Weak-lensing galaxy shape catalogue validation

# In[1]:


# Run the following notebooks in order:
# 1. main_set_up.ipynb (this ncotebook)
# 2. metacal_global.ipynb
# 3. psf_leakage.ipybn
# 4. write_cat.ipynb
# 5. cosmology.ipynb


# ## Main notebook, set-up, catalogue preparation
# 
# ### Contents
# 1. Set-up
# 2. Load data
# 3. Matching of stars
# 4. Select galaxies

# In[2]:


import sys
import os
import numpy as np
from astropy.io import fits


# In[3]:


from sp_validation.io import *
from sp_validation.cat import *
from sp_validation.survey import *


# In[4]:


sp_base= f"{os.environ['HOME']}/astro/repositories/github/sp_validation"

# The following commands will be replaced by import instructions, once the sp_validation scripts are stable
for sc in ['util', 'cat', 'survey', 'galaxy']:
    script = os.path.join(sp_base, 'sp_validation', sc)
    get_ipython().run_line_magic('run', '$script')


# ## 1. Set-up

# In[5]:


# Load parameters
get_ipython().run_line_magic('run', 'params.py')


# ### Create and open output files and directories

# In[6]:


make_out_dirs(plot_dir, plot_subdirs, verbose=verbose)
stats_file = open_stats_file(plot_dir, stats_file_name)


# ## 2. Load data
# 
# ### Load merged (final) galaxy catalogue

# In[7]:


dd = np.load(galaxy_cat_path, mmap_mode=mmap_mode)


# #### Print some quantities to check nothing obvious is wrong with catalogue

# In[8]:


print_some_quantities(dd, 'NGMIX_ELL_NOSHEAR', 2, stats_file, invalid=-10, verbose=verbose)


# #### Survey area and potential missing tiles
# The approximate observed area is the number of tiles $\times$ 0.25 deg$^2$ (ignoring overlaps and masking).

# In[9]:


area_deg2, area_amin2, tile_IDs = get_area(dd, area_tile, verbose=verbose)


# Identify missing tiles by comparing tile ID from catalogue to external input tile ID file.

# In[83]:


n_found, n_missing = missing_tiles(tile_IDs, path_tile_ID, path_missing_ID, verbose=verbose)


# ### Load star catalogue

# In[11]:


d_star = fits.getdata(star_cat_path, 1)


# In[12]:


print_some_quantities(d_star, ['E1_PSF_HSM', 'E2_PSF_HSM'], 1, stats_file, invalid=-10, verbose=verbose)


# ### 3. Matching of stars

# ### Matching of star catalogues
# Match the star catalogue `d_star` (selected on individual exposures using size-magnitude diagram) to catalogue from tile. Uses some simple criteria to select stars from tile catalogue such as SPREAD_CLASS.
# 
# This is mainly for testing, this match will not be used later.

# #### Match to all objects

# In[13]:


ind_star, mask_area_tiles, n_star_tot = check_matching(d_star, dd, ['RA', 'DEC'], ['XWIN_WORLD', 'YWIN_WORLD'], thresh,
                                                       stats_file, name=None, verbose=verbose)


# #### Refine: Match to valid, unflagged ngmix sample

# In[14]:


# Flags to indicate valid ngmix star sample
m_star_ngmix = (dd['FLAGS'][ind_star] == 0) &                (dd['IMAFLAGS_ISO'][ind_star] == 0) &                (dd['NGMIX_MCAL_FLAGS'][ind_star] == 0) &                (dd['NGMIX_ELL_PSFo_NOSHEAR'][:,0][ind_star] != -10)

print_stats('ngmix:', stats_file, verbose=verbose)

ra_star_ngmix, dec_star_ngmix, g_star_psf_ngmix =     match_subsample(dd, ind_star, m_star_ngmix, ['XWIN_WORLD', 'YWIN_WORLD'], 'NGMIX_ELL_PSFo_NOSHEAR',
                    n_star_tot, stats_file, verbose=verbose)


# #### Refine: Match to valid, unflagged galsim sample

# In[15]:


m_star_galsim = (dd['FLAGS'][ind_star] == 0) &                 (dd['IMAFLAGS_ISO'][ind_star] == 0) &                 (dd['GALSIM_PSF_ELL_ORIGINAL_PSF'][:,0][ind_star] != -10)

print_stats('galsim:', stats_file, verbose=verbose)

ra_star_galsim, dec_star_galsim, g_star_psf_galsim =     match_subsample(dd, ind_star, m_star_galsim, ['XWIN_WORLD', 'YWIN_WORLD'], 'GALSIM_PSF_ELL_ORIGINAL_PSF',
                    n_star_tot, stats_file, verbose=verbose)


# #### Refine: Match to ngmix SPREAD_CLASS samples

# In[16]:


print_stats('ngmix:', stats_file, verbose=verbose)
match_spread_class(dd, ind_star, m_star_ngmix, stats_file, len(ra_star_ngmix), verbose=verbose)


# #### Refine: Match to galsim SPREAD_CLASS samples

# In[17]:


print_stats('galsim:', stats_file, verbose=verbose)
match_spread_class(dd, ind_star, m_star_galsim, stats_file, len(ra_star_galsim), verbose=verbose)


# ## X. Check for objects with invalid PSF

# In[18]:


check_invalid(dd, ['NGMIX_ELL_PSFo_NOSHEAR', 'NGMIX_ELL_NOSHEAR'], [0, 0], [-10, -10],
              stats_file, name=['PSF', 'galaxy ellipticity'], verbose=verbose)


# ## 4. Select galaxies

# ### 4.1 Using the spread model parameter
# This parameter quantifies the size of an object with respect to the local PSF. Objects with larger spread model are more likely to be galaxies.

# #### Common flags and cuts
# First, set cuts common to ngmix and galsim:
#   - spread model: select objects well larger than the PSF
#   - magnitude: cut galaxies that are too faint (= too noisy, likely to be
#     artefacts), and too bright (might be too large for postage stamp)
#   - flags: cut objects that were flagged as invalid or masked
#   - n_epoch: select objects observed on at leatst one epoch (for safety,
#     to avoid potential errors with empty data)

# In[19]:


cut_common = classification_galaxy_base(dd)


# #### ngmix

# In[20]:


# add ngmix-specific cuts: select objects with valid flags, valid PSF ellipticity, valid moments,
# ngmix used at least one epoch

m_gal_ngmix = classification_galaxy_ngmix(dd, cut_common, stats_file, verbose=verbose)


# #### galsim

# In[24]:


# add ngmix-specific cuts: select objects with valid PSF ellipticity

m_gal_galsim = classification_galaxy_galsim(dd, cut_common, stats_file, verbose=verbose)


# coding: utf-8

# # Weak-lensing galaxy shape catalogue validation
# 
# ## Global metacalibration
# 
# Contents.
# - Metacalibration (global)
# - Additive bias, ellipticity, magnitude distributions

# The selection criteria for galaxies are
# - Flags = 0 to select valid objects
# - $\frac{T_{\rm gal}}{T_{\rm psf}}$ > 0.5  to select objects that are not too small compared to the PSF, thus not likely to be point-like
# - SNR > 10 to cut noisy objects
# - SNR < 500 to cut too bright objects, potentially too large for the postage stamp

# > **_NOTE:_** Before running this notebook, set kernel to `main_set.ipynb'

# In[25]:


import os


# In[26]:


from sp_validation.survey import *
from sp_validation.util import *
from sp_validation.basic import *
from sp_validation.plots import *
#from sp_validation.calibration import *

sp_base = '{}/astro/repositories/github/sp_validation'.format(os.environ['HOME'])

# The following commands will be replaced by import instructions, once the sp_validation scripts are stable
for sc in ['survey', 'io', 'cat', 'basic', 'util', 'plot_style', 'plots', 'calibration']:
    script = os.path.join(sp_base, 'sp_validation', sc)
    get_ipython().run_line_magic('run', '$script')


# ## metacalibration for galaxies

# ### ngmix

# In[27]:


gal_metacal_ngmix = metacal(dd, m_gal_ngmix, snr_min=gal_snr_min, snr_max=gal_snr_max, rel_size_min=gal_rel_size_min, verbose=verbose)


# #### Get quantities for plots

# In[28]:


g_corr_ngmix, w_ngmix, mask = get_calibrated_quantities(gal_metacal_ngmix)

# coordinates
ra_ngmix = dd['XWIN_WORLD'][m_gal_ngmix][mask]
dec_ngmix = dd['YWIN_WORLD'][m_gal_ngmix][mask]

# magnitude, from SExtractor
mag_ngmix = dd['MAG_AUTO'][m_gal_ngmix][mask]

# signal-to-noise ratio, from ngmix fitted flux and error
snr_ngmix = dd['NGMIX_FLUX_NOSHEAR'][m_gal_ngmix][mask]/dd['NGMIX_FLUX_ERR_NOSHEAR'][m_gal_ngmix][mask]


# In[29]:


# Number density
n_gal_ngmix = len(w_ngmix)
n_gal_ngmix_sm = len(np.where(m_gal_ngmix)[0])

print_stats('ngmix:', stats_file, verbose=verbose)
print_stats('Number of galaxies after metacal = {}/{} = {:.1f}%'             ''.format(n_gal_ngmix, n_gal_ngmix_sm, n_gal_ngmix / n_gal_ngmix_sm * 100), stats_file, verbose=verbose)
print_stats('Galaxy density = {:.2f} gal/arcmin2'.format(n_gal_ngmix / area_amin2), stats_file, verbose=verbose)


# #### Plot spatial distribution of objects

# In[30]:


x_label = 'R.A. [deg]'
y_label = 'DEC [deg]'
cbar_label_base = 'Density [$A_{\\rm pix}^{-1}$]'


# In[31]:


# Galaxies

ra = ra_ngmix
dec = dec_ngmix

Apix = 1 # [arcmin^2]
title = 'Galaxies'
out_name = 'galaxy_number_count_ngmix.png'

cbar_label = '{}, $A_{{\\rm pix}} \\approx {:.1g}$ arcmin$^2$'.format(cbar_label_base, Apix)
out_path = '{}/{}'.format(plot_dir, out_name)
n_grid = int(np.sqrt(area_amin2) / Apix)
if verbose:
    print('Number of pixels = {}^2'.format(n_grid))
plot_spatial_density(ra, dec, title, x_label, y_label, cbar_label, out_path, n_grid=n_grid, verbose=verbose)


# #### Plot galaxy signal-to-noise distribution
# Plot both ngmix and galsim selection

# In[32]:


print_stats('ngmix:', stats_file, verbose=verbose)

# Do not apply `mask_ns`, so use all galaxies
xs = [dd['NGMIX_FLUX_NOSHEAR'][m_gal_ngmix]/dd['NGMIX_FLUX_ERR_NOSHEAR'][m_gal_ngmix],
      dd['SNR_WIN'][m_gal_ngmix],
      dd['SNR_WIN'][m_gal_galsim],
     ]
labels = ['ngmix $F/\\sigma_F$', 'SExtractor ngmix selection', 'SExtractor galsim selection']

x_label = 'SNR'
y_label = 'Frequency'
title = 'Galaxies'
density = True
x_range = (0, 200)
n_bin = 500
out_name = 'hist_SNR_ngmix.pdf'
out_path = os.path.join(plot_dir, out_name)

x = 10
plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                vline_x=[x], vline_lab=['SNR = {}'.format(x)])


# ### galsim

# In[33]:


gal_metacal_galsim = metacal(dd, m_gal_galsim, prefix='GALSIM', snr_min=gal_snr_min, snr_max=gal_snr_max,
                             rel_size_min=gal_rel_size_min, verbose=verbose)


# #### Get quantities for plots

# In[34]:


g_corr_galsim, w_galsim, mask_galsim = get_calibrated_quantities(gal_metacal_galsim)

# coordinates
ra_galsim = dd['XWIN_WORLD'][m_gal_galsim][mask_galsim]
dec_galsim = dd['YWIN_WORLD'][m_gal_galsim][mask_galsim]

# magnitude, from SExtractor
mag_galsim = dd['MAG_AUTO'][m_gal_galsim][mask_galsim]

# No SNR estimate from galsim


# In[35]:


# Number density
n_gal_galsim = len(w_galsim)
n_gal_galsim_sm = len(np.where(m_gal_galsim)[0])


print_stats('galsim:', stats_file, verbose=verbose)
print_stats('Number of galaxies after metacal = {}/{} = {:.1f}%'             ''.format(n_gal_galsim, n_gal_galsim_sm, n_gal_galsim / n_gal_galsim_sm * 100), stats_file, verbose=verbose)
print_stats('Galaxy density = {:.2f} gal/arcmin2'.format(n_gal_galsim / area_amin2), stats_file, verbose=verbose)


# #### Plot spatial distribution of objects

# In[36]:


# Galaxies

ra = ra_galsim
dec = dec_galsim

Apix = 1 # [arcmin^2]
title = 'Galaxies'
out_name = 'galaxy_number_count_galsim.png'

cbar_label = '{}, $A_{{\\rm pix}} \\approx {:.1g}$ arcmin$^2$'.format(cbar_label_base, Apix)
out_path = '{}/{}'.format(plot_dir, out_name)
n_grid = int(np.sqrt(area_amin2) / Apix)
if verbose:
    print('Number of pixels n = {}^2'.format(n_grid))
plot_spatial_density(ra, dec, title, x_label, y_label, cbar_label, out_path, n_grid=n_grid, verbose=verbose)


# ### Common

# In[37]:


# All objects

ra = dd['XWIN_WORLD']
dec = dd['YWIN_WORLD']

Apix = 1 # [arcmin^2]
title = 'All objects'
out_name = 'object_number_count_ngmix.png'

cbar_label = '{}, $A_{{\\rm pix}} \\approx {:.1g}$ arcmin$^2$'.format(cbar_label_base, Apix)
out_path = '{}/{}'.format(plot_dir, out_name)
n_grid = int(np.sqrt(area_amin2) / Apix)
if verbose:
    print('Number of pixels = {}^2'.format(n_grid))
plot_spatial_density(ra, dec, title, x_label, y_label, cbar_label, out_path, n_grid=n_grid, verbose=verbose)


# ## Metacalibration for stars

# ### ngmix

# In[38]:


star_metacal_ngmix = metacal(dd[ind_star], m_star_ngmix, masking_type='star')


# #### Number density

# In[39]:


# mask for 'no shear' images
mask_ns_stars = star_metacal_ngmix.mask_dict['ns']

n_star_ngmix = len(star_metacal_ngmix.ns['g1'][mask_ns_stars])

print_stats('ngmix:', stats_file, verbose=verbose)
print_stats('Number of stars = {}'.format(n_star_ngmix), stats_file, verbose=verbose)
print_stats('Star density = {:.2f} stars/deg2'.format(n_star_ngmix / area_deg2), stats_file, verbose=verbose)


# ### galsim

# In[40]:


star_metacal_galsim = metacal(dd[ind_star], m_star_galsim, masking_type='star')


# #### Number density

# In[41]:


# mask for 'no shear' images
mask_ns_stars_galsim = star_metacal_galsim.mask_dict['ns']

n_star_galsim = len(star_metacal_galsim.ns['g1'][mask_ns_stars_galsim])

print_stats('galsim:', stats_file, verbose=verbose)
print_stats('Number of stars = {}'.format(n_star_galsim), stats_file, verbose=verbose)
print_stats('Star density = {:.2f} stars/deg2'.format(n_star_galsim / area_deg2), stats_file, verbose=verbose)


# ## Additive bias

# In[42]:


n_jack = 500

print_stats('additive bias', stats_file, verbose=verbose)


# In[43]:


print_stats('ngmix:', stats_file, verbose=verbose)

c_ngmix = np.zeros(2)
c_err_ngmix = np.zeros(2)
for comp in (0, 1):
    c_ngmix[comp], c_err_ngmix[comp] = jackknif_weighted_average(g_corr_ngmix[comp], w_ngmix, remove_size=0.05, n_realization=n_jack)
    c_dc = ufloat(c_ngmix[comp], c_err_ngmix[comp])
    print_stats('c_{} = {:.2eP}'.format(comp+1, c_dc), stats_file, verbose=verbose)


# In[44]:


print_stats('galsim:', stats_file, verbose=verbose)

c_galsim = np.zeros(2)
c_err_galsim = np.zeros(2)
for comp in (0, 1):
    c_galsim[comp], c_err_galsim[comp] = jackknif_weighted_average(g_corr_galsim[comp], np.ones_like(w_galsim),
                                                                   remove_size=0.05, n_realization=n_jack)
    c_dc = ufloat(c_galsim[comp], c_err_galsim[comp])
    print_stats('c_1 = {:.2eP}'.format(c_dc), stats_file, verbose=verbose)


# ## Response matrix

# ### Mean

# In[45]:


print_stats('ngmix galaxies:', stats_file, verbose=verbose)

print_stats('total response matrix:', stats_file, verbose=verbose)
rs = np.array2string(gal_metacal_ngmix.R)
print_stats(rs, stats_file, verbose=verbose)

print_stats('shear response matrix:', stats_file, verbose=verbose)
R_shear_ngmix = np.mean(gal_metacal_ngmix.R_shear, 2)
rs = np.array2string(R_shear_ngmix)
print_stats(rs, stats_file, verbose=verbose)

print_stats('selection response matrix:', stats_file, verbose=verbose)
rs = np.array2string(gal_metacal_ngmix.R_selection)
print_stats(rs, stats_file, verbose=verbose)


# In[46]:


print_stats('ngmix stars:', stats_file, verbose=verbose)

print_stats('total response matrix:', stats_file, verbose=verbose)
rs = np.array2string(star_metacal_ngmix.R)
print_stats(rs, stats_file, verbose=verbose)

print_stats('shear response matrix:', stats_file, verbose=verbose)
R_shear_stars_ngmix = np.mean(star_metacal_ngmix.R_shear, 2)
rs = np.array2string(R_shear_stars_ngmix)
print_stats(rs, stats_file, verbose=verbose)

print_stats('selection response matrix:', stats_file, verbose=verbose)
rs = np.array2string(star_metacal_ngmix.R_selection)
print_stats(rs, stats_file, verbose=verbose)


# In[47]:


print_stats('galsim galaxies:', stats_file, verbose=verbose)

print_stats('total response matrix:', stats_file, verbose=verbose)
rs = np.array2string(gal_metacal_galsim.R)
print_stats(rs, stats_file, verbose=verbose)

print_stats('shear response matrix:', stats_file, verbose=verbose)
R_shear_galsim = np.mean(gal_metacal_galsim.R_shear, 2)
rs = np.array2string(R_shear_galsim)
print_stats(rs, stats_file, verbose=verbose)

print_stats('selection response matrix:', stats_file, verbose=verbose)
rs = np.array2string(gal_metacal_ngmix.R_selection)
print_stats(rs, stats_file, verbose=verbose)


# In[48]:


print_stats('galsim stars:', stats_file, verbose=verbose)

print_stats('total response matrix:', stats_file, verbose=verbose)
rs = np.array2string(star_metacal_galsim.R)
print_stats(rs, stats_file, verbose=verbose)

print_stats('shear response matrix:', stats_file, verbose=verbose)
R_shear_stars_galsim = np.mean(star_metacal_galsim.R_shear, 2)
rs = np.array2string(R_shear_stars_galsim)
print_stats(rs, stats_file, verbose=verbose)

print_stats('selection response matrix:', stats_file, verbose=verbose)
rs = np.array2string(star_metacal_galsim.R_selection)
print_stats(rs, stats_file, verbose=verbose)


# ### Plot distribution of response matrix elements

# In[49]:


title = 'ngmix'
x_label = 'response matrix element'
y_label = 'Frequency'
x_range = (-3, 3)
n_bin = 500


# In[50]:


colors = ['blue', 'red','blue', 'red']
linestyles = ['-', '-', ':', ':'] 


# In[51]:


print_stats('ngmix:', stats_file, verbose=verbose)

xs = [gal_metacal_ngmix.R_shear[0,0],
      gal_metacal_ngmix.R_shear[1,1],
      star_metacal_ngmix.R_shear[0,0],
      star_metacal_ngmix.R_shear[1,1]
     ]
labels = ['$R_{11}$ galaxies',
          '$R_{22}$ galaixes',
          '$R_{11}$ stars',
          '$R_{22}$ stars'
         ]

out_name = 'R_ngmix_diag.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                colors=colors, linestyles=linestyles)


# In[52]:


print_stats('ngmix:', stats_file, verbose=verbose)

xs = [gal_metacal_ngmix.R_shear[0,1],
      gal_metacal_ngmix.R_shear[1,0],
      star_metacal_ngmix.R_shear[0,1],
      star_metacal_ngmix.R_shear[1,0]
     ]
labels = ['$R_{12}$ galaxies',
          '$R_{21}$ galaixes',
          '$R_{12}$ stars',
          '$R_{21}$ stars'
         ]

out_name = 'R_ngmix_offdiag.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                colors=colors, linestyles=linestyles)


# In[53]:


title = 'galsim'


# In[54]:


print_stats('galsim:', stats_file, verbose=verbose)

xs = [gal_metacal_galsim.R_shear[0,0],
      gal_metacal_galsim.R_shear[1,1],
      star_metacal_galsim.R_shear[0,0],
      star_metacal_galsim.R_shear[1,1]
     ]
labels = ['$R_{11}$ galaxies',
          '$R_{22}$ galaixes',
          '$R_{11}$ stars',
          '$R_{22}$ stars'
         ]

out_name = 'R_galsim_diag.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                colors=colors, linestyles=linestyles)


# In[55]:


print_stats('galsim:', stats_file, verbose=verbose)

xs = [gal_metacal_galsim.R_shear[0,1],
      gal_metacal_galsim.R_shear[1,0],
      star_metacal_galsim.R_shear[0,1],
      star_metacal_galsim.R_shear[1,0]
     ]
labels = ['$R_{12}$ galaxies',
          '$R_{21}$ galaixes',
          '$R_{12}$ stars',
          '$R_{21}$ stars'
         ]

out_name = 'R_galsim_offdiag.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                colors=colors, linestyles=linestyles)


# ## Ellipticities

# In[56]:


x_label = 'ellipticity'
y_label = 'Frequency'
x_range = (-1, 1)
n_bin = 500

labels = ['$e_1$', '$e_2$']
colors = ['blue', 'red']
linestyles = ['-', '-'] 


# In[57]:


print_stats('ngmix galaxies:', stats_file, verbose=verbose)

xs = [g_corr_ngmix[0], g_corr_ngmix[1]]
weights = [w_ngmix] * 2

title = 'ngmix galaxies'
out_name = 'ell_gal_ngmix.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                weights=weights, colors=colors, linestyles=linestyles)


# In[58]:


print_stats('galsim galaxies:', stats_file, verbose=verbose)

xs = [g_corr_galsim[0], g_corr_galsim[1]]
weights = [w_galsim] * 2

title = 'galsim galaxies'
out_name = 'ell_gal_galsim.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                weights=weights, colors=colors, linestyles=linestyles)


# In[59]:


print_stats('ngmix stars:', stats_file, verbose=verbose)

xs = [star_metacal_ngmix.ns['g1'][mask_ns_stars], star_metacal_ngmix.ns['g2'][mask_ns_stars]]
weights = [star_metacal_ngmix.ns['w'][mask_ns_stars]] * 2

title = 'ngmix stars'
out_name = 'ell_stars_ngmix.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                weights=weights, colors=colors, linestyles=linestyles)


# In[60]:


print_stats('galsim stars:', stats_file, verbose=verbose)

xs = [star_metacal_galsim.ns['g1'][mask_ns_stars_galsim], star_metacal_galsim.ns['g2'][mask_ns_stars_galsim]]
weights = [star_metacal_galsim.ns['w'][mask_ns_stars_galsim]] * 2
           
title = 'galsim stars'
out_name = 'ell_stars_galsim.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                weights=weights, colors=colors, linestyles=linestyles)


# In[61]:


x_range = (-0.15, 0.15)
n_bin = 250


# In[62]:


print_stats('ngmix PSF (uncorrected):', stats_file, verbose=verbose)

xs = [dd['NGMIX_ELL_PSFo_NOSHEAR'][:,0][mask_ns_stars], dd['NGMIX_ELL_PSFo_NOSHEAR'][:,1][mask_ns_stars]]
#weights = [star_metacal_ngmix.ns['w'][mask_ns_stars]] * 2

title = 'ngmix PSF'
out_name = 'ell_PSF_ngmix.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                colors=colors, linestyles=linestyles)


# In[63]:


xs = [dd['GALSIM_PSF_ELL_ORIGINAL_PSF'][:,0][mask_ns_stars_galsim], dd['GALSIM_PSF_ELL_ORIGINAL_PSF'][:,1][mask_ns_stars_galsim]]
#weights = [star_metacal_ngmix.ns['w'][mask_ns_stars]] * 2

title = 'galsim PSF'
out_name = 'ell_PSF_galsim.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                colors=colors, linestyles=linestyles)


# ## Magnitudes

# In[64]:


x_label = '$r$-band magnitude'
y_label = 'Frequency'
x_range = (19.8, 25.5)
n_bin = 500

labels = ['ngmix', 'galsim']
colors = ['blue', 'red']
linestyles = ['-', '-']


# In[65]:


print_stats('galaxies:', stats_file, verbose=verbose)

xs = [dd['MAG_AUTO'][m_gal_ngmix][mask], dd['MAG_AUTO'][m_gal_galsim][mask_galsim]]

title = 'galaxies'
out_name = 'mag_gal.pdf'
out_path = os.path.join(plot_dir, out_name)

plot_histograms(xs, labels, title, x_label, y_label, x_range, n_bin, out_path,
                colors=colors, linestyles=linestyles)


# ## Ellipticity dispersion

# In[66]:


print_stats('ngmix:', stats_file, verbose=verbose)

sig_eps = np.sqrt(np.var(g_corr_ngmix[0]) + np.var(g_corr_ngmix[1]))
print_stats('Dispersion of complex ellipticity = {:.3f}'             ''.format(sig_eps), stats_file, verbose=verbose)
print_stats('Dispersion of (average) single-component ellipticity = {:.3f} = {:.3f} / sqrt(2)'             ''.format(sig_eps /  np.sqrt(2), sig_eps), stats_file, verbose=verbose)


# In[67]:


print_stats('galsim:', stats_file, verbose=verbose)

sig_eps = np.sqrt(np.var(g_corr_galsim[0]) + np.var(g_corr_galsim[1]))
print_stats('Dispersion of complex ellipticity = {:.3f}'             ''.format(sig_eps), stats_file, verbose=verbose)
print_stats('Dispersion of (average) single-component ellipticity = {:.3f} = {:.3f} / sqrt(2)'             ''.format(sig_eps /  np.sqrt(2), sig_eps), stats_file, verbose=verbose)


# In[ ]:





# coding: utf-8

# # Weak-lensing galaxy shape catalogue validation 1
# 
# ## PSF leakage
# 
# Contents
# - Linear fit PSF leakage
# - Scale-dependent PSF leakage alpha
# - Cross-correlation function xi_sys

# In[68]:


import os
from uncertainties import ufloat
from shapepipe.utilities.canfar import download


# In[69]:


sp_base= f"{os.environ['HOME']}/astro/repositories/github/sp_validation"

from sp_validation.survey import *
from sp_validation.util import *
from sp_validation.basic import *
from sp_validation.plots import *
from sp_validation.galaxy import *

# The following commands will be replaced by import instructions, once the sp_validation scripts are stable
for sc in ['survey', 'io', 'cat', 'basic', 'util', 'plot_style', 'plots', 'galaxy', 'correlation', 'cosmology']:
    script = os.path.join(sp_base, 'sp_validation', sc)
    get_ipython().run_line_magic('run', '$script')


# ### Linear fit
# Fit galaxy ellipticity as function of (binned) PSF ellipticity.

# In[70]:


n_bin = 30


# #### ngmix

# In[71]:


plot_dir_leakage = '{}/psf_leak_ngmix'.format(plot_dir)

e1 = g_corr_ngmix[0] - c_ngmix[0]
e2 = g_corr_ngmix[1] - c_ngmix[1]
psf_e1 = dd['NGMIX_ELL_PSFo_NOSHEAR'][:,0][m_gal_ngmix][mask]
psf_e2 = dd['NGMIX_ELL_PSFo_NOSHEAR'][:,1][m_gal_ngmix][mask]
psf_fwhm = T_to_fwhm(dd['NGMIX_T_PSFo_NOSHEAR'][m_gal_ngmix][mask])
weights = w_ngmix

psf_e_corr(e1, e2, psf_e1, psf_e2, psf_fwhm,
           weights=weights, 
           n_bin=n_bin,
           save_plot=True,
           plot_dir=plot_dir_leakage)


# In[72]:


plot_dir_leakage = '{}/psf_leak_galsim'.format(plot_dir)

e1 = g_corr_galsim[0] - c_galsim[0]
e2 = g_corr_galsim[1] - c_galsim[1]
psf_e1 = dd['GALSIM_PSF_ELL_ORIGINAL_PSF'][:,0][m_gal_galsim][mask_galsim]
psf_e2 = dd['GALSIM_PSF_ELL_ORIGINAL_PSF'][:,1][m_gal_galsim][mask_galsim]
psf_fwhm = sigma_to_fwhm(dd['GALSIM_PSF_SIGMA_ORIGINAL_PSF'][m_gal_galsim][mask_galsim], pixel_size=pixel_size)
weights = None

psf_e_corr(e1, e2, psf_e1, psf_e2, psf_fwhm,
           weights=weights,
           n_bin=n_bin,
           save_plot=True,
           plot_dir=plot_dir_leakage)


# ### Scale-dependent leakage
# Compute scale-dependent leakage $\alpha(\theta)$ from cross-correlations betwen galaxy and PSF ellipticities,
# $$
#     \alpha(\theta) = \frac{\xi_{+}^{\rm gp}(\theta) - \langle e_{\rm gal} \rangle^{*} \langle e_{\rm PSF} \rangle}{\xi_{+}^{\rm pp}(\theta) - |\langle e_{\rm PSF} \rangle|^{2}}
# $$
# With g = galaxy and p = PSF.

# #### ngmix

# In[73]:


print_stats('ngmix:', stats_file, verbose=verbose)

e1_gal = g_corr_ngmix[0] - c_ngmix[0]
e2_gal = g_corr_ngmix[1] - c_ngmix[1]
weights = w_ngmix
e1_star = g_star_psf_ngmix[0]
e2_star = g_star_psf_ngmix[1]

# Correlation functions
r_corr_gp_ngmix, r_corr_pp_ngmix = correlation_12_22(ra_ngmix, dec_ngmix, e1_gal, e2_gal, weights,
                                                     ra_star_ngmix, dec_star_ngmix, e1_star, e2_star)

# Leakage
alpha_leak_ngmix, sig_alpha_leak_ngmix = alpha(r_corr_gp_ngmix, r_corr_pp_ngmix,
                                               e1_gal, e2_gal, weights, e1_star, e2_star)


# In[74]:


alpha_leak_mean_ngmix = transform_nan(np.average(alpha_leak_ngmix, weights=1/sig_alpha_leak_ngmix**2))
print_stats('Weighted average alpha = {:.3g}'.format(alpha_leak_mean_ngmix), stats_file, verbose=verbose)


# In[75]:


plot_dir_leakage = '{}/psf_leak_ngmix'.format(plot_dir)

x = [r_corr_gp_ngmix.meanr]
y = [alpha_leak_ngmix]
yerr = [sig_alpha_leak_ngmix]
xlabel = r'$\theta$ [arcmin]'
ylabel = r'$\alpha(\theta)$'
title = 'ngmix'
out_path = f'{plot_dir_leakage}/alpha_leakage_ngmix.png'
try:
    ylim  = leakage_alpha_ylim
except:
    ylim = None

plot_data_1d(x, y, yerr, title, xlabel, ylabel, out_path, xlog=True, ylim=ylim)


# #### galsim

# In[76]:


print_stats('galsim:', stats_file, verbose=verbose)

e1_gal = g_corr_galsim[0] - c_galsim[0]
e2_gal = g_corr_galsim[1] - c_galsim[1]
weights = w_galsim
e1_star = g_star_psf_galsim[0]
e2_star = g_star_psf_galsim[1]

# Correlation functions
r_corr_gp_galsim, r_corr_pp_galsim = correlation_12_22(ra_galsim, dec_galsim, e1_gal, e2_gal, weights,
                                                       ra_star_galsim, dec_star_galsim, e1_star, e2_star)

# Leakage
alpha_leak_galsim, sig_alpha_leak_galsim = alpha(r_corr_gp_galsim, r_corr_pp_galsim,
                                                 e1_gal, e2_gal, weights, e1_star, e2_star)

alpha_leak_mean_galsim = np.average(alpha_leak_galsim, weights=1/sig_alpha_leak_galsim**2)
print_stats('Weighted average alpha = {:.3g}'.format(alpha_leak_mean_galsim), stats_file, verbose=verbose)


# In[77]:


plot_dir_leakage = '{}/psf_leak_galsim'.format(plot_dir)

x = [r_corr_gp_galsim.meanr]
y = [alpha_leak_galsim]
yerr = [sig_alpha_leak_galsim]
xlabel = r'$\theta$ [arcmin]'
ylabel = r'$\alpha(\theta)$'
title = 'galsim'
out_path = '{}/alpha_leakage_galsim.png'.format(plot_dir_leakage)
try:
    ylim  = leakage_alpha_ylim
except:
    ylim = None

plot_data_1d(x, y, yerr, title, xlabel, ylabel, out_path, xlog=True, ylim=ylim)


# ### Cross-correlation PSF leakage
# Compute $\xi^{\rm sys}$, the normalised cross-correlation function of galaxy and star ellipticities:
# $$
# \xi^{\rm sys}(\theta) = \frac{\left( \xi_{\pm}^{\rm gp}(\theta) \right)^2}{\xi_{\pm}^{\rm pp}(\theta)}
# $$   
# With $g$: galaxy and $p$: PSF.

# In[78]:


C_sys_p_ngmix = r_corr_gp_ngmix.xip**2 / r_corr_pp_ngmix.xip
C_sys_m_ngmix = r_corr_gp_ngmix.xim**2 / r_corr_pp_ngmix.xim

C_sys_std_p_ngmix = np.abs(C_sys_p_ngmix) * np.sqrt((((2*r_corr_gp_ngmix.xip**2 * np.sqrt(r_corr_gp_ngmix.varxip))/r_corr_gp_ngmix.xip)/r_corr_gp_ngmix.xip**2.)**2 + (np.sqrt(r_corr_pp_ngmix.varxip)/r_corr_pp_ngmix.xip)**2.)
C_sys_std_m_ngmix = np.abs(C_sys_m_ngmix) * np.sqrt((((2*r_corr_gp_ngmix.xim**2 * np.sqrt(r_corr_gp_ngmix.varxim))/r_corr_gp_ngmix.xim)/r_corr_gp_ngmix.xim**2.)**2 + (np.sqrt(r_corr_pp_ngmix.varxim)/r_corr_pp_ngmix.xim)**2.)


# #### Theoretical prediction

# In[79]:


# Redshift distribution

nz_base = 'nz.CFHTLenSmatched.W3'
nz_ext = 'txt'
nz_date = '202012'
nz_version = 'v1'
nz_name = '{}_{}_{}.{}'.format(nz_base, nz_date, nz_version, nz_ext)
source = 'vos:cfis/cosmostat/cosmology/redshifts/{}'.format(nz_name)

download(source, nz_name, verbose=True)

z, nz = np.loadtxt(nz_name, unpack=True)


# In[80]:


# Cosmological parameters
Om = 0.3153
sig8 = 0.8111
ns = 0.9649
Ob = 0.0493
h = 0.6736

try:
    xi_p_planck, xi_m_planck = get_theo_xi(r_corr_gp_ngmix.meanr, z, nz, Omega_m=Om, h=h, Omega_b=Ob, sig8=sig8, ns=ns)
except:
    print('Computation of theoretical prediction failed')


# In[81]:


#### ngmix


# In[82]:


plot_dir_leakage = '{}/psf_leak_ngmix'.format(plot_dir)

x = [r_corr_gp_ngmix.meanr] * 2
y = [C_sys_p_ngmix, C_sys_m_ngmix]
yerr = [C_sys_std_p_ngmix, C_sys_std_m_ngmix]
labels = ['$\\xi^{\\rm sys}_+$', '$\\xi^{\\rm sys}_-$']

title = 'Cross-correlation leakage'
xlabel = '$\\theta$ [arcmin]'
ylabel = 'Correlation function'

try:
    ylim = leakage_xi_sys_ylim
except:
    ylim = None
out_path = '{}/xi_sys_ngmix.pdf'.format(plot_dir)
plot_data_1d(x, y, yerr, title, xlabel, ylabel, out_path, xlog=True, ylim=ylim, labels=labels)

try:
    ylim = leakage_xi_sys_log_ylim
except:
    ylim = None
out_path = '{}/xi_sys_log_ngmix.pdf'.format(plot_dir)
plot_data_1d(x, y, yerr, title, xlabel, ylabel, out_path, xlog=True, ylog=True, ylim=ylim, labels=labels)

