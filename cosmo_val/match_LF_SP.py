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

# # Match UNIONS ShapePipe and LensFit catalogues

# !pip install -U scikit-learn

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import match_coordinates_sky
import csv
import astropy.units as u
from astropy.coordinates import SkyCoord
from uncertainties import ufloat   
from sklearn import preprocessing

from sp_validation import cat

# ## Set-up

# Maximum separation between objects defined as match
max_sep = 0.5 * u.arcsec

# +
# Input UNIONS data

sp_version = '1.0'
lf_version = '1'
version = sp_version

path_to_unions_data = '/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/'
#path_to_unions_data = f'{os.environ["HOME"]}/astro/data/UNIONS/v{version}'

# Input catalogues
sp_cat_name = f'{path_to_unions_data}/ShapePipe/unions_shapepipe_extended_2022_v{sp_version}.fits'
lf_cat_name = f'{path_to_unions_data}/Lensfit/lensfit_goldshape_2022v{lf_version}.fits'

# Input masks
lf_mask_name = f'{path_to_unions_data}/Lensfit/masks/CFIS3500_THELI_mask_hp_4096.fits'
sp_mask_name = f'{path_to_unions_data}/ShapePipe/masks/healpix/nside_1024/mask_all.fits'

# +
# Output directory
direc = os.environ['HOME'] + '/astro/data/CFIS/v1.0/matched_LF_SP'

# Masked catalogue file names (here used as in- and out-put)
path_sp_masked = f'{direc}/masked_{os.path.basename(sp_cat_name)}'
path_lf_masked = f'{direc}/masked_{os.path.basename(lf_cat_name)}'
# -

# ## Masking

# !./check_footprint.py -m {lf_mask_name} -i {sp_cat_name} -g 0 --output_path {path_sp_masked}

# !./check_footprint.py -m {sp_mask_name} -i {lf_cat_name} -g 1 --output_path {path_lf_masked}

# ## Read catalogues

hdu_list_sp = fits.open(f'{path_sp_masked}')
sp_data = hdu_list_sp[1].data
sp_header = hdu_list_sp[0].header

hdu_list_lf = fits.open(f'{path_lf_masked}')
lf_data = hdu_list_lf[1].data
lf_header = hdu_list_lf[0].header

# ## Matching

coord_units = u.degree

# +
# Restrict SP catalogue to LF tiles.
# After above masking should have no effect.

w_in_LF = (sp_data['mask_extern'] == 0)
sp_data_ok = sp_data[w_in_LF]
#sp_data_ok = sp_data
# -

print(len(sp_data), len(sp_data_ok), len(lf_data))

sp_sc = SkyCoord(
    ra=sp_data_ok['RA'] * coord_units,
    dec=sp_data_ok['Dec'] * coord_units,
)

lf_sc = SkyCoord(
    ra=lf_data['RA'] * coord_units,
    dec=lf_data['Dec'] * coord_units,
)

# Match SDSS to UNIONS
idx, d2d, d3d = match_coordinates_sky(lf_sc, sp_sc) 

#test to see if we understand well match_coordinates_sky
print(idx[0], lf_sc[0], sp_sc[idx[0]])

# +
sep_constraint = d2d < max_sep

lf_matches = lf_data[sep_constraint]
sp_matches = sp_data_ok[idx[sep_constraint]]
# -

print(len(lf_matches), len(sp_matches))

print(len(sp_matches) / len(sp_data_ok))

# #### Plot matching distance histograms

d2d_arcsec = d2d.to('arcsec').value
plt.hist(d2d_arcsec, bins=500, range=(0, 5), density=True, log=True)
plt.xlabel('distance [arcsec]')
_ = plt.ylabel('frequency')
plt.xlim(1e-3, 5)
_ = plt.ylim(1e-5, 5e1)

plt.hist(d2d_arcsec, bins=500, range=(0, 3), density=True)
plt.xlabel('distance [arcsec]')
_ = plt.ylabel('frequency')
_ = plt.ylim(0, 1)

# Normalise weights
lf_w_n = preprocessing.normalize([lf_matches['w']])
sp_w_n = preprocessing.normalize([sp_matches['w']])

# Pearson correlation coefficient
r = stats.pearsonr(lf_matches['w'], sp_matches['w'])
r_n = stats.pearsonr(lf_w_n[0], sp_w_n[0])

print(r, r_n)

# +
nmax = -1 #10_000_000
fig, ax = plt.subplots()
ax.scatter(lf_matches['w'][:nmax], sp_matches['w'][:nmax], s=0.0001)

plt.title('Weights')
ax.set_xlabel('LensFit $w$')
ax.set_ylabel('ShapePipe $w$')

#ax.set_xlim(0, w_max)
#ax.set_ylim(0, w_max)
ax.set_box_aspect(1)

plt.show()

# +
w_max = 0.00022
nmax = -1 #10_000_000
fig, ax = plt.subplots()
ax.scatter(lf_w_n[0][:nmax], sp_w_n[0][:nmax], s=0.0001)

ax.set_xlabel(r'LensFit $w_{\rm{n}}$')
ax.set_ylabel(r'ShapePipe $w_{\rm{n}}$')
plt.title('Normalised weights')

ax.set_xlim(0, w_max)
ax.set_ylim(0, w_max)
ax.set_box_aspect(1)

plt.show()

# +
# Check additive bias
cw = {}
cw_err = {}

cat_names = ['LF', 'LF_matched', 'SP', 'SP_matched']

for cn in cat_names:
    cw[cn] = np.zeros(2)
    cw_err[cn] = np.zeros(2)

# LF
for comp in (0, 1):
    cw['LF'][comp] = np.average(lf_data[f'e{comp+1}'], weights=lf_data['w'])
    variance = np.average(
        lf_data[f'e{comp+1}'] - cw['LF'][comp]**2,
        weights=lf_data['w'],
    )
    cw_err['LF'][comp] = np.sqrt(variance / len(lf_data))

for comp in (0, 1):
    cw['LF_matched'][comp] = np.average(
        lf_matches[f'e{comp+1}'],
        weights=lf_matches['w'],
    )
    variance = np.average(
        lf_matches[f'e{comp+1}'] - cw['LF_matched'][comp]**2,
        weights=lf_matches['w'],
    )
    cw_err['LF_matched'][comp] = np.sqrt(variance / len(lf_matches))

# SP
for comp in (0, 1):
    cw['SP'][comp] = np.average(sp_data[f'e{comp+1}_uncal'], weights=sp_data['w'])
    variance = np.average(
        (sp_data[f'e{comp+1}_uncal'] - cw['SP'][comp])**2,
        weights=sp_data['w'],
    )
    cw_err['SP'][comp] = np.sqrt(variance / len(sp_data))

for comp in (0, 1):
    cw['SP_matched'][comp] = np.average(
        sp_matches[f'e{comp+1}_uncal'],
        weights=sp_matches['w'],
    )
    variance = np.average(
        (sp_matches[f'e{comp+1}_uncal'] - cw['SP_matched'][comp])**2,
        weights=sp_matches['w'],
    )
    cw_err['SP_matched'][comp] = np.sqrt(variance / len(sp_matches))

# Print results
for cn in cat_names:
    for comp in (0, 1):
        c_dc = ufloat(cw[cn][comp], cw_err[cn][comp])
        print(f'{cn} c_{comp+1} = {c_dc:.3eP}')

# +
# Metacal

R_g = {}

# Checkcd v
for cn in ['SP', 'SP_matched']:
    R_g[cn] = np.empty(shape=(2, 2))

for idx in (0, 1):
    for jdx in (0, 1):
        R_g['SP'][idx, jdx] = np.mean(sp_data[f'R_g{idx+1}{jdx+1}'])
        R_g['SP_matched'][idx, jdx] = np.mean(sp_matches[f'R_g{idx+1}{jdx+1}'])
        
print('Shear response matrices')
for cn in ['SP', 'SP_matched']:
    print(cn)
    print(np.matrix(R_g[cn]))

# +
# Selection response matrix

hdu_sp_orig = fits.open(sp_cat_name)
sp_header = hdu_sp_orig[0].header
# -

R_s = np.empty(shape=(2, 2))
for idx in (0, 1):
    for jdx in (0, 1):
        R_s[idx][jdx] = sp_header[f'R_S{idx+1}{jdx+1}']

# +
R = R_g['SP_matched'] + R_s

print('Total response matrix for matched SP cat R =')
print(np.matrix(R))
# -

# Test corrected additive bias
R_SP = np.matrix(R_g['SP'] + R_s)
print(R_SP)
Rm1_SP = np.linalg.inv(R_SP)
c_corr = Rm1_SP.dot(cw['SP'])
print(cw['SP'])
print(c_corr)

# +
# Apply metacal

e_uncal_minus_c = np.array([
    sp_matches['e1_uncal'] - cw['SP_matched'][0],
    sp_matches['e2_uncal'] - cw['SP_matched'][1]
])
Rm1 = np.linalg.inv(R)
e_cal = Rm1.dot(e_uncal_minus_c)

# +
# Write matched catalogues

# LF
out_path_lf_masked_matched = f'{direc}/masked_matched_{os.path.basename(lf_cat_name)}'

ra = lf_matches['ra']
dec = lf_matches['dec']
g = [lf_matches['e1'], lf_matches['e2']]
w = lf_matches['w']
mag = lf_matches['mag']
R = np.array([[0, 0], [0, 0]])
R_shear = R
R_select = R
c = cw['LF_matched']
c_err = cw_err['LF_matched']

cat.write_shape_catalog(
    out_path_lf_masked_matched,
    ra,
    dec,
    g,
    w,
    mag,
    R,
    R_shear,
    R_select,
    c,
    c_err,
)   

# +
# SP
out_path_sp_masked_matched = f'{direc}/masked_matched_{os.path.basename(sp_cat_name)}'

ra = sp_matches['ra']
dec = sp_matches['dec']
g = e_cal
w = sp_matches['w']
mag = sp_matches['mag']
R_shear = R_g['SP_matched']
R_select = R
c = cw['SP_matched']
c_err = cw_err['SP_matched']

cat.write_shape_catalog(
    out_path_sp_masked_matched,
    ra,
    dec,
    g,
    w,
    mag,
    R,
    R_shear,
    R_select,
    c,
    c_err,
) 
# -


