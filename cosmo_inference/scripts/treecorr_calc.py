#!/usr/bin/env python
# coding: utf-8


import sys
import os
import numpy as np
from astropy.io import fits
import treecorr

script_dir = os.path.dirname(os.path.abspath(sys.argv[0])) 

cat_name = sys.argv[1]
root = sys.argv[2]

hdu = fits.open(cat_name)
data = hdu[1].data

# Create TreeCorr catalogue
n_thread = 8
treecorr.set_omp_threads(n_thread)

sep_units = 'arcmin'
nbins = 20

TreeCorrConfig = {
        'ra_units': 'degrees',
        'dec_units': 'degrees',
        'max_sep': '200',
        'min_sep': '1',
        'sep_units': sep_units,
        'nbins': nbins,
        'var_method':'jackknife',
    }


cat_gal = treecorr.Catalog(
    ra=data['RA'],
    dec=data['Dec'],
    g1=data['e1_noleakage'], # for v1.4.1
    g2=data['e2_noleakage'], # for v1.4.1
    w=data['w'],
    ra_units='degrees',
    dec_units='degrees',
    npatch=50
)

gg = treecorr.GGCorrelation(TreeCorrConfig)

print("Running TreeCorr...")
gg.process(cat_gal)


lst = np.arange(1,nbins+1)

#create fits HDU with xi_p and xi_m data
col1 = fits.Column(name ='BIN1', format ='K', array = np.ones(len(lst)))
col2 = fits.Column(name ='BIN2', format ='K', array = np.ones(len(lst)))
col3 = fits.Column(name ='ANGBIN', format ='K', array = lst)
col4 = fits.Column(name ='VALUE', format ='D', array = gg.xip)
col5 = fits.Column(name ='ANG', format ='D', unit ='arcmin', array = gg.meanr)
coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
xiplus_hdu = fits.BinTableHDU.from_columns(coldefs,name ='XI_PLUS')


col4 = fits.Column(name ='VALUE', format ='D', array = gg.xim)
coldefs = fits.ColDefs([col1, col2, col3, col4, col5])
ximinus_hdu = fits.BinTableHDU.from_columns(coldefs,name ='XI_MINUS')

#append xi_p/xi_m header info 
xip_dict = {'2PTDATA':'T',
            'QUANT1':'G+R',
             'QUANT2':'G+R',
             'KERNEL_1':'NZ_SOURCE',
             'KERNEL_2':'NZ_SOURCE',
             'WINDOWS':'SAMPLE'}
for key in xip_dict:
    xiplus_hdu.header[key] = xip_dict[key]


xim_dict = {'2PTDATA':'T',
            'QUANT1':'G-R',
             'QUANT2':'G-R',
             'KERNEL_1':'NZ_SOURCE',
             'KERNEL_2':'NZ_SOURCE',
             'WINDOWS':'SAMPLE'}

for key in xim_dict:
    ximinus_hdu.header[key] = xim_dict[key]

ximinus_hdu.writeto('%s/../data/' %script_dir+root+'/ximinus_'+root+'.fits',overwrite=True)
xiplus_hdu.writeto('%s/../data/' %script_dir+root+'/xiplus_'+root+'.fits',overwrite=True)

print('Correlation functions written to {}'.format('%s/../data/' %script_dir+root+'/xiplus_minus_'+root+'.fits'))