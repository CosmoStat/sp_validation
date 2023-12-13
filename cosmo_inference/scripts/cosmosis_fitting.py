#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import sys

#transforms treecorr fits file of correlation functions into CosmoSIS-friendly 2pt FITS extension to be read by 2pt_likelihood
def treecorr_to_fits(xipm_file,root):
    
    xiplus_hdu = fits.open(xipm_file + '/xi_plus.fits')
    ximinus_hdu = fits.open(xipm_file + '/xi_minus.fits')
    
    return xiplus_hdu[1],ximinus_hdu[1]

#transforms text file of CosmoCov data into covmat HDU extension
def covdat_to_fits(filename):
    
    #read in cov txt data from CosmoCov
    covmat = np.loadtxt(filename)
    
    if len(covmat) != len(covmat[0]):
        print('Error: covmat not square!')
        exit()
     
    else:
        #create covmat ImageHDU
        cov_hdu = fits.ImageHDU(covmat)
        
        #create header
        cov_dict = {'COVDATA': 'True', 
                    'EXTNAME': 'COVMAT',
                    'NAME_0':'XI_PLUS',
                    'STRT_0': 0,
                    'NAME_1':'XI_MINUS',
                    'STRT_1' : int(len(covmat)/2)}
        for key in cov_dict:
            cov_hdu.header[key] = cov_dict[key] 

    
    return cov_hdu

#transforms nz data (that was used in CosmoCov format) into nzdat HDU extension
def nz_to_fits(filename):
    

    line= np.loadtxt(filename)
    nbins = len(line)-1
    
    z_low = np.loadtxt(filename, usecols=0)
    
    nstep = z_low[1] - z_low[0]
    
    z_mid = z_low + nstep/2
    z_high = np.append(z_low[1:],z_low[-1]+nstep)
    
    #create hdu for histogram bin 
    col1 = fits.Column(name ='Z_LOW', format ='D', array = z_low)
    col2 = fits.Column(name ='Z_MID', format ='D', array = z_mid)
    col3 = fits.Column(name ='Z_HIGH', format ='D', array = z_high) 
    cols = [col1,col2,col3]
    
    for i in range(nbins):
        bin_col = np.loadtxt(filename, usecols=i+1)
        hdu_col = fits.Column(name ='BIN%d' %(i+1), format ='D', array = bin_col)
        cols.append(hdu_col)
        
    coldefs = fits.ColDefs(cols)
    nz_hdu = fits.BinTableHDU.from_columns(coldefs,name ='NZDATA')
    
    #create n(z) header
    nz_lens_dict={'NZDATA' : 'T  ',                                                
                'EXTNAME' : 'NZ_SOURCE',
                'NBIN'    : nbins,
                'NZ'      : len(z_low) }

    for key in nz_lens_dict:
        nz_hdu.header[key]=nz_lens_dict[key]
        
    return nz_hdu


if __name__ == "__main__":
    
    
#combines all the data: 2pt correlation functions from treecorr, covmat from CosmoCov (must already be combined into 1 txt file), nz txt data 
#into 1 fits file to be read by CosmoSIS 2pt-likelihood function
#give file path of each of the 3 components as input, also file path of desired output FITS file
#outputs nothing, but writes a new FITS file with appropriate extensions

    root = sys.argv[1]
    two_pt_file = sys.argv[2]
    cov_file = sys.argv[3]        #in cosmocov combined txt format
    nz_file = sys.argv[4]         #in cosmocov format
    out_file = sys.argv[5] 

    
    #create the required FITS extensions
    print("Creating 2PT fits extension...\n")
    xip_hdu, xim_hdu = treecorr_to_fits(two_pt_file,root)
    print("Creating CovMat fits extension...\n")
    cov_hdu = covdat_to_fits(cov_file)
    print("Creating n(z) fits extension...\n")
    nz_hdu = nz_to_fits(nz_file)
    
    
    #create header for primary HDU
    pri_hdr_dict = {}
    
    pri_hdr = fits.Header()
    for key in pri_hdr_dict:
        pri_hdr[key]=pri_hdr_dict[key]
    
    #create primary HDU
    pri_hdu = fits.PrimaryHDU(header=pri_hdr)
    
    #create final FITS HDU
    print("Writing out combined FITS file...\n")
    hdul = fits.HDUList([pri_hdu, cov_hdu, nz_hdu, xip_hdu, xim_hdu])
    hdul.writeto(out_file,overwrite=True)
    print("FITS file written out to %s" %out_file)
    