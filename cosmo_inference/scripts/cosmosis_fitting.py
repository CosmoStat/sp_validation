#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
import sys

#transforms treecorr fits file of correlation functions into CosmoSIS-friendly 2pt FITS extension to be read by 2pt_likelihood
def treecorr_to_fits(filename1,filename2):
    
    xiplus_hdu = fits.open(filename1)
    ximinus_hdu = fits.open(filename2)
    
    return xiplus_hdu[1],ximinus_hdu[1]

def tau_to_fits(filename):

    tau_stats = fits.getdata(filename)

    ang = tau_stats['theta']
    nbins = len(ang)
    lst = np.arange(1, nbins+1)

    #Create fits HDU for tau_0_+
    col1 = fits.Column(name ='BIN1', format ='K', array = np.ones(len(lst)))
    col2 = fits.Column(name ='BIN2', format ='K', array = np.ones(len(lst)))
    col3 = fits.Column(name ='ANGBIN', format ='K', array = lst)
    col4 = fits.Column(name = 'VALUE', format = 'D', array = tau_stats['tau_0_p'])
    col5 = fits.Column(name = 'ANG', format = 'D', unit='arcmin', array = ang)
    coldefs = fits.ColDefs([col1,col2,col3,col4,col5])
    tau_0_p_hdu = fits.BinTableHDU.from_columns(coldefs,name ='TAU_0_PLUS')

    #Create fits HDU for tau_2_+
    col4 = fits.Column(name = 'VALUE', format = 'D', array = tau_stats['tau_2_p'])
    coldefs = fits.ColDefs([col1,col2,col3,col4,col5])
    tau_2_p_hdu = fits.BinTableHDU.from_columns(coldefs,name ='TAU_2_PLUS')

    #Append tau_0_p/tau_2_p header info
    tau_0_p_dict = {
        '2PTDATA' : 'T',
        'QUANT1' : 'G+R',
        'QUANT2' : 'P+R',
        'KERNEL_1': 'NZ_SOURCE',
        'KERNEL_2': 'NZ_SOURCE',
        'WINDOWS': 'SAMPLE'
    }

    for key in tau_0_p_dict:
        tau_0_p_hdu.header[key]=tau_0_p_dict[key]

    tau_2_p_dict = {
        '2PTDATA' : 'T',
        'QUANT1' : 'G+R',
        'QUANT2' : 'SR+R',
        'KERNEL_1': 'NZ_SOURCE',
        'KERNEL_2': 'NZ_SOURCE',
        'WINDOWS': 'SAMPLE'
    }

    for key in tau_2_p_dict:
        tau_2_p_hdu.header[key]=tau_2_p_dict[key]

    return tau_0_p_hdu, tau_2_p_hdu



#transforms text file of CosmoCov data into covmat HDU extension
def covdat_to_fits(filename_cov_xi, filename_cov_tau=None):
    
    #read in cov txt data from CosmoCov
    covmat_xi = np.loadtxt(filename_cov_xi)
    if filename_cov_tau is not None:
        covmat_tau = np.load(filename_cov_tau)
        nbins = int(len(covmat_tau)/3)
        covmat_tau = covmat_tau[:2*nbins, :2*nbins]
        covmat = np.block([[covmat_xi, np.zeros((len(covmat_xi), len(covmat_tau)))],
                            [np.zeros((len(covmat_tau), len(covmat_xi))), covmat_tau]])
    
    if len(covmat) != len(covmat[0]):
        print('Error: covmat not square!')
        exit()
     
    else:
        #create covmat ImageHDU
        cov_hdu = fits.ImageHDU(covmat)
        
        #create header
        if filename_cov_tau is None:
            cov_dict = {'COVDATA': 'True', 
                        'EXTNAME': 'COVMAT',
                        'NAME_0':'XI_PLUS',
                        'STRT_0': 0,
                        'NAME_1':'XI_MINUS',
                        'STRT_1' : int(len(covmat)/2)}
        else:
            cov_dict = {'COVDATA': 'True',
                        'EXTNAME': 'COVMAT',
                        'NAME_0':'XI_PLUS',
                        'STRT_0': 0,
                        'NAME_1':'XI_MINUS',
                        'STRT_1' : int(len(covmat_xi)/2),
                        'NAME_2':'TAU_0_PLUS',
                        'STRT_2': len(covmat_xi),
                        'NAME_3':'TAU_2_PLUS',
                        'STRT_3': len(covmat_xi)+int(len(covmat_tau)/2)}
        for key in cov_dict:
            cov_hdu.header[key] = cov_dict[key] 

    
    return cov_hdu


#transforms nz data (that was used in CosmoCov format) into nzdat HDU extension
def nz_to_fits(filename):
    

    line= np.loadtxt(filename, max_rows=1)
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

def rho_to_fits(filename):
    rho_stat_hdu = fits.open(filename)
    rho_stat_hdu[1].name = "RHO_STATS"
    return rho_stat_hdu[1]


if __name__ == "__main__":
    
#combines all the data: 2pt correlation functions from treecorr, covmat from CosmoCov (must already be combined into 1 txt file), nz txt data 
#into 1 fits file to be read by CosmoSIS 2pt-likelihood function
#give file path of each of the 3 components as input, also file path of desired output FITS file
#outputs nothing, but writes a new FITS file with appropriate extensions
    root = sys.argv[1]
    output_folder = sys.argv[2]
    
    two_pt_file_xip = output_folder+'/xi_plus_'+root+'.fits'
    two_pt_file_xim = output_folder+'/xi_minus_'+root+'.fits'
    cov_xi_file = sys.argv[3]        #in cosmocov combined txt format
    nz_file = sys.argv[4]         #in cosmocov format
    rho_stats_file = output_folder+'/rho_tau_stats/rho_stats_'+root+'.fits'
    out_file = sys.argv[5]
    use_tau_stats = sys.argv[6]
    use_tau_stats = True if use_tau_stats == 'y' else False
    tau_stats_file = output_folder +'rho_tau_stats/tau_stats_'+root+'.fits' if use_tau_stats else None
    cov_tau_file = output_folder + 'rho_tau_stats/cov_tau_'+root+'_th.npy' if use_tau_stats else None

    
    #create the required FITS extensions
    print("Creating 2PT fits extension...\n")
    if not (os.path.exists(two_pt_file_xip) and os.path.exists(two_pt_file_xim)):
        raise FileNotFoundError("2pt files not found. Please run cosmo_val.py first.")
    xip_hdu, xim_hdu = treecorr_to_fits(two_pt_file_xip, two_pt_file_xim)
    if not os.path.exists(tau_stats_file):
        raise Warning("Tau stats file not found. Please run cosmo_val.py first. Creating the FITS file without rho and tau statistics.")
        use_tau_stats= False
        cov_tau_file = None
    if use_tau_stats:
        print('Creating rho stats fits extension...\n')
        rho_hdu = rho_to_fits(rho_stats_file)
        print("Creating tau fits extensions...\n")
        tau_0_p_hdu, tau_2_p_hdu = tau_to_fits(tau_stats_file)
    print("Creating CovMat fits extension...\n")
    cov_hdu = covdat_to_fits(cov_xi_file, cov_tau_file)
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
    if use_tau_stats:
        hdu_list = [pri_hdu, cov_hdu, nz_hdu, xip_hdu, xim_hdu, tau_0_p_hdu, tau_2_p_hdu, rho_hdu]
    else:
        hdu_list = [pri_hdu, cov_hdu, nz_hdu, xip_hdu, xim_hdu]
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(out_file,overwrite=True)
    print("FITS file written out to %s" %out_file)
    




