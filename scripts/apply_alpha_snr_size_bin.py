"""Script apply_alpha_snr_size_bin.py

Add PSF leakage to galaxy ellipticity using fitted model of alpha.
Follows the methodology from Li et al.(2024) to add the columns.
Write FITS file with added columns.

:Authors: Sacha Guerrini
inspired from the code of Shun-Sheng Li
"""

import sys
import pickle

from optparse import OptionParser                                               

from lmfit import Parameters
from astropy.io import fits
from astropy.table import Table, Column
import matplotlib.pyplot as plt

import statsmodels.api as sm

from cs_util import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
import time

def params_default():
    """Params Default

    Set default parameter values.

    Returns
    -------
    dict
        default parameter values

    """
    param = {
        'e1_col': 'e1',
        'e2_col': 'e2',
        'e1_PSF_col': 'e1_PSF',
        'e2_PSF_col': 'e2_PSF',
        'snr_col': 'snr',
        'w_col': 'w',
        'TPSF_col': 'NGMIX_Tpsf_NOSHEAR',
        'T_col': 'NGMIX_T_NOSHEAR',
        'input_path_shear': 'SP/unions_shapepipe_extended_2022_v1.0.fits',
        'output_path': 'shape_cat_cor_alpha.fits',
        'plot_path': None
    }

    return param


def parse_options(p_def):                                                       
    """Parse Options

    Parse command line options and update parameter values accordingly.
                                                                                
    Parameters
    ----------                                                                  
    p_def: dict
        default parameter values                                                        
                                                                                
    Returns                                                                     
    -------                                                                     
    tuple                                                              
        Command line options                                                    
    string                                                                
        Command line string                                                     

    """                                                                         
    usage  = "%prog [OPTIONS]"                                                  
    parser = OptionParser(usage=usage)

    parser.add_option(                                                          
        '-i',                                                                   
        '--input_path_shear',                                                   
        dest='input_path_shear',                                                
        default=p_def['input_path_shear'],
        type='string',                                                          
        help='input path of the extended shear catalogue'                       
    )
    parser.add_option(                                                          
        '',                                                                     
        '--e1_col',                                                             
        dest='e1_col',                                                          
        default=p_def['e1_col'],                                                   
        type='string',                                                          
        help=(
            'e1 column name in galaxy catalogue,'
            + f' default=\'{p_def["e1_col"]}\''  
        ),
    )                                                                           
    parser.add_option(                                                          
        '',                                                                     
        '--e2_col',                                                             
        dest='e2_col',                                                          
        default=p_def['e2_col'],                                                   
        type='string',                                                          
        help=(
            'e2 column name in galaxy catalogue,'
            + f' default=\'{p_def["e2_col"]}\''  
        ),
    )                                                                           
    parser.add_option(                                                          
        '',                                                                     
        '--e1_PSF_col',                                                         
        dest='e1_PSF_col',                                                      
        default=p_def['e1_PSF_col'],                                               
        type='string',                                                          
        help=f'PSF e1 column name, default=\'{p_def["e1_PSF_col"]}\''              
    )                                                                           
    parser.add_option(                                                          
        '',                                                                     
        '--e2_PSF_col',                                                         
        dest='e2_PSF_col',                                                      
        default=p_def['e2_PSF_col'],                                               
        type='string',                                                          
        help=f'PSF e2 column name, default=\'{p_def["e2_PSF_col"]}\''              
    )
    parser.add_option(                                                          
        '',                                                                     
        '--snr_col',                                                         
        dest='snr_col',                                                      
        default=p_def['snr_col'],                                               
        type='string',                                                          
        help=f'SNR column name, default=\'{p_def["snr_col"]}\''              
    )
    parser.add_option(                                                          
        '',                                                                     
        '--w_col',                                                         
        dest='w_col',                                                      
        default=p_def['w_col'],                                               
        type='string',                                                          
        help=f'w column name, default=\'{p_def["w_col"]}\''              
    )
    parser.add_option(                                                          
        '',                                                                     
        '--TPSF_col',                                                         
        dest='TPSF_col',                                                      
        default=p_def['TPSF_col'],                                               
        type='string',                                                          
        help=f'T PSF column name, default=\'{p_def["TPSF_col"]}\''              
    )
    parser.add_option(                                                          
        '',                                                                     
        '--T_col',                                                         
        dest='T_col',                                                      
        default=p_def['T_col'],                                               
        type='string',                                                          
        help=f'T_gal column name, default=\'{p_def["T_col"]}\''              
    )
    parser.add_option(                                                          
        '-o',
        '--output_path',
        dest='output_path',
        default=p_def['output_path'],
        type='string',
        help=(
            'output path of the extended shear catalogue,'
            + f' default=\'{p_def["output_path"]}\''
        )
    )
    parser.add_option(
        '-p',
        '--plot_path',
        dest='plot_path',
        default=p_def['plot_path'],
        type='string',
        help=(
            'output path of the plots,' + f' default=\'{p_def["plot_path"]}\''
        )
    )

    options, args = parser.parse_args()                                         
                                                                                        
    return options, args


def main(argv=None):

    # Get default parameter values
    params = params_default()

    # Parse command line options
    options, args = parse_options(params)

    # Update parameter values
    for key in vars(options):
        params[key] = getattr(options, key)

    logging.log_command(argv)

    cat_gal = fits.getdata(params['input_path_shear'])

    #check consistency of the names of columns
    assert (params['e1_col'] in cat_gal.columns.names), (
        f'Column {params["e1_col"]} not found in galaxy catalogue'
    )
    assert (params['e2_col'] in cat_gal.columns.names), (
        f'Column {params["e2_col"]} not found in galaxy catalogue'
    )
    assert (params['e1_PSF_col'] in cat_gal.columns.names), (
        f'Column {params["e1_PSF_col"]} not found in galaxy catalogue'
    )
    assert (params['e2_PSF_col'] in cat_gal.columns.names), (
        f'Column {params["e2_PSF_col"]} not found in galaxy catalogue'
    )
    assert (params['snr_col'] in cat_gal.columns.names), (
        f'Column {params["snr_col"]} not found in galaxy catalogue'
    )
    assert (params['w_col'] in cat_gal.columns.names), (
        f'Column {params["w_col"]} not found in galaxy catalogue'
    )
    assert (params['TPSF_col'] in cat_gal.columns.names), (
        f'Column {params["TPSF_col"]} not found in galaxy catalogue'
    )
    assert (params['T_col'] in cat_gal.columns.names), (
        f'Column {params["T_col"]} not found in galaxy catalogue'
    )

    col_snr = params['snr_col']
    col_w = params['w_col']
    col_TPSF = params['TPSF_col']
    col_T = params['T_col']
    col_e1 = params['e1_col']
    col_e2 = params['e2_col']
    col_e1_PSF = params['e1_PSF_col']
    col_e2_PSF = params['e2_PSF_col']

    ratio_size = cat_gal[col_TPSF]/(cat_gal[col_TPSF] + cat_gal[col_T])

    #Create a dataframe
    df_gal = pd.DataFrame(np.array([
        np.array(cat_gal[col_e1], dtype=np.float32),
        np.array(cat_gal[col_e2], dtype=np.float32),
        np.array(cat_gal[col_e1_PSF], dtype=np.float32),
        np.array(cat_gal[col_e2_PSF], dtype=np.float32),
        np.array(cat_gal[col_snr], dtype=np.float32),
        np.array(cat_gal[col_w], dtype=np.float32),
        np.array(ratio_size, dtype=np.float32)
    ]).T, columns=['e1', 'e2', 'e1_PSF', 'e2_PSF', 'snr', 'w', 'ratio_size'])

    n_bins_snr = 20
    n_bins_r = 20

    print("Performing the binning of the catalog in size and SNR...")

    #Bin in size and snr
    df_gal.loc[:, 'bin_R'] = pd.qcut(df_gal['ratio_size'], n_bins_r, labels=False, retbins=False)

    #initialize bin snr
    df_gal.loc[:, 'bin_snr'] = -999

    for ibin_r in tqdm(range(n_bins_r)):
        #select galaxies in the bin
        mask_binr = df_gal['bin_R'].values == ibin_r

        #bin in snr
        df_gal.loc[mask_binr, 'bin_snr'] = pd.qcut(df_gal.loc[mask_binr, 'snr'], n_bins_snr, labels=False, retbins=False)

    #group by bin
    df_gal_grouped = df_gal.groupby(['bin_R', 'bin_snr'])
    ngroups = df_gal_grouped.ngroups

    print("Performing first round calibration...")
    start = time.time()
    alpha_df = pd.DataFrame(0.,
                            index = np.arange(ngroups),
                            columns=['R', 'SNR',
                            'alpha_1', 'alpha_1_err',
                            'alpha_2', 'alpha_2_err'])

    i_group = 0
    for name, group in df_gal_grouped:

        #Save weighted average
        alpha_df.loc[i_group, 'R'] = np.average(group['ratio_size'], weights=group['w'])
        alpha_df.loc[i_group, 'SNR'] = np.average(group['snr'], weights=group['w'])

        #Fit linear model to compute alpha
        e1_out = np.array(group['e1'])
        e2_out = np.array(group['e2'])
        weight_out = np.array(group['w'])
        e1_PSF = np.array(group['e1_PSF'])
        e2_PSF = np.array(group['e2_PSF'])
        del group

        #Fit e1
        mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, 'alpha_1'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_1_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del res_wls, mod_wls

        #Fit e2
        mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, 'alpha_2'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_2_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del weight_out, res_wls, mod_wls

        i_group += 1

    #Fit polynomial to remove general trend
    fitting_weight = 1./np.square(alpha_df['alpha_1_err'].values)
    A = np.vstack([fitting_weight*1,
    fitting_weight*np.power(alpha_df['SNR'].values, -2),
    fitting_weight*np.power(alpha_df['SNR'].values, -3),
    fitting_weight*alpha_df['R'].values,
    fitting_weight*alpha_df['R'].values*np.power(alpha_df['SNR'].values, -2)]).T
    poly1 = np.linalg.lstsq(A, fitting_weight*alpha_df['alpha_1'].values, rcond=None)[0]
    del fitting_weight, A

    fitting_weight = 1./np.square(alpha_df['alpha_2_err'].values)
    A = np.vstack([fitting_weight*1,
    fitting_weight*np.power(alpha_df['SNR'].values, -2),
    fitting_weight*np.power(alpha_df['SNR'].values, -3),
    fitting_weight*alpha_df['R'].values,
    fitting_weight*alpha_df['R'].values*np.power(alpha_df['SNR'].values, -2)]).T
    poly2 = np.linalg.lstsq(A, fitting_weight*alpha_df['alpha_2'].values, rcond=None)[0]
    del fitting_weight, A

    #Compute alpha to remove the general trend
    #e1
    alpha = poly1[0] \
        + poly1[1] * np.power(df_gal['snr'], -2) \
        + poly1[2] * np.power(df_gal['snr'], -3) \
        + poly1[3] * df_gal['ratio_size'] \
        + poly1[4] * df_gal['ratio_size'] * np.power(df_gal['snr'], -2)
    df_gal.loc[:, 'e1_cor'] = df_gal['e1'].values - alpha * df_gal['e1_PSF'].values
    del alpha, poly1

    #e2
    alpha = poly2[0] \
        + poly2[1] * np.power(df_gal['snr'], -2) \
        + poly2[2] * np.power(df_gal['snr'], -3) \
        + poly2[3] * df_gal['ratio_size'] \
        + poly2[4] * df_gal['ratio_size'] * np.power(df_gal['snr'], -2)
    df_gal.loc[:, 'e2_cor'] = df_gal['e2'].values - alpha * df_gal['e2_PSF'].values
    del alpha, poly2

    print(f'First round calibration took {time.time()-start} seconds')

    print("Performing second round of calibration...")
    start = time.time()
    #Initialise second run of calibration
    df_gal.loc[:, 'e1_cor_2'] = -999.
    df_gal.loc[:, 'e2_cor_2'] = -999.
    alpha_df.loc[:, 'alpha_1_corr'] = -999.
    alpha_df.loc[:, 'alpha_2_corr'] = -999.
    alpha_df.loc[:, 'alpha_1_corr_err'] = -999.
    alpha_df.loc[:, 'alpha_2_corr_err'] = -999.

    #Second run of calibration
    for i_group in range(n_bins_r*n_bins_snr):
        #Get the mask
        mask_group = (df_gal['bin_R'].values == i_group//n_bins_snr) & (df_gal['bin_snr'].values == i_group%n_bins_snr)
        #Fit linear model to compute alpha
        e1_out = np.array(df_gal.loc[mask_group, 'e1_cor'])
        e2_out = np.array(df_gal.loc[mask_group, 'e2_cor'])
        weight_out = np.array(df_gal.loc[mask_group, 'w'])
        e1_PSF = np.array(df_gal.loc[mask_group, 'e1_PSF'])
        e2_PSF = np.array(df_gal.loc[mask_group, 'e2_PSF'])
        
        #Fit e1
        mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, 'alpha_1_corr'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_1_corr_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del res_wls, mod_wls

        #Fit e2
        mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
        res_wls = mod_wls.fit()
        alpha_df.loc[i_group, 'alpha_2_corr'] = res_wls.params[1]
        alpha_df.loc[i_group, 'alpha_2_corr_err'] = np.sqrt(res_wls.cov_params()[1, 1])
        del weight_out, res_wls, mod_wls

        df_gal.loc[mask_group, 'e1_cor_2'] = e1_out - alpha_df.loc[i_group, 'alpha_1_corr'] * e1_PSF
        df_gal.loc[mask_group, 'e2_cor_2'] = e2_out - alpha_df.loc[i_group, 'alpha_2_corr'] * e2_PSF

    print(f'Second round calibration took {time.time()-start} seconds')
    #Save the corrected ellipticities
    t = Table(cat_gal)
    t.add_column(Column(name='e1_cor', data=df_gal['e1_cor'].values))
    t.add_column(Column(name='e2_cor', data=df_gal['e2_cor'].values))
    t.add_column(Column(name='e1_cor_2', data=df_gal['e1_cor_2'].values))
    t.add_column(Column(name='e2_cor_2', data=df_gal['e2_cor_2'].values))
    print(
        'Saving alpha-corrected ellipticities to' +
        f' {params["output_path"]}'
    )
    t.write(params['output_path'], overwrite=True, format='fits')


    if params['plot_path'] is not None:

        print("Plotting...")
        #Compute alpha of the calibrated catalog
        alpha_df.loc[:, 'alpha_1_corr_2'] = -999.
        alpha_df.loc[:, 'alpha_2_corr_2'] = -999.
        alpha_df.loc[:, 'alpha_1_corr_2_err'] = -999.
        alpha_df.loc[:, 'alpha_2_corr_2_err'] = -999.

        for i_group in range(n_bins_r*n_bins_snr):
            #Get the mask
            mask_group = (df_gal['bin_R'].values == i_group//n_bins_snr) & (df_gal['bin_snr'].values == i_group%n_bins_snr)
            #Fit linear model to compute alpha
            e1_out = np.array(df_gal.loc[mask_group, 'e1_cor_2'])
            e2_out = np.array(df_gal.loc[mask_group, 'e2_cor_2'])
            weight_out = np.array(df_gal.loc[mask_group, 'w'])
            e1_PSF = np.array(df_gal.loc[mask_group, 'e1_PSF'])
            e2_PSF = np.array(df_gal.loc[mask_group, 'e2_PSF'])
            
            #Fit e1
            mod_wls = sm.WLS(e1_out, sm.add_constant(e1_PSF), weights=weight_out)
            res_wls = mod_wls.fit()
            alpha_df.loc[i_group, 'alpha_1_corr_2'] = res_wls.params[1]
            alpha_df.loc[i_group, 'alpha_1_corr_2_err'] = np.sqrt(res_wls.cov_params()[1, 1])
            del res_wls, mod_wls

            #Fit e2
            mod_wls = sm.WLS(e2_out, sm.add_constant(e2_PSF), weights=weight_out)
            res_wls = mod_wls.fit()
            alpha_df.loc[i_group, 'alpha_2_corr_2'] = res_wls.params[1]
            alpha_df.loc[i_group, 'alpha_2_corr__2err'] = np.sqrt(res_wls.cov_params()[1, 1])
            del weight_out, res_wls, mod_wls

        print(
            'Plotting the results and saving to directory' + f' {params["plot_path"]}'
        )

        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        #Plot scatter in SNR vs r plane
        SNR_bins_val = (alpha_df['SNR'].values)
        R_bins_val = (alpha_df['R'].values)

        alpha = 0.5*((alpha_df['alpha_1'].values)+(alpha_df['alpha_2'].values))
        vmin, vmax = np.min(alpha), np.max(alpha)
        vmin, vmax = -np.max([np.abs(vmin), np.abs(vmax)]), np.max([np.abs(vmin), np.abs(vmax)])

        cset1 = axs[0,0].scatter(SNR_bins_val, R_bins_val, c=alpha, cmap='seismic', vmin=vmin, vmax=vmax)
        axs[0,0].set_xlabel('SNR')
        axs[0,0].set_ylabel('R')
        axs[0,0].set_ylim(max(R_bins_val)+0.01, min(R_bins_val)-0.01)
        axs[0,0].set_title('Leakage - no calibration')
        axs[0,0].set_xscale('log')
        fig.colorbar(cset1, ax=axs[0,0])

        alpha = 0.5*((alpha_df['alpha_1_corr'].values)+(alpha_df['alpha_2_corr'].values))
        cset2 = axs[1,0].scatter(SNR_bins_val, R_bins_val, c=alpha, cmap='seismic', vmin=vmin, vmax=vmax)
        axs[1,0].set_xlabel('SNR')
        axs[1,0].set_ylabel('R')
        axs[1,0].set_ylim(max(R_bins_val)+0.01, min(R_bins_val)-0.01)
        axs[1,0].set_title('Leakage - Calibration Ro 1')
        axs[1,0].set_xscale('log')
        fig.colorbar(cset2, ax=axs[1,0])

        alpha = 0.5*((alpha_df['alpha_1_corr_2'].values)+(alpha_df['alpha_2_corr_2'].values))
        cset3 = axs[2,0].scatter(SNR_bins_val, R_bins_val, c=alpha, cmap='seismic', vmin=vmin, vmax=vmax)
        axs[2,0].set_xlabel('SNR')
        axs[2,0].set_ylabel('R')
        axs[2,0].set_ylim(max(R_bins_val)+0.01, min(R_bins_val)-0.01)
        axs[2,0].set_title('Leakage - Calibration Ro 2')
        axs[2,0].set_xscale('log')
        fig.colorbar(cset3, ax=axs[2,0])

        #Plot leakage as a function of SNR for alpha_e1
        SNR = alpha_df['SNR'].values
        alpha = alpha_df['alpha_1'].values
        alpha_err = alpha_df['alpha_1_err'].values
        axs[0,1].errorbar(SNR, alpha, yerr=alpha_err, fmt='o')
        axs[0,1].axhline(0, color='black', linestyle='--')
        axs[0,1].set_xlabel('SNR')
        axs[0,1].set_xscale('log')
        axs[0,1].set_ylabel(r'$\alpha_1$')
        axs[0,1].set_title('Leakage vs SNR - no calibration')

        alpha = alpha_df['alpha_1_corr'].values
        alpha_err = alpha_df['alpha_1_corr_err'].values
        axs[1,1].errorbar(SNR, alpha, yerr=alpha_err, fmt='o')
        axs[1,1].axhline(0, color='black', linestyle='--')
        axs[1,1].set_xlabel('SNR')
        axs[1,1].set_xscale('log')
        axs[1,1].set_ylabel(r'$\alpha_1$')
        axs[1,1].set_title('Leakage vs SNR - Calibration Ro 1')

        alpha = alpha_df['alpha_1_corr_2'].values
        alpha_err = alpha_df['alpha_1_corr_2_err'].values
        axs[2,1].errorbar(SNR, alpha, yerr=alpha_err, fmt='o')
        axs[2,1].axhline(0, color='black', linestyle='--')
        axs[2,1].set_xlabel('SNR')
        axs[2,1].set_xscale('log')
        axs[2,1].set_ylabel(r'$\alpha_1$')
        axs[2,1].set_title('Leakage vs SNR - Calibration Ro 2')

        #Plot leakage as a function of size for alpha_e1
        R = alpha_df['R'].values
        alpha = alpha_df['alpha_1'].values
        alpha_err = alpha_df['alpha_1_err'].values
        axs[0,2].errorbar(R, alpha, yerr=alpha_err, fmt='o')
        axs[0,2].axhline(0, color='black', linestyle='--')
        axs[0,2].set_xlabel('R')
        axs[0,2].set_ylabel(r'$\alpha_1$')
        axs[0,2].set_title('Leakage vs R - no calibration')

        alpha = alpha_df['alpha_1_corr'].values
        alpha_err = alpha_df['alpha_1_corr_err'].values
        axs[1,2].errorbar(R, alpha, yerr=alpha_err, fmt='o')
        axs[1,2].axhline(0, color='black', linestyle='--')
        axs[1,2].set_xlabel('R')
        axs[1,2].set_ylabel(r'$\alpha_1$')
        axs[1,2].set_title('Leakage vs R - Calibration Ro 1')

        alpha = alpha_df['alpha_1_corr_2'].values
        alpha_err = alpha_df['alpha_1_corr_2_err'].values
        axs[2,2].errorbar(R, alpha, yerr=alpha_err, fmt='o')
        axs[2,2].axhline(0, color='black', linestyle='--')
        axs[2,2].set_xlabel('R')
        axs[2,2].set_ylabel(r'$\alpha_1$')
        axs[2,2].set_title('Leakage vs R - Calibration Ro 2')

        plt.tight_layout()
        fig.savefig(params['plot_path'], dpi=600)

    print("Done!")


    return 0
                                                                                
                                                                                
if __name__ == "__main__":                                                      
    sys.exit(main(sys.argv)) 