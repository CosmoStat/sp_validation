"""
Script homogenize_cat_extended.py

Overwrite the extended catalog to replace the columns e1 and e2 from the extended catalog with the columns
e1 and e2 from the non-extended one. Currently, the calibration of the columns e1 and e2 of the extended catalog
are calibrated per patch, while the non-extended catalog is calibrated on the whole footprint.

:Authors: Sacha Guerrini, Martin Kilbinger
"""

import sys

from optparse import OptionParser

from cs_util import logging

from astropy.io import fits
from astropy.table import Table, Column

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
        'input_path_shear': 'SP/unions_shapepipe_extended_2022_v1.0.fits',
        'input_path_shear_no_ext': 'SP/unions_shapepipe_2022_v1.0.fits',
        'output_path': 'shape_cat_homogenize.fits',
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
        '--input_path_shear_no_ext',                                                   
        dest='input_path_shear_no_ext',                                                
        default=p_def['input_path_shear_no_ext'],
        type='string',                                                          
        help='input path of the non extended shear catalogue'                       
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
        '-o',
        '--output_path',
        dest='output_path',
        default=p_def['output_path'],
        type='string',
        help=(
            'input path of the extended shear catalogue,'
            + f' default=\'{p_def["output_path"]}\''
        )
    )

    options, args = parser.parse_args()                                         
                                                                                        
    return options, args

def main(argv=None):

    #Get default parameter values
    params = params_default()

    #Parse command line options
    options, args = parse_options(params)

    #Update parameters values
    for key in vars(options):
        params[key] = getattr(options, key)

    logging.log_command(argv)

    #Read input files
    dat_shear = fits.getdata(params['input_path_shear'])
    dat_shear_no_ext = fits.getdata(params['input_path_shear_no_ext'])

    t = Table(dat_shear)
    t.remove_columns([params['e1_col'], params['e2_col']])
    t.add_columns([Column(dat_shear_no_ext[params['e1_col']], name=params['e1_col']),
                   Column(dat_shear_no_ext[params['e2_col']], name=params['e2_col'])])
    print(
        'Saving the homogenized catalog to'+f' {params["output_path"]}'
    )
    t.write(params['output_path'], overwrite=True, format='fits')

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))