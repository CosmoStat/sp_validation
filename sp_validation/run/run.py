from cs_util import logging


# To cs_util
def parse_options(p_def, short_options, types, help_strings):
    """Parse command line options.

    Parameters
    ----------
    p_def : dict
        default parameter values
    help_strings : dict
        help strings for options

    Returns
    -------
    options: tuple
        Command line options
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    for key in p_def:
        if key in help_strings:

            if key in short_options:
                short = short_options[key]
            else:
                short = ''

            if key in types:
                typ = types[key]
            else:
                typ = 'string'

            parser.add_option(
                short,
                f'--{key}',
                dest=key,
                type=typ,
                default=p_def[key],
                help=help_strings[key].format(p_def[key]),
            )

    parser.add_option(
        '-v',
        '--verbose',
        dest='verbose',
        action='store_true',
        help=f'verbose output'
    )

    options, _ = parser.parse_args()
        
    return options


class LeakageScale:

    def __init__(self):

        # Set default parameters
        self.params_default()

        # Read command line options
        self.options = parse_options(self._p_def, self._short_options, self._types, self._help_strings)

        # Check options
        if self.check_options() is False:                                         
            return 1                                                                

        # Update parameter values from options
        for key in vars(options):
            params[key] = getattr(options, key)

        # del options ?
                                                                                
        # Save calling command                                                      
        logging.log_command(argv)

    def params_default():
        """Set default parameter values.

        """
        self._params = {
            'input_path_shear': None,
            'e1_col': 'e1_uncal',
            'e2_col': 'e2_uncal',
            'input_path_PSF': None,
            'hdu_psf': 1,
            'ra_star_col': 'RA',
            'dec_star_col': 'Dec',
            'e1_PSF_star_col': 'E1_PSF_HSM',
            'e2_PSF_star_col': 'E2_PSF_HSM',
            'output_dir': '.',
            'sh': 'ngmix',
            'theta_min_amin': 1,
            'theta_max_amin': 300,
            'n_theta': 20,
            'leakage_alpha_ylim': [-0.03, 0.1],
            'leakage_xi_sys_ylim': [-4e5, 5e5],
            'leakage_xi_sys_log_ylim': [2e-13, 5e-5],
        }

        self._short_options = {
            'input_path_shear': '-i',
            'input_path_PSF': '-I',
            'output_dir': '-o',
            'shapes': '-s',
            'close_pair_tolerance': '-t',
            'close_pair_mode': '-m',
        }

        self._types = {
            'hdu_psf': 'int',
            'theta_min_amin': 'float',
            'theta_max_amin': 'float',
            'n_theta': 'int',
        }

        self._help_strings = {
            'input_path_shear': 'input path of the shear catalogue',
            'e1_col': 'e1 column name in galaxy catalogue',
        1   'e2_col': 'e2 column name in galaxy catalogue',
            'input_path_PSF': 'input path of the PSF catalogue',
            'hdu_PSF': 'HDU number of PSF catalogue',
            'ra_star_col': 'right ascension column name in star catalogue',
            'dec_star_col': 'declination column name in star catalogue',
            'e1_PSF_star_col': 'e1 PSF column name in star catalogue',
            'e2_PSF_star_col': 'e2 PSF column name in star catalogue',
            'output_dir': 'output_directory',
            'sh': 'shape measurement method'.
            'close_pair_tolerance': 'tolerance angle for close objects in star catalogue',
            'close_pair_mode': 'mode for close objects in star catalogue, allowed are '\'remove\', \'average\''
            'cut': 'list of criteria (white-space separated, do not use \'_\') to cut data, e.g. \'w>0_mask!=0\''
            'theta_min_amin': 'mininum angular scale [arcmin], default={}',
            'theta_max_amin': 'maximum angular scale [arcmin], default={}',
            'n_theta': 'number of angular scales on input, default={}',
        }

    def check_options(self):

        # if self.options['some_key'] != some_value:
        #    return False 

        return True


def run_leakage_scale(*args):

    # Create object for scale-dependent leakage calculations
    obj = LeakgeScale()

    obj.run()
