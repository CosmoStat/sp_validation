#!/usr/bin/env python3

import sys
import numpy as np
import copy
import matplotlib.pylab as plt

from optparse import OptionParser


class param:
    """Param Class.

    General class to store (default) variables.

    """

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def print(self, **kwds):
        """Print."""
        print(self.__dict__)

    def var_list(self, **kwds):
        """Get Variable List."""
        return vars(self)



def params_default():
    """Set default parameter values.

    Parameters
    ----------
    None

    Returns
    -------
    p_def: class param
        parameter values
    """

    p_def = param(
        survey = 'v1'
    )

    return p_def


def parse_options(p_def):
    """Parse command line options.

    Parameters
    ----------
    p_def: class param
        parameter values

    Returns
    -------
    options: tuple
        Command line options
    args: string
        Command line string
    """

    usage  = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    # I/O
    parser.add_option(
        '-i',
        '--input',
        dest='input',
        type='string',
        help='input file'
    )
    parser.add_option(
        '-s',
        '--survey',
        dest='survey',
        type='string',
        help='survey'
    )

    options, args = parser.parse_args()

    return options, args


def check_options(options):
    """Check command line options.

    Parameters
    ----------
    options: tuple
        Command line options

    Returns
    -------
    erg: bool
        Result of option check. False if invalid option value.
    """

    if not options.input and not options.survey:
        print('Either input or survey need to be specified')
        return False

    return True


def update_param(p_def, options):
    """Return default parameter, updated and complemented according to options.
    
    Parameters
    ----------
    p_def:  class param
        parameter values
    optiosn: tuple
        command line options
    
    Returns
    -------
    param: class param
        updated paramter values
    """

    param = copy.copy(p_def)

    # Update keys in param according to options values
    for key in vars(param):
        if key in vars(options):
            setattr(param, key, getattr(options, key))

    # Add remaining keys from options to param
    for key in vars(options):
        if not key in vars(param):
            setattr(param, key, getattr(options, key))

    return param



def main(argv=None):
    """Main

    Main program
    """

    # Set default parameters
    p_def = params_default()

    # Command line options
    options, args = parse_options(p_def)
    # Without option parsing, this would be: args = argv[1:]

    if check_options(options) is False:
        return 1

    param = update_param(p_def, options)

    sh = 'ngmix'

    # Get input file paths
    print('Retrieving input file paths')
    input_files = []
    if param.survey == 'v1':
        n_patch = 7
        patches = [f'P{x}' for x in np.arange(n_patch) + 1]
        for patch in patches:
            path = f'{patch}/sp_output/tile_id_gal_counts_{sh}.txt'
            input_files.append(path)
    else:
        input_files = [param.input]

    print(f'Found {len(input_files)} input files')

    # Read from input files number of detections, galaxies, shapes
    print('Reading input detection, galaxy, shape numbers')
    dat = {}
    n_det_arr = []
    n_gal_arr = []
    n_shape_arr = []

    for patch, input_path in zip(patches, input_files):

        dat[patch] = np.loadtxt(input_path)
        tile_ID = dat[patch][:, 0]
        n_det = dat[patch][:, 1]
        n_gal = dat[patch][:, 2]
        n_shape = dat[patch][:, 3]

        n_det_arr.extend(n_det)
        n_gal_arr.extend(n_gal)
        n_shape_arr.extend(n_shape)


    # Write tile IDs with number of shapes > 0
    print('Writing tile IDs with n_shapes>0')
    for patch in patches:
        out_path = f'{patch}/sp_output/found_ID_wshapes.txt'
        with open(out_path, 'w') as f_out:
            mask_n_shape = dat[patch][:, 3] > 0
            tile_ID_masked = dat[patch][mask_n_shape, 0]
            for ID in tile_ID_masked:
                print(f'{ID:07.3f}', file=f_out)

    # Plot histograms
    print('Plotting histograms')
    print(f'Using {len(n_det_arr)} tiles')
    bins = np.arange(0, 40000, 200)
    density = False
    alpha = 0.5

    plt.ylabel('frequency')
    plt.xlabel('#obj/tile')

    plt.hist(n_det_arr, bins=bins, label='detections', density=density, alpha=alpha)

    plt.hist(n_gal_arr, bins=bins, label='selected galaxies', density=density, alpha=alpha)

    plt.hist(n_shape_arr, bins=bins, label='measured shapes', density=density, alpha=alpha)

    plt.legend()
    plt.savefig('hist_obj_per_tile.png')

    figure_mosaic = """
    A
    B
    C
    """
    fig, axes = plt.subplot_mosaic(mosaic=figure_mosaic, figsize=(10, 6))
    axes['A'].hist(n_det_arr, bins=bins, label='detections', density=density, alpha=alpha, color='b')
    axes['B'].hist(n_gal_arr, bins=bins, label='selected galaxies', density=density, alpha=alpha, color='orange')
    axes['C'].hist(n_shape_arr, bins=bins, label='measured shapes', density=density, alpha=alpha, color='green')

    axes['C'].set_xlabel('#obj/tile')

    for panel in axes:
        axes[panel].legend()
        axes[panel].set_ylabel('frequency')

    plt.savefig('hist_obj_per_tile_3.png')

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
