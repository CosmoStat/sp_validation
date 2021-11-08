#!/usr/bin/env python3


import sys
import os
import re

import numpy as np
import uncertainties as unc


def get_match(stats_files, patch, pattern, previous=None, n_previous=1, typ=str):

    prev_ok = False

    for idx, line in enumerate(stats_files[patch]):
        m = re.search(pattern, line)
        if m:
            if (previous and prev_ok) or not previous:
                if typ == 'ufloat':
                    u = unc.ufloat_fromstr(m[1])
                    return u.nominal_value, u.std_dev
                else:
                    return typ(m[1])

        if previous:
            # Line index n_previous earlier, +1 since next line will be read in
            # next loop
            idx_prev = idx - n_previous + 1

            # Look for pattern in previous line
            m_prev = re.search(previous, stats_files[patch][idx - n_previous + 1])
            if m_prev:
                prev_ok = True

    raise ValueError(f'No match of \'{pattern}\' in patch {patch} (previous=\'{previous}\')')


def read_stats_files(patches, path, verbose=False):

    stats_files = {}
    for p in patches:

        fname = f'{p}/{path}'
        if os.path.exists(fname):
            if verbose:
                print(f'Reading stats file \'{fname}\'')
            with open(fname) as f:
                stats_files[p] = f.readlines()

    return stats_files


def combine(results):

    for key in results['type']:

        # Values
        values = np.array(list(results['value'][key].values()))

        if results['type'][key] == 'sum':
            # Compute sum
            results['value']['all'] = sum(values)

        elif results['type'][key] == 'w_avg':
            # Weight key
            key_w = results['extra'][key]

            # Weight values
            w = np.array(list(results['value'][key_w].values()))
            print(w)

            # Compute weighted average
            results['value']['all'] = sum(values * w) / sum(w)
        

def main(argv=None):
    """Main

    Main program
    """

    n_patch = 7
    patches = [f'P{x}' for x in np.arange(n_patch) + 1]

    directory = 'sp_output/plots'
    fname = 'stats_file.txt'
    path = f'{directory}/{fname}'

    shape = 'ngmix'

    verbose = True

    stats_files = read_stats_files(patches, path, verbose=verbose)
    n_patch_found = len(stats_files)

    results = {
        'value' : {},
        'type' : {},
        'extra' : {}
    }

    # Number of galaxies
    key = 'N_gal'
    results['type'][key] = 'sum'
    results['value'][key] = {}
    for patch in stats_files:
        results['value'][key][patch] = get_match(stats_files, patch, 'Number of galaxies after metacal = (\d*)/', previous=f'^{shape}$', typ=int)

    # Additive bias
    for comp in (1, 2):
        key = f'c_{comp}'
        results['type'][key] = 'w_avg'
        results['value'][key] = {}
        results['extra'][key] = 'N_gal'
        for patch in stats_files:
            c, dc = get_match(stats_files, patch, f'c_{comp} = (\S*)', typ='ufloat', previous=f'^{shape}:$', n_previous=comp)
            results['value'][key][patch] = c
            # TODO: dc

    # Ellipticity dispersion
    key = 'sigma_eps'
    results['type'][key] = 'sqr_w_avg'
    results['value'][key] = {}
    results['extra'][key] = 'N_gal'
    for patch in stats_files:
        results['value'][key][patch] = get_match(stats_files, patch, 'Dispersion of complex ellipticity = (\d*)/', previous=f'^{shape}$', typ=int)



    combine(results)
    print(results)

if __name__ == "__main__":
    sys.exit(main(sys.argv))

