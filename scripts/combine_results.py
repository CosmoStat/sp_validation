#!/usr/bin/env python3


import sys
import os
import re

import numpy as np
import uncertainties as unc


def get_match(stats_files, patch, pattern, previous=None, n_previous=[1], typ=str):

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
            prev_ok = False
            for prev, n_prev in zip(previous, n_previous):

                # Line index n_previous earlier, +1 since next line will be read in
                # next loop
                idx_prev = idx - n_prev + 1

                # Look for pattern in previous line
                m_prev = re.search(prev, stats_files[patch][idx - n_prev + 1])
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
            results['all'][key] = sum(values)

        elif results['type'][key] == 'avg':

            # Compute average
            results['all'][key] = sum(values) / len(values)

        elif results['type'][key] == 'w_avg':
            # Weight key
            key_w = results['extra'][key]

            # Weight values
            w = np.array(list(results['value'][key_w].values()))

            # Compute weighted average
            results['all'][key] = sum(values * w) / sum(w)

        elif results['type'][key] == 'sqr_w_avg':
            # Weight key
            key_w = results['extra'][key]

            # Weight values
            w = np.array(list(results['value'][key_w].values()))

            # Square values
            v_sqr = values * values    

            # Compute weighted average
            res_tmp = sum(v_sqr * w) / sum(w)

            # Take square root
            results['all'][key] = np.sqrt(res_tmp)


def init_key(results, key, typ, extra=None):

    results['type'][key] = typ
    results['value'][key] = {}
    if extra:
        results['extra'][key] = extra


def print_all(results, stats_files):

    # Header
    print(' '*12, end=' ')
    for key in results['type']:
        print(f'{key:>12s}', end=' ')
    print()

    # Loop over patches
    for patch in stats_files:

        print(f'{patch:12s}', end=' ')

        # Write value for each key
        for key in results['type']:
            val = results['value'][key][patch]
            if key == 'N_gal':
                print(f'{val:>12d}', end=' ')
            else:
                print(f'{val:12.5g}', end=' ')
        print()

    # Write total
    p = 'all'
    print(f'{p:12s}', end=' ')
    for key in results['type']:
        val = results['all'][key]
        if key == 'N_gal':
            print(f'{val:>12d}', end=' ')
        else:
            print(f'{val:12.5g}', end=' ')
    print()


def main(argv=None):
    """Main

    Main program
    """

    #n_patch = 7
    #patches = [f'P{x}' for x in np.arange(n_patch) + 1]
    patches = ['P7', 'W3', 'S4']

    # Validate with combined catalogue
    #patches = ['comb']

    ##n_patch = len(patches)

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
        'extra' : {},
        'all' : {}
    }

    # Number of galaxies
    key = 'N_gal'
    init_key(results, key, 'sum')
    for patch in stats_files:
        results['value'][key][patch] = get_match(stats_files, patch, 'Number of galaxies after metacal = (\d+)/', previous=[f'^{shape}$'], typ=int)

    # Additive bias
    for comp in (1, 2):
        key = f'c_{comp}'
        init_key(results, key, 'w_avg', extra='N_gal')
        for patch in stats_files:
            c, dc = get_match(stats_files, patch, f'c_{comp} = (\S+)', previous=[f'^{shape}:$'], n_previous=[comp], typ='ufloat')
            results['value'][key][patch] = c
            # TODO: dc

    # Ellipticity dispersion
    key = 'sigma_eps'
    init_key(results, key, 'sqr_w_avg', extra='N_gal')
    for patch in stats_files:
        results['value'][key][patch] = get_match(stats_files, patch, 'Dispersion of complex ellipticity = (\S+)', previous=[f'^{shape}$'], typ=float)

    # Galaxy shear response matrix
    keys = ['R_tot_11', 'R_tot_12', 'R_tot_21', 'R_tot_22']
    for key in keys:
        init_key(results, key, 'w_avg', extra='N_gal')
    for patch in stats_files:
        tmp = get_match(stats_files, patch, '\[\[(\s?\S+)\s+\S+]', previous=['ngmix galaxies:', 'total response matrix:'], n_previous=[2, 1], typ=float)
        results['value']['R_tot_11'][patch] = tmp
        tmp = get_match(stats_files, patch, '\[\[\s?\S+\s+(\S+)]', previous=['ngmix galaxies:', 'total response matrix:'], n_previous=[2, 1], typ=float)
        results['value']['R_tot_12'][patch] = tmp
        tmp = get_match(stats_files, patch, '\[(\s?\S+)\s+\S+\]\]', previous=['ngmix galaxies', 'total response matrix:'], n_previous=[3, 2], typ=float)
        results['value']['R_tot_21'][patch] = tmp
        tmp = get_match(stats_files, patch, ' \[\s?\S+\s+(\S+)\]\]', previous=['ngmix galaxies:', 'total response matrix:'], n_previous=[3, 2], typ=float)
        results['value']['R_tot_22'][patch] = tmp

    # Object-wise PSF leakage
    keys = ['m_11', 'm_12', 'm_21', 'm_22', 'm_s1', 'm_s2']
    for key in keys:
        init_key(results, key, 'w_avg', extra='N_gal')
    for patch in stats_files:
        m, dm = get_match(stats_files, patch, '\$e_\{1\}\^\{\\\\rm PSF\}\$: m_1=(\S*)', previous=['ngmix'], n_previous=[1], typ='ufloat')
        results['value']['m_11'][patch] = m
        m, dm = get_match(stats_files, patch, '\$e_\{1\}\^\{\\\\rm PSF\}\$: m_2=(\S*)', previous=['ngmix'], n_previous=[2], typ='ufloat')
        results['value']['m_12'][patch] = m
        m, dm = get_match(stats_files, patch, '\$e_\{2\}\^\{\\\\rm PSF\}\$: m_1=(\S*)', previous=['ngmix'], n_previous=[3], typ='ufloat')
        results['value']['m_21'][patch] = m
        m, dm = get_match(stats_files, patch, '\$e_\{2\}\^\{\\\\rm PSF\}\$: m_2=(\S*)', previous=['ngmix'], n_previous=[4], typ='ufloat')
        results['value']['m_22'][patch] = m
        m, dm = get_match(stats_files, patch, '\$\\\\mathrm\{FWHM\}\^\{\\\\rm PSF\}\$ \[arcsec]: m_1=(\S+)', previous=['ngmix'], n_previous=[5], typ='ufloat')
        results['value']['m_s1'][patch] = m
        m, dm = get_match(stats_files, patch, '\$\\\\mathrm\{FWHM\}\^\{\\\\rm PSF\}\$ \[arcsec]: m_2=(\S+)', previous=['ngmix'], n_previous=[6], typ='ufloat')
        results['value']['m_s2'][patch] = m

    # Scale-dependent PSF leakage
    key = 'alpha'
    init_key(results, key, 'w_avg', extra='N_gal')
    for patch in stats_files:
        results['value'][key][patch] = get_match(stats_files, patch, 'ngmix: Weighted average alpha =(\s?\S+)', typ=float)

    combine(results)
    print_all(results, stats_files)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

