#!/usr/bin/env python3


import sys
import os
import re

import numpy as np
import uncertainties as unc
from pathlib import Path
from astropy.io import ascii


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

    raise ValueError(f'No match of \'{pattern}\' in patch {patch} (prev=\'{previous}\'), n_prev={n_previous}')


def read_stats_files(patches, path, verbose=False):

    stats_files = {}
    for p in patches:

        fname = f'{p}/{path}'
        if os.path.exists(fname):
            if verbose:
                print(f'Reading stats file \'{fname}\'')
            with open(fname) as f:
                stats_files[p] = f.readlines()
        else:
            print(f'Warning: stats file {fname} not found')

    if len(stats_files) == 0:
        raise ValueError('No stats file found')

    print(f'Found {len(stats_files)} stats files')

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
            s = sum(w)
            if s > 0:
                results['all'][key] = sum(values * w) / s
            else:
                raise ZeroDivisionError(f'sum(w) = {s} for key={key}, key_w={key_w}')

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

        elif results['type'][key] == 'var':
            # Combined variance

            # Weight key
            key_w = results['extra'][key][0]

            # Mean key
            key_m = results['extra'][key][1]

            # Weight values
            w = np.array(list(results['value'][key_w].values()))

            # Patch mean values
            m = np.array(list(results['value'][key_m].values()))

            # Overall mean
            m_all = results['all'][key_m]

            # Square values
            v_sqr = values * values    

            # Compute weighted average
            res_tmp = sum( (v_sqr + (m - m_all)**2) * w ) / sum(w)

            # Take square root
            results['all'][key] = np.sqrt(res_tmp)

        elif results['type'][key] == 'var_m':
            # Combined variance of mean

            # Weight key
            key_w = results['extra'][key][0]

            # Mean key
            key_m = results['extra'][key][1]

            # Weight values
            w = np.array(list(results['value'][key_w].values()))

            # Patch mean values
            m = np.array(list(results['value'][key_m].values()))

            # Overall mean
            m_all = results['all'][key_m]

            # Square values
            v_sqr = values * values

            # Compute weighted average, note extra factor of w in v_sqr
            res_tmp = sum( (v_sqr * w + (m - m_all)**2) * w ) / sum(w)

            # Take square root
            results['all'][key] = np.sqrt(res_tmp / sum(w))


def init_key(results, key, typ, extra=None):

    results['type'][key] = typ
    results['value'][key] = {}
    if extra:
        results['extra'][key] = extra


def print_all(results, stats_files, use_keys=None, fout=sys.stdout, header=True, all=True):

    if use_keys:
        keys = use_keys
    else:
        keys = results['type']

    # Header
    if header:
        print('# patch', ' '*3, end=' ', file=fout)
        for key in keys:
            print(f'{key:>11s}', end=' ', file=fout)
        print(file=fout)

    # Loop over patches
    for patch in stats_files:

        print(f'{patch:11s}', end=' ', file=fout)

        # Write value for each key
        for key in keys:
            val = results['value'][key][patch]
            if key == 'N_gal':
                print(f'{val:>11.0f}', end=' ', file=fout)
            elif key in ('w_tot', 'n_gal_am2'):
                print(f'{val:>11.2f}', end=' ', file=fout)
            elif key in ['xi_sys_p', 'xi_sys_m']:
                print(f'{val:>11.3e}', end=' ', file=fout)
            else:
                print(f'{val:>11.7f}', end=' ', file=fout)
        print(file=fout)

    # Write total
    if all:
        p = 'all'
        if p in results:
            print(f'{p:11s}', end=' ', file=fout)
            for key in keys:
                val = results[p][key]
                if key == 'N_gal':
                    print(f'{val:>11d}', end=' ', file=fout)
                elif key == 'w_tot':
                    print(f'{val:>11.1f}', end=' ', file=fout)
                else:
                    print(f'{val:>11.7f}', end=' ', file=fout)
                    #print(f'{val:>11.3e}', end=' ', file=fout)
            print(file=fout)


def get_area(fname):

    if os.path.exists(fname):

        with open(fname) as f:
            lines = f.readlines()
        for line in lines:
            m = re.search('nmasked patch area without overlap = (.*) deg', line)
            if m:
                return float(m[1])

    else:
        print(f'Warning: No file {fname} found to obtain area')
        return 1


def get_values(results, stats_files, shape, use_keys, area_deg2=-1):
    """Get Values

    Get values from stats files for all patches.

    Parameters
    ----------
    results : dict
        results dictionary
    stats_files : dict of array of str
        stats files content for each patch
    shape : str
        shape measurement method
    use_keys : dict
        keys to include
    area_deg2 : float, optional
        area in square degree, optional is -1 (to be retrieved
        for each patch)
    """
    # Number of galaxies
    key = 'N_gal'
    if use_keys[key]:
        init_key(results, key, 'sum')
        for patch in stats_files:
            results['value'][key][patch] = get_match(stats_files, patch, 'Number of galaxies after metacal = (\d+)/', previous=[f'^{shape}$'], typ=int)

        area_deg2_tot = 0
        key_der = 'n_gal_am2'
        init_key(results, key_der, 'w_avg', extra='N_gal')
        for patch in stats_files:
            if area_deg2 < 0:
                area_deg2_patch = get_area(f'{patch}/area.txt')
                area_deg2_tot += area_deg2_patch
                print(f'area({patch}) = {area_deg2_patch} deg^2')
            else:
                area_deg2_patch = area_deg2
            results['value'][key_der][patch] = results['value']['N_gal'][patch] / (area_deg2_patch * 3600)

    if area_deg2 < 0:
        with open('area_deg2_tot.txt', 'w') as f:
            print(area_deg2_tot, file=f)

    # Sum of weights
    key = 'w_tot'
    if use_keys[key]:
        init_key(results, key, 'sum')
        for patch in stats_files:
            results['value'][key][patch] = get_match(stats_files, patch, 'Sum of weights = (\S+)', typ=float, previous=[f'^{shape}$'])

    # Additive bias (unweighted mean)
    key_base = 'c_'
    if use_keys[key_base]:
        for comp in (1, 2):
            key = f'{key_base}{comp}'
            init_key(results, key, 'w_avg', extra='N_gal')
            for patch in stats_files:
                c = get_match(stats_files, patch, f'{key_base}{comp} = (\S+)', previous=[f'^{shape}:$'], n_previous=[2*comp - 1], typ=float)
                results['value'][key][patch] = c

    # Additive bias (unweighted error)
    key_base = 'dc_'
    if use_keys[key_base]:
        for comp in (1, 2):
            key = f'{key_base}{comp}'
            init_key(results, key, 'var', extra=['N_gal', f'c_{comp}'])
            for patch in stats_files:
                dc = get_match(stats_files, patch, f'{key_base}{comp} = (\S+)', typ=float, previous=[f'^{shape}:$'], n_previous=[2*comp + 7])
                results['value'][key][patch] = dc


    # Additive bias (unweighted error of mean)
    key_base = 'dmc_'
    if use_keys[key_base]:
        for comp in (1, 2):
            key = f'{key_base}{comp}'
            init_key(results, key, 'var_m', extra=['N_gal', f'c_{comp}'])
            for patch in stats_files:
                dmc = get_match(stats_files, patch, f'{key_base}{comp} = (\S+)', typ=float, previous=[f'^{shape}:$'], n_previous=[2*comp + 7])
                results['value'][key][patch] = dmc

    # Additive bias (weighted mean)
    key_base = 'cw_'
    if use_keys[key_base]:
        for comp in (1, 2):
            key = f'{key_base}{comp}'
            init_key(results, key, 'w_avg', extra='w_tot')
            for patch in stats_files:
                c = get_match(stats_files, patch, f'{key_base}{comp} = (\S+)', previous=[f'^{shape}:$'], n_previous=[comp*2], typ=float)
                results['value'][key][patch] = c

    # Additive bias (weighted error of mean)
    key_base = 'dmcw_'
    if use_keys[key_base]:
        for comp in (1, 2):
            key = f'{key_base}{comp}'
            init_key(results, key, 'var_m', extra=['N_gal', f'cw_{comp}'])
            for patch in stats_files:
                dmc = get_match(stats_files, patch, f'{key_base}{comp} = (\S+)', typ=float, previous=[f'^{shape}:$'], n_previous=[2*comp + 8])
                results['value'][key][patch] = dmc

    # Additive bias (jackknife)
    key_base = 'cjk_'
    if use_keys[key_base]:
        for comp in (1, 2):

            # Mean
            key = f'{key_base}{comp}'
            key_m = key
            init_key(results, key, 'w_avg', extra='w_tot')

            # Error
            key = f'd{key_base}{comp}'
            key_s = key
            init_key(results, key, 'var_m', extra=['w_tot', f'cjk_{comp}'])

            for patch in stats_files:
                c, dc = get_match(stats_files, patch, f'{key_base}{comp} = (\S+)', previous=[f'^{shape}:$'], n_previous=[comp], typ='ufloat')
                results['value'][key_m][patch] = c
                results['value'][key_s][patch] = dc

    # Ellipticity dispersion
    key = 'sigma2_epsilon'
    if use_keys[key]:
        init_key(results, key, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key][patch] = get_match(stats_files, patch, 'Dispersion of complex ellipticity = (\S+)', previous=[f'^{shape}$'], typ=float)

    # Galaxy total response matrix
    key_base = 'R_tot_'
    if use_keys[key_base]:
        keys = [f'{key_base}11', f'{key_base}12', f'{key_base}21', f'{key_base}22']
        for key in keys:
            init_key(results, key, 'w_avg', extra='N_gal')
        for patch in stats_files:
            tmp = get_match(stats_files, patch, '\[\[(\s?\S+)\s+\S+]', previous=['ngmix galaxies:', 'total response matrix:'], n_previous=[2, 1], typ=float)
            results['value'][keys[0]][patch] = tmp
            tmp = get_match(stats_files, patch, '\[\[\s?\S+\s+(\S+)]', previous=['ngmix galaxies:', 'total response matrix:'], n_previous=[2, 1], typ=float)
            results['value'][keys[1]][patch] = tmp
            tmp = get_match(stats_files, patch, '\[(\s?\S+)\s+\S+\]\]', previous=['ngmix galaxies', 'total response matrix:'], n_previous=[3, 2], typ=float)
            results['value'][keys[2]][patch] = tmp
            tmp = get_match(stats_files, patch, ' \[\s?\S+\s+(\S+)\]\]', previous=['ngmix galaxies:', 'total response matrix:'], n_previous=[3, 2], typ=float)
            results['value'][keys[3]][patch] = tmp

        # Normalised trace = mean diagonal
        key_der = 'trN_R_tot'
        init_key(results, key_der, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key_der][patch] = (results['value']['R_tot_11'][patch] + results['value']['R_tot_22'][patch]) / 2

        # Sum of absolute off-diagonal
        key_der = 'abs_off_R_tot'
        init_key(results, key_der, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key_der][patch] = np.abs(results['value']['R_tot_12'][patch]) + np.abs(results['value']['R_tot_21'][patch])

    # Galaxy shear response matrix
    key_base = 'R_shear_'
    if use_keys[key_base]:
        keys = [f'{key_base}11', f'{key_base}12', f'{key_base}21', f'{key_base}22']
        for key in keys:
            init_key(results, key, 'w_avg', extra='N_gal')
        for patch in stats_files:
            tmp = get_match(stats_files, patch, '\[\[(\s?\S+)\s+\S+]', previous=['ngmix galaxies:', 'shear response matrix:'], n_previous=[5, 1], typ=float)
            results['value'][keys[0]][patch] = tmp
            tmp = get_match(stats_files, patch, '\[\[\s?\S+\s+(\S+)]', previous=['ngmix galaxies:', 'shear response matrix:'], n_previous=[5, 1], typ=float)
            results['value'][keys[1]][patch] = tmp
            tmp = get_match(stats_files, patch, '\[(\s?\S+)\s+\S+\]\]', previous=['ngmix galaxies', 'shear response matrix:'], n_previous=[6, 2], typ=float)
            results['value'][keys[2]][patch] = tmp
            tmp = get_match(stats_files, patch, ' \[\s?\S+\s+(\S+)\]\]', previous=['ngmix galaxies:', 'shear response matrix:'], n_previous=[6, 2], typ=float)
            results['value'][keys[3]][patch] = tmp

        # Normalised trace = mean diagonal
        key_der = 'trN_R_shear'
        init_key(results, key_der, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key_der][patch] = (results['value']['R_shear_11'][patch] + results['value']['R_shear_22'][patch]) / 2

        # Sum of absolute off-diagonal
        key_der = 'abs_off_R_shear'
        init_key(results, key_der, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key_der][patch] = np.abs(results['value']['R_shear_12'][patch]) + np.abs(results['value']['R_shear_21'][patch])


    # Galaxy selection response matrix
    key_base = 'R_select_'
    if use_keys[key_base]:
        keys = [f'{key_base}11', f'{key_base}12', f'{key_base}21', f'{key_base}22']
        for key in keys:
            init_key(results, key, 'w_avg', extra='N_gal')
        for patch in stats_files:
            tmp = get_match(stats_files, patch, '\[\[(\s?\S+)\s+\S+]', previous=['ngmix galaxies:', 'selection response matrix:'], n_previous=[8, 1], typ=float)
            results['value'][keys[0]][patch] = tmp
            tmp = get_match(stats_files, patch, '\[\[\s?\S+\s+(\S+)]', previous=['ngmix galaxies:', 'selection response matrix:'], n_previous=[8, 1], typ=float)
            results['value'][keys[1]][patch] = tmp
            tmp = get_match(stats_files, patch, '\[(\s?\S+)\s+\S+\]\]', previous=['ngmix galaxies', 'selection response matrix:'], n_previous=[9, 2], typ=float)
            results['value'][keys[2]][patch] = tmp
            tmp = get_match(stats_files, patch, ' \[\s?\S+\s+(\S+)\]\]', previous=['ngmix galaxies:', 'selection response matrix:'], n_previous=[9, 2], typ=float)
            results['value'][keys[3]][patch] = tmp

        # Normalised trace = mean diagonal
        key_der = 'trN_R_select'
        init_key(results, key_der, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key_der][patch] = (results['value']['R_select_11'][patch] + results['value']['R_select_22'][patch]) / 2

        # Sum of absolute off-diagonal
        key_der = 'abs_off_R_select'
        init_key(results, key_der, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key_der][patch] = np.abs(results['value']['R_select_12'][patch]) + np.abs(results['value']['R_select_21'][patch])


    # Object-wise PSF leakage
    key_base = 'm_'
    if use_keys[key_base]:
        keys = [f'{key_base}11', f'{key_base}12', f'{key_base}21', f'{key_base}22', f'{key_base}s1', f'{key_base}s2']
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
    if use_keys[key]:
        init_key(results, key, 'w_avg', extra='N_gal')
        for patch in stats_files:
            results['value'][key][patch] = get_match(stats_files, patch, 'ngmix: Weighted average alpha =(\s?\S+)', typ=float)

    # xi_sys
    key_base = 'xi_sys'
    if use_keys[key_base]:
        init_key(results, 'xi_sys_p', 'w_avg', extra='N_gal')
        init_key(results, 'xi_sys_m', 'w_avg', extra='N_gal')
        for patch in stats_files:
            tmp = get_match(stats_files, patch, 'ngmix: <\|xi_sys_\+\|> = (\S*)', typ=float)
            results['value']['xi_sys_p'][patch] = tmp
            tmp = get_match(stats_files, patch, 'ngmix: <\|xi_sys_\-\|> = (\S*)', typ=float)
            results['value']['xi_sys_m'][patch] = tmp


def latex_table(file_base, cols=None, col_names=None):

    dat = ascii.read(f'{file_base}.txt')

    fout = open(f'{file_base}.tex', 'w')

    n_lines = len(dat)

    n_dat = len(cols)

    print(r'\begin{tabular}{l', end='', file=fout)
    print('c'*n_dat, end='', file=fout)
    #print('r@{}l'*n_dat, end='', file=fout)
    print(r'}\hline\hline', file=fout)

    # Table header
    str_line = 'patch\t&'
    for name in col_names:
        str_line = f'{str_line} ${name}$\t&'
        #str_line = f'{str_line} \\multicolumn{{2}}{{c}}{{${name}$}}\t&'
    print(rf'{str_line[:len(str_line)-2]} \\ \hline', file=fout)

    # Loop over input lines
    for nl in range(n_lines):
        str_line = ''

        str_line = f'{str_line}{dat["patch"][nl]}\t&'

        for col in cols:
            if len(col) == 2:
                # value and error bar
                val_err = unc.ufloat(dat[col[0]][nl], dat[col[1]][nl])
                str_line = f'{str_line} ${val_err:+.2eL}$\t&'
            else:
                if dat[col][nl] < 0:
                    pref = ''
                else:
                    pref = '\phantom{-}'
                str_line = f'{str_line} ${pref}{dat[col][nl]:#.4f}$\t&'
        print(rf'{str_line[:len(str_line)-2]} \\', file=fout)

    print(r'\hline', file=fout)
    print(r'\end{tabular}', file=fout)

    fout.close()


def get_matrix_elements(base, me):

    res = []

    for m in me:
        res.append(f'{base}_{{{m}}}')

    return res


def main(argv=None):
    """Main

    Main program
    """
    if len(argv) == 1:
        argv.append('snr')

    all = True

    if argv[1] == 'snr':
        # All directories
        patches = [f.path for f in os.scandir('.') if f.is_dir()]
        all = False
    elif argv[1] == 'v1':
        n_patch = 7
        patches = [f'P{x}' for x in np.arange(n_patch) + 1]
    elif argv[1] == 'v1.5':
        n_patch = 8
        patches = [f'P{x}' for x in np.arange(n_patch) + 1]
    elif argv[1] == 'test':
        patches = ['P7', 'W3', 'S4']
    elif argv[1] == 'comb':
        # Validate with combined catalogue
        patches = ['comb']
    else:
        patches = argv[1].split("+")

    n_patch = len(patches)

    print('combine_results.py:', patches)

    directory = 'sp_output/plots'
    fbase = 'stats_file'
    fname = f'{fbase}.txt'
    path = f'{directory}/{fname}'

    shape = 'ngmix'

    verbose = False

    stats_files = read_stats_files(patches, path, verbose=verbose)
    n_patch_found = len(stats_files)

    results = {
        'value' : {},
        'type' : {},
        'extra' : {},
        'all' : {}
    }

    use_keys = {
        'N_gal' : 1,
        'w_tot' : 1,
        'c_' : 1,
        'cw_' : 1,
        'dc_' : 0,
        'dmc_' : 0,
        'dmcw_' : 1,
        'cjk_' : 0,
        'sigma2_epsilon' : 1,
        'R_tot_' : 1,
        'R_shear_' : 1,
        'R_select_' : 1,
        'm_' : 1,
        'alpha' : 1,
        'xi_sys' : 0,
    }
    use_keys_m = {
        'N_gal' : 0,
        'w_tot' : 0,
        'c_' : 0,
        'cw_' : 0,
        'dc_' : 0,
        'dmc_' : 0,
        'dmcw_' : 0,
        'cjk_' : 0,
        'sigma2_epsilon' : 0,
        'R_tot_' : 0,
        'R_shear_' : 0,
        'R_select_' : 0,
        'm_' : 1,
        'alpha' : 0,
        'xi_sys' : 0,
    }


    if argv[1] == 'snr':
        area_deg2 = get_area('area.txt')
    else:
        area_deg2 = -1
    get_values(results, stats_files, shape, use_keys, area_deg2=area_deg2)

    # Combine all keys
    combine(results)

    # Print all keys to terminal
    print_all(results, stats_files, all=all)


    # Get value of combined run for precision check
    if argv[1] == 'test':
        stats_file_comb = read_stats_files(['comb'], path, verbose=verbose)
        n_patch_comb = len(stats_files)
        results_comb = {
            'value' : {},
            'type' : {},
            'extra' : {},
        }
        get_values(results_comb, stats_file_comb, shape, use_keys)
        print('\nCombined data:')
        print_all(results_comb, stats_file_comb, header=False)

        # Compute fractional difference
        for key in results['type']:
            results_comb['value'][key]['comb'] = (results_comb['value'][key]['comb'] / results['all'][key] - 1) * 100
        print('\nFractional difference [%]:')
        print_all(results_comb, stats_file_comb, header=False)
    else:
        results_comb = None

    # Get value of entire-sample run
    if argv[1] == 'v1':
        try:
            stats_file_comb = read_stats_files(['joint'], f'leakage/{fbase}_leakage.txt', verbose=verbose)
            n_patch_comb = len(stats_files)
            results_comb = {
                'value' : {},
                'type' : {},
                'extra' : {},
            }
            get_values(results_comb, stats_file_comb, shape, use_keys_m, area_deg2=1)
        except:
            print("leakage stats file of joint catalogue not found, skipping")
 

    # Print some key (combinations) to text and LaTeX file
    if argv[1] not in ['comb']:

        file_base_arr = []

        file_base = 'c'
        file_base_arr.append(file_base)
        f = open(f'{file_base}.txt', 'w')
        print_all(results, stats_files, use_keys=['cw_1', 'dmcw_1', 'cw_2', 'dmcw_2'], fout=f, all=all)
        f.close()
        latex_table(file_base, cols=[['cw_1', 'dmcw_1'], ['cw_2', 'dmcw_2']], col_names=['c_1', 'c_2'])

        key = 'sigma2_epsilon'
        if use_keys[key]:
            file_base = key
            file_base_arr.append(file_base)
            f = open(f'{file_base}.txt', 'w')
            print_all(results, stats_files, use_keys=[key], fout=f, all=all)
            f.close()
            latex_table(file_base, cols=[key], col_names=['\sigma^2_\epsilon'])
        

        me = ['11', '12', '21', '22']

        key_base = 'R_tot_'
        if use_keys[key_base]:
            file_base = 'R'
            file_base_arr.append(file_base)
            f = open(f'{file_base}.txt', 'w')
            use_keys=[f'{key_base}11', f'{key_base}12', f'{key_base}21', f'{key_base}22']
            print_all(results, stats_files, use_keys=use_keys, fout=f, all=all)
            f.close()
            col_names = get_matrix_elements('R^{\\textrm{tot}}', me)
            latex_table(file_base, cols=use_keys, col_names=col_names)

        file_base = 'R_shear'
        file_base_arr.append(file_base)
        f = open(f'{file_base}.txt', 'w')
        key_base = 'R_shear_'
        use_keys=[f'{key_base}11', f'{key_base}12', f'{key_base}21', f'{key_base}22']
        print_all(results, stats_files, use_keys=use_keys, fout=f, all=all)
        f.close()
        col_names = get_matrix_elements('R^{\\textrm{shear}}', me)
        latex_table(file_base, cols=use_keys, col_names=col_names)

        file_base = 'R_select'
        file_base_arr.append(file_base)
        f = open(f'{file_base}.txt', 'w')
        key_base = 'R_select_'
        use_keys=[f'{key_base}11', f'{key_base}12', f'{key_base}21', f'{key_base}22']
        print_all(results, stats_files, use_keys=use_keys, fout=f, all=all)
        f.close()
        col_names = get_matrix_elements('R^{\\textrm{select}}', me)
        latex_table(file_base, cols=use_keys, col_names=col_names)

    if argv[1] not in ['comb']:

        file_base = 'summary_Rc'
        file_base_arr.append(file_base)
        f = open(f'{file_base}.txt', 'w')
        use_keys=['cw_1', 'dmcw_1', 'cw_2', 'dmcw_2', 'trN_R_tot', 'abs_off_R_tot', 'trN_R_shear', 'trN_R_select']
        print_all(results, stats_files, use_keys=use_keys, fout=f, all=all)
        f.close()
        col_names = ['c_1', '\Delta c_1', 'c_2', '\Delta c_2', '\langle R^{\\rm tot}_{ii} \\rangle',
                     '\\sum | R^{\\ tot}_{i \\ne j}|', '\langle R^{\\rm shear}_{ii} \\rangle', '\langle R^{\\rm select}_{ii} \\rangle']
        latex_table(file_base, cols=use_keys, col_names=col_names)

        file_base = 'summary_leakage'
        file_base_arr.append(file_base)
        f = open(f'{file_base}.txt', 'w')
        use_keys=['n_gal_am2', 'm_11', 'm_22', 'm_12', 'm_21', 'm_s1', 'm_s2', 'alpha']
        print_all(results, stats_files, use_keys=use_keys, fout=f, all=all)
        f.close()
        col_names = ['n_{\\rm gal} [{\\rm am}^{-2}]', 'm_{11}', 'm_{22}', 'm_{12}', 'm_{21}', 'm_{\\rm s1}', 'm_{\\rm s2}', '\\alpha']
        latex_table(file_base, cols=use_keys, col_names=col_names)

        file_base = 'summary_leakage_m'
        file_base_arr.append(file_base)
        f = open(f'{file_base}.txt', 'w')
        use_keys=['m_11', 'm_22', 'm_12', 'm_21', 'm_s1', 'm_s2']
        print_all(results, stats_files, use_keys=use_keys, fout=f, all=False)

        # Add joint sample results
        if results_comb:
            print_all(results_comb, stats_file_comb, use_keys=use_keys, fout=f, all=False)
            f.close()
            col_names = ['m_{11}', 'm_{22}', 'm_{12}', 'm_{21}', 'm_{\\rm s1}', 'm_{\\rm s2}']
            latex_table(file_base, cols=use_keys, col_names=col_names)


    for file_base in file_base_arr:

        #os.system(f'~/txt2tex.pl {file_base}.tex > {file_base}_out.tex')
        #print(f'Creating LaTeX file {file_base}.tex')
        #os.system(f'pdflatex {file_base}_out 2&>/dev/null')
        print("Skipping calling LaTeX")

if __name__ == "__main__":
    sys.exit(main(sys.argv))
