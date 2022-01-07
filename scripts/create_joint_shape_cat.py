#!/usr/bin/env python3

import sys

import numpy as np
from astropy.io import ascii

from sp_validation.cat import *


def get_R(fname_base, key_base=None):

    dat = ascii.read(f'{fname_base}.txt')
    if dat[-1]['patch'] != 'all':
        raise ValueError(
            f'Invalid file {fname}, last row does not correspond to patch=\'all\''
        )
    R = np.empty(shape=(2, 2))

    if not key_base:
        key_base = fname_base

    R[0, 0] = dat[-1][f'{key_base}_11']
    R[0, 1] = dat[-1][f'{key_base}_12']
    R[1, 0] = dat[-1][f'{key_base}_21']
    R[1, 1] = dat[-1][f'{key_base}_22']

    return R


def main(argv=None):
    """Main

    Main program
    """

    if argv[1] == 'v1':
        n_patch = 7
        patches = [f'P{x}' for x in np.arange(n_patch) + 1]
    elif argv[1] == 'test':
        patches = ['P7', 'W3', 'S4']

    sh = 'ngmix'

    R = get_R('R', key_base='R_tot')
    R_shear = get_R('R_shear')
    R_select = get_R('R_select')

    print('R - R_shear + R_select = 0 ?')
    print(R - R_shear - R_select)

    alpha = 0.0

    fname = 'c.txt'
    dat = ascii.read(fname)
    if dat[-1]['patch'] != 'all':
        raise ValueError(
            f'Invalid file {fname}, last row does not correspond to patch=\'all\''
        )
    c = np.empty(2)
    c_err = np.empty(2)
    c[0] = dat[-1]['cw_1']
    c[1] = dat[-1]['cw_2']
    c_err[0] = dat[-1]['dmcw_1']
    c_err[1] = dat[-1]['dmcw_2']


    # Invert total response matrix
    Rm1 = np.linalg.inv(R)

    ra_all = np.array([])
    dec_all = np.array([])
    g1_corr_mc_all = np.array([])
    g2_corr_mc_all = np.array([])
    w_all = np.array([])
    mag_all = np.array([])
    snr_all = np.array([])

    for patch in patches:

        print(patch)

        input_path = f'{patch}/sp_output/shape_catalog_{sh}.fits'
        ra, dec, g1, g2, w, mag, snr = read_shape_catalog(input_path)

        ra_all = np.append(ra_all, ra)
        dec_all = np.append(dec_all, dec)
        w_all = np.append(w_all, w)
        mag_all = np.append(mag_all, mag)
        snr_all = np.append(snr_all, snr)
        
        g = np.array([g1, g2])

        # Calibrate with global R and c
        c_corr = Rm1.dot(c)

        g_corr_mc = Rm1.dot(g)
        for comp in (0, 1):
            g_corr_mc[comp] = g_corr_mc[comp] - c_corr[comp]
        g1_corr_mc_all = np.append(g1_corr_mc_all, g_corr_mc[0])
        g2_corr_mc_all = np.append(g2_corr_mc_all, g_corr_mc[1])

    output_path = 'joint.fits'
    g_corr_mc_all = np.array([g1_corr_mc_all, g2_corr_mc_all])
    write_shape_catalog(output_path, ra_all, dec_all, g_corr_mc_all, w_all, mag_all, snr_all, R, R_shear, R_select, c, c_err, alpha)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

