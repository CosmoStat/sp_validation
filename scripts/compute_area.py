#!/usr/bin/env python3


import sys
import os
import re

import numpy as np
from uncertainties import ufloat
import glob


def main(argv=None):
    """Main

    Main program
    """

    # File variables
    tile_ID_path = 'tile_numbers.txt'

    random_log_path = 'output/run_sp_Rc/random_cat_runner/logs'
    log_file_base = 'process'

    # Get expected number of tiles in patch
    num_lines = sum(1 for _ in open(tile_ID_path))
    print(f'Found {num_lines} tiles in ID file {tile_ID_path}')

    # Get all log files of random module run
    log_files = glob.glob(f'{random_log_path}/{log_file_base}*')
    n_logs = len(log_files)
    print(f'Found {n_logs} log files')

    area_deg2_non_overl = np.empty(len(log_files)) * np.nan
    area_deg2_non_overl_wgal = np.empty(len(log_files)) * np.nan
    area_deg2_eff_non_overl = np.empty(len(log_files)) * np.nan
    area_deg2_eff_non_overl_wgal = np.empty(len(log_files)) * np.nan
    ratio_unmasked_tot = np.empty(len(log_files)) * np.nan

    # Open stats tile ID gal count file
    sh = 'ngmix'
    tile_ID_from = 'random'
    n_gal = {}
    if tile_ID_from == 'final':
        dat = np.loadtxt(f'sp_output/tile_id_gal_counts_{sh}.txt')
        tile_ID = dat[:, 0]
        for idx, t_ID in enumerate(tile_ID):
            n_gal[f'{t_ID:07.3f}'] = dat[idx, 2]
    elif tile_ID_from == 'random':
        dat = np.loadtxt(f'sp_output_random/found_ID.txt')
        tile_ID = dat
        for idx, t_ID in enumerate(tile_ID):
            n_gal[f'{t_ID:07.3f}'] = 1

    n_tile_no_gal = 0
    n_tile_wgal = 0

    # Loop over log files
    for idx, log_file in enumerate(log_files):

        no_gal = False
        pattern = re.compile(f'.*{log_file_base}(.*)-(.*)\.log')
        m = re.match(pattern, log_file)
        if m:
            this_id = f'{m[1]}.{m[2]}'
            if tile_ID_from == 'final':
                if this_id in n_gal:
                    if n_gal[this_id] == 0:
                        n_tile_no_gal += 1
                        no_gal = True
            else:
                if this_id not in n_gal:
                    no_gal = True
        else:
            raise ValueError('Invalid log file format')

        if not no_gal:
            n_tile_wgal += 1

        with open(log_file) as fin:
            log_content = fin.readlines()

        # Loop over lines in log file
        for line in log_content:
            m = re.search('Total area without overlap = (\S+) deg', line)
            if m:
                area_deg2_non_overl[idx] = float(m[1])
                if not no_gal:
                    area_deg2_non_overl_wgal[idx] = float(m[1])
                else:
                    area_deg2_non_overl_wgal[idx] = 0

            m = re.search('Unmaskewd area without overlap = (\S+) deg', line)
            if m:
                area_deg2_eff_non_overl[idx] = float(m[1])
                if not no_gal:
                    area_deg2_eff_non_overl_wgal[idx] = float(m[1])
                else:
                    area_deg2_eff_non_overl_wgal[idx] = 0

            m = re.search('Ratio masked to total pixels = (\S+)', line)
            if m:
                ratio_unmasked_tot[idx] = float(m[1])

    print(f'Tiles with zero galaxies = {n_tile_no_gal}')
    print(f'Tiles with galaxies = {n_tile_wgal}')

    area_deg2_non_overl_total = np.sum(area_deg2_non_overl)
    area_deg2_non_overl_total_wgal = np.sum(area_deg2_non_overl_wgal)
    area_deg2_non_overl_tile = ufloat(np.mean(area_deg2_non_overl), np.std(area_deg2_non_overl))
    print(f'Patch area without overlap = {area_deg2_non_overl_total:.3f} deg^2')
    print(f'Patch area without overlap and no 0 gal = {area_deg2_non_overl_total_wgal:.3f} deg^2')
    print(f'Tile area without overlap = {area_deg2_non_overl_tile:.3fP} deg^2')

    area_deg2_eff_non_overl_total = np.sum(area_deg2_eff_non_overl)
    area_deg2_eff_non_overl_total_wgal = np.sum(area_deg2_eff_non_overl_wgal)
    area_deg2_eff_non_overl_tile = ufloat(np.mean(area_deg2_eff_non_overl), np.std(area_deg2_eff_non_overl))
    print(f'Unmasked patch area without overlap = {area_deg2_eff_non_overl_total:.3f} deg^2')
    print(f'Unmasked patch area without overlap and no 0 gal = {area_deg2_eff_non_overl_total_wgal:.3f} deg^2')
    print(f'Unmasked tile area without overlap = {area_deg2_eff_non_overl_tile:.3fP} deg^2')

    ratio_unmasked_tile = ufloat(np.mean(ratio_unmasked_tot), np.std(ratio_unmasked_tot))
    print(f'Ratio of unmasked to total pixels = {ratio_unmasked_tile:.3fP} (min = {np.min(ratio_unmasked_tot):.3f})')

if __name__ == "__main__":
    sys.exit(main(sys.argv))
