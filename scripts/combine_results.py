#!/usr/bin/env python3


import sys
import os
import re

import numpy as np


def get_match(stats_files, patch, pattern):

    for line in stats_files[patch]:
        m = re.search(pattern, line)
        if m:
            res = m[1]
            return res

    raise ValueError(f'No match of \'{pattern}\' in patch {patch}')



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
        

def main(argv=None):
    """Main

    Main program
    """

    n_patch = 7
    patches = [f'P{x}' for x in np.arange(n_patch) + 1]

    directory = 'sp_output/plots'
    fname = 'stats_file.txt'
    path = f'{directory}/{fname}'

    verbose = True

    stats_files = read_stats_files(patches, path, verbose=verbose)
    n_patch_found = len(stats_files)

    for patch in stats_files:
        res = get_match(stats_files, patch, 'Number of galaxies after metacal = (\d*)/')
        print(patch, res)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

