#!/usr/bin/env python

import sys
import re
import numpy as np


def main(argv=None):

    types = ['star', 'gal', 'other']
    text = 'Number of stars selected as'

    n_patch = 7                                                             
    patches = [f'P{x}' for x in np.arange(n_patch) + 1]

    ntyp = {}
    ntot = {}
    for typ in types:
        ntyp[typ] = 0
        ntot[typ] = 0

    for patch in patches:
        #print(patch)
        path = f'{patch}/sp_output/plots/stats_file.txt'
        with open(path, 'r') as fin:
            lines = fin.readlines()
            for typ in types:
                for line in lines:
                    pattern = f'{text} {typ}.*= (\d+)/(\d+)'
                    m = re.search(pattern, line)
                    if m:
                        ntyp_patch = int(m.group(1))
                        ntot_patch = int(m.group(2))
                        #print(typ, m.group(1), m.group(2))
                        ntyp[typ] += ntyp_patch
                        ntot[typ] += ntot_patch

    for typ in types:
        print(f'{text} {typ} = {ntyp[typ]}/{ntot[typ]} = {ntyp[typ]/ntot[typ]:.2%}')


if __name__ == "__main__":                                                      
    sys.exit(main(sys.argv))
