import numpy as np
import sys
import re

def main(argv=None):

    survey = 'v1'

    if survey == 'v1':
        n_patch = 7
        patches = [f'P{x}' for x in np.arange(n_patch) + 1]

    IDs_sp_base = 'found_ID_wshapes.txt' 
    tile_ID_gal_counts_sp_base = 'tile_id_gal_counts_ngmix.txt'

    n_SP_not_in_LF_all = 0
    n_LF_not_in_SP_all = 0

    for patch in patches:

        print(patch)

        # ShapePipe
        path = f'{patch}/sp_output/{IDs_sp_base}'

        with open(path) as f:
            dat = f.readlines()
        ID_SP = []
        for line in dat:
            ID_SP.append(line.rstrip())
        print(f' #SP = {len(ID_SP)}')

        # LensFit
        path = f'CFIS3500_THELI_{patch}.list'
        with open(path) as f:
            dat = f.readlines()
        ID_LF = []
        for line in dat:
            m = re.match('.*CFIS\.(\d{3}\.\d{3})\.r', line)
            if m:
                ID_LF.append(m[1].rstrip())
        print(f' #LF = {len(ID_LF)}')

        # tile stats for SP
        dat = np.loadtxt(f'{patch}/sp_output/{tile_ID_gal_counts_sp_base}')
        tile_ID = []
        for my_ID in dat[:, 0]:
            tile_ID.append(f'{my_ID:07.3f}')
        n_det = dat[:, 1]
        n_gal = dat[:, 2]
        n_shape = dat[:, 3]

        # ShapePipe tiles not contained in LensFit
        n_SP_not_in_LF = 0
        for ID in ID_SP:
            if ID not in ID_LF:
                # Print number of galaxies on those tiles not contained in LF
                if ID in tile_ID:
                    idx = tile_ID.index(ID)
                    #print(ID, tile_ID[idx], n_det[idx], n_gal[idx], n_shape[idx])
                n_SP_not_in_LF += 1
                if n_SP_not_in_LF == -1:
                    print(f'  SP {ID} not in LF')
        n_SP_not_in_LF_all += n_SP_not_in_LF

        # LensFit tiles not contained in ShapePipe
        n_LF_not_in_SP = 0
        for ID in ID_LF:
            if ID not in ID_SP:
                n_LF_not_in_SP += 1
                if n_LF_not_in_SP == -1:
                    print(f'  LF {ID} not in SP')
                    print(f'   [{ID}]')
        n_LF_not_in_SP_all += n_LF_not_in_SP

        print(f' # SP not in LF = {n_SP_not_in_LF}')
        print(f' # LF not in SP = {n_LF_not_in_SP}')

    print()
    print('All patches')
    print(f' # SP not in LF = {n_SP_not_in_LF_all}')
    print(f' # LF not in SP = {n_LF_not_in_SP_all}')

    return 0
        
if __name__ == "__main__":
    sys.exit(main(sys.argv))

