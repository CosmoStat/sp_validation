"""
  
:Name: survey.py

:Description: This script contains methods to deal with the survey:
    geometry, missing tiles, area, ...

:Author: Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

import os


def get_area(dd, area_tile, verbose=False):
    """Get area
    Return survey area.

    Parameters
    ----------
    dd : dict
        galaxy catalog
    area_tile : float
        area per tile in square degree
    verbose : bool
        verbose output if True

    Returns
    -------
    area_deg2 : float
        area in square degrees
    area_amin2 : float
        area in square arcmin
    tile_IDs : array of float
        tile IDs
    """

    if 'TILE_ID' in dd.dtype.names:
        # Set nnumer of tiles to number of unique IDs found in cat
        tile_IDs  = set(dd['TILE_ID'])
        n_tile = len(tile_IDs)
        if verbose:
            print('Number of tiles found in galaxy catalogue = {}'.format(n_tile))
    else:    
        # Set number of tiles by hand if information is not in catalogue
        tile_IDs = None
        n_tiles = 1

    area_deg2 = n_tile * area_tile
    area_amin2 = area_deg2 * 3600

    if verbose:
        print('Area [deg^2] = {}'.format(area_deg2))

    return area_deg2, area_amin2, tile_IDs


def missing_tiles(tile_IDs, path_tile_ID, path_missing_ID, verbose=False):
    """Missing tiles
    Compute completeness and identify missing tiles

    Parameters
    ----------
    tile_IDs : list of string
        input tile IDs in catalogue
    path_tile_ID : string
        input tile ID path to match
    path_missing_ID : string
        output missing tile ID path
    verbose : bool
        verbose output if True

    Returns
    -------
    n_found : int
        number of tiles found
    n_missing : int
        number of tiles missing
    """

    if os.path.exists(path_tile_ID):
        
        missing_IDs = []
        found_IDs = []
        with open(path_tile_ID) as f_in:
            for line in f_in:
                ID = line.rstrip()
                if float(ID) not in tile_IDs:
                    missing_IDs.append(ID)
                else:
                    found_IDs.append(ID)
                    
        n_missing = len(missing_IDs)
        n_found = len(found_IDs)

        if verbose:
            n_tile = len(tile_IDs)
            print('{}/{} = {:.2g}% tiles missing'
                ''.format(n_missing, n_tile, n_missing / n_tile * 100))

        if n_missing > 0:
            if verbose:
                print('Creating file \'{}\''.format(path_missing_ID))
            with open(path_missing_ID, 'w') as f_out:
                for ID in missing_IDs:
                    print(ID, file=f_out)
            f_out.close()

        return n_found, n_missing
            
    else:
        if verbose:
            print('Tile ID file \'{}\' not found'
                ''.format(path_tile_ID))

        return None, None
