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

def get_footprint(patch, ra, dec):
    """Get Footprint

    Return coordinates within footprint of patch

    Parameters
    ----------
    patch : str
        patch name
    ra : array of float
        R,A, coordintates
    dec : array of float
        DEC coordinates

    Returns
    -------
    ra, dec : arrays of float
        list of coordinates withint footprint
    """

    ra_14 = 157.5
    ra_45 = 207
    ra2_45 = 220
    ra_36 = 230
    dec_3456 = 48

    ra2_34 = 190

    dec_min = 29
    dec_max = 60

    if patch == 'P1':

        return (ra > 100) & (ra < ra_14) & (dec > dec_min) & (dec < dec_max)

    elif patch == 'P2':

        # -30 < ra < 60
        return (
            ((ra > 0) & (ra < 60))
            | ((ra > 330) & (ra < 360))
            & (dec > dec_min) & (dec < dec_max)
        )

    elif patch == 'P3':

        return (ra > ra2_34) & (ra < ra_36) & (dec > dec_3456) & (dec < 70)

    elif patch == 'P4':

        return (
            ((ra > ra_14) & (ra < ra_45) & (dec > dec_min) & (dec < dec_3456))
            | ((ra > ra_14) & (ra < ra2_34) & (dec > dec_min) & (dec < 70))
            | ((ra > ra_45) & (ra < ra2_45) * (dec > dec_min) & (dec < 36))
        )

    elif patch == 'P5':

        return (
            ((ra > ra2_45) & (ra < 330) & (dec > dec_min) & (dec < dec_3456))
            | ((ra > ra_45) & (ra < ra2_45) & (dec > 36) & (dec < dec_3456))
        )

    elif patch == 'P6':

        return (ra > ra_36) & (ra < 330) & (dec > dec_3456) & (dec > 70)

    elif patch == 'P7':

        return (ra > 60) & (ra < 180) & (dec > 60) & (dec < 90)

    elif patch == 'W3':

        return (ra > 208) & (ra < 221) & (dec > 51) & (dec < 58)

    else:

        return (dec > dec_min)
