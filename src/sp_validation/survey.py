"""SURVEY.

:Description: This file contains methods to deal with the survey geometry.
Some methods are UNIONS-/CFIS-specific.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""


import os

from collections import Counter


def get_area(dd, area_tile, verbose=False):
    """Get area.

    Return total survey area. This ignores masking as well as
    overlaps between tiles, and provides therefore an overestimation.
    of the actual observed area.

    Parameters
    ----------
    dd : numpy.ndarray
        galaxy catalog
    area_tile : float
        area per tile in square degree
    verbose : bool, optional
        verbose output if `True`, default is `False`

    Returns
    -------
    float
        area in square degrees
    float
        area in square arc minutes
    list of float
        tile IDs

    """
    if 'TILE_ID' in dd.dtype.names:
        # Get unique tile IDs
        tile_IDs = set(dd['TILE_ID'])
        n_tile = len(tile_IDs)
        if verbose:
            print(f'Number of tiles found in galaxy catalogue = {n_tile}')
    else:
        # Set to dummy values
        tile_IDs = None
        n_tiles = 1

    # Compute area
    area_deg2 = n_tile * area_tile
    area_amin2 = area_deg2 * 3600

    if verbose:
        print('Area [deg^2] = {}'.format(area_deg2))

    return area_deg2, area_amin2, tile_IDs


def missing_tiles(
    tile_IDs,
    path_tile_ID,
    path_found_ID,
    path_missing_ID,
    verbose=False,
):
    """Missing tiles.

    Compute completeness, and identify missing tiles.

    Parameters
    ----------
    tile_IDs : list of string
        input tile IDs in catalogue
    path_tile_ID : string
        input tile ID path to match
    path_found_ID : string
        output found tile ID path
    path_missing_ID : string
        output missing tile ID path
    verbose : bool
        verbose output if True

    Returns
    -------
    int
        number of tiles found, -1 if ID file path not found
    int
        number of tiles missing, -1 if ID file path not found
    """
    if os.path.exists(path_tile_ID):

        # Loop over input tile ID file
        found_IDs = []
        missing_IDs = []
        with open(path_tile_ID) as f_in:
            for line in f_in:
                ID = line.rstrip()

                if float(ID) not in tile_IDs:
                    # Add ID to missing if not in input ID file
                    missing_IDs.append(ID)
                else:
                    # Add ID to found if not in input ID file
                    found_IDs.append(ID)

        n_found = len(found_IDs)
        n_missing = len(missing_IDs)

        if verbose:
            n_tile = len(tile_IDs)
            print(
                f'{n_missing}/{n_tile} = {n_missing / n_tile:.2%}'
                + ' tiles missing'
            )

        # Create output files with found and missing IDs
        if n_found > 0:
            if verbose:
                print(f'Creating file \'{path_found_ID}\'')
            with open(path_found_ID, 'w') as f_out:
                for ID in found_IDs:
                    print(ID, file=f_out)

        if n_missing > 0:
            if verbose:
                print('Creating file \'{path_missing_ID}\'')
            with open(path_missing_ID, 'w') as f_out:
                for ID in missing_IDs:
                    print(ID, file=f_out)

    else:
        if verbose:
            print(f'Tile ID file \'{path_tile_ID}\' not found')

        # Set to dummy values
        n_found = -1
        n_missing = -1

    return n_found, n_missing


def write_tile_id_gal_counts(detection_IDs, galaxy_IDs, shape_IDs, fname):
    """Write Tile ID Galaxy Counts.

    Write number of galaxies per tile ID to file.

    Parameters
    ----------
    detection_IDs : list
        input tile ID for each detected object
    galaxy_IDs : list
        input tile ID for each selected galaxy
    shape_IDs : list
        input tile ID for each galaxy with measured shape
    fname : str
        output file name

    """
    # Create Counter objects of input ID lists, to get number of objects
    # on each tile that are detected / selected as galaxy / for which
    # a shape is measured
    detection_counts = Counter(detection_IDs)
    galaxy_counts = Counter(galaxy_IDs)
    shape_counts = Counter(shape_IDs)

    with open(fname, 'w') as f:

        # Loop over tile IDs of detected objects
        for tile_id in detection_counts:

            # Write tile ID in CFIS tile format (`ABC.XYZ`)
            print(f'{tile_id:007.3f}', end=' ', file=f)

            # Loop over counters
            for x in detection_counts, galaxy_counts, shape_counts:

                # Get number of objects in corresponding counter
                if tile_id in x:
                    num = x[tile_id]
                else:
                    num = 0

                # Write number to file
                print(num, end=' ', file=f)
            print(file=f)


def get_footprint(patch, ra, dec):
    """Get Footprint.

    Return coordinates within footprint of patch.

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
    list of float
        list of coordinates withint footprint

    """
    # Set boundary coordinates between some of the patches
    ra_14 = 157.5
    ra_45 = 207
    ra2_45 = 220
    ra_36 = 230
    dec_3456 = 48

    ra2_34 = 190

    dec_min = 29
    dec_max = 60

    # Check whether input matches one of the seven CFIS patch name.
    # Return coordinates within the patch
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
