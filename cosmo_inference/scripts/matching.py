# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:37:27 2023
@author: fh272693
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import treecorr
from astropy.coordinates import match_coordinates_sky
import astropy.units as u
from astropy.coordinates import SkyCoord

Cat1=fits.open('/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues/unions_shapepipe_2022_v1.0.fits')
Cat2=fits.open('/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues/lensfit_goldshape_2022v1.fits')

coord_units = u.degree
Cat1_coord = SkyCoord(
    ra=Cat1[1].data['ra'] * coord_units,
    dec=Cat1[1].data['dec'] * coord_units
)
Cat2_coord = SkyCoord(
    ra=Cat2[1].data['ra'] * coord_units,
    dec=Cat2[1].data['dec'] * coord_units
)
idx, d2d, d3d = match_coordinates_sky(Cat1_coord, Cat2_coord) 
max_sep = 1. * u.arcsec
sep_constraint = d2d < max_sep

#Important here is that the first catalogue of match_coordinates_sky has 
#indices [sep_constraint] and the second[idx[sep_constraint]]
Cat1_matches = Cat1[1].data[sep_constraint]

# np.save('/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues/shapepipe_unmatches_ra.npy',Cat1[1].data['ra'][sep_constraint])
# np.save('/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues/shapepipe_unmatches_dec.npy',Cat1[1].data['dec'][sep_constraint])
# np.save('/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues/shapepipe_unmatches_e1.npy',Cat1[1].data['e1'][sep_constraint])
# np.save('/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues/shapepipe_unmatches_e2.npy',Cat1[1].data['e2'][sep_constraint])
# np.save('/feynman/work/dap/lcs/lg268561/UNIONS/Catalogues/shapepipe_unmatches_w.npy',Cat1[1].data['w'][sep_constraint])

print('there are ',len(Cat1_matches),' matching galaxies in catalogue', Cat1)
# print('there are ',len(Cat2_matches),' matching galaxies in catalogue', Cat2)