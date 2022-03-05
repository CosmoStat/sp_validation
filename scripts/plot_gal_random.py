#!/usr/bin/env python3

import numpy as np
import matplotlib.pylab as plt

from astropy.io import fits

shear_cat_name = 'sp_output/shape_catalog_ngmix.fits'
random_cat_name = 'sp_output_random/random_catalog.fits'

gal = fits.getdata(shear_cat_name)
random = fits.getdata(random_cat_name)

print(gal.dtype.names)
print(random.dtype.names)

plt.figure(figsize=(20, 20))

plt.plot(gal['ra'], gal['dec'], '.', markersize=1)
plt.plot(random['ra'], random['dec'], '.', markersize=1)
plt.xlabel('R.A. [deg]')
plt.ylabel('DEC [deg]')
plt.title(f'{len(gal)} galaxies')
plt.title(f'{len(random)} random objects')
plt.savefig('gal+rand.png')
plt.show()
