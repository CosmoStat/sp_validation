#!/usr/bin/env python3

import numpy as np
import matplotlib.pylab as plt

from sp_validation import galaxy

rand = np.load('random_cat.npy')
print(rand.dtype.names)

markersize = 0.01

plt.figure(figsize=(10, 10))

plt.plot(rand['RA'], rand['DEC'], '.', markersize=markersize)
plt.xlabel('R.A. [deg]')
plt.ylabel('DEC [deg]')
plt.title(f'{len(rand)} random objects with overlap')
plt.savefig('rand_wo.png')
plt.show()

plt.clf()

# Duplicate objects due to tile overlaps
cut_overlap = galaxy.classification_galaxy_overlap_ra_dec(rand, ra_key='RA', dec_key='DEC')
rand_no = rand[cut_overlap]

plt.plot(rand_no['RA'], rand_no['DEC'], '.', markersize=markersize)
plt.xlabel('R.A. [deg]')
plt.ylabel('DEC [deg]')
plt.title(f'{len(rand_no)} random objects overlap removed')
plt.savefig('rand_no.png')
plt.show()
