#!/usr/bin/env python3

import numpy as np
import matplotlib.pylab as plt

rand = np.load('random_cat.npy')
print(rand.dtype.names)

print(rand['DEC'])

# Duplicate objects due to tile overlaps
cut_overlap = (
    rand['FLAG_TILING'] == 1
)
rand_no = rand[cut_overlap]

plt.figure(figsize=(10, 10))

plt.plot(rand['RA'], rand['DEC'], '.', markersize=1)
plt.xlabel('R.A. [deg]')
plt.ylabel('DEC [deg]')
plt.title(f'W3, {len(rand_no)} random objects')
plt.savefig('rand.png')
plt.show()
