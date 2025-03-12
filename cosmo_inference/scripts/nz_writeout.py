#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits

nz_hdu = sys.argv[1]
root = sys.argv[2]
blind = sys.argv[3]

hdu = fits.open(nz_hdu)
z = hdu[1].data['Z_%s' %blind]

zmax = 5.0

(n,bins,_)= plt.hist(z, bins=200, range=(0,zmax), density=True, weights=None)

print("zmin = ",min(z))
print("zmax = ",max(z))

np.savetxt('data/'+root+'/nz_'+root+'.txt',np.column_stack((bins[:-1],n)))