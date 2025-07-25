#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import sys
from pathlib import Path

def get_cov(filename):

	data = np.loadtxt(filename)
	ndata = int(np.max(data[:,0]))+1

	print("Dimension of cov: %dx%d"%(ndata,ndata))

	ndata_min = int(np.min(data[:,0]))
	cov_g = np.zeros((ndata,ndata))
	cov_ng = np.zeros((ndata,ndata))
	for i in range(0,data.shape[0]):
		cov_g[int(data[i,0]),int(data[i,1])] =data[i,8]
		cov_g[int(data[i,1]),int(data[i,0])] =data[i,8]
		cov_ng[int(data[i,0]),int(data[i,1])] =data[i,9]
		cov_ng[int(data[i,1]),int(data[i,0])] =data[i,9]

	return cov_g, cov_ng, ndata


if __name__ == '__main__':
	
	if len(sys.argv) != 3:
		print("Usage: python cosmocov_process.py <input_file> <output_stub>")
		sys.exit(1)
	
	covfile = sys.argv[1]
	output_base = sys.argv[2]
	
	c_g, c_ng, ndata = get_cov(covfile)
	
	cov = c_ng+c_g
	cov_g = c_g

	b = np.sort(LA.eigvals(cov))
	print("min+max eigenvalues cov: %e, %e"%(np.min(b), np.max(b)))
	if(np.min(b)<=0.):
		print("non-positive eigenvalue encountered! Covariance Invalid!")
		exit()

	print("Covariance is postive definite!")

	pp_var = []
	for i in range(ndata):
		pp_var.append(cov[i][i])

	np.savetxt(str(output_base)+'.txt',cov)
	print("covmat saved as %s" %(str(output_base)+'.txt'))
    
	np.savetxt(str(output_base)+'_g.txt',cov_g)
	print("Gaussian covmat saved as %s" %(str(output_base)+'_g.txt'))

	cmap = 'seismic'

	pp_norm = np.zeros((ndata,ndata))
	for i in range(ndata):
		for j in range(ndata):
			pp_norm[i][j] = cov[i][j]/ np.sqrt(cov[i][i]*cov[j][j])

	print("Plotting correlation matrix ...")

	plot_path = str(output_base)+'_plot.pdf'
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1) 
	extent = (0, ndata, ndata, 0)
	im3 = ax.imshow(pp_norm, cmap=cmap, vmin=-1, vmax=1, extent=extent)
 
	plt.axvline(x=int(ndata/2),color='black',linewidth=1.0)
	plt.axhline(y=int(ndata/2),color='black',linewidth=1.0)

	fig.colorbar(im3, orientation='vertical')

	ax.text(int(ndata/4), ndata+5, r'$\xi_+^{ij}(\theta)$', fontsize=12)
	ax.text(3*int(ndata/4), ndata+5, r'$\xi_-^{ij}(\theta)$', fontsize=12)
	ax.text(-9, int(ndata/4), r'$\xi_+^{ij}(\theta)$', fontsize=12)
	ax.text(-9, 3*int(ndata/4), r'$\xi_-^{ij}(\theta)$', fontsize=12)

	plt.savefig(plot_path,dpi=2000)
	plt.show()
	print("Plot saved as %s"%(plot_path))

