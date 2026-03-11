"""
Scripts to postprocess the CosmoSIS chains
Author: Sacha Guerrini
"""
import numpy as np

def compute_average(chain, param_name):
    """
    Compute the average of a parameter from a CosmoSIS chain
    """
    margestats = chain.getMargeStats()
    param_stats = margestats.parWithName(param_name)
    return param_stats.mean

def compute_map_1D(chain, param_name, num_bins=1000):
    """
    Compute the MAP value of a parameter from a CosmoSIS chain using 1D KDE
    """
    param_names_getdist = chain.getParamNames()
    par = param_names_getdist.parWithName(param_name)
    kde = chain.get1DDensity(par, num_bins=num_bins)
    kde_map = kde.x[np.argmax(kde.P)]
    return kde_map

def compute_map_2D(chain, param_name_x, param_name_y, num_bins=1000):
    """
    Compute the MAP value of two parameters from a CosmoSIS chain using 2D KDE
    """
    param_names_getdist = chain.getParamNames()
    par_x = param_names_getdist.parWithName(param_name_x)
    par_y = param_names_getdist.parWithName(param_name_y)
    kde = chain.get2DDensity(par_x, par_y, fine_bins_2D=num_bins)
    kde_map_index = np.unravel_index(np.argmax(kde.P), kde.P.shape)
    return kde.x[kde_map_index[1]], kde.y[kde_map_index[0]]

def compute_limits(chain, param_name):
    """
    Compute the 68% and 95% confidence limits of a parameter from a CosmoSIS chain.
    """
    margestats = chain.getMargeStats()
    param_stats = margestats.parWithName(param_name)
    return param_stats.limits[0].upper, param_stats.limits[0].lower, param_stats.limits[1].upper, param_stats.limits[1].lower

def load_samples_and_write_paramnames(path_samples, path_paramnames):
    """
    Load the samples from a CosmoSIS chain and write the parameter names to a file
    """
    with open(path_samples, 'r') as file:
        params = file.readline()[1:].split('\t')[:-4]
        file.close()

    with open(path_paramnames, 'w') as file:
        for i in range(len(params)):
            if len(params[i].split('--')) > 1:
                file.write(params[i].split('--')[1] + '\n')
            else:
                file.write(params[i].split('--')[0] + '\n')
        file.close()
    return 0

def write_samples_getdist_format(path_samples, path_gd, chain_type='polychord'):
    """
    Load the samples from a CosmoSIS chain and write them in GetDist format
    """
    samples = np.loadtxt(
        path_samples
    )
    if chain_type == 'nautilus':
        samples = np.column_stack((np.exp(samples[:,-3]),samples[:,-1]-samples[:,-2],samples[:,0:-3]))
    else:
        samples = np.column_stack((samples[:,-1],samples[:,-2],samples[:,0:-4]))
    np.savetxt(path_gd, samples)
    return 0

