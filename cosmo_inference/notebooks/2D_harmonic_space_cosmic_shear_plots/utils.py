import numpy as np
import scipy.stats as stats

def get_chi2_and_pte(data_vector, cov, verbose=True):
    """
    Calculate chi2 and pte for a given data vector and covariance matrix.
    """
    # Calculate chi2
    chi2 = data_vector @ np.linalg.inv(cov) @ data_vector
    if verbose:
        print(f"Chi2: {chi2:.4f}")
    
    #Calculate the reduced chi^2
    dof = len(data_vector)
    reduced_chi2 = chi2 / dof
    if verbose:
        print(f"Reduced Chi2: {reduced_chi2:.4f}")
    
    # Calculate pte
    pte = 1 - stats.chi2.cdf(chi2, dof)
    if verbose:
        print(f"PTE: {pte:.4f}")
    
    return chi2, reduced_chi2, pte