"""

:Name: basic.py

:Description: This file contains methods for calibration (metacalibration) and
              basic validation of a weak-lensing shape catalogue
              independent of cosmology.

:Author: Axel Guinot, Martin Kilbinger

:Date: 2021

:Package: sp_validation

"""

from astropy.io import fits
from astropy import units as u
from astropy import coordinates as coords
from astropy.wcs import WCS
from astropy.table import Table

#import galsim

import numpy as np

import operator as op
import itertools as itools

from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit, minimize
from scipy.special import gamma

from shapepipe.pipeline import file_io as io

from tqdm import tqdm
from joblib import Parallel, delayed


from lenspack.utils import bin2d
from lenspack.image.inversion import ks93
from lenspack.geometry.projections.gnom import radec2xy

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch


def mad(data, axis=None):
    return np.median(np.abs(data - np.median(data, axis)), axis)*1.4826


def weighted_std(x, w):
    """
    """

    return np.sqrt(len(x)/(len(x) - 1) * np.average((x - np.average(x, weights=w))**2, weights=w))

def weighted_median(values, weights):
    ''' Compute the weighted median of values list. 
    The weighted median is computed as follows:
    1- sort both lists (values and weights) based on values.
    2- select the 0.5 point from the weights and return the corresponding values as results
    e.g. values = [1, 3, 0] and weights=[0.1, 0.3, 0.6] assuming weights are probabilities.
    sorted values = [0, 1, 3] and corresponding sorted weights = [0.6,     0.1, 0.3] the 0.5 point on
    weight corresponds to the first item which is 0. so the weighted     median is 0.
    '''

    #convert the weights into probabilities
    sum_weights = sum(weights)
    weights = np.array([(w*1.0)/sum_weights for w in weights])
    #sort values and weights based on values
    values = np.array(values)
    sorted_indices = np.argsort(values)
    values_sorted  = values[sorted_indices]
    weights_sorted = weights[sorted_indices]
    #select the median point
    it = np.nditer(weights_sorted, flags=['f_index'])
    accumulative_probability = 0
    median_index = -1
    while not it.finished:
        accumulative_probability += it[0]
        if accumulative_probability > 0.5:
            median_index = it.index
            return values_sorted[median_index]
        elif accumulative_probability == 0.5:
            median_index = it.index
            it.iternext()
            next_median_index = it.index
            return np.mean(values_sorted[[median_index, next_median_index]])
        it.iternext()

    return values_sorted[median_index]


def get_Isaac_nz(mag, weight, z, alpha=1.79, sigma=1.3, mu=1):
    """
    """

    def A(m):
        """
        """

        return -0.4154*m**2.+19.1734*m-220.261

    def z0(m):
        """
        """
        
        return 0.1081*m-1.9417

    m = weighted_median(mag, weight)

    term1 = A(m) * (z**alpha * np.exp(-(z/z0(m))**alpha))/(z0(m)**(alpha+1)/alpha * gamma(alpha+1/alpha))
    term2 = (1-A(m)) * np.exp(-(z-mu)**2./(2*sigma**2.))/2.5046

    return term1 + term2


class metacal:
    """
    """

    def __init__(self, data, mask, masking_type='gal', step=0.01, stat_operator=np.mean, prefix='NGMIX'):

        self._prefix = prefix
        if prefix == 'NGMIX':
            self._read_data_ngmix(data, mask, prefix)
        elif prefix == 'GALSIM':
            self._read_data_galsim(data, mask, prefix)
        if masking_type == 'gal':
            self._masking_gal()
        elif masking_type == 'galmom':
            self._masking_gal_mom()
        elif masking_type == 'star':
            self._masking_star()
        self._shear_response(step)
        self._selection_response(stat_operator)
        self._total_response(stat_operator)

    def _read_data_ngmix(self, data, mask, prefix):
        """
        """

        m1 = {}
        p1 = {}
        m2 = {}
        p2 = {}
        ns = {}
        for name_shear, dict_tmp in zip(['1m', '1p', '2m', '2p', 'noshear'], [m1, p1, m2, p2, ns]):
            print('Extracting {}'.format(name_shear))
            dict_tmp['flag'] = data['{}_FLAGS_{}'.format(prefix, name_shear.upper())][mask]
            dict_tmp['g1'] = data['{}_ELL_{}'.format(prefix, name_shear.upper())][:,0][mask]
            dict_tmp['g2'] = data['{}_ELL_{}'.format(prefix, name_shear.upper())][:,1][mask]
            # dict_tmp['s2n'] = data['{}_SNR_{}'.format(prefix, name_shear.upper())][mask]
            if prefix == 'NGMIX':
                dict_tmp['flux'] = data['{}_FLUX_{}'.format(prefix, name_shear.upper())][mask]
                dict_tmp['flux_err'] = data['{}_FLUX_ERR_{}'.format(prefix, name_shear.upper())][mask]
            dict_tmp['T'] = data['{}_T_{}'.format(prefix, name_shear.upper())][mask]
            dict_tmp['T_err'] = data['{}_T_ERR_{}'.format(prefix, name_shear.upper())][mask]
            dict_tmp['Tpsf'] = data['{}_Tpsf_{}'.format(prefix, name_shear.upper())][mask]
            # dict_tmp['C11'] = data['{}_ELL_ERR_{}'.format(prefix, name_shear.upper())][:,0][mask]
            # dict_tmp['C22'] = data['{}_ELL_ERR_{}'.format(prefix, name_shear.upper())][:,1][mask]
            # dict_tmp['w'] = 1./(2*0.34**2. + dict_tmp['C11'] + dict_tmp['C22'])
        if prefix == 'GALSIM':
            self.snr_sextractor = data['SNR_WIN'][mask]
        ns['C11'] = data['{}_ELL_ERR_NOSHEAR'.format(prefix)][:,0][mask]
        ns['C22'] = data['{}_ELL_ERR_NOSHEAR'.format(prefix)][:,1][mask]
        ns['w'] = 1./(2*0.34**2. + dict_tmp['C11'] + dict_tmp['C22'])

        self.m1 = m1
        self.p1 = p1
        self.m2 = m2
        self.p2 = p2
        self.ns = ns

    def _read_data_galsim(self, data, mask, prefix):
        """
        """
    
        prefix2 = prefix
        prefix = 'GALSIM_GAL'

        m1 = {}
        p1 = {}
        m2 = {}
        p2 = {}
        ns = {}
        for name_shear, dict_tmp in zip(['1m', '1p', '2m', '2p', 'noshear'], [m1, p1, m2, p2, ns]):
            print('Extracting {}'.format(name_shear))
            dict_tmp['flag'] = data['{}_FLAGS_{}'.format(prefix2, name_shear.upper())][mask]
            dict_tmp['g1'] = data['{}_ELL_UNCORR_{}'.format(prefix, name_shear.upper())][:,0][mask]
            dict_tmp['g2'] = data['{}_ELL_UNCORR_{}'.format(prefix, name_shear.upper())][:,1][mask]
            # dict_tmp['s2n'] = data['{}_SNR_{}'.format(prefix, name_shear.upper())][mask]
            if prefix == 'NGMIX':
                dict_tmp['flux'] = data['{}_FLUX_{}'.format(prefix, name_shear.upper())][mask]
                dict_tmp['flux_err'] = data['{}_FLUX_ERR_{}'.format(prefix, name_shear.upper())][mask]
            dict_tmp['T'] = data['{}_SIGMA_{}'.format(prefix, name_shear.upper())][mask]
            dict_tmp['Tpsf'] = data['{}_PSF_SIGMA_{}'.format(prefix2, name_shear.upper())][mask]
            # dict_tmp['C11'] = data['{}_ELL_ERR_{}'.format(prefix, name_shear.upper())][:,0][mask]
            # dict_tmp['C22'] = data['{}_ELL_ERR_{}'.format(prefix, name_shear.upper())][:,1][mask]
            # dict_tmp['w'] = 1./(2*0.34**2. + dict_tmp['C11'] + dict_tmp['C22'])
        if prefix2 == 'GALSIM':
            self.snr_sextractor = data['SNR_WIN'][mask]
        ns['C11'] = data['{}_ELL_ERR_NOSHEAR'.format(prefix)][:,0][mask]
        ns['C22'] = data['{}_ELL_ERR_NOSHEAR'.format(prefix)][:,1][mask]
        ns['w'] = 1./(2*0.34**2. + dict_tmp['C11'] + dict_tmp['C22'])

        self.m1 = m1
        self.p1 = p1
        self.m2 = m2
        self.p2 = p2
        self.ns = ns

    def _masking_gal(self):
        """
        """

        self.mask_dict = {}
        for data, name in zip([self.ns, self.m1, self.p1, self.m2, self.p2], ['ns', 'm1', 'p1', 'm2', 'p2']):
            Tr_tmp = data['T'] * (1 - (data['g1']**2. + data['g2']**2.))/(1 + (data['g1']**2. + data['g2']**2.))
            #Tr_err_tmp = data['T_err'] * (1 - (data['g1']**2. + data['g2']**2.))/(1 + (data['g1']**2. + data['g2']**2.))
            if hasattr(self, 'snr_sextractor'):
                snr_flux = self.snr_sextractor
            else:
                snr_flux = data['flux']/data['flux_err']
            # T_psf = np.sqrt(data['Tpsf']/2) * 1.17741
            # mask_tmp = (data['flag'] == 0) & (data['s2n'] > 20) & (data['s2n'] < 10000) & (data['T'] + np.sqrt(data['T_err']) > 0.02) & (Tr_tmp/data['Tpsf'] > 0.5)  & (data['T'] < 2. * np.sqrt(20*0.187/2.355)) #& (dict_tmp['C11'] < 0.7) & (dict_tmp['C22'] < 0.7)

            mask_tmp = (data['flag'] == 0) & (Tr_tmp/data['Tpsf'] > 0.5) & (snr_flux > 10) & (snr_flux < 500) #& (data['s2n'] > 100) & (data['s2n'] < 5000)

            # mask_tmp = (data['flag'] == 0) & (Tr_tmp/T_psf > 1.) & (snr_flux > 10) & (snr_flux < 500)

            # mask_tmp = (data['flag'] == 0) & (np.sqrt((Tr_tmp + 3.*Tr_err_tmp)/2.) * 2.355 > 0.2) & (np.sqrt((Tr_tmp + 3.*Tr_err_tmp)/2.) * 2.355 < 25*0.2) & (np.sqrt(data['Tpsf']/2.)*2.355 > 0.2) & (np.sqrt(data['Tpsf']/2.)*2.355 < 0.9) & ((Tr_tmp +3.*Tr_err_tmp)/data['Tpsf'] > 0.5) & (np.sqrt(data['C11']) < 0.5) & (np.sqrt(data['C22']) < 0.5) & (data['s2n'] > 50) & (data['s2n'] < 5000) #((data['T'] +3.*np.sqrt(data['T_err']))/data['Tpsf'] > 0.5) & (np.sqrt((data['T'] + 3.*np.sqrt(data['T_err']))/2.) * 2.355 > 0.2) & (np.sqrt((data['T'] + 3.*np.sqrt(data['T_err']))/2.) * 2.355 < 25*0.186)

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            # mask_dict[name] = np.array(list(set(np.arange(len(data))) - set(ind_masked)))
            self.mask_dict[name] = ind_masked

    def _masking_gal_mom(self):
        """
        """

        self.mask_dict = {}
        for data, name in zip([self.ns, self.m1, self.p1, self.m2, self.p2], ['ns', 'm1', 'p1', 'm2', 'p2']):
            Tr_tmp = data['T'] * (1 - (data['g1']**2. + data['g2']**2.))/(1 + (data['g1']**2. + data['g2']**2.))
            # Tr_err_tmp = data['T_err'] * (1 - (data['g1']**2. + data['g2']**2.))/(1 + (data['g1']**2. + data['g2']**2.))
            snr_flux = data['flux']/data['flux_err']
            # mask_tmp = (data['flag'] == 0) & (data['s2n'] > 20) & (data['s2n'] < 10000) & (data['T'] + np.sqrt(data['T_err']) > 0.02) & (Tr_tmp/data['Tpsf'] > 0.5)  & (data['T'] < 2. * np.sqrt(20*0.187/2.355)) #& (dict_tmp['C11'] < 0.7) & (dict_tmp['C22'] < 0.7)
            mask_tmp = (data['flag'] == 0)  & (data['s2n'] > 10) & (data['s2n'] < 500) & (data['g1'] != -10) & (data['g1'] != 0) & (1-data['Tpsf']/data['T'] > 0.4) #& (data['s2n'] > 100) & (data['s2n'] < 5000)
            # mask_tmp = (data['flag'] == 0) & (np.sqrt((Tr_tmp + 3.*Tr_err_tmp)/2.) * 2.355 > 0.2) & (np.sqrt((Tr_tmp + 3.*Tr_err_tmp)/2.) * 2.355 < 25*0.2) & (np.sqrt(data['Tpsf']/2.)*2.355 > 0.2) & (np.sqrt(data['Tpsf']/2.)*2.355 < 0.9) & ((Tr_tmp +3.*Tr_err_tmp)/data['Tpsf'] > 0.5) & (np.sqrt(data['C11']) < 0.5) & (np.sqrt(data['C22']) < 0.5) & (data['s2n'] > 50) & (data['s2n'] < 5000) #((data['T'] +3.*np.sqrt(data['T_err']))/data['Tpsf'] > 0.5) & (np.sqrt((data['T'] + 3.*np.sqrt(data['T_err']))/2.) * 2.355 > 0.2) & (np.sqrt((data['T'] + 3.*np.sqrt(data['T_err']))/2.) * 2.355 < 25*0.186)

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            # mask_dict[name] = np.array(list(set(np.arange(len(data))) - set(ind_masked)))
            self.mask_dict[name] = ind_masked

    def _masking_star(self):
        """
        """

        self.mask_dict = {}
        for data, name in zip([self.ns, self.m1, self.p1, self.m2, self.p2], ['ns', 'm1', 'p1', 'm2', 'p2']):
            Tr_tmp = data['T'] * (1 - (data['g1']**2. + data['g2']**2.))/(1 + (data['g1']**2. + data['g2']**2.))
            if hasattr(self, 'snr_sextractor'):
                snr_flux = self.snr_sextractor
            else:
                snr_flux = data['flux']/data['flux_err']
            mask_tmp = (data['flag'] == 0) & (snr_flux > 10) & (snr_flux < 500) #& (data['T'] + np.sqrt(data['T_err']) > 0.02) & (Tr_tmp/data['Tpsf'] > 0.5)  & (data['T'] < 2. * np.sqrt(20*0.187/2.355)) #& (dict_tmp['C11'] < 0.7) & (dict_tmp['C22'] < 0.7)

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            # mask_dict[name] = np.array(list(set(np.arange(len(data))) - set(ind_masked)))
            self.mask_dict[name] = ind_masked

    def _shear_response(self, step=0.01):
        """
        """
        
        sign = 1
        if self._prefix == 'GALSIM':
            sign = -1

        self.R11 = (self.p1['g1'][self.mask_dict['ns']] - self.m1['g1'][self.mask_dict['ns']])/(2.*step)
        self.R22 = sign*(self.p2['g2'][self.mask_dict['ns']] - self.m2['g2'][self.mask_dict['ns']])/(2.*step)
        self.R12 = (self.p2['g1'][self.mask_dict['ns']] - self.m2['g1'][self.mask_dict['ns']])/(2.*step)
        self.R21 = (self.p1['g2'][self.mask_dict['ns']] - self.m1['g2'][self.mask_dict['ns']])/(2.*step)

        self.R_shear = np.array([[self.R11, self.R12], [self.R21, self.R22]])

    def _shear_response_w(self, step=0.01, stat_operator=np.average):
        """
        """

        self.R11_w = (stat_operator(self.p1['g1'][self.mask_dict['ns']], weights=self.p1['w'][self.mask_dict['ns']]) - stat_operator(self.m1['g1'][self.mask_dict['ns']], weights=self.m1['w'][self.mask_dict['ns']]))/(2.*step)
        self.R22_w = (stat_operator(self.p2['g2'][self.mask_dict['ns']], weights=self.p2['w'][self.mask_dict['ns']]) - stat_operator(self.m2['g2'][self.mask_dict['ns']], weights=self.m2['w'][self.mask_dict['ns']]))/(2.*step)
        self.R12_w = (stat_operator(self.p2['g1'][self.mask_dict['ns']], weights=self.p2['w'][self.mask_dict['ns']]) - stat_operator(self.m2['g1'][self.mask_dict['ns']], weights=self.m2['w'][self.mask_dict['ns']]))/(2.*step)
        self.R21_w = (stat_operator(self.p1['g2'][self.mask_dict['ns']], weights=self.p1['w'][self.mask_dict['ns']]) - stat_operator(self.m1['g2'][self.mask_dict['ns']], weights=self.m1['w'][self.mask_dict['ns']]))/(2.*step)

        self.R_shear_w = np.array([[self.R11_w, self.R12_w], [self.R21_w, self.R22_w]])

    def _selection_response(self, stat_operator=np.mean):
        """
        """

        sign = 1
        if self._prefix == 'GALSIM':
            sign = -1

        self.R11_s = (stat_operator(self.ns['g1'][self.mask_dict['p1']]) - stat_operator(self.ns['g1'][self.mask_dict['m1']]))/0.02
        self.R22_s = sign*(stat_operator(self.ns['g2'][self.mask_dict['p2']]) - stat_operator(self.ns['g2'][self.mask_dict['m2']]))/0.02
        self.R12_s = (stat_operator(self.ns['g1'][self.mask_dict['p2']]) - stat_operator(self.ns['g1'][self.mask_dict['m2']]))/0.02
        self.R21_s = (stat_operator(self.ns['g2'][self.mask_dict['p1']]) - stat_operator(self.ns['g2'][self.mask_dict['m1']]))/0.02

        self.R_selection = np.array([[self.R11_s, 0.], [0., self.R22_s]])

    def _selection_response_weighted(self, stat_operator=np.average):
        """
        """

        self.R11_s_w = (stat_operator(self.ns['g1'][self.mask_dict['p1']], weights=self.ns['w'][self.mask_dict['p1']]) - stat_operator(self.ns['g1'][self.mask_dict['m1']], weights=self.ns['w'][self.mask_dict['m1']]))/0.02
        self.R22_s_w = (stat_operator(self.ns['g2'][self.mask_dict['p2']], weights=self.ns['w'][self.mask_dict['p2']]) - stat_operator(self.ns['g2'][self.mask_dict['m2']], weights=self.ns['w'][self.mask_dict['m2']]))/0.02
        self.R12_s_w = (stat_operator(self.ns['g1'][self.mask_dict['p2']], weights=self.ns['w'][self.mask_dict['p2']]) - stat_operator(self.ns['g1'][self.mask_dict['m2']], weights=self.ns['w'][self.mask_dict['m2']]))/0.02
        self.R21_s_w = (stat_operator(self.ns['g2'][self.mask_dict['p1']], weights=self.ns['w'][self.mask_dict['p1']]) - stat_operator(self.ns['g2'][self.mask_dict['m1']], weights=self.ns['w'][self.mask_dict['m1']]))/0.02

        self.R_selection_w = np.array([[self.R11_s_w, 0.], [0., self.R22_s_w]])


    def _total_response(self, stat_operator=np.mean):
        """
        """

        R_ws = stat_operator(self.R_shear, 2)
        self.R = R_ws + self.R_selection

    def _return():
        """
        """

        return self.m1, self.p1, self.p1, self.p2, self.ns, self.R


class footprint_mask:

    def __init__(self):
        pass

    def correct_ra(ra, dec):
        """
        It redefines RA according to a sinusoidal projection of the sky,
        so that all pixels have the same area (https://en.wikipedia.org/wiki/Sinusoidal_projection)
        """

        return (ra - np.mean(ra))*np.cos(dec/180.*np.pi)

    def binning(self):
        pass


def get_mask_footprint(ra, dec):
    """
    """

    return ((ra > 0) & (ra < 39) & (dec > 30.6) & (dec < 36.5)) \
           | ((ra > 341) & (ra < 360) & (dec > 30.6) & (dec < 36.5)) \
           | ((ra > 112) & (ra < 200) & (dec > 30.6) & (dec < 40.4)) \
           | ((ra > 225.5) & (ra < 270) & (dec > 30.3) & (dec < 40)) \
           | ((ra > 197) & (ra < 237.5) & (dec > 54) & (dec < 61.7))

def get_mask_footprint_2(ra, dec):
    """
    """

    return (ra > 112) & (ra < 200) & (dec > 30.6) & (dec < 40.4)

def get_mask_footprint_2A(ra, dec):
    """
    """

    return (ra > 112) & (ra < 160) & (dec > 30.6) & (dec < 40.4)

def get_mask_footprint_2B(ra, dec):
    """
    """

    return (ra > 160) & (ra < 200) & (dec > 30.6) & (dec < 40.4)

def get_mask_footprint_3(ra, dec):
    """
    """

    return (ra > 225.5) & (ra < 270) & (dec > 30.3) & (dec < 40)

def get_mask_footprint_P1(ra, dec):
    """
    """

    return (ra > 107) & (ra < 171) & (dec > 29) & (dec < 60)

def get_mask_footprint_P2(ra, dec):
    """
    """

    return (((ra > 0) & (ra < 38)) | ((ra > 340) & (ra < 360))) & (dec > 29) & (dec < 37)

def get_mask_footprint_P22(ra, dec):
    """
    """

    # ra2 = ra + 100
    # m_ra = np.where(ra2 > 360)
    # ra2[m_ra] -= 360

    return (ra > 80) & (ra < 138) & (dec > 29) & (dec < 37)

def get_mask_footprint_P3(ra, dec):
    """
    """

    return (ra > 190) & (ra < 240) & (dec > 47) & (dec < 65)

def get_mask_footprint_P4(ra, dec):
    """
    """

    return (ra > 159) & (ra < 201) & (dec > 29) & (dec < 45)

def get_mask_footprint_W3(ra, dec):
    """
    """

    return (ra > 208) & (ra < 221) & (dec > 51) & (dec < 58)

def get_mask_footprint_full(ra, dec):
    """
    """

    return ((ra > 107) & (ra < 171) & (dec > 29) & (dec < 60)) \
           | ((((ra > 0) & (ra < 38)) | ((ra > 340) & (ra < 360))) & (dec > 29) & (dec < 37)) \
           | ((ra > 190) & (ra < 240) & (dec > 47) & (dec < 65)) \
           | ((ra > 159) & (ra < 201) & (dec > 29) & (dec < 45))



def get_random(n_obj):
    """
    """

    final_size = 0
    while True:
        ra = np.random.rand(int(1e6))*(360. - 0.) + 0.
        dec = np.random.rand(int(1e6))*(90. - 0.) + 0.

        m = get_mask_footprint(ra, dec)

        current_size = len(ra[m]) + final_size

        if current_size >= n_obj:
            if final_size == 0:
                final_ra = ra[m][:n_obj]
                final_dec = dec[m][:n_obj]
                break
            else:
                final_ra = np.concatenate((final_ra, ra[m][:n_obj-final_size]))
                final_dec = np.concatenate((final_dec, dec[m][:n_obj-final_size]))
                break
        else:
            if final_size == 0:
                final_ra = ra[m]
                final_dec = dec[m]
                final_size = len(final_ra)
            else:
                final_ra = np.concatenate((final_ra, ra[m]))
                final_dec = np.concatenate((final_dec, dec[m]))
                final_size = len(final_ra)

    return {'ra': final_ra, 'dec': final_dec}


def random_cross(n_obj, ra_cat, dec_cat, e1_cat, e2_cat, w_cat = None, n_sample = 100):
    """
    """

    res = []
    for i in tqdm(range(n_sample)):
        cluster_rand = get_random(n_obj)
        res.append(gamma_T_tc(cluster_rand['ra'], cluster_rand['dec'], ra_cat, dec_cat, e1_cat, e2_cat, w_cat))
    res = np.array(res)

    return res


############
############

def correct_ra(ra, dec, mean_ra=None):
    """
    It redefines RA according to a sinusoidal projection of the sky,
    so that all pixels have the same area (https://en.wikipedia.org/wiki/Sinusoidal_projection)
    """

    if mean_ra == None:
        mean_ra = np.mean(ra)

    #return (ra - mean_ra)*np.cos(dec/180.*np.pi)
    return ra*np.cos(dec/180.*np.pi)

def correct_ra2(ra, dec, mean_dec=None):
    """
    It redefines RA according to a sinusoidal projection of the sky,
    so that all pixels have the same area (https://en.wikipedia.org/wiki/Sinusoidal_projection)
    """

    if mean_dec == None:
        mean_dec = np.mean(dec)

    #return (ra - mean_ra)*np.cos(dec/180.*np.pi)
    return ra*np.cos(mean_dec/180.*np.pi)

def get_ang_smoothing(params):
    """
    it translates the smoothing scale to number of pixels.
    """
    smooth = max(1,int(round(params['smooth']/params['pix'], 0)))
    print("smoothing pixel: ", smooth)
    return smooth


############
############

def stack_cluster(cluster_cat, ra, dec, g1, g2, w, z_bin_size=0.1, final_size=100, d_max=2, tree=None):
    """
    """
    n_cluster = len(cluster_cat['ra'])
    mean_dec = np.mean(dec)

    # Make cKDTree
    print("make tree..")
    ra_corr = correct_ra2(ra, dec)
    if tree == None:
        tree = cKDTree(np.array([ra_corr, dec]).T)
    print('Done.')


    h = 0.7
    cosmo = cosmology.FlatLambdaCDM(H0=h*100., Om0=0.3)
    deg_to_rad = np.pi / 180. #/60.

    R_mpc = cluster_cat['R']/h


    final_ra = np.array([])
    final_dec = np.array([])
    final_g1 = np.array([])
    final_g2 = np.array([])
    final_w = np.array([])
    for i in range(n_cluster):
        print('#######')
        print('{}'.format(i))#, end='\r')

        d_ang = cosmo.angular_diameter_distance(cluster_cat['z'][i]).value

        R_ang = R_mpc[i] / d_ang / deg_to_rad

        ra_tmp = correct_ra2(cluster_cat['ra'][i], cluster_cat['dec'][i], mean_dec)

        print(3.*R_ang)

        pos = coords.SkyCoord(ra=cluster_cat['ra'][i]*u.degree, dec=cluster_cat['dec'][i]*u.degree, frame='icrs')
        xid = SDSS.query_region(pos, radius=3.*R_ang*u.degree, spectro=True)
        if xid == None:
            continue
        print(len(xid['ra']))
        res = tree.query(np.array([correct_ra2(xid['ra'], xid['dec'], mean_dec), xid['dec']]).T, k=1, n_jobs=-1)
        ind_z = np.where(res[0]<0.0005)
        ind_tmp = res[1][ind_z]

        #res = tree.query(np.array([ra_tmp, cluster_cat['dec'][i]]).T, k = 100000, n_jobs=-1)

        #mask_size = np.where(res[0] <= 10. * R_ang)
        #mask_size = np.where(res[0] <= 0.5)

        if len(ind_tmp) == 0:
            continue

        ra_gal = ra[ind_tmp]
        dec_gal = dec[ind_tmp]
        z_gal = xid['z'][ind_z]

        mask_z = (z_gal > cluster_cat['z'][i] - z_bin_size) & (z_gal < cluster_cat['z'][i] + z_bin_size)

        if len(z_gal[mask_z]) == 0:
            continue

        ra_gal = ra_gal[mask_z]
        dec_gal = dec_gal[mask_z]
        g1_gal = g1[ind_tmp][mask_z]
        g2_gal = g2[ind_tmp][mask_z]
        w_gal= w[ind_tmp][mask_z]

        print(len(ra_gal))

        ra_gal_corr = correct_ra2(ra_gal, dec_gal, mean_dec)

        #d_gal = np.sqrt((ra_gal_corr - ra_tmp)**2. + (dec_gal - cluster_cat['dec'][i])**2.)
        d_gal = np.sqrt((ra_gal - cluster_cat['ra'][i])**2. + (dec_gal - cluster_cat['dec'][i])**2.)


        #ind_tmp = res[1][mask_size]
        #d_max_tmp = np.max(res[0][mask_size])

        d_max_tmp = np.max(d_gal)

        final_g1 = np.concatenate((final_g1, g1_gal))
        final_g2 = np.concatenate((final_g2, g2_gal))
        final_w = np.concatenate((final_w, w_gal))
        #ra_gal_tmp = (ra_gal_corr-ra_tmp) * d_max / d_max_tmp
        ra_gal_tmp = (ra_gal - cluster_cat['ra'][i]) * d_max / d_max_tmp
        dec_gal_tmp = (dec_gal-cluster_cat['dec'][i]) * d_max / d_max_tmp
        final_ra = np.concatenate((final_ra, ra_gal_tmp))
        final_dec = np.concatenate((final_dec, dec_gal_tmp))

    #ke, kb, params = make_mass_map(final_ra, final_dec, final_g1, final_g2, final_w)

    return final_ra, final_dec, final_g1, final_g2, final_w

def stack_cluster2(cluster_cat, ra, dec, g1, g2, w, z_bin_size=0.1, final_size=100, d_max=2, tree=None):
    """
    """
    n_cluster = len(cluster_cat['ra'])
    mean_dec = np.mean(dec)

    # Make cKDTree
    print("make tree..")
    ra_corr = correct_ra2(ra, dec)
    if tree == None:
        tree = cKDTree(np.array([ra_corr, dec]).T)
    print('Done.')


    h = 0.7
    cosmo = cosmology.FlatLambdaCDM(H0=h*100., Om0=0.3)
    deg_to_rad = np.pi / 180. #/60.

    R_mpc = cluster_cat['R']/h


    final_ra = np.array([])
    final_dec = np.array([])
    final_g1 = np.array([])
    final_g2 = np.array([])
    final_w = np.array([])
    for i in range(n_cluster):
        print('{}'.format(i), end='\r')

        d_ang = cosmo.angular_diameter_distance(cluster_cat['z'][i]).value

        R_ang = R_mpc[i] / d_ang / deg_to_rad

        ra_tmp = correct_ra2(cluster_cat['ra'][i], cluster_cat['dec'][i], mean_dec)

        res = tree.query(np.array([ra_tmp, cluster_cat['dec'][i]]).T, k = 500000, n_jobs=-1)

        #mask_size = np.where(res[0] <= 10. * R_ang)
        mask_size = np.where(res[0] <= 3.*R_ang)


        ind_tmp = res[1][mask_size]
        if len(ind_tmp) == 0:
            continue
        d_max_tmp = np.max(res[0][mask_size])

        ra_gal = ra_corr[ind_tmp]
        dec_gal = dec[ind_tmp]
        g1_gal = g1[ind_tmp]
        g2_gal = g2[ind_tmp]
        w_gal= w[ind_tmp]

        final_g1 = np.concatenate((final_g1, g1_gal))
        final_g2 = np.concatenate((final_g2, g2_gal))
        final_w = np.concatenate((final_w, w_gal))
        ra_gal_tmp = (ra_gal-ra_tmp) * d_max / d_max_tmp
        #ra_gal_tmp = (ra_gal - cluster_cat['ra'][i]) * d_max / d_max_tmp
        dec_gal_tmp = (dec_gal-cluster_cat['dec'][i]) * d_max / d_max_tmp
        final_ra = np.concatenate((final_ra, ra_gal_tmp))
        final_dec = np.concatenate((final_dec, dec_gal_tmp))

    #ke, kb, params = make_mass_map(final_ra, final_dec, final_g1, final_g2, final_w)

    return final_ra, final_dec, final_g1, final_g2, final_w


############
############

def jackknif_stat(data, stat_est=np.mean, remove_size=0.1, n_realization=100):
    """
    """

    samp_size = len(data)
    keep_size_pc = 1-remove_size

    if keep_size_pc < 0:
        raise ValueError('remove size should be in [0, 1]')

    subsamp_size = int(samp_size*keep_size_pc)

    all_est = []
    for i in tqdm(range(n_realization), total=n_realization):
        sub_data = np.random.choice(data, subsamp_size)

        all_est.append(stat_est(sub_data))

    all_est = np.array(all_est)

    return np.mean(all_est), np.std(all_est)

def jackknif_weighted_average(data, weights, remove_size=0.1, n_realization=100):
    """
    """

    samp_size = len(data)
    keep_size_pc = 1-remove_size

    if keep_size_pc < 0:
        raise ValueError('remove size should be in [0, 1]')

    subsamp_size = int(samp_size*keep_size_pc)

    all_ind = np.arange(samp_size)

    all_est = []
    for i in range(n_realization):
        sub_data_ind = np.random.choice(all_ind, subsamp_size)

        all_est.append(np.average(data[sub_data_ind], weights=weights[sub_data_ind]))

    all_est = np.array(all_est)

    return np.mean(all_est), np.std(all_est)



def psf_e_corr(e1, e2, psf_e1, psf_e2, psf_size, weights=None, n_bin=30, save_plot=False, plot_dir=None):
    """
    """

    def lin(x, a, b):
        return a * x + b

    def gauss(x, x0, sig):
        return 1. / (sig * np.sqrt(2. * np.pi)) * np.exp(-0.5 * ((x - x0)/sig)**2.)

    if weights is None:
        weights = np.ones_like(e1)

    size_all = len(e1)
    size_bin = int(size_all/n_bin)
    diff_size = size_all-size_bin

    psf_e1_arg_sort = np.argsort(psf_e1)
    psf_e2_arg_sort = np.argsort(psf_e2)
    psf_size_arg_sort = np.argsort(psf_size)

    psf1 = []
    psf2 = []
    m_e1 = []
    m_e12 = []
    m_e2 = []
    m_e21 = []
    s_e1 = []
    s_e12 = []
    s_e2 = []
    s_e21 = []
    psfs = []
    m_s1 = []
    m_s2 = []
    s_s1 = []
    s_s2 = []
    for i in tqdm(range(n_bin), total=n_bin):
        if i < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind_1 = psf_e1_arg_sort[starter + i * bin_size_tmp : starter + (i + 1) * bin_size_tmp]
        ind_2 = psf_e2_arg_sort[starter + i * bin_size_tmp : starter + (i + 1) * bin_size_tmp]
        ind_s = psf_size_arg_sort[starter + i * bin_size_tmp : starter + (i + 1) * bin_size_tmp]

        ####
        psf1.append(np.mean(psf_e1[ind_1]))
        # m_e1.append(np.average(e1[ind_1], weights=weights[ind_1]))
        # m_e21.append(np.average(e2[ind_1], weights=weights[ind_1]))

        r_jk = jackknif_weighted_average(e1[ind_1], weights[ind_1], remove_size=0.2, n_realization=50)
        m_e1.append(r_jk[0])
        s_e1.append(r_jk[1])
        r_jk = jackknif_weighted_average(e2[ind_1], weights[ind_1], remove_size=0.2, n_realization=50)
        m_e21.append(r_jk[0])
        s_e21.append(r_jk[1])

        # hist_err = np.histogram(e1[ind_1], 50, weights=weights[ind_1])[0]
        # hist_tmp = np.histogram(e1[ind_1], 50, weights=weights[ind_1], density=True)
        # res = curve_fit(gauss, (hist_tmp[1][:-1] + hist_tmp[1][1:])/2., hist_tmp[0], sigma=1./np.sqrt(hist_err))
        # m_e1.append(res[0][0])
        # hist_err = np.histogram(e2[ind_1], 50, weights=weights[ind_1])[0]
        # hist_tmp = np.histogram(e2[ind_1], 50, weights=weights[ind_1], density=True)
        # res = curve_fit(gauss, (hist_tmp[1][:-1] + hist_tmp[1][1:])/2., hist_tmp[0], sigma=1./np.sqrt(hist_err))
        # m_e21.append(res[0][0])
        # m_e1.append(np.median(e1[ind_1]))
        # m_e21.append(np.median(e2[ind_1]))

        # s_e1.append(np.std(e1[ind_1])/np.sqrt(len(e1[ind_1])))
        # s_e21.append(np.std(e2[ind_1])/np.sqrt(len(e2[ind_1])))

        # s_e1.append(weighted_std(e1[ind_1], weights[ind_1])/np.sqrt(len(e1[ind_1])))
        # s_e21.append(weighted_std(e2[ind_1], weights[ind_1])/np.sqrt(len(e2[ind_1])))
        ####

        ####
        psf2.append(np.mean(psf_e2[ind_2]))
        # m_e2.append(np.average(e2[ind_2], weights=weights[ind_2]))
        # m_e12.append(np.average(e1[ind_2], weights=weights[ind_2]))

        r_jk = jackknif_weighted_average(e2[ind_2], weights[ind_2], remove_size=0.2, n_realization=50)
        m_e2.append(r_jk[0])
        s_e2.append(r_jk[1])
        r_jk = jackknif_weighted_average(e1[ind_2], weights[ind_2], remove_size=0.2, n_realization=50)
        m_e12.append(r_jk[0])
        s_e12.append(r_jk[1])

        # hist_err = np.histogram(e2[ind_2], 50, weights=weights[ind_2])[0]
        # hist_tmp = np.histogram(e2[ind_2], 50, weights=weights[ind_2], density=True)
        # res = curve_fit(gauss, (hist_tmp[1][:-1] + hist_tmp[1][1:])/2., hist_tmp[0], sigma=1./np.sqrt(hist_err))
        # m_e2.append(res[0][0])
        # hist_err = np.histogram(e1[ind_2], 50, weights=weights[ind_2])[0]
        # hist_tmp = np.histogram(e1[ind_2], 50, weights=weights[ind_2], density=True)
        # res = curve_fit(gauss, (hist_tmp[1][:-1] + hist_tmp[1][1:])/2., hist_tmp[0], sigma=1./np.sqrt(hist_err))
        # m_e12.append(res[0][0])
        # m_e2.append(np.median(e2[ind_2]))
        # m_e12.append(np.median(e1[ind_2]))

        # s_e2.append(np.std(e2[ind_2])/np.sqrt(len(e2[ind_2])))
        # s_e12.append(np.std(e1[ind_2])/np.sqrt(len(e1[ind_2])))

        # s_e2.append(weighted_std(e2[ind_2], weights[ind_2])/np.sqrt(len(e2[ind_2])))
        # s_e12.append(weighted_std(e1[ind_2], weights[ind_2])/np.sqrt(len(e1[ind_2])))
        ####

        ####
        psfs.append(np.mean(psf_size[ind_s]))
        # m_s1.append(np.average(e1[ind_s], weights=weights[ind_s]))
        # m_s2.append(np.average(e2[ind_s], weights=weights[ind_s]))

        r_jk = jackknif_weighted_average(e1[ind_s], weights[ind_s], remove_size=0.2, n_realization=50)
        m_s1.append(r_jk[0])
        s_s1.append(r_jk[1])
        r_jk = jackknif_weighted_average(e2[ind_s], weights[ind_s], remove_size=0.2, n_realization=50)
        m_s2.append(r_jk[0])
        s_s2.append(r_jk[1])

        # m_s1.append(np.median(e1[ind_s]))
        # m_s2.append(np.median(e2[ind_s]))

        # s_s1.append(np.std(e1[ind_s])/np.sqrt(len(e1[ind_s])))
        # s_s2.append(np.std(e2[ind_s])/np.sqrt(len(e2[ind_s])))

        # s_s1.append(weighted_std(e1[ind_s], weights[ind_s])/np.sqrt(len(e1[ind_s])))
        # s_s2.append(weighted_std(e2[ind_s], weights[ind_s])/np.sqrt(len(e2[ind_s])))
        ####

    #     plt.figure()
    #     plt.title('psf e1 = {}'.format(psf1[-1]))
    #     plt.hist(e2[ind_1], 500, weights=weights[ind_1])
    #     plt.vlines(m_e21[-1], ymin=plt.ylim()[0], ymax=plt.ylim()[1], label='m = {:.5f}'.format(m_e21[-1]))
    #     plt.legend()
    #
    # plt.show()


    psf1 = np.array(psf1)
    s_e1 = np.array(s_e1)
    s_e21 = np.array(s_e21)
    m_e1 = np.array(m_e1)
    m_e21 = np.array(m_e21)
    psf2 = np.array(psf2)
    s_e2 = np.array(s_e2)
    s_e12 = np.array(s_e12)
    m_e2 = np.array(m_e2)
    m_e12 = np.array(m_e12)
    psfs = np.array(psfs)
    s_s1 = np.array(s_s1)
    s_s2 = np.array(s_s2)
    m_s1 = np.array(m_s1)
    m_s2 = np.array(m_s2)

    # res = curve_fit(lin, psf_e1, e2, sigma=1./np.sqrt(weights))
    # plt.figure()
    # plt.hexbin(psf_e1, e2, gridsize=300)
    # plt.plot(psf2, lin(psf2, *res[0]), c='k', label = 'm = {:.3f}'.format(res[0][0]))
    # plt.show()

    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['font.size'] = 35
    mpl.rcParams['xtick.minor.size'] = 7
    mpl.rcParams['ytick.minor.size'] = 7
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 3
    mpl.rcParams['ytick.major.width'] = 3
    mpl.rcParams['boxplot.boxprops.linewidth'] = 2.
    mpl.rcParams['boxplot.medianprops.linewidth'] = 2.
    mpl.rcParams['boxplot.flierprops.markersize'] = 12
    mpl.rcParams['boxplot.whiskerprops.linewidth'] = 2.
    mpl.rcParams['boxplot.capprops.linewidth'] = 2.


    mpl.rcParams['axes.xmargin'] = mpl.rcParamsDefault['axes.xmargin']

    plt.figure(figsize=(30,20))
    res = curve_fit(lin, psf_e1, e1, sigma=1./np.sqrt(weights))
    # plt.plot(psf1, lin(psf1, *res[0]), c='b', label = 'm = {:.3f}  c = {:.3f}'.format(res[0][0], res[0][1]))
    plt.plot(psf1, lin(psf1, *res[0]), c='b', label = 'm = {:.2g} +/- {:.2g}'.format(res[0][0], np.sqrt(res[1][0,0])))
    # res = curve_fit(lin, psf1, m_e1, sigma=s_e1, p0=[0.,0.])
    # plt.plot(psf1, lin(psf1, *res[0])-res[0][1], '--b', label = 'm = {:.3f} a = {:.3f} from point'.format(*res[0]))
    plt.errorbar(psf1, m_e1, yerr=s_e1, c='b', fmt='.', label='e1')

    res = curve_fit(lin, psf_e1, e2, sigma=1./np.sqrt(weights))
    # plt.plot(psf1, lin(psf1, *res[0]), c='r', label = 'm = {:.3f}  c = {:.3f}'.format(res[0][0], res[0][1]))
    plt.plot(psf1, lin(psf1, *res[0]), c='r', label = 'm = {:.2g} +/- {:.2g}'.format(res[0][0], np.sqrt(res[1][0,0])))
    # res = curve_fit(lin, psf1, m_e21, sigma=s_e21, p0=[0.,0.])
    # plt.plot(psf1, lin(psf1, *res[0])-res[0][1], '--r', label = 'm = {:.3f} a = {:.3f} from point'.format(*res[0]))
    plt.errorbar(psf1, m_e21, yerr=s_e21, c='r', fmt='.', label='e2')
    plt_xmin, plt_xmax = plt.xlim()
    plt.hlines(y=0, xmin=plt_xmin*5, xmax=plt_xmax*5, linestyles='dashed', color='k')
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(r'$e_{1}^{PSF}$')
    plt.ylabel(r'$<e_{1,2}^{gal}>$')
    plt.legend()
    if save_plot and (plot_dir is not None):
        plt.savefig(plot_dir + 'PSF_e1_vs_e_gal.png')

    plt.figure(figsize=(30,20))
    res = curve_fit(lin, psf_e2, e2, sigma=1./np.sqrt(weights))
    # plt.plot(psf2, lin(psf2, *res[0]), c='r', label = 'm = {:.3f}  c = {:.3f}'.format(res[0][0], res[0][1]))
    plt.plot(psf2, lin(psf2, *res[0]), c='r', label = 'm = {:.2g} +/- {:.2g}'.format(res[0][0], np.sqrt(res[1][0,0])))
    # res = curve_fit(lin, psf2, m_e2, sigma=s_e2, p0=[0.,0.])
    # plt.plot(psf2, lin(psf2, *res[0])-res[0][1], '--r', label = 'm = {:.3f} a = {:.3f} from point'.format(*res[0]))
    plt.errorbar(psf2, m_e2, yerr=s_e2, c='r', fmt='.', label='e2')

    res = curve_fit(lin, psf_e2, e1, sigma=1./np.sqrt(weights))
    # plt.plot(psf2, lin(psf2, *res[0]), c='b', label = 'm = {:.3f}  c = {:.3f}'.format(res[0][0], res[0][1]))
    plt.plot(psf2, lin(psf2, *res[0]), c='b', label = 'm = {:.2g} +/- {:.2g}'.format(res[0][0], np.sqrt(res[1][0,0])))
    # res = curve_fit(lin, psf2, m_e12, sigma=s_e12, p0=[0.,0.])
    # plt.plot(psf2, lin(psf2, *res[0])-res[0][1], '--b', label = 'm = {:.3f} a = {:.3f} from point'.format(*res[0]))
    plt.errorbar(psf2, m_e12, yerr=s_e12, c='b', fmt='.', label='e1')
    plt_xmin, plt_xmax = plt.xlim()
    plt.hlines(y=0, xmin=plt_xmin*5, xmax=plt_xmax*5, linestyles='dashed', color='k')
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(r'$e_{2}^{PSF}$')
    plt.ylabel(r'$<e_{1,2}^{gal}>$')
    plt.legend()
    if save_plot and (plot_dir is not None):
        plt.savefig(plot_dir + 'PSF_e2_vs_e_gal.png')


    plt.figure(figsize=(30,20))
    res = curve_fit(lin, psf_size, e1, sigma=1./np.sqrt(weights))
    # plt.plot(psfs, lin(psfs, *res[0]), c='b', label = 'm = {:.3f}  c = {:.3f}'.format(res[0][0], res[0][1]))
    plt.plot(psfs, lin(psfs, *res[0]), c='b', label = 'm = {:.2g} +/- {:.2g}'.format(res[0][0], np.sqrt(res[1][0,0])))
    plt.errorbar(psfs, m_s1, yerr=s_s1, c='b', fmt='.', label='e1')

    res = curve_fit(lin, psf_size, e2, sigma=1./np.sqrt(weights))
    # plt.plot(psfs, lin(psfs, *res[0]), c='r', label = 'm = {:.3f}  c = {:.3f}'.format(res[0][0], res[0][1]))
    plt.plot(psfs, lin(psfs, *res[0]), c='r', label = 'm = {:.2g} +/- {:.2g}'.format(res[0][0], np.sqrt(res[1][0,0])))
    plt.errorbar(psfs, m_s2, yerr=s_s2, c='r', fmt='.', label='e2')
    plt_xmin, plt_xmax = plt.xlim()
    plt.hlines(y=0, xmin=plt_xmin*(-5), xmax=plt_xmax*5, linestyles='dashed', color='k')
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(r'$\mathrm{FWHM}^{PSF}$ (arcsec)')
    plt.ylabel(r'$<e_{1,2}^{gal}>$')
    plt.legend()
    if save_plot and (plot_dir is not None):
        plt.savefig(plot_dir + 'PSF_size_vs_e_gal.png')

    plt.show()


def cluster_mass_plot(ra_gal, dec_gal, e1, e2, weights, ra_cluster, dec_cluster, m_cluster, n_bin=4):
    """
    """

    if weights is None:
        weights = np.ones_like(e1)

    size_all = len(m_cluster)
    size_bin = int(size_all/n_bin)
    diff_size = size_all-size_bin

    m_cluster_arg_sort = np.argsort(m_cluster)


    logR_p_a = []
    gamT_p_a = []
    gamX_p_a = []
    gam_sig_p_a = []
    mean_m = []

    for i in tqdm(range(n_bin), total=n_bin):
        if i < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind_1 = m_cluster_arg_sort[starter + i * bin_size_tmp : starter + (i + 1) * bin_size_tmp]

        ####
        mean_m.append(np.mean(m_cluster[ind_1]))

        R_p, logR_p, gamT_p, gamX_p, gam_sig_p = gamma_T_tc(ra_cluster[ind_1], dec_cluster[ind_1], ra_gal, dec_gal, e1, e2, weights)

        logR_p_a.append(logR_p)
        gamT_p_a.append(gamT_p)
        gamX_p_a.append(gamX_p)
        gam_sig_p_a.append(gam_sig_p)



    logR_p_a = np.array(logR_p_a)
    gamT_p_a = np.array(gamT_p_a)
    gamX_p_a = np.array(gamX_p_a)
    gam_sig_p_a = np.array(gam_sig_p_a)
    mean_m = np.array(mean_m)

    # res = curve_fit(lin, psf_e1, e2, sigma=1./np.sqrt(weights))
    # plt.figure()
    # plt.hexbin(psf_e1, e2, gridsize=300)
    # plt.plot(psf2, lin(psf2, *res[0]), c='k', label = 'm = {:.3f}'.format(res[0][0]))
    # plt.show()

    plt.figure(figsize=(30,20))
    # plt.errorbar(logR_p, gamT_p, yerr=gam_sig_p, capsize = 3, label='gamma_T')
    # plt.boxplot(res[:,2,:], positions=logR_p, widths=0.2)
    for i in range(n_bin):
        errplot = plt.errorbar(logR_p_a[i], gamT_p_a[i], yerr=gam_sig_p_a[i], capsize = 3, label=r'$\gamma_t$ M = {}'.format(mean_m[i]))
    #plt.boxplot(res[:,2,:], positions=logR_p, widths=0.2)
    plt.hlines(y=0, xmin=np.min(logR_p), xmax=np.max(logR_p), linestyles='dashed')
    # plt.xscale('log')
    plt.title(r'Lensing by clusters')
    plt.ylabel(r'$\gamma$')
    plt.xlabel(r'$log(\theta$) (arcmin)')
    #plt.xticks(fontsize=20)
    #plt.yticks(fontsize=20)

    plt.legend()

    # plt.gca().xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    # plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    # plt.xlim(plt.xlim()[0]+0.3,plt.xlim()[1]-0.3)

    # plt.legend([errplot, Patch(facecolor='w', edgecolor='k')], [r'$\gamma_{t}$ known clusters', r'random positions'])
    # plt.savefig('new_cat/cluster_profile_bp.png')
    plt.show()


def match_stars(ra_gal, dec_gal, ra_star, dec_star, thresh=0.0005):
    """
    """

    # Get right area
    m_foot_all = get_mask_footprint_W3(ra_gal, dec_gal)
    m_foot_star = get_mask_footprint_W3(ra_star, dec_star)

    mean_ra = np.mean(ra_gal)
    mean_dec = np.mean(dec_gal)
    xx_gal, yy_gal = radec2xy(mean_ra, mean_dec, ra_gal, dec_gal)
    xx_star, yy_star = radec2xy(mean_ra, mean_dec, ra_star, dec_star)

    # Make tree with galaxies
    print("Creating tree...")
    tree = cKDTree(np.array([xx_gal[m_foot_all], yy_gal[m_foot_all]]).T)

    # Match stars
    print("Matching stars...")
    res_matching = tree.query(np.array([xx_star[m_foot_star], yy_star[m_foot_star]]).T, k=1, n_jobs=-1)

    # Actual match
    i_match = np.where(res_matching[0] < thresh*np.pi/180)

    ind_stars = res_matching[1][i_match]

    return ind_stars

def match_stars2(ra_gal, dec_gal, ra_star, dec_star, thresh=0.0002):
    """
    """

    gal_coord = coords.SkyCoord(ra=ra_gal*u.degree, dec=dec_gal*u.degree)
    star_coord = coords.SkyCoord(ra=ra_star*u.degree, dec=dec_star*u.degree)

    res_coord = coords.match_coordinates_sky(star_coord, gal_coord)

    ind_stars = res_coord[0][np.where(res_coord[1].value < thresh)]

    return ind_stars

def get_cluster_mass(R_200, z, H_0=72*1e3, Omega_m=0.3,Omega_lambda=0.7,G=6.674e-11):
    """
    Retrun M_200
    """

    def H2(z):
        return H_0**2. * (Omega_m * (1 + z)**3. + Omega_lambda)

    return 100. * R_200**3.  * H2(z) / G * 3.086e22


def c_DuttonMaccio(z, m, h):
    """Concentration from c(M) relation in Dutton & Maccio (2014).
    Parameters
    ----------
    z : float or array_like
        Redshift(s) of halos.
    m : float or array_like
        Mass(es) of halos (m200 definition), in units of solar masses.
    h : float, optional
        Hubble parameter. Default is from Planck13.
    Returns
    ----------
    ndarray
        Concentration values (c200) for halos.
    References
    ----------
    Calculation from Planck-based results of simulations presented in:
    A.A. Dutton & A.V. Maccio, "Cold dark matter haloes in the Planck era:
    evolution of structural parameters for Einasto and NFW profiles,"
    Monthly Notices of the Royal Astronomical Society, Volume 441, Issue 4,
    p.3359-3374, 2014.
    """

    # z, m = _check_inputs(z, m)

    a = 0.52 + 0.385 * np.exp(-0.617 * (z**1.21))  # EQ 10
    b = -0.101 + 0.026 * z                         # EQ 11

    logc200 = a + b * np.log10(m * h / (10.**12))  # EQ 7

    concentration = 10.**logc200

    return concentration

def c_XRAY(z, m, h):
    """ 
    Paper: https://arxiv.org/pdf/1510.01961.pdf
    Table2, X-ray values 
    """

    M_star = 1.3*1e13/h

    alpha = -0.105
    b = 2.494

    A = 10**(b + alpha * np.log10(M_star))

    return A/(1+z) * (m/M_star)**alpha


def write_catalog(output_path, ra, dec, g1, g2, w, mag, snr, R, R_select, c1, c2, c1_err, c2_err, alpha_leak):
    """
    """

    # Data HDU
    c_ra = fits.Column(name='ra', array=ra, format='D', unit='deg')
    c_dec = fits.Column(name='dec', array=dec, format='D', unit='deg')
    c_g1 = fits.Column(name='g1', array=g1, format='D')
    c_g2 = fits.Column(name='g2', array=g2, format='D')
    c_w = fits.Column(name='w', array=w, format='D')
    c_mag = fits.Column(name='mag', array=mag, format='D')
    c_snr = fits.Column(name='snr', array=snr, format='D')
    table_hdu = fits.BinTableHDU.from_columns([c_ra, c_dec, c_g1, c_g2, c_w, c_mag, c_snr])

    table_hdu.header['TTYPE3'] = ('g1', 'Reduced shear (a-b)/(a+b)')
    table_hdu.header['TTYPE4'] = ('g2', 'Reduced shear (a-b)/(a+b)')
    table_hdu.header['TTYPE5'] = ('w', 'Ellipticity weight')
    table_hdu.header['TTYPE6'] = ('mag', 'From SExtractor: MAG_AUTO')
    table_hdu.header['TTYPE7'] = ('snr', 'SNR compute on metacal noshear: flux/flux_std')

    # Primary HDU with information in header
    primary_header = fits.Header()

    primary_header['R'] = (r'<R>', r'Full response: <R> = <R_shear> + <R_select>')

    primary_header['R1_1'] = (R[0,0], 'Full response matrix')
    primary_header['R1_2'] = (R[0,1], 'Full response matrix')
    primary_header['R2_1'] = (R[1,0], 'Full response matrix')
    primary_header['R2_2'] = (R[1,1], 'Full response matrix')

    primary_header['RS'] = (r'<RS>', r'Selection response matrix: <R_select>')
    primary_header['RS1_1'] = (R_select[0,0], 'Selection response matrix')
    primary_header['RS1_2'] = (R_select[0,1], 'Selection response matrix')
    primary_header['RS2_1'] = (R_select[1,0], 'Selection response matrix')
    primary_header['RS2_2'] = (R_select[1,1], 'Selection response matrix')

    primary_header['g_corr'] = ('Apply correction', r'g_corr = R**(-1).g')

    primary_header['c'] = ('', 'Additive bias computed with JackKnife')
    primary_header['c1'] = (c1, 'Additive bias for g1')
    primary_header['c1_err'] = (c1_err, 'Standard deviation for c1')
    primary_header['c2'] = (c2, 'Additive bias for g2')
    primary_header['c2_err'] = (c2_err, 'Standard deviation for c2')

    primary_header['w'] = ('Ellipticity weight', r'1 / (2*sig_SN**2 + g1_var + g2_var)')
    primary_header['sig_SN'] = (0.34, 'Sigma_shape_noise')

    primary_header['alpha'] = (alpha_leak, 'PSF leakage alpha') 

    primary_hdu = fits.PrimaryHDU(header=primary_header)

    # Final file
    hdu_list = fits.HDUList([primary_hdu, table_hdu])

    hdu_list.writeto(output_path, overwrite=True)


def NegDash(x_in, y_in, yerr_in, plot_name='', vertical_lines=True,
            xlabel='', ylabel='',
            ylim=None, semilogx=False, semilogy=False,
            **kwargs):
    """ This function is for making plots with vertical errorbars, where negative values
    are shown in absolute value as dashed lines. The resulting plot can either be saved
    by specifying a file name as `plot_name', or be kept as a pyplot instance (for instance
    to combine several NegDashes).
    """
    x = np.copy(x_in)
    y = np.copy(y_in)
    yerr = np.copy(yerr_in)
    # catch and separate errorbar-specific keywords from Lines2D ones
    safekwargs = dict(kwargs)
    errbkwargs = dict()
    if 'linestyle' in kwargs.keys():
        print("""Warning: linestyle was provided but that would kind of defeat the purpose,
         so I'll just ignore it. Sorry.""")
        del safekwargs['linestyle']
    for errorbar_kword in ['fmt', 'ecolor', 'elinewidth', 'capsize', 'barsabove', 'errorevery']:
        if errorbar_kword in kwargs.keys():
            #posfmt = '-'+kwargs['fmt']
            #negfmt = '--'+kwargs['fmt']
            errbkwargs[errorbar_kword] = kwargs[errorbar_kword]
            del safekwargs[errorbar_kword]
    errbkwargs = dict(errbkwargs, **safekwargs)
    """else:
        posfmt = '-'
        negfmt = '--'
        safekwargs = kwargs"""
    # plot up to next change of sign
    current_sign = np.sign(y[0])
    first_change = np.argmax(current_sign*y < 0)
    while first_change:
        if current_sign>0:
            plt.errorbar(x[:first_change], y[:first_change], yerr=yerr[:first_change],
                         linestyle='-', **errbkwargs)
            if vertical_lines:
                plt.vlines(x[first_change-1], 0, y[first_change-1], linestyle='-', **safekwargs)
                plt.vlines(x[first_change], 0, np.abs(y[first_change]), linestyle='--', **safekwargs)
        else:
            plt.errorbar(x[:first_change], np.abs(y[:first_change]), yerr=yerr[:first_change],
                         linestyle='--', **errbkwargs)
            if vertical_lines:
                plt.vlines(x[first_change-1], 0, np.abs(y[first_change-1]), linestyle='--',
                            **safekwargs)
                plt.vlines(x[first_change], 0, y[first_change], linestyle='-', **safekwargs)
        x = x[first_change:]
        y = y[first_change:]
        yerr = yerr[first_change:]
        current_sign *= -1
        first_change = np.argmax(current_sign*y < 0)
    # one last time when `first_change'==0 ie no more changes:
    if current_sign>0:
        plt.errorbar(x, y, yerr=yerr, linestyle='-', **errbkwargs)
    else:
        plt.errorbar(x, np.abs(y), yerr=yerr, linestyle='--', **errbkwargs)
    if semilogx:
        plt.xscale('log')
    if semilogy:
        plt.yscale('log')
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if plot_name:
        plt.savefig(plot_name)
        plt.close()



# d = vx.open('final_cat.arrow')
# dd = {key:d[key].values for key in d.column_names}
# # (dd['SPREAD_CLASS'] == 0) & (dd['MAG_AUTO'] > 17) & (dd['MAG_AUTO'] < 21) \
# # (dd['SPREAD_CLASS'] == 1) & (dd['MAG_AUTO'] > 21) & (dd['MAG_AUTO'] < 30) \
# m = (dd['SPREAD_CLASS'] == 1) & (dd['MAG_AUTO'] > 21) & (dd['MAG_AUTO'] < 25) \
#     & (dd['FLAGS'] == 0) & (dd['IMAFLAGS_ISO'] == 0) & (dd['NGMIX_MCAL_FLAGS'] == 0) \
#     & (dd['PSF_E1'] != 0) \
#     & (dd['PSF_E2'] != 0) \
#     & (dd['N_EPOCH'] >= 3) \
#     & (dd['NGMIX_N_EPOCH'] >= 1)
#     # & (dd['XWIN_WORLD'] > 113.6) & (dd['XWIN_WORLD'] < 270.8) & (dd['YWIN_WORLD'] > 30.2) & (dd['YWIN_WORLD'] < 32.8)
#     # & (dd['XWIN_WORLD'] > 193) & (dd['XWIN_WORLD'] < 240) & (dd['YWIN_WORLD'] > 54) & (dd['YWIN_WORLD'] < 64)
#
#
# print('Getting data ...')
# m1 = {}
# p1 = {}
# m2 = {}
# p2 = {}
# ns = {}
# for name_shear, dict_tmp in zip(['1m', '1p', '2m', '2p', 'noshear'], [m1, p1, m2, p2, ns]):
#     print('Extracting {}'.format(name_shear))
#     dict_tmp['flag'] = dd['NGMIX_FLAGS_{}'.format(name_shear.upper())][m]
#     dict_tmp['g1'] = dd['NGMIX_E1_{}'.format(name_shear.upper())][m]
#     dict_tmp['g2'] = dd['NGMIX_E2_{}'.format(name_shear.upper())][m]
#     dict_tmp['s2n'] = dd['NGMIX_SNR_{}'.format(name_shear.upper())][m]
#     dict_tmp['T'] = dd['NGMIX_T_{}'.format(name_shear.upper())][m]
#     dict_tmp['T_err'] = dd['NGMIX_T_ERR_{}'.format(name_shear.upper())][m]
#     dict_tmp['Tpsf'] = dd['NGMIX_Tpsf_{}'.format(name_shear.upper())][m]
#     dict_tmp['C11'] = dd['NGMIX_E1_ERR_{}'.format(name_shear.upper())][m]
#     dict_tmp['C22'] = dd['NGMIX_E2_ERR_{}'.format(name_shear.upper())][m]
#     dict_tmp['w'] = 1./(2*0.34**2. + dict_tmp['C11'] + dict_tmp['C22'])
# print('Done.')
#
# psf_e1 = dd['PSF_E1'][m]
# psf_e2 = dd['PSF_E2'][m]
# psf_e = np.sqrt(psf_e1**2. + psf_e2**2.)
#
#
# # Shear Response
# R11 = (p1['g1'] - m1['g1'])/0.02
# R22 = (p2['g2'] - m2['g2'])/0.02
# R12 = (p2['g1'] - m2['g1'])/0.02
# R21 = (p1['g2'] - m1['g2'])/0.02
#
# # Masking
# mask_dict = {}
# for data, name in zip([ns, m1, p1, m2, p2], ['ns', 'm1', 'p1', 'm2', 'p2']):
#     Tr_tmp = data['T'] * (1 - (data['g1']**2. + data['g2']**2.))/(1 + (data['g1']**2. + data['g2']**2.))
#     mask_tmp = (data['flag'] == 0) & (data['s2n'] > 20) & (data['s2n'] < 10000) & (data['T'] + np.sqrt(data['T_err']) > 0.02) & (Tr_tmp/data['Tpsf'] > 0.5)  & (data['T'] < 2. * np.sqrt(20*0.187/2.355)) #& (dict_tmp['C11'] < 0.7) & (dict_tmp['C22'] < 0.7)
#
#     # Take care of rotated version
#     ind_masked = np.where(mask_tmp == True)[0]
#
#     # mask_dict[name] = np.array(list(set(np.arange(len(data))) - set(ind_masked)))
#     mask_dict[name] = ind_masked
#
# # Final shear response
# R_shear = np.array([[R11[mask_dict['ns']], R12[mask_dict['ns']]], [R21[mask_dict['ns']], R22[mask_dict['ns']]]])
# # R_value = np.mean([R11[mask_dict['ns']], R22[mask_dict['ns']]], 0)
# # R_shear = np.array([[R_value, np.zeros_like(R_value)], [np.zeros_like(R_value), R_value]])
# # R_shear = np.array([[R11, R12], [R21, R22]])
#
# # m_R_shear = (np.abs(R_shear[0,0,:]) < 3) & (np.abs(R_shear[1,0,:]) < 3) & (np.abs(R_shear[0,1,:]) < 3) & (np.abs(R_shear[1,1,:]) < 3)
#
# # Selection response
# # R11_s = (np.mean(ns['g1'][mask_dict['p1']]) - np.mean(ns['g1'][mask_dict['m1']]))/0.02
# # R22_s = (np.mean(ns['g2'][mask_dict['p2']]) - np.mean(ns['g2'][mask_dict['m2']]))/0.02
# # R12_s = (np.mean(ns['g1'][mask_dict['p2']]) - np.mean(ns['g1'][mask_dict['m2']]))/0.02
# # R21_s = (np.mean(ns['g2'][mask_dict['p1']]) - np.mean(ns['g2'][mask_dict['m1']]))/0.02
# R11_s = (np.median(ns['g1'][mask_dict['p1']]) - np.median(ns['g1'][mask_dict['m1']]))/0.02
# R22_s = (np.median(ns['g2'][mask_dict['p2']]) - np.median(ns['g2'][mask_dict['m2']]))/0.02
# R12_s = (np.median(ns['g1'][mask_dict['p2']]) - np.median(ns['g1'][mask_dict['m2']]))/0.02
# R21_s = (np.median(ns['g2'][mask_dict['p1']]) - np.median(ns['g2'][mask_dict['m1']]))/0.02
# # R11_s = (np.average(ns['g1'][mask_dict['p1']], weights=ns['w'][mask_dict['p1']]) - np.average(ns['g1'][mask_dict['m1']], weights=ns['w'][mask_dict['m1']]))/0.02
# # R22_s = (np.average(ns['g2'][mask_dict['p2']], weights=ns['w'][mask_dict['p2']]) - np.average(ns['g2'][mask_dict['m2']], weights=ns['w'][mask_dict['m2']]))/0.02
# # R12_s = (np.average(ns['g1'][mask_dict['p2']], weights=ns['w'][mask_dict['p2']]) - np.average(ns['g1'][mask_dict['m2']], weights=ns['w'][mask_dict['m2']]))/0.02
# # R21_s = (np.average(ns['g2'][mask_dict['p1']], weights=ns['w'][mask_dict['p1']]) - np.average(ns['g2'][mask_dict['m1']], weights=ns['w'][mask_dict['m1']]))/0.02
#
# # Final selection response
# R_selection = np.array([[R11_s, R12_s], [R21_s, R22_s]])
#
# # Final response
# # R_ws = np.mean(R_shear, 2)
# R_ws = np.median(R_shear, 2)
# R = R_ws + R_selection
#
# # all_R['R'].append(R)
# # all_R['R_select'].append(R_selection)
# # all_R['R'].append(R_shear)

def guess_surface(ra, dec, min_nbins=100, max_nbins=1e6, n_nbins=100):
    """
    """

    n_obj = len(ra)
    ra_size = np.max(ra) - np.min(ra)
    dec_size = np.max(dec) - np.min(dec)

    ra_dec_ratio = ra_size / dec_size
    area_eff = []
    bins = np.linspace(min_nbins, max_nbins, n_nbins)
    for tmp_nbins in tqdm(bins, total=n_nbins):

        if ra_dec_ratio >= 1:
            if ra_dec_ratio > 2.:
                ra_dec_ratio = 2.
            dec_nbins = tmp_nbins
            ra_nbins = int(ra_dec_ratio * tmp_nbins)
        else:
            ra_dec_ratio = 1. / ra_dec_ratio
            if ra_dec_ratio > 2.:
                ra_dec_ratio = 2.
            ra_nbins = tmp_nbins
            dec_nbins = int(ra_dec_ratio * tmp_nbins)

        ra_pix_size = ra_size / ra_nbins
        dec_pix_size = dec_size / dec_nbins

        pixel_area = ra_pix_size * dec_pix_size

        map_2d, edges = np.histogramdd(np.array([dec,ra]).T, bins=(dec_nbins, ra_nbins))

        n_pix_used = len(np.where(map_2d != 0)[0])
        print(n_pix_used)

        area_eff.append(n_pix_used * pixel_area)

    return area_eff, bins
