"""BASIC.

:Name: basic.py

:Description: This file contains methods for calibration (metacalibration) and
              basic validation of a weak-lensing shape catalogue
              independent of cosmology.

:Author: Axel Guinot, Martin Kilbinger

"""

import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from scipy.special import gamma

import matplotlib.pyplot as plt

from tqdm import tqdm
import operator as op
import itertools as itools
from joblib import Parallel, delayed

from astropy.io import fits
from astropy import units as u
from astropy import coordinates as coords
from astropy.wcs import WCS
from astropy.table import Table

from lenspack.geometry.projections.gnom import radec2xy


class metacal:
    """Metacal.

    Metacalibration.

    Parameters
    ----------
    data :
        input galaxy catalogue
    mask : array of bool
        mask according to galaxy selection, e.g. spread_model
    masking_type : string, optional, default='gal'
        masking type, one in 'gal', 'gal_mom', 'star'
    step : float, optional, default=0.01
        step h in finite differences
    stat_operator : optional, default=np.mean
        summary statistic for response matrices
    prefix : string, optional, default='NGMIX'
        to specify columns in input catalogue
    snr_min : float, optional, default=10
        signal-to-noise minimum
    snr_max; float, optional, default=500
        signal-to-noise maximum
    rel_size_min : float, optional, default=0.5
        relative size minimum
    size_corr_ell : bool, optional, default=True
    sigma_eps : float, optional
        ellipticity dispersion (one component) for computation
        of weights; default is 0.34
    verbose : bool, optional, default=False
        verbose output if True

    """

    def __init__(
        self,
        data,
        mask,
        masking_type='gal',
        step=0.01,
        stat_operator=np.mean,
        prefix='NGMIX',
        snr_min=10,
        snr_max=500,
        rel_size_min=0.5,
        size_corr_ell=True,
        sigma_eps=0.34,
        verbose=False,
    ):

        self._masking_type = masking_type
        self._step = step
        self._stat_operator = stat_operator

        # Cuts
        self._snr_min = snr_min
        self._snr_max = snr_max
        self._rel_size_min = rel_size_min
        self._size_corr_ell = size_corr_ell
        if verbose:
            print(
                f'Metacal cuts: {snr_min}<snr<{snr_max}, '
                + f'rel_size_min={rel_size_min}, '
                + f'size_corr_ell={size_corr_ell}'
            )

        self._sigma_eps = sigma_eps

        self._verbose = verbose

        self._prefix = prefix

        self._read_data(data, mask)
        self._compute_calibration()

    def _read_data(self, data, mask):
        """Read Data.

        Read relevant data columns.
        """
        m1 = {}
        p1 = {}
        m2 = {}
        p2 = {}
        ns = {}

        if self._prefix == 'NGMIX':
            m1, p1, m2, p2, ns = self._read_data_ngmix(
                data,
                mask,
                m1,
                p1,
                m2,
                p2,
                ns,
            )
        elif self._prefix == 'GALSIM':
            m1, p1, m2, p2, ns = self._read_data_galsim(
                data,
                mask,
                m1,
                p1,
                m2,
                p2,
                ns,
            )

        self.m1 = m1
        self.p1 = p1
        self.m2 = m2
        self.p2 = p2
        self.ns = ns

    def _read_data_ngmix(self, data, mask, m1, p1, m2, p2, ns):
        """Read Data Ngmix.

        Read data from ngmix catalogue.
        """
        for name_shear, dict_tmp in zip(
            ['1m', '1p', '2m', '2p', 'noshear'],
            [m1, p1, m2, p2, ns]
        ):

            if self._verbose:
                print('Extracting {}'.format(name_shear))

            dict_tmp['flag'] = (
                data[f'{self._prefix}_FLAGS_{name_shear.upper()}'][mask]
            )
            dict_tmp['g1'] = (
                data[f'{self._prefix}_ELL_{name_shear.upper()}'][:, 0][mask]
            )
            dict_tmp['g2'] = (
                data[f'{self._prefix}_ELL_{name_shear.upper()}'][:, 1][mask]
            )

            dict_tmp['flux'] = (
                data[f'{self._prefix}_FLUX_{name_shear.upper()}'][mask]
            )
            dict_tmp['flux_err'] = (
                data[f'{self._prefix}_FLUX_ERR_{name_shear.upper()}'][mask]
            )

            dict_tmp['T'] = (
                data[f'{self._prefix}_T_{name_shear.upper()}'][mask]
            )
            dict_tmp['T_err'] = (
                data[f'{self._prefix}_T_ERR_{name_shear.upper()}'][mask]
            )
            dict_tmp['Tpsf'] = (
                data[f'{self._prefix}_Tpsf_{name_shear.upper()}'][mask]
            )

        ns['C11'] = data[f'{self._prefix}_ELL_ERR_NOSHEAR'][:, 0][mask]
        ns['C22'] = data[f'{self._prefix}_ELL_ERR_NOSHEAR'][:, 1][mask]
        ns['w'] = (
            1. / (2 * self._sigma_eps ** 2 + dict_tmp['C11'] + dict_tmp['C22'])
        )

        return m1, p1, m2, p2, ns

    def _read_data_galsim(self, data, mask, m1, p1, m2, p2, ns):
        """Read Data Galsim.

        Read data from galsim catalogue.
        """
        prefix_mom = 'GALSIM_GAL'

        for name_shear, dict_tmp in zip(
            ['1m', '1p', '2m', '2p', 'noshear'],
            [m1, p1, m2, p2, ns]
        ):

            if self._verbose:
                print('Extracting {}'.format(name_shear))

            dict_tmp['flag'] = (
                data[f'{self._prefix}_FLAGS_{name_shear.upper()}'][mask]
            )
            dict_tmp['g1'] = data[
                f'{prefix_mom}_ELL_UNCORR_{name_shear.upper()}'
            ][:, 0][mask]
            dict_tmp['g2'] = data[
                f'{prefix_mom}_ELL_UNCORR_{name_shear.upper()}'
            ][:, 1][mask]

            dict_tmp['T'] = (
                data[f'{prefix_mom}_SIGMA_{name_shear.upper()}'][mask]
            )
            dict_tmp['Tpsf'] = (
                data[f'{self._prefix}_PSF_SIGMA_{name_shear.upper()}'][mask]
            )

        self.snr_sextractor = data['SNR_WIN'][mask]
        ns['C11'] = data[f'{prefix_mom}_ELL_ERR_NOSHEAR'][:, 0][mask]
        ns['C22'] = data[f'{prefix_mom}_ELL_ERR_NOSHEAR'][:, 1][mask]
        ns['w'] = (
            1. / (2 * self._sigma_eps ** 2 + dict_tmp['C11'] + dict_tmp['C22'])
        )

        return m1, p1, m2, p2, ns

    def _compute_calibration(self):
        """Compute Calibration.

        Perform masking and compute calibration.
        """
        if self._masking_type == 'gal':
            self._masking_gal()
        elif self._masking_type == 'galmom':
            self._masking_gal_mom()
        elif self._masking_type == 'star':
            self._masking_star()
        else:
            raise ValueError(f'Invalid masking type \'{self._masking_type}\'')

        self._shear_response()
        self._selection_response()
        self._total_response()
        # self._shear_response_std(stat_operator=lambda x:
        # jackknif_weighted_average(x, np.ones_like(x)))

    def add_cuts(self, snr_min=10, snr_max=500, rel_size_min=0.5):
        """Add Cuts.

        Apply additional cuts to metacal galaxy catalogue.
        """
        if snr_min < self._snr_min or \
           snr_max > self._snr_max or \
           rel_size_min < self._rel_size_min:
            print(
                'At least on cut is less stringend than existing one, '
                + 'skipping...'
            )
            return

        self._snr_min = snr_min
        self._snr_max = snr_max
        self._rel_size_min = rel_size_min
        if self._verbose:
            print(
                f'Metacal new cuts: {snr_min}<snr<{snr_max}, '
                + f'rel_size_min={rel_size_min}'
            )

        self._compute_calibration()

    def _masking_gal(self):
        """Masking Gal.

        Mask metacal catalogue, i.e. apply cuts.
        """
        self.mask_dict = {}
        for data, name in zip(
            [self.ns, self.m1, self.p1, self.m2, self.p2],
            ['ns', 'm1', 'p1', 'm2', 'p2']
        ):
            Tr_tmp = data['T']
            if self._size_corr_ell:
                Tr_tmp *= (
                    (1 - (data['g1'] ** 2 + data['g2'] ** 2))
                    / (1 + (data['g1'] ** 2 + data['g2'] ** 2))
                )
            if hasattr(self, 'snr_sextractor'):
                snr_flux = self.snr_sextractor
            else:
                snr_flux = data['flux'] / data['flux_err']

            mask_tmp = (
                (data['flag'] == 0)
                & (Tr_tmp / data['Tpsf'] > self._rel_size_min)
                & (snr_flux > self._snr_min)
                & (snr_flux < self._snr_max)
            )

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            self.mask_dict[name] = ind_masked

    def _masking_gal_mom(self):
        """Add docstring.

        ...

        """
        self.mask_dict = {}
        for data, name in zip(
            [self.ns, self.m1, self.p1, self.m2, self.p2],
            ['ns', 'm1', 'p1', 'm2', 'p2']
        ):
            Tr_tmp = data['T']
            if self._size_corr_ell:
                Tr_tmp *= (
                    (1 - (data['g1'] ** 2 + data['g2'] ** 2))
                    / (1 + (data['g1'] ** 2 + data['g2'] ** 2))
                )
            snr_flux = data['flux'] / data['flux_err']

            mask_tmp = (
                (data['flag'] == 0)
                & (1 - data['Tpsf'] / data['T'] > self._rel_size_min)
                & (data['s2n'] > self._snr_min)
                & (data['s2n'] < self._snr_max)
                & (data['g1'] != -10)
                & (data['g1'] != 0)
            )

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            self.mask_dict[name] = ind_masked

    def _masking_star(self):
        """Add docstring.

        ...

        """
        self.mask_dict = {}
        for data, name in zip(
            [self.ns, self.m1, self.p1, self.m2, self.p2],
            ['ns', 'm1', 'p1', 'm2', 'p2']
        ):
            if hasattr(self, 'snr_sextractor'):
                snr_flux = self.snr_sextractor
            else:
                snr_flux = data['flux'] / data['flux_err']
            mask_tmp = (data['flag'] == 0) & (snr_flux > 10) & (snr_flux < 500)

            # Take care of rotated version
            ind_masked = np.where(mask_tmp == True)[0]

            self.mask_dict[name] = ind_masked

    def _shear_response(self):
        """Shear Response.

        Compute shear response matrix
        """
        sign = 1
        if self._prefix == 'GALSIM':
            sign = -1

        ma = self.mask_dict['ns']
        h2 = 2 * self._step

        self.R11 = (self.p1['g1'][ma] - self.m1['g1'][ma]) / h2
        self.R22 = sign * (self.p2['g2'][ma] - self.m2['g2'][ma]) / h2
        self.R12 = (self.p2['g1'][ma] - self.m2['g1'][ma]) / h2
        self.R21 = (self.p1['g2'][ma] - self.m1['g2'][ma]) / h2

        self.R_shear = np.array([[self.R11, self.R12], [self.R21, self.R22]])

    def _shear_response_std(
        self,
        stat_operator=lambda x: jackknif_weighted_average2(x, np.ones_like(x))
    ):
        """Shear Response Std.

        Standard deviation of shear response
        """
        ma = self.mask_dict['ns']
        h2 = 2 * self._step

        if len(self.ns['g1'][ma]) == 0:
            self.R_shear_std = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        else:
            self.R11_stds = stat_operator(
                (self.p1['g1'][ma] - self.m1['g1'][ma]) / h2
            )[1]
            self.R22_stds = stat_operator(
                (self.p2['g2'][ma] - self.m2['g2'][ma]) / h2
            )[1]
            self.R12_stds = stat_operator(
                (self.p2['g1'][ma] - self.m2['g1'][ma]) / h2
            )[1]
            self.R21_stds = stat_operator(
                (self.p1['g2'][ma] - self.m1['g2'][ma]) / h2
            )[1]

            self.R_shear_std = np.array([
                [self.R11_stds, self.R12_stds],
                [self.R21_stds, self.R22_stds]
            ])

    def _selection_response(self):
        """Add docstring.

        ...

        """
        sign = 1
        if self._prefix == 'GALSIM':
            sign = -1

        ma_p1 = self.mask_dict['p1']
        ma_m1 = self.mask_dict['m1']
        ma_p2 = self.mask_dict['p2']
        ma_m2 = self.mask_dict['m2']
        h2 = 2 * self._step

        self.R11_s = (
            self._stat_operator(self.ns['g1'][ma_p1])
            - self._stat_operator(self.ns['g1'][ma_m1])
        ) / h2
        self.R22_s = sign * (
            self._stat_operator(self.ns['g2'][ma_p2])
            - self._stat_operator(self.ns['g2'][ma_m2])
        ) / h2
        self.R12_s = (
            self._stat_operator(self.ns['g1'][ma_p2])
            - self._stat_operator(self.ns['g1'][ma_m2])
        ) / h2
        self.R21_s = (
            self._stat_operator(self.ns['g2'][ma_p1])
            - self._stat_operator(self.ns['g2'][ma_m1])
        ) / h2

        self.R_selection = np.array([
            [self.R11_s, self.R12_s],
            [self.R21_s, self.R22_s]
        ])

    def _total_response(self):
        """Add docstring.

        ...

        """
        R_ws = self._stat_operator(self.R_shear, 2)
        self.R = R_ws + self.R_selection

    def _return():
        """Add docstring.

        ...

        """
        return (
            self.m1,
            self.p1,
            self.p1,
            self.p2,
            self.ns,
            self.R,
            self.R_selection_std,
            self.R_shear_std
        )


def jackknif_weighted_average2(
    data,
    weights,
    remove_size=0.1,
    n_realization=100,
):
    """Add docstring.

    ...

    """
    samp_size = len(data)
    keep_size_pc = 1 - remove_size

    if keep_size_pc < 0:
        raise ValueError('remove size should be in [0, 1]')

    subsamp_size = int(samp_size * keep_size_pc)

    all_ind = np.arange(samp_size)

    all_est = []
    for i in range(n_realization):
        sub_data_ind = np.random.choice(all_ind, subsamp_size)

        if (sum(data[sub_data_ind]) == 0):
            all_est.append(np.nan)
        else:
            all_est.append(
                np.average(data[sub_data_ind], weights=weights[sub_data_ind])
            )

    all_est = np.array(all_est)

    return np.mean(all_est), np.std(all_est)
