"""CALIBRATE JOINT CAT.

This module implements the class to calibrate a joint comprehensive catalogue.

:Authors: Martin Kilbinger
"""

import sys
import os
import numpy as np
from scipy import stats

from optparse import OptionParser

import h5py
from astropy.io import fits
from astropy.table import Column

from cs_util import logging
from cs_util import cat
from cs_util import args as cs_args


class BaseCat(object):
    """Base_Cat.

    Basic catalogue class.
    
    """

    def __init__(self):
        self.params_default()

    def set_params_from_command_line(self, args):
        """Set Params From Command Line.

        Only use when calling using python from command line.
        Does not work from ipython or jupyter.

        """
        # Read command line options
        options = cs_args.parse_options(
            self._params,
            self._short_options,
            self._types,
            self._help_strings,
        )

        # Save calling command
        logging.log_command(args)

    def read_cat(self, load_into_memory=False, mode="r"):
        """Read Cat.
        
        Read input catalogue, either FITS or HDF5.
        
        Parameters
        ----------
        load_into_memory: bool, optional
            load data into memory (potentially slow) of ``True``;
            default is ``False``
        mode: bool, optional
            HDF5 read mode, default is "r"

        Returns
        -------
        list
            Catalogue data
            
        Raises
        ------
        IOError
            If file extension is not .fits or .hd5

        """
        fpath = self._params["input_path"]
        verbose = self._params["verbose"]
        
        extension = os.path.splitext(fpath)[1]
        if extension == ".fits":
            if verbose:
                print(f"Reading FITS file {fpath}, HDU {hdu}...")
                
            hdu = 1
            dat = fits.getdata(fpath, hdu)
            
        elif extension in (".hdf5", ".hd5"):
            if verbose:
                print(f"Reading HDF5 file {fpath}...")

            self._hd5file = h5py.File(fpath, mode)
            try:
                dat = self._hd5file["data"]
            except:
                print(f"Error while reading file {fpath}")
                raise
            if load_into_memory:
                return dat[()]
            else:
                return dat
        else:
            raise IOError(f"Unknown file extension {extension}")

    def close_hd5(self):
        """Close HD5.

        Close HDF5 file.

        """
        self._hd5file.close()


class ApplyHspMasks(BaseCat):
    """Apply Hsp Masks."""

    def __init__(self):
        # Set default parameters
        self.params_default()
        
    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "input_path": None,
            "mask_dir": ".",
            "nside": 131072,
            "file_base": "mask_r_",
            "bits": 1,
        }
        self._short_options = {
            "input_path": "-i",
            "mask_dir": "-d",
            "nside": "-n",
            "file_base": "-f",
            "bits": "-b",
        }
        self._types = {
            "nside": "int",
            "bits" : "int",
        }
        self._help_strings = {
            "input_path": "path input FITS catalogue, default={}",
            "mask_dir": "directory with mask files, default={}",
            "nside": "healsparse resolution parameter, default={}",
            "file_base": "base name of mask files, default={}",
            "bits": "bits to apply, default={}",
        }       
    
class CalibrateCat(BaseCat):
    """Calibrate Cat.

    Class to calibrate joint catalogue.

    """

    def __init__(self):
        # Set default parameters
        self.params_default()

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "input_path": None,
            "cmatrices": False,
        }
        self._short_options = {
            "input_path": "-i",
            "cmatrices": "-C",
        }
        self._types = {
            "cmatrices": "bool",
        }
        self._help_strings = {
            "input_path": "path input FITS catalogue",
            "cmatrices": "compute correlation and confusion matrices",
        }

    def run(self):
        """Run.

        Main processing function.

        """

def confusion_matrix(mask, confidence_level=0.9):
    
    n_key = len(mask)

    cm = np.empty((n_key, n_key))
    r_val = np.zeros_like(cm)
    r_cl = np.empty((n_key, n_key, 2))

    for idx, key1 in enumerate(mask):
        for jdx, key2 in enumerate(mask):
            res = stats.pearsonr(mask[key1], mask[key2])
            r_val[idx][jdx] = res.statistic
            r_cl[idx][jdx] = res.confidence_interval(confidence_level=confidence_level)
            
    return r_val, r_cl       

def correlation_matrix(mask):
    
    n_tot = len(mask)
    n_key = len

    cm = np.empty((n_key, n_key))
    r = np_like(cm)

    for idx, key1 in enumerate(mask):
        for jdx, key2 in enumerate(mask):
            r[idx][jdx] = stats.pearsonr(mask[key1], mask[key2])
            
    return r
            


def run_calibrate_comprehensive_cat(*args):
    """Run Calibrate Comprehensive Cat.

    Run class to calibrate joint comprehensive catalogue from command line.

    """
    obj = CalibrateCat()

    obj.set_params_from_command_line(args)

    obj.run()

def run_apply_hsp_masks(*args):
    """Run Apply Healsparse Masks.

    Run class to apply healsparse masks.

    """
    obj = ApplyHspMasks()

    obj.set_params_from_command_line(args)

    obj.run()
