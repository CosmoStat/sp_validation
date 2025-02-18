"""CALIBRATE JOINT CAT.

This module implements the class to calibrate a joint comprehensive catalogue.

:Authors: Martin Kilbinger
"""

import sys
import os
import numpy as np

from optparse import OptionParser

import h5py
from astropy.io import fits
from astropy.table import Column

from cs_util import logging
from cs_util import cat
from cs_util import args as cs_args


class CalibrateCat:
    """Calibrate Cat.

    Class to calibrate joint catalogue.

    """

    def __init__(self):
        # Set default parameters
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

        # Update parameter values from options
        self._params.update(options)

        # Save calling command
        logging.log_command(args)

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "input_path": None,
        }
        self._short_options = {
            "input_path": "-i",
        }
        self._types = {
        }
        self._help_strings = {
            "input_path": "path input FITS catalogue",
        }

    def read_cat(self):
        """Read Cat.
        
        Read input catalogue, either FITS or HDF5.
        
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

            with h5py.File(fpath, "r") as f:
                dat = f["data"][:]
        else:
            raise IOError(f"Unknown file extension {extension}")

        return dat

    def run(self):
        """Run.

        Main processing function.

        """
        

def run_calibrate_comprehensive_cat(*args):
    """Run Calibrate Comprehensive Cat.

    Run class to calibrate joint comprehensive catalogue from command line.

    """
    obj = CalibrateCat()

    obj.set_params_from_command_line(args)

    obj.run()
