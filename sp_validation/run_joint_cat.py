"""RUN JOINT CAT.

This module implements the class to create a joint comprehensive catalogue.

:Authors: Martin Kilbinger
"""

import sys
import os
import numpy as np
from tqdm import tqdm

from optparse import OptionParser
from importlib.metadata import version 


import h5py
from astropy.io import fits
from astropy.table import Column

from cs_util import logging
from cs_util import cat
from cs_util import args as cs_args

from . import util


class JointCat:
    """Joint Cat.

    Class to create joint weak-lensing catalogues.

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
            "patches": "v1",
            "sh": "ngmix",
            "survey": "unions",
            "year": "2024",
            "version": "1.4.2",
            "pipeline": "shapepipe",
            "hdu": 1,
        }
        self._short_options = {
            "patches": "-p",
            "sh": "-g",
            "survey": "-s",
            "year": "-y",
            "version": "-V",
        }
        self._types = {
            "hdu": "int",
        }
        self._help_strings = {
            "patches": "list of patches separated by '+', or shortcut (allowed are 'v1'), default={}",
            "sh": "shape measurement method, default={}",
            "survey": "survey name, default={}",
            "year": "year of processing, default={}",
            "version": "catalogue version, default={}",
        }

    def get_patches(self):
        """Get Patches.

        Return list of patches according to option parameter value.

        Returns
        -------
        list
            patches, list of str

        """
        if self._params["patches"] == "v1":
            n_patch = 7
            patches = [f"P{x}" for x in np.arange(n_patch) + 1]
        else:
            patches = self._params["patches"].split("+")

        return patches

    def get_n_obj(self, patches, base_path, input_sub_path):
        """Get N Obj.

        Get number of objects from FITS file headers.

        Parameters
        ----------
        patches : list
            input patches, type is str
        base_path : str
            input base directory, root dir of patches
        input_sub_path : str
            input file name; input path is base_path/patch/input_sub_path
        
        Raises:
            ValueError: if input file canont be read

        Returns:
            list
                HDUs
            list
                number of objects per file
            int
                total number of objects

        """
        if self._params["verbose"]:
            print("Getting number of objects")
        n_obj_list = []
        n_obj = 0
        hdu_lists = []
        for patch in patches:
            input_path = f"{base_path}/{patch}/{input_sub_path}"
            try:
                hdu_list = fits.open(input_path)
            except:
                raise ValueError(
                    f"Could not open file {input_path} at HDU #{self._params['hdu']}"
                )
            hdu_lists.append(hdu_list)

            this_n = int(hdu_list[self._params["hdu"]].header["NAXIS2"])
            n_obj_list.append(this_n)
            n_obj += this_n

        if self._params["verbose"]:
            print(f"Found a total of {n_obj} (~{util.millify(n_obj)}) objects.")

        return hdu_lists, n_obj_list, n_obj

    def get_col_info(self, dat):
        """Get Col Info.
        
        Return information of input columns.

        Parameters
        ----------
        dat : numpy.ndarray
            input data
        
        Returns
        -------
        list
            column names
        list
            column formats
        int
            number of columns

        """
        col_names = dat.dtype.names

        n_col = 0
        formats = {}
        ndim = {}
        for name in col_names:
            formats[name] = dat.dtype.fields[name][0]
            ndim[name] = dat[name].ndim
            n_col += ndim[name]
        # Add one for patch
        n_col += 1

        if self._params["verbose"]:
            print(f"Number of input (output) columns = {len(col_names)} ({n_col})")

        return col_names, formats, ndim, n_col

    def init_data(self, n_col, n_obj, ndim, dat):
        """Init Data.
        
        Initialize empty structured data.
        
        Parameters
        ----------
        n_col : int
            number of columns
        n_obj : int
            number of objects (rows)
        ndim : dict
            dimension of input columns
        dat : numpy.ndarray
            example data
            
        Returns
        -------
        numpy.ndarray
            combined structure data, (n_col x n_obj) array

        """
        if self._params["verbose"]:
            print(
                f"Allocating {n_col * n_obj * 8 / 1024**3:.1f}"
                + f" Gb memory for the ({n_col} x {n_obj}) data array ...",
                end="",
            )
        
        # Create dtypes from input column names and types.
        # Transform multi-D columns into 1D columns
        dtype_tmp_list = []
        for name in ndim:
            if ndim[name] == 1:
                dtype_tmp_list.append((name, dat[name].dtype))
            else:
                for jdx in range(ndim[name]):
                    dtype_tmp_list.append((f"{name}_{jdx}", dat[name].dtype))
        dtype_tmp_list.append(('patch', np.int32))
        dtype_tmp_struct = np.dtype(dtype_tmp_list)
        dat_all = np.empty((n_obj,), dtype=dtype_tmp_struct)

        if self._params["verbose"]:
            print("done")

        return dat_all
    
    def write_header(self, hd5file, patches):
        """Write Header.
        
        Write header information to HDF5 file.
        
        Parameters
        ----------
        hd5file : h5py.File
            input HDF5 file
        patches : list
            input patches, list of str
            
        """ 
        author = os.getenv("USER")
        software_name = "sp_validation"
        software_version = version(software_name)
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patches_str = " ".join(patches)
        
        hd5file.attrs["author"] = author
        hd5file.attrs["softname"] = software_name
        hd5file.attrs["softver"] = software_version
        hd5file.attrs["date"] = date
        hd5file.attrs["patches"] = patches_str

        
    def write_hdf5_file(self, dat_all, patches):
        """Write HDF5 File.
        
        Write data to HDF5 file.
        
        Parameters
        ----------
        dat_all i numpy.ndarray

        """
        if self._params["verbose"]:
            print("Creating hdf5 file")
        output_path = (
            f"{self._params['survey']}_{self._params['pipeline']}"
            + f"_comprehensive_{self._params['year']}_"
            + f"v{self._params['version']}.hdf5"
        )

        with h5py.File(output_path, "w") as f:

            self.write_header(f, patches)

            dset = f.create_dataset("data", data=dat_all)
            dset[:] = dat_all

        if self._params["verbose"]:
            print(f"Done.")

    def merge_catalogues(self, patches, base_path="."):
        """Merge Catalogues.

        Merge individual patch-based catalogues.

        Parameters
        ----------
        patches : list
            input patches; list of `str`
        base_path : str, optional
            input base directory path; default is "."

        """
        input_sub_path = (
            f"sp_output/shape_catalog_comprehensive_{self._params['sh']}.fits"
        )

        # Get input FITS files    
        hdu_lists, n_obj_list, n_obj = self.get_n_obj(
            patches,
            base_path,
            input_sub_path,
        )

        # Read data
        start = end = 0
        for idx, patch in enumerate(patches):

            input_path = f"{base_path}/{patch}/{input_sub_path}"
            try:
                dat = fits.getdata(input_path, self._params["hdu"])
                #dat = hdu_lists[idx][self._params["hdu"]].data

                hdu_lists[idx].close()
            except:
                raise ValueError(
                    f"Could not read data of file {input_path} at HDU #{self._params['hdu']}"
                )

            # Create empty lists if first patch
            if idx == 0:

                col_names, formats, ndim, n_col = self.get_col_info(dat)
                dat_all = self.init_data(n_col, n_obj, ndim, dat)

            # Append new data for that patch (between start and end)
            end += n_obj_list[idx]
            
            # Copy data
            i_col = 0
            names_out = dat_all.dtype.names
            for name in col_names:
                if ndim[name] == 1:
                    # Copy 1D column
                    dat_all[names_out[i_col]][start:end] = dat[name]
                else:
                    # Copy all components of multi-D column
                    for jdx in range(ndim[name]):
                        dat_all[names_out[i_col + jdx]][start:end] = dat[name][:, jdx]
                i_col += ndim[name]
            # Add patch number
            dat_all["patch"][start:end] = patch[1:]
            
            if i_col + 1 != n_col:
                raise ValueError(
                    "Inconsistent number of columns, {i_col + 1}"
                    + f" != {n_col}"
                )
            if self._params["verbose"]:
                print(
                    f"{patch}: Added {len(dat)} (~{util.millify(len(dat))})"
                    + f" objects (from {start} to {end-1})."
                )
            start = end
    
        del dat

        self.write_hdf5_file(dat_all, patches)

    def run(self):
        """Run.

        Main processing function.

        """
        patches = self.get_patches()
        if self._params["verbose"]:
            print("Merging patches", patches)

        self.merge_catalogues(patches)


class ReadCat:

    def __init__(self):
        self.params_default()

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "input": "input_cat.hdf5",
            "n_row": None,
        }
        self.short_options = {
            "input": "-i",
            "n_row": "-n",
        }
        self._help_string = {
            "input": "input file, default={}",
            "n_row": "print first N_ROW rows only",
        }

    def run(self):

        obj = self
        with h5py.File(obj._params["input"], "r") as hdf5_file:
            pass



def run_joint_comprehensive_cat(*args):
    """Run Joint Comprehensive Cat.

    Run class to create joint comprehensive catalogue from command line.

    """
    obj = JointCat()

    obj.set_params_from_command_line(args)

    obj.run()
