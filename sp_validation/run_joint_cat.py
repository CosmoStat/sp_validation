"""RUN JOINT CAT.

This module implements classes to create, mask, and calibrate joint
comprehensive catalogues.

:Authors: Martin Kilbinger
"""

import sys
import os

import numpy as np
from scipy import stats

import datetime
from tqdm import tqdm

from optparse import OptionParser
from importlib.metadata import version 


import h5py
import healsparse as hsp

from astropy.io import fits
from astropy.table import Column

from cs_util import logging
from cs_util import cat
from cs_util import args as cs_args

from . import util


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
        
    def write_hdf5_header(self, hd5file, patches=None):
        """Write HDF5 Header.
        
        Write header information to HDF5 file.
        
        Parameters
        ----------
        hd5file : h5py.File
            input HDF5 file
        patches : list, optional
            input patches, list of str, default is ``None``
            
        """ 
        author = os.getenv("USER")
        software_name = "sp_validation"
        software_version = version(software_name)
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        hd5file.attrs["author"] = author
        hd5file.attrs["softname"] = software_name
        hd5file.attrs["softver"] = software_version
        hd5file.attrs["date"] = date
        
        if patches is not None:
            patches_str = " ".join(patches)
            hd5file.attrs["patches"] = patches_str
            
    def write_hdf5_file(self, dat, output_path=None, patches=None):
        """Write HDF5 File.
        
        Write HDF5 data to file.
        
        Parameters
        ----------
        dat : numpy.ndarray
            input data
        output_path : str, optional
            output file path; when ``None`` (default) use
            self._params['output_path']
        patches : list, optional
            input patches, list of str, default is ``None``
        
        """
        if output_path is None:
             output_path = self._params["output_path"]

        if self._params["verbose"]:
            print("Creating hdf5 file")

        with h5py.File(output_path, "w") as f:

            self.write_hdf5_header(f, patches=patches)

            dset = f.create_dataset("data", data=dat)
            dset[:] = dat

        if self._params["verbose"]:
            print(f"Done.")

    def close_hd5(self):
        """Close HD5.

        Close HDF5 file.

        """
        self._hd5file.close()

class JointCat(BaseCat):
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
    
        
    def write_hdf5_file(self, dat_all, patches):
        """Write HDF5 File.
        
        Write data to HDF5 file.
        
        Parameters
        ----------
        dat_all : numpy.ndarray
            input data
        patches : list
            input patches, list of str

        """
        output_path = (
            f"{self._params['survey']}_{self._params['pipeline']}"
            + f"_comprehensive_{self._params['year']}_"
            + f"v{self._params['version']}.hdf5"
        )

        super().write_hdf5_file(
            dat_all,
            output_path=output_path, 
            patches=patches
        )

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


class ApplyHspMasks(BaseCat):
    """Apply Hsp Masks."""

    def __init__(self):
        # Set default parameters
        self.params_default()

    @classmethod
    def get_label_struct(cls, bit):
        
        # Labels of bit-coded structural masks
        labels_struct = {
            1 : "Faint_star_halos",
            2 : "Bright_star_halos",
            4 : "Stars",
            8 : "Manual",
            16 : "u",
            32 : "g",
            64 : "r",
            128 : "i",
            256 : "z",
            512 : "Tile_RA_DEC_cut",
            1024 : "Maximask",
        }
        
        return label_struct[bit] 

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
        
    def read_hsp_mask(self, path):
        """Read Hsp Mask.
    
        Parameters
        ----------
        path : str
            Path to the mask file.
        
        Returns
        -------
        np.ndarray
            Mask array.
        """
    
        if self._params["verbose"]:
            print(f"Reading healsparse mask file {path}...")
        return hsp.HealSparseMap.read(path)

    def reverse_bit_list(self):
        """Reverse Bit List.
        
        Split bit-coded integer into bits.
        
        Parameters
        ----------
        bit : int
            Bit-coded integer
            
        Returns
        -------
        list
            List of bits
            
        """
        bit_list = []
        bits = self._params["bits"]
        while bits:
            lowest_bit = bits & -bits  # Extract lowest set bit
            bit_list.append(lowest_bit)
            bits -= lowest_bit  # Remove this bit from bit

        return bit_list

    def get_paths(self):
        """Get Paths.
        
        Return paths of mask files.
        
        Returns
        -------
        dict
            Dictionary with bit as key and path as value.
        
        """
        paths = {}
        bit_list = self.reverse_bit_list()
        for bit in bit_list:
            paths[bit] = (
                f"{self._params['mask_dir']}/{self._params['file_base']}"
                + f"nside{self._params['nside']}_n{bit}.hsp"
        )
        return paths

    def get_masks(self, dat=None):
        """Get Masks.
        
        Returns masks for all bits.
        
        Parameters
        ----------
        dat: numpy.ndarray, optional
            input data; if not given (default), data will be read from
            input file

        Returns
        -------
        dict
            masks
        
        """
        masks = {}
    
        # Get mask file paths
        paths = self.get_paths()
    
        # Get coordinates from data
        if dat is None:
            dat = self.read_cat()
        if self._params["verbose"]:
            print("Reading coordinates from data...")
        ra = dat["RA"]
        dec = dat["Dec"]
    
        # Read healsparse files and apply masks to coordinate
        for bit in paths:
            if self._params["verbose"]:
                print(f"Reading mask for bit {bit}...")
            hsp_mask = hsp.HealSparseMap.read(paths[bit])
    
            if self._params["verbose"]:
                print(f"Computing mask bit={bit}...")
            masks[bit] = ~hsp_mask.get_values_pos(ra, dec, lonlat=True)
        
        return masks
    
    @classmethod
    def get_mask_col_name(cls, bit):
        
        return f"{bit}_{cls.get_labels_struct(bit)}"
        
    
    def append_masks(self, dat, masks):
        """Append Masks.
        
        Add mask information as columns to data.
        
        Parameters
        ----------
        dat: numpy.ndarray
            input data
        masks: dict
            mask information

        Returns
        --------
        numpy.ndarray
            updated data
 
        """
        n_masks = len(masks)
        new_dtype = np.dtype(
            dat.dtype.descr
            + [(self.get_mask_col_name(bit), np.bool_) for bit in masks]
        )
        new_data = np.zeros(dat.shape, dtype=new_dtype)

        # Copy previous columns
        for name in dat.dtype.names:
            new_data[name] = dat[name]
        
        # Copy masks as new columns
        for bit in masks:
            new_data[self.get_mask_col_name(bit)] = masks[bit]

        return new_data

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
