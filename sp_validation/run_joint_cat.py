"""RUN JOINT CAT.

This module implements classes to create, mask, and calibrate joint
comprehensive catalogues.

:Authors: Martin Kilbinger
"""

import sys
import os

import numpy as np
from scipy import stats
import yaml

import datetime
from tqdm import tqdm

from optparse import OptionParser
from importlib.metadata import version

import h5py
import healsparse as hsp
import numpy as np

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
        
    def read_config_set_params(self, fpath):
        
        # Load YAML configuration file.
        with open("config_mask.yaml", "r") as f:
            config = yaml.safe_load(f)
    
        # Read general parameters from configuration and remove
        if "params" in config:
            params = config.pop("params")

            # Copy parameters to object
            for key in params:
                self._params[key] = params[key]
                
        return config

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

    def write_hdf5_header(self, hd5file):
        """Write HDF5 Header.

        Write basic header information to HDF5 file.

        Parameters
        ----------
        hd5file : h5py.File
            input HDF5 file

        """
        author = os.getenv("USER")
        software_name = "sp_validation"
        software_version = version(software_name)
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        hd5file.attrs["author"] = author
        hd5file.attrs["softname"] = software_name
        hd5file.attrs["softver"] = software_version
        hd5file.attrs["date"] = date

    def write_hdf5_file(self, dat, output_path=None):
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

            self.write_hdf5_header(f)

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

        tracemalloc.start()

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
            "reduce_mem": False,
        }
        self._short_options = {
            "patches": "-p",
            "sh": "-g",
            "survey": "-s",
            "year": "-y",
            "version": "-V",
            "reduce_mem": "-r",
        }
        self._types = {
            "hdu": "int",
            "reduce_mem": "bool",
        }
        self._help_strings = {
            "patches": "list of patches separated by '+', or shortcut (allowed are 'v1'), default={}",
            "sh": "shape measurement method, default={}",
            "survey": "survey name, default={}",
            "year": "year of processing, default={}",
            "version": "catalogue version, default={}",
            "reduce_mem": "output some columns in lower precision to reduce memory",
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
        elif self._params["patches"] == "v1.5":
            n_patch = 8
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
                    f"Could not open file {input_path} at HDU"
                    + f" #{self._params['hdu']}"
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
            print(
                f"Number of input (output) columns = {len(col_names)} ({n_col})"
            )

        return col_names, formats, ndim, n_col

    def dtype_out(self, name, dtype_in):
        """Set output dtype.

        Parameters
        ----------
        name : str
            column name
        dtype_in : np.dtype
            input dtype

        Returns
        -------
        np.dtype
            output dtype

        """
        cols_keep_dtype = [
            "RA",
            "Dec",
            "FLAGS",
            "IMAFLAGS_ISO",
        ]
        if self._params["reduce_mem"] == False:
            return dtype_in
        elif name not in cols_keep_dtype:
            if dtype_in.kind == "f" and dtype_in.itemsize == 8:
                return np.float32
            if dtype_in.kind == "i" and dtype_in.itemsize == 4:
                return np.int8

        return dtype_in

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

        # Create dtypes from input column names and types.
        # Reduce memory if flag set.
        # Transform multi-D columns into 1D columns
        dtype_tmp_list = []
        for name in ndim:
            if ndim[name] == 1:
                dtype_tmp_list.append(
                    (name, self.dtype_out(name, dat[name].dtype))
                )
            else:
                for jdx in range(ndim[name]):
                    dtype_tmp_list.append(
                        (f"{name}_{jdx}", self.dtype_out(name, dat[name].dtype))
                    )
        dtype_tmp_list.append(("patch", np.int8))
        dtype_tmp_struct = np.dtype(dtype_tmp_list)

        if self._params["verbose"]:
            memory = n_obj * dtype_tmp_struct.itemsize
            print(
                f"Allocating <= {memory / 1024**3:.1f}"
                + f" Gb memory for the ({n_col} x {n_obj}) input data array ...",
                end="",
            )

        dat_all = np.empty((n_obj,), dtype=dtype_tmp_struct)

        if self._params["verbose"]:
            print("done")

        return dat_all

    def write_hdf5_file(self, dat, patches):
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

        with h5py.File(output_path, "w") as f:

            self.write_hdf5_header(f)

            dset = f.create_dataset("data", data=dat)
            dset[:] = dat

        # super().write_hdf5_file(dat, output_path=output_path)

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
        super().write_hdf5_header(hd5file)

        if patches is not None:
            patches_str = " ".join(patches)
            hd5file.attrs["patches"] = patches_str

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
                # dat = hdu_lists[idx][self._params["hdu"]].data

                hdu_lists[idx].close()
            except:
                raise ValueError(
                    f"Could not read data of file {input_path} at HDU"
                    + f" #{self._params['hdu']}"
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
                        dat_all[names_out[i_col + jdx]][start:end] = dat[name][
                            :, jdx
                        ]
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
        """Get Label Struct.

        Return label of bit-coded mask.

        Parameters
        ----------
        bit: int
            input bit

        Returns
        -------
        str
            label

        """
        # Labels of bit-coded structural masks
        labels_struct = {
            1: "Faint_star_halos",
            2: "Bright_star_halos",
            4: "Stars",
            8: "Manual",
            16: "u",
            32: "g",
            64: "r",
            128: "i",
            256: "z",
            512: "Tile_RA_DEC_cut",
            1024: "Maximask",
        }

        return labels_struct[bit]

    @classmethod
    def get_mask_col_name(cls, bit):
        """Get Mask Col Name.

        Return column name of mask corresponding to input bit.

        Parameters
        ----------
        bit : int
            input bit

        Returns
        -------
        str
            column name

        """
        return f"{bit}_{cls.get_label_struct(bit)}"

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "input_path": None,
            "output_path": "output.hdf5",
            "mask_dir": ".",
            "nside": 131072,
            "file_base": "mask_r_",
            "bits": 1,
            "aux_mask_files": None,
            "aux_mask_labels": None,
        }
        self._short_options = {
            "input_path": "-i",
            "output_path": "-o",
            "mask_dir": "-d",
            "nside": "-n",
            "file_base": "-f",
            "bits": "-b",
        }
        self._types = {
            "nside": "int",
            "bits": "int",
        }
        self._help_strings = {
            "input_path": "path of input hdf5 catalogue, default={}",
            "output_path": "path of output hdf5 catalogue, default={}",
            "mask_dir": "directory with mask files, default={}",
            "nside": "healsparse resolution parameter, default={}",
            "file_base": "base name of mask files, default={}",
            "bits": "bits to apply, default={}",
            "aux_mask_files": "auxiliary mask files separated with '\\', defualt={}",
            "aux_mask_labels": "auxiliary mask column names separated with '\\'",
        }

    def check_params(self):
        """Check Params.

        Check whether parameter values are valid.

        Raises
        ------
        ValueError
            if a parameter value is not valid

        """
        if (self._params["aux_mask_files"] is None) != (
            self._params["aux_mask_labels"] is None
        ):
            raise ValueError("Both or none of the 'aux_mask_*' can be None")

    def update_params(self):
        """Update Params.

        Update and transform parameter values.

        """
        if self._params["aux_mask_files"] is not None:
            self._params["aux_mask_file_list"] = cs_args.my_string_split(
                self._params["aux_mask_files"],
                verbose=self._params["verbose"],
                stop=True,
                sep="\\",
            )
            self._params["aux_mask_num"] = len(
                self._params["aux_mask_file_list"]
            )
            self._params["aux_mask_label_list"] = cs_args.my_string_split(
                self._params["aux_mask_labels"],
                verbose=self._params["verbose"],
                stop=True,
                num=self._params["aux_mask_num"],
                sep="\\",
            )
        else:
            self._params["aux_mask_file_list"] = []

    def read_hsp_mask(self, path):
        """Read Hsp Mask.

        Parameters
        ----------
        path : str
            Path to the mask file.

        Returns
        -------
        np.ndarray
            Mask
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

    def get_paths_bit_masks(self):
        """Get Paths Bit Masks.

        Return paths of bit-coded mask files.

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

        Returns per-object masks for all bits.

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

        # Get bit-coded mask file paths
        paths = self.get_paths_bit_masks()

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

            label = self.get_mask_col_name(bit)

            if self._params["verbose"]:
                print(f"Computing mask bit={bit}...")
            masks[label] = hsp_mask.get_values_pos(ra, dec, lonlat=True)

        # Read auxiliary mask files"
        for idx, path in enumerate(self._params["aux_mask_file_list"]):
            label = self._params["aux_mask_label_list"][idx]
            if self._params["verbose"]:
                print(f"Reading aux mask for label {label}...")
            hsp_mask = hsp.HealSparseMap.read(path)
            masks[label] = hsp_mask.get_values_pos(ra, dec, lonlat=True)

        return masks

    def run(self):
        """Run.

        Main processing function.

        """
        obj = self

        # Check parameters
        obj.check_params()

        # Update parameters
        obj.update_params()

        # Read input data
        dat = obj.read_cat(load_into_memory=True, mode="r")

        # Get masks
        masks = obj.get_masks(dat=dat)

        # Append masks to data
        dat_ext = obj.append_masks(dat, masks)

        # Write extended data to new HDF5 file
        obj.write_hdf5_file(dat_ext)

        # Close input HDF5 catalogue file
        obj.close_hd5()

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
        labels = [label for label in masks]
        dtypes = [masks[label].dtype for label in masks]

        # Create a structured dtype
        structured_dtype = np.dtype(
            [(label, dtype) for label, dtype in zip(labels, dtypes)]
        )

        new_data = np.zeros(dat.shape, dtype=structured_dtype)

        # Copy masks as new columns
        for label in masks:
            new_data[label] = masks[label]

        return new_data

    def write_hdf5_file(self, dat, dat_new):

        with h5py.File(self._params["output_path"], "w") as f:

            self.write_hdf5_header(f)

            dset = f.create_dataset(
                "data", shape=dat.shape, dtype=dat.dtype, chunks=True
            )
            dset_new = f.create_dataset(
                "data_ext",
                shape=dat_new.shape,
                dtype=dat_new.dtype,
                chunks=True,
            )

            chunk_size = 10000
            nrow = dat.shape[0]

            for i in range(0, nrow, chunk_size):
                end = min(i + chunk_size, nrow)
                dset[i:end] = dat[i:end]  # Write chunk to dataset
                dset_new[i:end] = dat_new[i:end]  # Write chunk to dataset

    def write_hdf5_header(self, hd5file):
        """Write HDF5 Header.

        Write header information to HDF5 file.

        Parameters
        ----------
        hd5file : h5py.File
            input HDF5 file
        patches : list, optional
            input patches, list of str, default is ``None``

        """
        super().write_hdf5_header(hd5file)

        hd5file.attrs["hsp_nside"] = self._params["nside"]

        # Bit-mask file paths
        paths = self.get_paths_bit_masks()
        for bit in paths:
            hd5file.attrs[f"hsp_path_{bit}"] = paths[bit]

        # Auxiliary mask file paths
        for idx, path in enumerate(self._params["aux_mask_file_list"]):
            label = self._params["aux_mask_label_list"][idx]
            hd5file.attrs[f"hsp_path_{label}"] = path


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

    def read_cat(self, load_into_memory=False):
        """Read Cat.

        Read input HDF5 catalogue.

        Parameters
        ----------
        load_into_memory: bool, optional
            load data into memory (potentially slow) of ``True``;
            default is ``False``

        Returns
        -------
        list
            Catalogue data
        list_ext
            Extended catalogue data

        """
        fpath = self._params["input_path"]
        verbose = self._params["verbose"]

        if verbose:
            print(f"Reading HDF5 file {fpath}...")

        self._hd5file = h5py.File(fpath, "r")
        try:
            dat = self._hd5file["data"]
            dat_ext = self._hd5file["data_ext"]
        except:
            print(f"Error while reading file {fpath}")
            raise
        if load_into_memory:
            return dat[()], dat_ext[()]
        else:
            return dat, dat_ext

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
            r_cl[idx][jdx] = res.confidence_interval(
                confidence_level=confidence_level
            )

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

class Mask():
    """Mask.
    
    Class to handle masking of catalogues.
    
    Parameters
    ----------
    col_name : str
        column name
    label : str
        label
    kind : str
        operation type, allowed are "equal", "not_equal, ""greater_equal", "range"
    value : float or list
        value(s) to be used in mask operation
    dat : numpy.ndarray, optional
        input data, default is `None`; apply mask if given
    verbose : bool, optional
        verbose output if ``True``; default is ``False``

    """

    def __init__(self, col_name, label, kind="equal", value=0, dat=None, verbose=False):
        
        self._col_name = col_name
        self._label = label
        self._value = value
        self._kind = kind
        self._num_ok = None

        if verbose:
            print("Initialising mask:", self)

        if dat is not None:
            self.apply(dat)
            
    def __repr__(self):
        
        return (
            f"Mask(col_name={self._col_name}, label={self._label}, kind={self._kind},"
            + f" value={self._value})"
        )
         
    @classmethod    
    def from_list(cls, masks, label="combined"):
        my_mask = cls(label, label, kind="combined", value=None)

        my_mask._mask = np.logical_and.reduce([m._mask for m in masks])

        return my_mask

    def apply(self, dat):
        if self._kind == "equal":
            self._mask = dat[self._col_name] == self._value
        elif self._kind == "not_equal":
            self._mask = dat[self._col_name] != self._value
        elif self._kind == "greater_equal":
            self._mask = dat[self._col_name] >= self._value
        elif self._kind == "range":
            self._mask = (dat[self._col_name] >= self._value[0]) & (dat[self._col_name] <= self._value[1])
        else:
            raise ValueError(f"Invalid kind {kind}")
        
    @classmethod
    def print_strings(cls, coln, lab, num, fnum):
        print(f"{coln:30s} {lab:30s} {num:10s} {fnum:10s}")
        
    def print_stats(self, num_obj):
        if self._num_ok is None:
            self._num_ok = sum(self._mask)

        si = f"{self._num_ok:10d}"
        sf = f"{self._num_ok/num_obj:10.2%}"
        self.print_strings(self._col_name, self._label, si, sf)
        
    def print_summary(self, f_out):
        print(f"[{self._label}]\t\t\t", file=f_out, end="")
        sign = None
        if self._kind =="equal":
            sign = "="
        elif self._kind =="not_equal":
            sign = "!="
        elif self._kind =="greater_equal":
            sign = ">="
        if sign is not None:
            print(f"{self._col_name} {sign} {self._value}", file=f_out)
            
        if self._kind == "range":
            print(f"{self._value[0]} <= {self._col_name} <= {self._value[1]}", file=f_out)

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
