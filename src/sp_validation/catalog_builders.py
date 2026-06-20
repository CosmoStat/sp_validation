"""CATALOG BUILDERS.

This module implements classes to create, mask, and calibrate joint
comprehensive catalogues, plus the run_* entry-point functions.

:Author: Martin Kilbinger
"""

import sys
import os

import numpy as np
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

from . import format
from . import calibration
from . import cat as sp_cat

# Spatial-masking primitives now live in ``masks``; re-exported here so external
# code using ``from sp_validation import catalog_builders as sp_joint`` keeps
# resolving ``sp_joint.Mask``, ``sp_joint.get_masks_from_config``, etc.
from sp_validation.masks import (
    Mask,
    get_masks_from_config,
    print_mask_stats,
    correlation_matrix,
    confusion_matrix,
)


class BaseCat(object):
    """Base_Cat.

    Basic catalogue class.

    """

    def __init__(self):
        pass

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
        """Read Config Set Params.
        
        Read configuration file and sets class parameters.
        
        Parameters
        ----------
        fpath : str
            inpput file path
            
        Returns
        -------
        dict
            configuration

        """
        # Load YAML configuration file.
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        # Read general parameters from configuration and remove
        if "params" in config:
            params = config.pop("params")

            # Copy parameters to object
            for key in params:
                self._params[key] = params[key]
                
        return config

    def read_cat(self, load_into_memory=False, mode="r", hdu=1, name="data"):
        """Read Cat.

        Read input catalogue, either FITS or HDF5.

        Parameters
        ----------
        load_into_memory: bool, optional
            load data into memory (potentially slow) of ``True``;
            default is ``False``
        mode: bool, optional
            HDF5 read mode, default is "r"
        hdu: int, optional
            HDU number (for FITS file); default is 1
        name: str, optional
            dataset name, default is 'data'

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
                dat = self._hd5file[name]
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

    def get_header(self, path=None):
        """Get Header.

        Return header of hd5 file.

        Parameters
        ----------
        path : str, optional
            input path; if not ``None`` (default) use `output_path` of
            self._params dict

        Returns
        -------
        dict
            header

        """
        if path is None:
            path = self._params["output_path"]
            
        with h5py.File(path, "r") as f:
            header = dict(f.attrs)
        return header

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
            "verbose": False,
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
            print(f"Found a total of {n_obj} (~{format.millify(n_obj)}) objects.")

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
        # Specify columns for which original (high-precision) format
        # needs to be kept and not reduced to lower precision
        cols_keep_dtype = [
            "RA",
            "Dec",
            "FLAGS",
            "IMAFLAGS_ISO",
            "NUMBER",
        ]
        if dtype_in.kind == "U":
            # Transform unicode to string of equal length
            return np.dtype(f"S{dtype_in.itemsize // 4}")

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

        with h5py.File(output_path, "w") as f:

            self.write_hdf5_header(f)

            dset = f.create_dataset("data", data=dat_all)
            dset[:] = dat_all

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
                    f"{patch}: Added {len(dat)} (~{format.millify(len(dat))})"
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

    # Labels of bit-coded structural masks
    _labels_struct = {
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
        2048: "z2",
    }

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
        return cls._labels_struct[bit]

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
            "verbose": False,
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
            
        if "verbose" not in self._params:
            self._params["verbose"] = False

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
    
    def get_mask(self, path):
        """Get Mask.
        
        Read from file and return healsparse mask.
        
        Parameters
        ----------
        path: str
            input path
        
        Returns
        -------
        hsp.HealSparseMap
            mask
    
        """
        if self._params["verbose"]:
            print(f"Reading mask file {path}...")
        return hsp.HealSparseMap.read(path)

    def apply_mask(self, ra, dec, hsp_mask, label):
        """Apply Mask.

        Apply mask to input coordinates.

        Parameters
        ----------
        hsp_mask : hsp.HealSparseMap
            input mask
        ra : numpy.ndarray
            input right ascension
        dec : numpy.ndarray
            input declination

        Returns
        -------
        numpy.ndarray
            mask values

        """
        if self._params["verbose"]:
            print(f"Applying mask {label}...")

        return hsp_mask.get_values_pos(ra, dec, lonlat=True)

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
            hsp_mask = self.get_mask(paths[bit])

            label = self.get_mask_col_name(bit)
            masks[label] = self.apply_mask(ra, dec, hsp_mask, label)

        # Read auxiliary mask files"
        for idx, path in enumerate(self._params["aux_mask_file_list"]):
            hsp_mask = self.get_mask(path)
            label = self._params["aux_mask_label_list"][idx]
            masks[label] = self.apply_mask(ra, dec, hsp_mask, label)

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

    def write_hdf5_file(self, dat, dat_new=None, masks=None):
        """Write HDF5 File.
        
        Save data to a hdf5 file on disk.
        
        Parameters
        ----------
        dat : h5py dataset
            input dataset
        dat_new : h5py dataset, optional
            second dataset; unused if ``None``
        masks : list, optional
            masks, to be added to header information

        Returns
        -------
        """
        with h5py.File(self._params["output_path"], "w") as f:

            self.write_hdf5_header(f)

            dset = f.create_dataset(
                "data", shape=dat.shape, dtype=dat.dtype, chunks=True
            )
            if dat_new is not None:
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
                if dat_new is not None:
                    dset_new[i:end] = dat_new[i:end]  # Write chunk to dataset
                    
            if masks is not None:
                # Adding mask descriptions to header
                dtype = np.dtype([("expr", "S20"), ("desc", "S20")])
                descr_arr = np.zeros(len(masks), dtype=dtype)
                for idx, mask in enumerate(masks):
                    descr_arr[idx] = (
                        (mask._descr.encode("utf-8"), mask._label.encode("utf-8"))
                    )
                f.create_dataset("applied_masks", data=descr_arr)
                

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
        if "aux_mask_file_list" in self._params:
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
            "verbose": False,
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
            Extended catalogue data if exists in input file

        """
        fpath = self._params["input_path"]
        verbose = self._params["verbose"]

        if verbose:
            print(f"Reading HDF5 file {fpath}...")

        self._hd5file = h5py.File(fpath, "r")
        try:
            dat = self._hd5file["data"]
            if "data_ext" in self._hd5file:
                dat_ext = self._hd5file["data_ext"]
            else:
                dat_ext = None
        except:
            print(f"Error while reading file {fpath}")
            raise

        if verbose:
            print(
                f"Found {len(dat)} (~{format.millify(len(dat))}) objects"
                + " in catalogue"
            )

        if load_into_memory:
            if dat_ext:
                return dat[()], dat_ext[()]
            else:
                return dat[()]
        else:
            return dat, dat_ext
        
    def add_params_to_FITS_header(self, header, cm=None):

        header_new = fits.Header()

        # General information
        keys = ["input_path"]
        descriptions = ["input comprehensive catalogue"]
        for key, descr in zip(keys, descriptions):
            header_new[key] = (key, descr)
            
        # Metacal parameters
        if cm is not None:
            for idx, (descr, value) in enumerate(cm.items()):
                key = f"mc_par_{idx}"
                header_new[key] = (descr, value)             
        
        header.update(header_new)
        
    def run(self):
        """Run.

        Main processing function.

        """


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
            "verbose": False,
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


def compute_weights_gatti(
    cat_gal,
    g_uncorr,
    gal_metacal,
    dat,
    mask_combined,
    mask_metacal,
    num_bins=20,
    snr_min=10,
    snr_max=500,
    size_ratio_min=0.707,
    size_ratio_max=3,
):
    """Compute Weights Gatti.
    
    Compute Gatti et al. (2021) DES-like weights.
    
    """
    calibration.fill_cat_gal(
        cat_gal,
        dat,
        g_uncorr,
        gal_metacal,
        mask_combined._mask,
        mask_metacal,
        purpose="weights"
    )

    cat_gal["w_des"] = calibration.get_w_des(
        cat_gal,
        num_bins,
        snr_min=snr_min,
        snr_max=snr_max,
        size_ratio_min=size_ratio_min,
        size_ratio_max=size_ratio_max,
    )


def compute_PSF_leakage(
    cat_gal,
    g_corr_mc,
    dat,
    mask_combined,
    mask_metacal,
    num_bins=20,
):
    """Compute PSF Leakage.
    
    """
    cat_gal["e1"] = g_corr_mc[0]
    cat_gal["e2"] = g_corr_mc[1]
    cat_gal["e1_PSF"] = sp_cat.get_col(
        dat, "e1_PSF", mask_combined._mask, mask_metacal
    )
    cat_gal["e2_PSF"] = sp_cat.get_col(
        dat, "e2_PSF", mask_combined._mask, mask_metacal
    )

    weight_type = "des"
    key = f"w_{weight_type}"
    if key not in cat_gal:
        raise KeyError("Key '{key}' not found in cat_gal")

    try:
        alpha_1, alpha_2 = calibration.get_alpha_leakage_per_object(
            cat_gal, num_bins, weight_type
        )
    except:
        alpha_1, alpha_2 = -99, -99

    return alpha_1, alpha_2


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
