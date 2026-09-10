"""CATALOG BUILDERS.

Catalogue construction pipeline — the runner classes that create, mask, and
calibrate joint comprehensive catalogues, plus the ``run_*`` entry-point
functions. Built on the catalogue data layer in ``catalog`` (imported here as
``sp_cat``), which supplies the read/write/column-access/matching primitives.

:Author: Martin Kilbinger
"""

import datetime
import os
from importlib.metadata import version

import h5py
import healsparse as hsp
import numpy as np
import yaml
from astropy.io import fits
from cs_util import args as cs_args
from cs_util import logging

# Spatial-masking primitives now live in ``masks``; re-exported here so external
# code using ``from sp_validation import catalog_builders as sp_joint`` keeps
# resolving ``sp_joint.Mask``, ``sp_joint.get_masks_from_config``, etc.
from sp_validation.masks import (
    Mask,
    confusion_matrix,
    correlation_matrix,
    get_masks_from_config,
    print_mask_stats,
)

from . import calibration, format
from . import catalog as sp_cat

# Names re-exported for external code that resolves them off this module.
__all__ = [
    "Mask",
    "get_masks_from_config",
    "print_mask_stats",
    "correlation_matrix",
    "confusion_matrix",
]


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
        # Read command line options (parses sys.argv; may exit on --help)
        cs_args.parse_options(
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
            print("Done.")

    def close_hd5(self):
        """Close HD5.

        Close HDF5 file.

        """
        self._hd5file.close()


def _promote(dtype_a, dtype_b):
    """Return a dtype that holds both input dtypes without truncation."""
    if dtype_a == dtype_b:
        return dtype_a
    if dtype_a.kind in "SU" and dtype_b.kind in "SU":
        kind = "U" if "U" in (dtype_a.kind, dtype_b.kind) else "S"
        size_a = dtype_a.itemsize // (4 if dtype_a.kind == "U" else 1)
        size_b = dtype_b.itemsize // (4 if dtype_b.kind == "U" else 1)
        return np.dtype(f"{kind}{max(size_a, size_b)}")

    return np.promote_types(dtype_a, dtype_b)


def _checked_assign(target, start, end, values, name):
    """Assign `values` into `target[start:end]`, refusing a lossy cast."""
    target[start:end] = values
    written = target[start:end]
    if written.dtype == values.dtype:
        return
    if written.dtype.kind in "fc":
        bad = np.isfinite(values) & ~np.isfinite(written)
    else:
        bad = written != values
    if np.any(bad):
        raise ValueError(
            f"Column {name!r} cannot be stored as {written.dtype}:"
            + f" {int(np.sum(bad))} value(s) overflow or are truncated."
            + " Disable reduce_mem or widen the output dtype."
        )


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
            "input_paths": None,
            "sh": "ngmix",
            "survey": "unions",
            "year": "2024",
            "version": "1.4.2",
            "pipeline": "shapepipe",
            "param_path": None,
            "reduce_mem": False,
            "verbose": False,
        }
        self._short_options = {
            "input_paths": "-i",
            "sh": "-g",
            "survey": "-s",
            "year": "-y",
            "version": "-V",
            "param_path": "-p",
            "reduce_mem": "-r",
        }
        self._types = {
            "reduce_mem": "bool",
        }
        self._help_strings = {
            "input_paths": (
                "campaign catalogue files (final_cat_<campaign>.hdf5) to merge,"
                + " separated by '+'"
            ),
            "sh": "shape measurement method, default={}",
            "survey": "survey name, default={}",
            "year": "year of processing, default={}",
            "version": "catalogue version, default={}",
            "param_path": "path to parameter file listing columns to keep",
            "reduce_mem": "output some columns in lower precision to reduce memory",
        }

    def get_input_paths(self):
        """Get Input Paths.

        Return the list of campaign catalogue files to merge.

        Returns
        -------
        list of str
            input file paths

        """
        input_paths = self._params["input_paths"]
        if not input_paths:
            raise ValueError(
                "No input campaign catalogues given; set 'input_paths' to one"
                + " or more final_cat_<campaign>.hdf5 files separated by '+'"
            )
        if isinstance(input_paths, str):
            input_paths = input_paths.split("+")

        return [path.strip() for path in input_paths if path.strip()]

    @staticmethod
    def campaign_name(input_path):
        """Campaign Name.

        Return the campaign name encoded in a catalogue file name,
        ``final_cat_<campaign>.hdf5`` -> ``<campaign>``.

        Parameters
        ----------
        input_path : str
            input file path

        Returns
        -------
        str
            campaign name

        """
        stem = os.path.splitext(os.path.basename(input_path))[0]
        prefix = "final_cat_"
        return stem[len(prefix) :] if stem.startswith(prefix) else stem

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
            "DEC",
            "FLAGS",
            "NUMBER",
        ]
        if dtype_in.kind == "U":
            # Transform unicode to string of equal length
            return np.dtype(f"S{dtype_in.itemsize // 4}")

        if not self._params["reduce_mem"]:
            return dtype_in
        if name not in cols_keep_dtype:
            if dtype_in.kind == "f" and dtype_in.itemsize == 8:
                return np.dtype(np.float32)
            if dtype_in.kind == "i" and dtype_in.itemsize == 4:
                # int32 -> int16, not int8: int8 cannot hold e.g. N_EPOCH or
                # CCD_NB values and wrapped them silently. Values that do not
                # fit are caught at assignment time by ``_checked_assign``.
                return np.dtype(np.int16)

        return dtype_in

    def output_dtype(self, dtypes_in, n_char_campaign):
        """Output Dtype.

        Return the merged-catalogue dtype: the input columns (possibly
        reduced in precision, and promoted to a common type across all input
        campaigns) plus a ``campaign`` column.

        Parameters
        ----------
        dtypes_in : numpy.dtype or list of numpy.dtype
            structured dtype(s) of the input campaign catalogues
        n_char_campaign : int
            width of the campaign name column

        Returns
        -------
        numpy.dtype
            output structured dtype

        Raises
        ------
        ValueError
            if the inputs have different column sets, or a column is
            multi-dimensional (campaign catalogues are scalar-column only)

        """
        if isinstance(dtypes_in, np.dtype):
            dtypes_in = [dtypes_in]

        names = dtypes_in[0].names
        for dtype_in in dtypes_in[1:]:
            if set(dtype_in.names) != set(names):
                raise ValueError(
                    "Campaign catalogues have incompatible column sets:"
                    + f" {sorted(names)} vs {sorted(dtype_in.names)}"
                )

        fields = []
        for name in names:
            subs = [dtype_in[name] for dtype_in in dtypes_in]
            for sub in subs:
                if sub.subdtype is not None:
                    raise ValueError(
                        f"Column {name!r} is multi-dimensional (shape"
                        + f" {sub.subdtype[1]}); campaign catalogues are"
                        + " expected to hold scalar columns only."
                    )
            # Promote across campaigns so a wider string or integer column in
            # a later file is not silently truncated or overflowed.
            promoted = subs[0]
            for sub in subs[1:]:
                promoted = _promote(promoted, sub)
            fields.append((name, self.dtype_out(name, promoted)))
        fields.append(("campaign", np.dtype(f"S{n_char_campaign}")))

        return np.dtype(fields)

    def write_hdf5_file(self, dat_all, campaigns=None):
        """Write HDF5 File.

        Write data to HDF5 file.

        Parameters
        ----------
        dat_all : numpy.ndarray
            input data
        campaigns : list, optional
            input campaign names, list of str

        """
        output_path = (
            f"{self._params['survey']}_{self._params['pipeline']}"
            + f"_comprehensive_{self._params['year']}_"
            + f"v{self._params['version']}.hdf5"
        )

        with h5py.File(output_path, "w") as f:
            self.write_hdf5_header(f, campaigns=campaigns)

            dset = f.create_dataset("data", data=dat_all)
            dset[:] = dat_all

    def write_hdf5_header(self, hd5file, campaigns=None):
        """Write HDF5 Header.

        Write header information to HDF5 file.

        Parameters
        ----------
        hd5file : h5py.File
            input HDF5 file
        campaigns : list, optional
            input campaign names, list of str, default is ``None``

        """
        super().write_hdf5_header(hd5file)

        if campaigns is not None:
            hd5file.attrs["campaigns"] = " ".join(campaigns)

    def merge_catalogues(self, input_paths):
        """Merge Catalogues.

        Merge a list of campaign catalogues into one joint catalogue, adding
        a ``campaign`` column that records each object's origin.

        Parameters
        ----------
        input_paths : list of str
            campaign catalogue files (final_cat_<campaign>.hdf5)

        Returns
        -------
        numpy.ndarray
            merged catalogue

        """
        param_list = (
            sp_cat.read_param_file(
                self._params["param_path"], verbose=self._params["verbose"]
            )
            if self._params["param_path"]
            else None
        )

        campaigns = [self.campaign_name(path) for path in input_paths]
        n_char_campaign = max(len(name) for name in campaigns)

        # First pass over file metadata only (row counts and dtypes), so the
        # merged array is allocated once and filled in place, instead of
        # concatenating per-campaign copies (peak memory 2x the output).
        shapes = [
            sp_cat.campaign_shape(path, param_list=param_list)
            for path in input_paths
        ]
        n_total = sum(n_rows for n_rows, _ in shapes)
        dtype_out = self.output_dtype([dtype for _, dtype in shapes], n_char_campaign)

        dat_all = np.empty(n_total, dtype=dtype_out)
        start = 0
        for input_path, campaign in zip(input_paths, campaigns):
            dat = sp_cat.read_campaign_catalogue(
                input_path,
                param_list=param_list,
                verbose=self._params["verbose"],
            )

            end = start + len(dat)
            for name in dat.dtype.names:
                _checked_assign(dat_all[name], start, end, dat[name], name)
            dat_all["campaign"][start:end] = campaign.encode()
            start = end

            if self._params["verbose"]:
                print(
                    f"{campaign}: added {len(dat)}"
                    + f" (~{format.millify(len(dat))}) objects."
                )
            del dat

        if self._params["verbose"]:
            print(
                f"Merged {len(dat_all)} (~{format.millify(len(dat_all))})"
                + f" objects from {len(campaigns)} campaign(s)."
            )

        return dat_all

    def run(self):
        """Run.

        Main processing function.

        """
        input_paths = self.get_input_paths()
        if self._params["verbose"]:
            print("Merging campaigns", input_paths)

        dat_all = self.merge_catalogues(input_paths)
        self.write_hdf5_file(dat_all, [self.campaign_name(p) for p in input_paths])


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
            self._params["aux_mask_num"] = len(self._params["aux_mask_file_list"])
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
                        mask._descr.encode("utf-8"),
                        mask._label.encode("utf-8"),
                    )
                f.create_dataset("applied_masks", data=descr_arr)

    def write_hdf5_header(self, hd5file):
        """Write HDF5 Header.

        Write header information to HDF5 file.

        Parameters
        ----------
        hd5file : h5py.File
            input HDF5 file
        campaigns : list, optional
            input campaign names, list of str, default is ``None``

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

        # Image-simulation path: a single per-run comprehensive catalogue in
        # FITS, not the joined multi-campaign HDF5 the data path builds. Read the
        # FITS table directly into memory; there is no separate data_ext group.
        extension = os.path.splitext(fpath)[1]
        if extension == ".fits":
            if verbose:
                print(f"Reading FITS file {fpath}, HDU 1...")
            dat = fits.getdata(fpath, 1)
            dat_ext = None
            if verbose:
                print(
                    f"Found {len(dat)} (~{format.millify(len(dat))}) objects"
                    + " in catalogue"
                )
            return dat, dat_ext

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
        purpose="weights",
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
    """Compute PSF Leakage."""
    cat_gal["e1"] = g_corr_mc[0]
    cat_gal["e2"] = g_corr_mc[1]
    cat_gal["e1_PSF"] = sp_cat.get_col(dat, "e1_PSF", mask_combined._mask, mask_metacal)
    cat_gal["e2_PSF"] = sp_cat.get_col(dat, "e2_PSF", mask_combined._mask, mask_metacal)

    weight_type = "des"
    key = f"w_{weight_type}"
    if key not in cat_gal:
        raise KeyError("Key '{key}' not found in cat_gal")

    try:
        alpha_1, alpha_2 = calibration.get_alpha_leakage_per_object(
            cat_gal, num_bins, weight_type
        )
    except Exception:
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
