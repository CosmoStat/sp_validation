"""RUN JOINT CAT.

This module implements the class to create a joint comprehensive catalogue.

:Authors: Martin Kilbinger
"""

import sys
import numpy as np

from optparse import OptionParser

from astropy.io import fits
from astropy.table import Column

from cs_util import logging
from cs_util import cat
from cs_util import args as cs_args


class JointCat:
    """Joint Cat.

    Class to create joint catalogue.

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
        if self._params["patches"] == "v1":
            n_patch = 7
            patches = [f"P{x}" for x in np.arange(n_patch) + 1]
        else:
            patches = self._params["patches"].split("+")

        return patches

    def merge_catalogues(self, patches, base_path="."):
        
        input_sub_path = (
            f"sp_output/shape_catalog_comprehensive_{self._params['sh']}.fits"
        )
        output_path = (
            f"{self._params['survey']}_{self._params['pipeline']}"
            + f"_comprehensive_{self._params['year']}_v{self._params['version']}.fits"
        )
        
        dat_all = {}

        # Loop over patches
        for idx, patch in enumerate(patches):
            if self._params["verbose"]:
                print(patch, end=" ")

            input_path = f"{base_path}/{patch}/{input_sub_path}"
            try:
                dat = fits.getdata(input_path, self._params["hdu"])
            except:
                raise ValueError(
                    "Could not read data of file {input_path} at HDU #{self._params['hdu']}"
                )

            # Create empty lists if first patch
            if idx == 0:
                col_names = dat.dtype.names
                formats = {}
                for name in col_names:
                    formats[name] = dat.dtype.fields[name][0]
                for name in col_names:
                    dat_all[name] = []
                dat_all["patch"] = []

            # Append new data
            for name in col_names:
                dat_all[name].append(dat[name])
            if self._params["verbose"]:
                print(f"Added {len(dat)} objects.")

            # Add patch number
            dat_all["patch"].append([idx + 1] * len(dat))
        
        col_names = col_names + ("patch",)
        formats["patch"] = "I"

        for name in col_names:
            dat_all[name] = np.concatenate(dat_all[name], axis=0)

        column_all = []
        for name in col_names:
            if dat_all[name].ndim == 1:
                column = fits.Column(
                    name=name, array=dat_all[name], format=formats[name]
                )
                column_all.append(column)
            else:
                column_2d = [
                    fits.Column(
                        name=f"{name}_{idx}",
                        array=dat_all[name][:, idx],
                        format="D",
                    ) for idx in range(dat_all[name].shape[1])
                ]
                for column in column_2d:
                    column_all.append(column)
        
        print(f"Final total length = {len(dat_all[col_names[0]])}.")
        
        # Write to disk
        cat.write_fits_BinTable_file(column_all, output_path)

        table_hdu = fits.BinTableHDU.from_columns(column_all)

    def run(self):
        """Run.

        Main processing function.

        """
        patches = self.get_patches()
        if self._params["verbose"]:
            print("Merging patches", patches)

        self.merge_catalogues(patches)
        

def run_joint_comprehensive_cat(*args):
    """Run Joint Comprehensive Cat.

    Run class to create joint comprehensive catalogue from command line.

    """
    obj = JointCat()

    obj.set_params_from_command_line(args)

    obj.run()
