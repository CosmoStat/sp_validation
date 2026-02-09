#!/usr/bin/env python3
"""MERGE PSF CAT.

Merge PSF catalogues (psf_catalog_ngmix.fits) from different patches
into a single FITS file.

:Author: Martin Kilbinger
"""

import sys

import numpy as np
from astropy.io import fits

from cs_util import logging
from cs_util import cat
from cs_util import args as cs_args


class MergePsfCat:
    """Merge Psf Cat.

    Class to merge PSF catalogues from multiple patches.

    """

    def __init__(self):
        self.params_default()

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
            "base_path": ".",
            "hdu": 1,
            "verbose": False,
        }
        self._short_options = {
            "patches": "-p",
            "sh": "-g",
            "survey": "-s",
            "year": "-y",
            "version": "-V",
            "base_path": "-b",
            "hdu": "-H",
        }
        self._types = {
            "hdu": "int",
        }
        self._help_strings = {
            "patches": (
                "list of patches separated by '+', or shortcut "
                "(allowed are 'v1', 'v1.5'), default={}"
            ),
            "sh": "shape measurement method, default={}",
            "survey": "survey name, default={}",
            "year": "year of processing, default={}",
            "version": "catalogue version, default={}",
            "base_path": "base path containing patch directories, default={}",
            "hdu": "HDU number to read from input FITS files, default={}",
        }

    def set_params_from_command_line(self, args):
        """Set Params From Command Line.

        Only use when calling using python from command line.
        Does not work from ipython or jupyter.

        """
        options = cs_args.parse_options(
            self._params,
            self._short_options,
            self._types,
            self._help_strings,
        )

        self._params.update(options)

        logging.log_command(args)

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
        elif self._params["patches"] == "v1.6":
            n_patch = 9
            patches = [f"P{x}" for x in np.arange(n_patch) + 1]
        else:
            patches = self._params["patches"].split("+")

        return patches

    def merge_catalogues(self, patches):
        """Merge Catalogues.

        Merge PSF catalogues from sub-patches into one FITS file.

        Parameters
        ----------
        patches : list of str
            list of patches/sub-directories

        """
        base_path = self._params["base_path"]
        sh = self._params["sh"]
        hdu_in = self._params["hdu"]
        verbose = self._params["verbose"]

        input_sub_path = f"sp_output/psf_catalog_{sh}.fits"
        output_path = (
            f"{self._params['survey']}_{self._params['pipeline']}_star_"
            f"{self._params['year']}_v{self._params['version']}.fits"
        )

        dat_all = {}
        for idx, patch in enumerate(patches):

            if verbose:
                print(f"  {patch}")

            input_path = f"{base_path}/{patch}/{input_sub_path}"
            try:
                dat = fits.getdata(input_path, hdu_in)
            except Exception:
                print(f"No data found in file {input_path} at HDU #{hdu_in}")
                print(f"Trying at HDU #{hdu_in - 1}")
                dat = fits.getdata(input_path, hdu_in - 1)

            if idx == 0:
                col_names = dat.dtype.names
                for name in col_names:
                    dat_all[name] = []
                dat_all["patch"] = []

            for name in col_names:
                dat_all[name] = np.append(dat_all[name], dat[name])

            dat_all["patch"] = np.append(dat_all["patch"], [idx + 1] * len(dat))

        col_names = col_names + ("patch",)

        column_all = []
        for name in col_names:
            if name != "patch":
                my_format = "D"
            else:
                my_format = "I"
            column = fits.Column(name=name, array=dat_all[name], format=my_format)
            column_all.append(column)

        if verbose:
            print(f"Writing file {output_path}")
        cat.write_fits_BinTable_file(column_all, output_path)

    def run(self):
        """Run.

        Main processing function.

        """
        patches = self.get_patches()

        if self._params["verbose"]:
            print("Merging PSF catalogues from patches:", patches)

        self.merge_catalogues(patches)


def main(argv=None):
    """Main.

    Main program.

    """
    obj = MergePsfCat()

    obj.set_params_from_command_line(argv)

    obj.run()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
