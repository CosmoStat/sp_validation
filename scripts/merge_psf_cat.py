#!/usr/bin/env python3
"""MERGE PSF CAT.

Merge PSF catalogues (psf_catalog_ngmix.fits) from different campaigns
into a single FITS file.

:Author: Martin Kilbinger
"""

import sys

import numpy as np
from astropy.io import fits
from cs_util import args as cs_args
from cs_util import cat, logging


class MergePsfCat:
    """Merge Psf Cat.

    Class to merge PSF catalogues from multiple campaigns.

    """

    def __init__(self):
        self.params_default()

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "campaigns": None,
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
            "campaigns": "-p",
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
            "campaigns": "list of campaigns separated by '+'",
            "sh": "shape measurement method, default={}",
            "survey": "survey name, default={}",
            "year": "year of processing, default={}",
            "version": "catalogue version, default={}",
            "base_path": "base path containing campaign directories, default={}",
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

    def get_campaigns(self):
        """Get Campaigns.

        Return list of campaigns according to option parameter value.

        Returns
        -------
        list
            campaigns, list of str

        """
        campaigns = self._params["campaigns"]
        if not campaigns:
            raise ValueError("No campaigns given; set 'campaigns'")

        return campaigns.split("+")

    def merge_catalogues(self, campaigns):
        """Merge Catalogues.

        Merge PSF catalogues from campaigns into one FITS file.

        Parameters
        ----------
        campaigns : list of str
            list of campaigns / sub-directories

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
        for idx, campaign in enumerate(campaigns):
            if verbose:
                print(f"  {campaign}")

            input_path = f"{base_path}/{campaign}/{input_sub_path}"
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
                dat_all["campaign"] = []

            for name in col_names:
                dat_all[name] = np.append(dat_all[name], dat[name])

            dat_all["campaign"] = np.append(dat_all["campaign"], [idx + 1] * len(dat))

        col_names = col_names + ("campaign",)

        column_all = []
        for name in col_names:
            if name != "campaign":
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
        campaigns = self.get_campaigns()

        if self._params["verbose"]:
            print("Merging PSF catalogues from campaigns:", campaigns)

        self.merge_catalogues(campaigns)


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
