import os
from optparse import OptionParser

import numpy as np
import pandas as pd
from astropy.io import fits

from cs_util import cat as cat_csu
from cs_util import logging
from cs_util import plots

from sp_validation import cat as cat_sp
from sp_validation import correlation as corr
from sp_validation import cosmology as cosmology
from sp_validation import io
from sp_validation import util


# To cs_util
def parse_options(p_def, short_options, types, help_strings):
    """Parse command line options.

    Parameters
    ----------
    p_def : dict
        default parameter values
    help_strings : dict
        help strings for options

    Returns
    -------
    dict
        Command line options
    """

    usage = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    for key in p_def:
        if key in help_strings:
            if key in short_options:
                short = short_options[key]
            else:
                short = ""

            if key in types:
                typ = types[key]
            else:
                typ = "string"

            parser.add_option(
                short,
                f"--{key}",
                dest=key,
                type=typ,
                default=p_def[key],
                help=help_strings[key].format(p_def[key]),
            )

    parser.add_option(
        "-v", "--verbose", dest="verbose", action="store_true", help=f"verbose output"
    )

    options, _ = parser.parse_args()

    return options


# Maybe put the following function somewhere else, unions_wl/default.py?
def get_theo_xi_planck(theta, dndz_path):

    Om = 0.3153
    sig8 = 0.8111
    ns = 0.9649
    Ob = 0.0493
    h = 0.6736

    z, nz, _ = cat_csu.read_dndz(dndz_path)

    xi_p, xi_m = cosmology.get_theo_xi(
        theta,
        z,
        nz,
        Omega_m=Om,
        h=h,
        Omega_b=Ob,
        sig8=sig8,
        ns=ns
    )

    return xi_p, xi_m


class LeakageScale:
    """Leakage Scale.

    Class to compute scale-dependent PSF leakage.

    """

    def __init__(self):
        # Set default parameters
        self.params_default()

    def set_params_from_command_line(self, args):
        """Set Params From Command line.

        Only use when calling using python from command line.
        Does not work from ipython or jupyter.

        """
        # Read command line options
        options = parse_options(
            self._params,
            self._short_options,
            self._types,
            self._help_strings,
        )

        # Update parameter values from options
        for key in vars(options):
            self._params[key] = getattr(options, key)

        # del options ?

        # Save calling command
        logging.log_command(args)

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "input_path_shear": None,
            "e1_col": "e1_uncal",
            "e2_col": "e2_uncal",
            "input_path_PSF": None,
            "hdu_psf": 1,
            "ra_star_col": "RA",
            "dec_star_col": "Dec",
            "e1_PSF_star_col": "E1_PSF_HSM",
            "e2_PSF_star_col": "E2_PSF_HSM",
            "dndz_path" : None,
            "output_dir": ".",
            "sh": "ngmix",
            "close_pair_tolerance": None,
            "close_pair_mode": None,
            "cut": None,
            "theta_min_amin": 1,
            "theta_max_amin": 300,
            "n_theta": 20,
            "leakage_alpha_ylim": [-0.03, 0.1],
            "leakage_xi_sys_ylim": [-4e5, 5e5],
            "leakage_xi_sys_log_ylim": [2e-13, 5e-5],
        }

        self._short_options = {
            "input_path_shear": "-i",
            "input_path_PSF": "-I",
            "output_dir": "-o",
            "shapes": "-s",
            "close_pair_tolerance": "-t",
            "close_pair_mode": "-m",
        }

        self._types = {
            "hdu_psf": "int",
            "theta_min_amin": "float",
            "theta_max_amin": "float",
            "n_theta": "int",
        }

        self._help_strings = {
            "input_path_shear": "input path of the shear catalogue",
            "e1_col": "e1 column name in galaxy catalogue, default={}",
            "e2_col": "e2 column name in galaxy catalogue, default={}",
            "input_path_PSF": "input path of the PSF catalogue",
            "hdu_PSF": "HDU number of PSF catalogue, default={}",
            "ra_star_col": "right ascension column name in star catalogue, default={}",
            "dec_star_col": "declination column name in star catalogue, default={}",
            "e1_PSF_star_col": "e1 PSF column name in star catalogue, default={}",
            "e2_PSF_star_col": "e2 PSF column name in star catalogue, default={}",
            "dndz_path" : (
                "path to galaxy redshift distribution file, for xi_sys ratio"
            ),
            "output_dir": "output_directory, default={}",
            "sh": "shape measurement method, default={}",
            "close_pair_tolerance": (
                "tolerance angle for close objects in star catalogue, default={}"
            ),
            "close_pair_mode": (
                "mode for close objects in star catalogue, allowed are"
                + f" 'remove', 'average'"
            ),
            "cut": (
                "list of criteria (white-space separated, do not use '_')"
                + f" to cut data, e.g. 'w>0_mask!=0'"
            ),
            "theta_min_amin": "mininum angular scale [arcmin], default={}",
            "theta_max_amin": "maximum angular scale [arcmin], default={}",
            "n_theta": "number of angular scales on input, default={}",
        }

    def check_params(self):
        """Check Params.

        Check whether parameter values are valid.

        Raises
        ------
        ValueError
            if a parameter value is not valid

        """
        if not self._params["input_path_shear"]:
            raise ValueError("No input shear catalogue given")
        if not self._params["input_path_PSF"]:
            raise ValueError("No input star/PSF catalogue given")
        if not self._params["dndz_path"]:
            raise ValueError("No input n(z) file given")

        if "verbose" not in self._params:
            self._params["verbose"] = False

    def read_data(self):
        """Read Data.

        Read input galaxy and PSF catalogues.
        """

        # Read input shear
        dat_shear = self.read_shear_cat()

        # Apply cuts to galaxy catalogue if required
        dat_shear = cat_sp.cut_data(
            dat_shear,
            self._params["cut"],
            self._params["verbose"]
        )

        # Read star catalogue
        dat_PSF = io.open_fits_or_npy(
            self._params["input_path_PSF"],
            hdu_no=self._params["hdu_psf"],
        )

        # Deal with close objects in PSF catalogue (= stars on same position
        # from different exposures)
        dat_PSF = self.handle_close_objects(dat_PSF)

        # Set instance variables
        self.dat_shear = dat_shear
        self.dat_PSF = dat_PSF

    def prepare_output(self):
        """Prepare Output.

        Prepare output directory and stats file.

        """
        if not os.path.exists(self._params["output_dir"]):
            os.mkdir(self._params["output_dir"])
        self._stats_file = io.open_stats_file(
            self._params["output_dir"], "stats_file_leakage.txt"
        )

    def run(self):
        """Run.

        Main processing of scale-dependent leakage.

        """
        # Check parameter validity
        self.check_params()

        # Read input data
        self.read_data()

        # Prepare output
        self.prepare_output()

        # compute auto- and cross-correlation functions including alpha
        self.compute_corr_gp_pp_alpha()

        # alpha leakage
        self.do_alpha()

        # xi_sys function
        self.do_xi_sys()

    def read_shear_cat(self):
        # Read galaxy catalogue
        in_path = self._params["input_path_shear"]
        _, file_extension = os.path.splitext(in_path)
        if file_extension == '.parquet':
            df = pd.read_parquet(in_path, engine='pyarrow')
            sep_array = df['Separation'].to_numpy()
            idx = np.argwhere(np.isfinite(sep_array))
            dat_shear = {}
            for col in df:
                dat_shear[col] = df[col].to_numpy()[idx].flatten()
        else:
            hdu_list = fits.open(in_path)
            dat_shear = hdu_list[1].data
        n_shear = len(dat_shear)
        io.print_stats(
            f"{n_shear} galaxies found in shear catalogue",
            self._stats_file,
            verbose=self._params["verbose"],
        )

        return dat_shear

    def handle_close_objects(self, dat_PSF):
        """Handle Close Objects.

        Deal with close objects in PSF catalogue.

        Parameters
        ----------
        dat_PSF : FITS.record
            input PSF data

        Returns
        -------
        FITS.record
            processed PSF data

        """
        if not self._params["close_pair_tolerance"]:
            return dat_PSF

        n_star = len(dat_PSF)

        tolerance_angle = coords.Angle(self._params["close_pair_tolerance"])

        io.print_stats(
            f"close object distance = {tolerance_angle}",
            self._stats_file,
            verbose=self._params["verbose"],
        )

        # Create SkyCoord object from star positions
        coordinates = coords.SkyCoord(
            ra=dat_PSF[self._params["ra_star_col"]],
            dec=dat_PSF[self._params["dec_star_col"]],
            unit="deg",
        )

        # Search PSF catalogue in itself around tolerance angle
        indices1, indices2, d2d, d3d = coordinates.search_around_sky(
            coordinates, tolerance_angle
        )

        # Count multiplicity of indices = number of matches of search
        count = np.bincount(indices1)
        dat_PSF_proc = {}

        # Copy unique objects (multiplicity of unity)
        for col in dat_PSF.dtype.names:
            dat_PSF_proc[col] = dat_PSF[col][count == 1]
        n_non_close = len(dat_PSF_proc[self._params["ra_star_col"]])
        io.print_stats(
            f"found {n_non_close}/{n_star} = {n_non_close / n_star:.1%} "
            + "non-close objects",
            self._stats_file,
            verbose=self._params["verbose"],
        )

        # Deal with repeated objects (multiplicity > 1)
        multiples = count != 1
        if not multiples.any():
            # No multiples found -> no action
            io.print_stats(
                "no close objects found",
                self._stats_file,
                verbose=self._params["verbose"],
            )

        else:
            # Get index list of multiple objects
            idx_mult = np.where(multiples)[0]
            if self._params["mode"] == "average":
                # Initialise additional data vector
                dat_PSF_mult = {}
                for col in dat_PSF.dtype.names:
                    dat_PSF_mult[col] = []

                done = np.array([])
                n_avg_rem = 0

                # Loop over repeated indices
                for idx in idx_mult:
                    # If already used: ignore this index
                    if idx in done:
                        continue

                    # Get indices in data index list corresponding to
                    # this multiple index
                    w = np.where(indices1 == idx)[0]

                    # Get indices in data
                    ww = indices2[w]

                    # Append mean to additional data vector
                    for col in dat_PSF.dtype.names:
                        mean = np.mean(dat_PSF[col][ww])
                        dat_PSF_mult[col].append(mean)

                    # Register indixes to avoid repetition
                    done = np.append(done, ww)
                    n_avg_rem += len(ww) - 1

                n_avg = len(dat_PSF_mult[ra_star_col])
                io.print_stats(
                    f"adding {n_avg}/{n_star} = {n_avg / n_star:.1%} "
                    + "averaged objects",
                    self._stats_file,
                    verbose=self._params["verbose"],
                )

                for col in dat_PSF.dtype.names:
                    dat_PSF_proc[col] = np.append(dat_PSF_proc[col], dat_PSF_mult[col])
            elif mode == "remove":
                n_rem = len(idx_mult)
                io.print_stats(
                    f"removing {n_rem}/{n_star} = {n_rem / n_star:.1%} "
                    + "close objects",
                    self._stats_file,
                    verbose=self._params["verbose"],
                )

        # Test
        coordinates_proc = coords.SkyCoord(
            ra=dat_PSF_proc[self._params["ra_star_col"]],
            dec=dat_PSF_proc[self._params["dec_star_col"]],
            unit="deg",
        )
        idx, d2d, d3d = coords.match_coordinates_sky(
            coordinates_proc, coordinates_proc, nthneighbor=2
        )
        non_close = (d2d > tolerance_angle).all()
        io.print_stats(
            f"Check: all remaining distances > {tolerance_angle}? {non_close}",
            self._stats_file,
            verbose=self._params["verbose"],
        )
        if mode == "average":
            io.print_stats(
                f"Check: n_non_close + n_avg + n_avg_rem = n_star? "
                + f"{n_non_close} + {n_avg} + {n_avg_rem} = "
                + f"{n_non_close + n_avg + n_avg_rem} ({n_star})",
                self._stats_file,
                verbose=self._params["verbose"],
            )
        elif mode == "remove":
            io.print_stats(
                f"Check: n_non_close + n_rem = n_star? {n_non_close} "
                + f"+ {n_rem} = {n_non_close + n_rem} ({n_star})",
                self._stats_file,
                verbose=self._params["verbose"],
            )

        n_in = len(dat_PSF[self._params["ra_star_col"]])
        n_out = len(dat_PSF_proc[self._params["dec_star_col"]])

        if n_in == n_out:
            io.print_stats(
                f"keeping all {n_out} stars",
                self._stats_file,
                verbose=self._params["verbose"],
            )
        else:
            io.print_stats(
                f"keeping {n_out}/{n_in} = {n_out/n_in:.1%} stars",
                self._stats_file,
                verbose=self._params["verbose"],
            )

        return dat_PSF_proc

    def compute_corr_gp_pp_alpha(self):
        """Compute Corr GP PP Alpha

        Compute and plot scale-dependent PSF leakage functions.

        """
        ra = self.dat_shear["RA"]
        dec = self.dat_shear["Dec"]
        e1_gal = self.dat_shear[self._params["e1_col"]]
        e2_gal = self.dat_shear[self._params["e2_col"]]
        weights = self.dat_shear["w"]

        ra_star = self.dat_PSF[self._params["ra_star_col"]]
        dec_star = self.dat_PSF[self._params["dec_star_col"]]
        e1_star = self.dat_PSF[self._params["e1_PSF_star_col"]]
        e2_star = self.dat_PSF[self._params["e2_PSF_star_col"]]

        # Correlation functions
        r_corr_gp, r_corr_pp = corr.correlation_12_22(
            ra,
            dec,
            e1_gal,
            e2_gal,
            weights,
            ra_star,
            dec_star,
            e1_star,
            e2_star,
            theta_min_amin=self._params["theta_min_amin"],
            theta_max_amin=self._params["theta_max_amin"],
            n_theta=self._params["n_theta"],
        )

        # Check consistency of angular scales
        if any(
            np.abs(r_corr_gp.meanr - r_corr_pp.meanr) / r_corr_gp.meanr
            > 0.1
        ):
            print("Warning: angular scales not consistent")

        # Set instance variables
        self.r_corr_gp = r_corr_gp
        self.r_corr_pp = r_corr_pp

    def compute_alpha_mean(self):
        """Compute Alpha Mean

        Compute weighted mean of the leakage function alpha

        """
        self.alpha_leak_mean = util.transform_nan(
            np.average(self.alpha_leak, weights=1 / self.sig_alpha_leak**2)
        )
        io.print_stats(
            f"{self._params['sh']}: Weighted average alpha"
            + f" = {self.alpha_leak_mean:.3g}",
            self._stats_file,
            verbose=self._params["verbose"],
        )

    def compute_xi_sys(self):
        """Compute Xi Sys

        Compute galaxy - PSF systematics correlation function

        """

        C_sys_p = self.r_corr_gp.xip**2 / self.r_corr_pp.xip
        C_sys_m = self.r_corr_gp.xim**2 / self.r_corr_pp.xim

        C_sys_std_p = np.abs(C_sys_p) * np.sqrt(
            (
                (
                    (2 * self.r_corr_gp.xip**2 * np.sqrt(self.r_corr_gp.varxip))
                    / self.r_corr_gp.xip
                )
                / self.r_corr_gp.xip**2
            )
            ** 2
            + (np.sqrt(self.r_corr_pp.varxip) / self.r_corr_pp.xip) ** 2
        )

        C_sys_std_m = np.abs(C_sys_m) * np.sqrt(
            (
                (
                    (2 * self.r_corr_gp.xim**2 * np.sqrt(self.r_corr_gp.varxim))
                    / self.r_corr_gp.xim
                )
                / self.r_corr_gp.xim**2
            )
            ** 2
            + (np.sqrt(self.r_corr_pp.varxim) / self.r_corr_pp.xim) ** 2
        )

        self.C_sys_p = C_sys_p
        self.C_sys_m = C_sys_m
        self.C_sys_std_p = C_sys_std_p
        self.C_sys_std_m = C_sys_std_m

    def plot_alpha_leakage(self):
        """Plot Alpha Leakage.

        Plot scale-dependent leakage function alpha(theta)

        """
        plot_dir_leakage = self._params["output_dir"]

        theta = [self.r_corr_gp.meanr]
        alpha_theta = [self.alpha_leak]
        yerr = [self.sig_alpha_leak]
        xlabel = r"$\theta$ [arcmin]"
        ylabel = r"$\alpha(\theta)$"
        title = self._params["sh"]
        out_path = (
            f"{self._params['output_dir']}"
            + f"/alpha_leakage_{self._params['sh']}.png"
        )
        xlim = [self._params["theta_min_amin"], self._params["theta_max_amin"]]
        ylim = self._params["leakage_alpha_ylim"]
        plots.plot_data_1d(
            theta,
            alpha_theta,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            xlim=xlim,
            ylim=ylim,
        )

    def plot_xi_sys(self):
        """Plot Xi Sys.

        Plot galaxy - PSF systematics correlation function.

        """
        labels = ["$\\xi^{\\rm sys}_+$", "$\\xi^{\\rm sys}_-$"]

        title = "Cross-correlation leakage"
        xlabel = "$\\theta$ [arcmin]"
        ylabel = "Correlation function"

        theta = [self.r_corr_gp.meanr] * 2
        xi = [self.C_sys_p, self.C_sys_m]
        yerr = [self.C_sys_std_p, self.C_sys_std_m]

        comp_arr = [0, 1]
        symb_arr = ["+", "-"]
        for comp, symb in zip(comp_arr, symb_arr):
            mean = np.mean(np.abs(xi[comp]))
            msg = f"{self._params['sh']}: <|xi_sys_{symb}|> = {mean}"
            io.print_stats(
                msg,
                self._stats_file,
                verbose=self._params["verbose"]
            )

        ylim = self._params["leakage_xi_sys_ylim"]
        out_path = (
            f"{self._params['output_dir']}/xi_sys_{self._params['sh']}.pdf"
        )
        plots.plot_data_1d(
            theta,
            xi,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            ylim=ylim,
            labels=labels,
        )

        ylim = self._params["leakage_xi_sys_log_ylim"]
        out_path = (
            f"{self._params['output_dir']}/xi_sys_log_{self._params['sh']}.pdf"
        )
        plots.plot_data_1d(
            theta,
            xi,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            ylog=True,
            ylim=ylim,
            labels=labels,
        )

    def plot_xi_sys_ratio(self, xi_p_theo, xi_m_theo):
        """Plot Xi Sys Ratio.

        Plot xi_sys relative to theoretical model of cosmological
        xi_pm.

        Parameters
        ----------
        xi_p_theo : list
            theoretical model of xi+
        xi_m_theo : list
            theoretical model of xi-

        """
        labels = [
            '$\\xi^{\\rm sys}_+ / \\xi_+$',
            '$\\xi^{\\rm sys}_- / \\xi_-$',
        ]

        title = 'Cross-correlation leakage ratio'
        xlabel = '$\\theta$ [arcmin]'
        ylabel = 'Correlation function ratio'

        theta = [self.r_corr_gp.meanr] * 2
        xi = [self.C_sys_p / xi_p_theo, self.C_sys_m / xi_m_theo]
        yerr = [self.C_sys_std_p / xi_p_theo, self.C_sys_std_m / xi_m_theo]

        comp_arr = [0, 1]
        symb_arr = ['+', '-']
        for comp, symb in zip(comp_arr, symb_arr):
            mean = np.mean(np.abs(xi[comp]))
            msg = (
                f"{self._params['sh']}: <|xi_sys_{symb}| / xi_{symb}> = {mean}"
            )
            io.print_stats(
                msg,
                self._stats_file,
                verbose=self._params['verbose']
            )

        out_path = (
            f"{self._params['output_dir']}"
            + f"/xi_sys_{self._params['sh']}_ratio.pdf"
        )

        ylim = [0, 0.5]

        plots.plot_data_1d(
            theta,
            xi,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            ylim=ylim,
            labels=labels
        )

    def do_alpha(self):
        """Do Alpha.

        Compute, plot, and save alpha leakage function.
        """
        # Get input catalogues for averages
        e1_gal = self.dat_shear[self._params["e1_col"]]
        e2_gal = self.dat_shear[self._params["e2_col"]]
        weights = self.dat_shear["w"]

        e1_star = self.dat_PSF[self._params["e1_PSF_star_col"]]
        e2_star = self.dat_PSF[self._params["e2_PSF_star_col"]]

        # Compute alpha leakage function
        self.alpha_leak, self.sig_alpha_leak = corr.alpha(
            self.r_corr_gp,
            self.r_corr_pp,
            e1_gal,
            e2_gal,
            weights,
            e1_star,
            e2_star,
        )
        self.compute_alpha_mean()

        # Plot
        self.plot_alpha_leakage()

        # Write to disk
        io.save_alpha(
            self.r_corr_gp.meanr,
            self.alpha_leak,
            self.sig_alpha_leak,
            self._params["sh"],
            self._params["output_dir"],
        )


    def do_xi_sys(self):
        """Do Xi Sys.

        Compute, plot, and save xi_sys function.
        """

        # Compute xi_sys
        self.compute_xi_sys()

        # Compute theoretical model for the 2PCF
        xi_p_theo, xi_m_theo = get_theo_xi_planck(
            self.r_corr_gp.meanr,
            self._params["dndz_path"]
        )

        # Plot
        self.plot_xi_sys()
        self.plot_xi_sys_ratio(xi_p_theo, xi_m_theo)

        # Write to disk
        io.save_xi_sys(
            self.r_corr_gp.meanr,
            self.C_sys_p,
            self.C_sys_m,
            self.C_sys_std_p,
            self.C_sys_std_m,
            xi_p_theo,
            xi_m_theo,
            self._params['sh'],
            self._params['output_dir'],
        )


def run_leakage_scale(*args):
    """Run Leakage Scale.

    Run scale-dependent PSF leakage as python script from commane line.

    """
    # Create object for scale-dependent leakage calculations
    obj = LeakageScale()

    obj.set_params_from_command_line(args)

    obj.run()
