"""PLOTS.

:Name: plots.py

:Description: This script contains methods for plots.

:Author: Martin Kilbinger


"""

from collections import Counter

import healpy as hp
import healsparse as hsp
import matplotlib.pylab as plt
import numpy as np
import skyproj
from astropy import units as u
from astropy.coordinates import SkyCoord
from cs_util import plots
from lenspack.geometry.projections.gnom import radec2xy

from sp_validation.plot_style import *


def plot_spatial_density(
    ra, dec, title, x_label, y_label, cbar_label, out_path, n_grid=1000, verbose=False
):
    """Plot Spatial Density.

    Plot spatial density distribution of objects.

    Parameters
    ----------
    ra, dec : array of float
        coordinates
    title : string
        plot title
    x_label, y_label : string
        x-/y-axis label
    cbar_label : string
        color bar label
    out_path : string
        output file path
    n_grid : int, optional, default=1000
        number of hex grid points
    verbose : bool, optional, default=False
        verbose output if True
    """
    plots.figure(figsize=(30, 30))

    if max(ra) > 360:
        ra_plot = ra - 360
    else:
        ra_plot = ra
    plt.hexbin(ra_plot, dec, gridsize=n_grid)

    cbar = plt.colorbar()
    cbar.set_label(cbar_label, rotation=270, labelpad=40)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plots.savefig(out_path)


def get_ticks(loc, N, new_min, new_max):
    """Get ticks.

    Return formatted axis ticks for plots.

    Parameters
    ----------
    loc : array of floats
        original tick locations
    N : number of pixels (in origina coordinates)
    new_min : float
        new coordinate minimum
    new_max : float
        new coordinate maximum

    Returns
    -------
    loc_new : array of floats
        new tick locations
    labels_new : array of strings
        new tick labels
    """
    loc_new = []
    labels_new = []

    for i in range(1, len(loc) - 1):
        lab = loc[i] / N * (new_max - new_min) + new_min
        loc_new.append(loc[i])
        labels_new.append(f"{lab:.1f}")

    return loc_new, labels_new


def plot_map(
    m,
    ra,
    dec,
    min_x,
    max_x,
    min_y,
    max_y,
    Nx,
    Ny,
    title,
    out_path,
    vlim=None,
    grid=True,
    clusters=None,
    map_cut_coords=None,
    dpi=100,
    colorbar=True,
):
    """Plot Map.

    Plots 2D map.

    Parameters
    ----------
    m : 2D array of float
        map
    ra, dec : array of float
        coordinates, for axis ticks
    title : string
        plot title
    out_path : string
        output file path
    vlim : array(2) of float, optional, default=None
        limits of map values, if not given compute from map
    grid : bool, optional
        if `True` (default) plot grid lines
    clusters :
        dictionary of cluster information, optional, default=None
    """
    plots.figure(figsize=(10, 10))

    # plot image
    plt.imshow(m)

    # Transform axis labels to ra, dec
    ra_min, ra_max = ra.min(), ra.max()
    ra_mean = np.mean(ra)
    dec_min, dec_max = dec.min(), dec.max()
    dec_mean = np.mean(dec)

    # save image limits
    xlim = plt.xlim()
    ylim = plt.ylim()

    # Set colorbar
    if not vlim:
        vlim = plt.gci().get_clim()
    else:
        plt.gci().set_clim(vlim)
    if colorbar:
        plt.colorbar()

    loc, labels = plt.xticks()
    loc_ra, labels_ra = get_ticks(loc, Nx, ra_min, ra_max)
    plt.xticks(loc_ra, labels=labels_ra)

    loc, labels = plt.yticks()
    loc_dec, labels_dec = get_ticks(loc, Ny, dec_min, dec_max)
    plt.yticks(loc_dec, labels=labels_dec)

    mean_x = (min_x + max_x) / 2
    mean_y = (min_y + max_y) / 2

    # plot grid
    if grid:
        grid_lines_ra = []
        grid_lines_dec = []
        n_per_line = 200

        # create lines of constant ra and varying dec, and vice versa

        # extend beyond projected image limits, to avoid image edges
        # without grid lines
        d = 2
        gl_ra = np.linspace(ra_min - d, ra_max + d, num=n_per_line)
        gl_dec = np.linspace(dec_min - d, dec_max + d, num=n_per_line)
        ra_list = np.arange(np.floor(ra_min - d), np.ceil(ra_max + d))
        dec_list = np.arange(np.floor(dec_min - d), np.ceil(dec_max + d))
        for ra in ra_list:
            grid_lines_ra.append([ra] * n_per_line)
            grid_lines_dec.append(gl_dec)
        for dec in dec_list:
            grid_lines_dec.append([dec] * n_per_line)
            grid_lines_ra.append(gl_ra)

        for grid_line_ra, grid_line_dec in zip(grid_lines_ra, grid_lines_dec):
            x, y = radec2xy(ra_mean, dec_mean, grid_line_ra, grid_line_dec)
            xx = (x + mean_x - min_x) / (max_x - min_x) * Nx
            yy = (y + mean_y - min_y) / (max_y - min_y) * Ny
            plt.plot(xx, yy, "w:", linewidth=0.5)

    # cut out if required
    if map_cut_coords:
        x_cut, y_cut = radec2xy(
            ra_mean,
            dec_mean,
            [map_cut_coords[0], map_cut_coords[1]],
            [map_cut_coords[2], map_cut_coords[3]],
        )
        xx = (x_cut + mean_x - min_x) / (max_x - min_x) * Nx
        yy = (y_cut + mean_y - min_y) / (max_y - min_y) * Ny
        xlim = plt.xlim(xx)
        ylim = plt.ylim(yy)

    # mark cluster positions
    if clusters:
        x_cluster = (clusters["x"] + mean_x - min_x) / (max_x - min_x) * Nx
        y_cluster = (clusters["y"] + mean_y - min_y) / (max_y - min_y) * Ny
        dy = 0.02
        plt.plot(
            x_cluster,
            y_cluster,
            "ro",
            mfc="none",
            markeredgewidth=0.9,
            markersize=12,
        )

    # go back to image limits
    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.xlabel("R.A. [deg]")
    plt.ylabel("Dec [deg]")

    plt.title(title)

    plots.savefig(out_path)

    return vlim


def plot_map_stacked(kappa, title, radius, output_path, vlim=None):
    """Plot Map Stacked.

    Plot stacked convergence map.

    Parameters
    ----------
    kappa : image
        map values
    title : string
        plot title
    output_path : string
        figure output file path

    vlim : array(2) of float, optional, default=None
        map limits; min and max of kappa if not given

    Returns
    -------
    array(2) of float
        map limits

    """
    plots.figure(figsize=(10, 10))

    # plot image
    plt.imshow(kappa)

    # set colorbar
    if not vlim:
        vlim = plt.gci().get_clim()
    else:
        plt.gci().set_clim(vlim)
    plt.colorbar()

    npix = kappa.shape[0]

    # mark center
    plt.plot(npix / 2 - 1, npix / 2 - 1, "+")

    # axes ticks
    n_ticks = 4
    loc = np.arange(0, npix + npix / n_ticks, step=npix / n_ticks)
    lab = np.round(
        np.arange(
            -radius,
            radius + radius * 2 / n_ticks,
            step=radius * 2 / n_ticks,
        ),
        1,
    )
    plt.xticks(loc, labels=lab)
    plt.yticks(loc, labels=lab)

    plt.xlabel(r"separation $R$ [Mpc]")
    plt.ylabel(r"separation $R$ [Mpc]")

    plt.title(title)

    plots.savefig(output_path)

    return vlim


def plot_binned_one(
    ax,
    quantity,
    bin_edges_x,
    bin_edges_y,
    vmin=None,
    vmax=None,
    title=None,
    xlabel=None,
    ylabel=None,
):

    # Note: transpose R slice to match (y, x) shape required by pcolormesh
    pcm = ax.pcolormesh(
        bin_edges_x, bin_edges_y, quantity, vmin=vmin, vmax=vmax, shading="auto"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.colorbar(pcm, ax=ax)


def plot_binned(
    quantities,
    key,
    bin_edges_x,
    bin_edges_y,
    title_base,
    vmin=None,
    vmax=None,
    xlabel=None,
    ylabel=None,
):

    len_shape = len(quantities[key].shape)

    fig_size = 2 * len_shape
    fig = plt.figure(figsize=(fig_size, fig_size))

    if len_shape == 2:

        ax = plt.subplot2grid((1, 1), (0, 0))
        plot_binned_one(
            ax,
            quantities[key].T,
            bin_edges_x,
            bin_edges_y,
            vmin=vmin,
            vmax=vmax,
            title=title_base,
            xlabel=xlabel,
            ylabel=ylabel,
        )

    elif len_shape == 4:
        for idx in (0, 1):
            for jdx in (0, 1):
                ax = plt.subplot2grid((2, 2), (idx, jdx))

                if vmin is not None and vmax is not None:
                    if idx == jdx:
                        my_vmin = vmin["diag"]
                        my_vmax = vmax["diag"]
                    else:
                        my_vmin = vmin["offdiag"]
                        my_vmax = vmax["offdiag"]
                else:
                    my_vmin = None
                    my_vmax = None

                plot_binned_one(
                    ax,
                    quantities[key][:, :, idx, jdx].T,
                    bin_edges_x,
                    bin_edges_y,
                    vmin=my_vmin,
                    vmax=my_vmax,
                    title=f"${title_base}_{{{idx+1}{jdx+1}}}$",
                    xlabel=xlabel,
                    ylabel=ylabel,
                )

    plt.tight_layout()
    plots.savefig(f"{key}_binned.png")


class FootprintPlotter:
    """Class to create footprint plots.

    Parameters
    -----------
    nside_coverage: int, optional
        basic resolution of map; default is 32
    nside_map:
        fine resolution for plotting; default is 2048

    """

    # Dictionary storing region parameters
    _regions = {
        "NGC": {"ra_0": 180, "extend": [120, 270, 20, 70], "vmax": 60},
        "SGC": {"ra_0": 15, "extend": [-20, 45, 20, 45], "vmax": 60},
        "fullsky": {"ra_0": 150, "extend": [0, 360, -90, 90], "vmax": 60},
    }

    def __init__(self, nside_coverage=32, nside_map=2048):

        self._nside_coverage = nside_coverage
        self._nside_map = nside_map

    def create_hsp_map(self, ra, dec):
        """Create Hsp Map.

        Create healsparse map.

        Parameters
        ----------
        ra : numpy.ndarray
            right ascension values
        dec : numpy.ndarray
            declination values

        Returns
        -------
        hsp.HealSparseMap
            map

        """
        # Create empty map
        hsp_map = hsp.HealSparseMap.make_empty(
            self._nside_coverage, self._nside_map, dtype=np.float32, sentinel=np.nan
        )

        # Get pixel list corresponding to coordinates
        hpix = hp.ang2pix(self._nside_map, ra, dec, nest=True, lonlat=True)

        # Get count of objects per pixel
        pixel_counts = Counter(hpix)

        # List of unique pixels
        unique_hpix = np.array(list(pixel_counts.keys()))

        # Number of objects
        values = np.array(list(pixel_counts.values()), dtype=np.float32)

        # Create maps with numbers per pixel
        hsp_map[unique_hpix] = values

        return hsp_map

    def plot_area(
        self,
        hsp_map,
        ra_0=0,
        extend=[120, 270, 29, 70],
        vmax=60,
        projection=None,
        outpath=None,
        title=None,
    ):
        """Plot Area.

        Plot catalogue in an area on the sky.

        Parameters
        ----------
        hsp_map : hsp_HealSparseMap
            input map
        ra_0 : float, optional
            anchor point in R.A.; default is 0
        extend : list, optional
            sky region, extend=[ra_low, ra_high, dec_low, dec_high];
            default is [120, 270, 29, 70]
        vmax : float, optional
            maximum pixel value to plot with color; default is 60
        projection : skyproj.McBrydeSkyproj
            if ``None`` (default), a new plot is created
        outpath : str, optional
            output path, default is ``None``
        title : str, optional
            print title if not ``None`` (default)

        Returns
        --------
        skyproj.McBrydeSkyproj
            projection instance
        plt.axes.Axes
            axes instance

        Raises
        ------
        ValueError
            if no object found in region

        """
        if not projection:

            # Create new figure and axes
            fig, ax = plt.subplots(figsize=(10, 10))

            # Create new projection
            projection = skyproj.McBrydeSkyproj(
                ax=ax, lon_0=ra_0, extent=extend, autorescale=True, vmax=vmax
            )
        else:
            ax = None

        try:
            _ = projection.draw_hspmap(
                hsp_map, lon_range=extend[0:2], lat_range=extend[2:]
            )
        except ValueError:
            msg = "No object found in region to draw"
            print(f"{msg}, continuing...")
            # raise ValueError(msg)

        projection.draw_milky_way(width=25, linewidth=1.5, color="black", linestyle="-")

        if title:
            plt.title(title, pad=5)

        if outpath:
            plt.savefig(outpath)

        return projection, ax

    def plot_region(self, hsp_map, region, projection=None, outpath=None, title=None):

        return self.plot_area(
            hsp_map,
            region["ra_0"],
            region["extend"],
            region["vmax"],
            projection=projection,
            outpath=outpath,
            title=title,
        )

    def plot_all_regions(self, hsp_map, outbase=None):

        for region in self._regions:
            if outbase:
                outpath = f"{outbase}_{region}.png"
            else:
                outpath = None
            self.plot_region(hsp_map, self._regions[region], outpath=outpath)

    @classmethod
    def hp_pixel_centers(cls, nside, nest=False):

        # Get number of pixels for given nside
        npix = hp.nside2npix(nside)

        # Get pixel indices
        pix_indices = np.arange(npix)

        # Get coordinates of pixel centers
        ra, dec = hp.pix2ang(nside, pix_indices, nest=nest, lonlat=True)

        return ra, dec, npix

    @classmethod
    def plot_footprint_as_hp(cls, hsp_map, nside, outpath=None, title=None):

        ra, dec, npix = cls.hp_pixel_centers(nside)

        # Create an empty HEALPix map
        m = np.full(npix, np.nan)

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the HEALPix grid
        hp.mollview(m, title=title, coord="C", notext=True, rot=(180, 0, 0))

        # Define the Galactic Plane: l = [0, 360], b = 0°
        for l0, ls in zip((-5, 0, 5), (":", "-", ":")):
            l_values = np.linspace(0, 360, 500)  # 500 points along the plane
            b_values = np.zeros_like(l_values)  # Galactic latitude is 0 (the plane)

            # Convert (l, b) to (λ, β) - Ecliptic coordinates
            coords = SkyCoord(
                l=l_values * u.degree, b=b_values * u.degree, frame="galactic"
            )
            ecl_coords = coords.transform_to(
                "barycentrictrueecliptic"
            )  # Ecliptic frame

            # Extract Ecliptic longitude (λ) and latitude (β)
            lambda_ecl = ecl_coords.lon.deg  # Ecliptic longitude
            beta_ecl = ecl_coords.lat.deg  # Ecliptic latitude

            # Convert to HEALPix projection coordinates (colatitude, longitude)
            theta = np.radians(90 - beta_ecl)  # HEALPix uses colatitude
            phi = np.radians(lambda_ecl)  # HEALPix uses longitude

            # Create a healpy Mollweide projection in Ecliptic coordinates
            hp.projplot(
                theta, phi, linestyle=ls, color="black", linewidth=1
            )  # Plot the outline

        # Apply mask
        mask_values = hsp_map.get_values_pos(ra, dec, valid_mask=True, lonlat=True)

        ok = np.where(mask_values == False)[0]
        # nok = np.where(mask_values == False)[0]

        hp.projscatter(ra[ok], dec[ok], lonlat=True, color="green", s=1, marker=".")
        # hp.projscatter(ra[nok], dec[nok], lonlat=True, color="red", s=1, marker=".")

        plt.tight_layout()

        if outpath:
            plt.savefig(outpath)

        plt.show()


def hsp_map_logical_or(maps, verbose=False):
    """
    Hsp Map Logical Or.

    Logical AND of HealSparseMaps.

    """
    if verbose:
        print("Combine all maps...")

    # Ensure consistency in coverage and data type
    nside_coverage = maps[0].nside_coverage
    nside_sparse = maps[0].nside_sparse
    dtype = maps[0].dtype

    for m in maps:
        # MKDEBUG TODO: Change nside if possible
        if m.nside_coverage != nside_coverage:
            raise ValueError(
                f"Coverage nside={m.nside_coverage} does not match {nside_coverage}"
            )
        if m.dtype != dtype:
            raise ValueError(f"Data type {m.dtype} does not match {dtype}")

    # Create an empty HealSparse map
    map_comb = hsp.HealSparseMap.make_empty(nside_coverage, nside_sparse, dtype=dtype)
    for idx, m in enumerate(maps):
        map_comb |= m

        if verbose:
            valid_pixels = map_comb.valid_pixels
            n_tot = np.sum(valid_pixels)
            n_true = np.count_nonzero(valid_pixels)
            n_false = n_tot - n_true
            print(
                f"after map {idx}: frac_true={n_true / n_tot:g}, frac_false={n_false / n_tot:g}"
            )

    return map_comb
