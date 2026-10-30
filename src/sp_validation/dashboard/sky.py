"""Footprint and field maps in a projection that suits the survey.

The footprint is split into its disconnected parts (e.g. the main patch and
a long thin strip). Each part gets its own Lambert azimuthal equal-area
(zenithal equal-area) projection centred on it, rotated so its long axis is
horizontal unless it is the main part, which keeps north up; RA increases to
the left as on the sky. All parts are drawn at the same angular scale, the
main part on top and any others below it.

Map products are ``field_maps_<version>.npz`` (see
:mod:`sp_validation.dashboard.field_maps`) and the cosmo_val footprint
``footprint_<version>.npz``. Quantities are ratios of pixel sums, coarsened
by summing nested children before dividing.
"""

from dataclasses import dataclass

import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from . import style as S

DEG = np.pi / 180

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "parchment_teal", ["#f3ead6", "#9cc3b5", "#3F8278", "#1d4a44", "#0f2421"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "cobalt_cinnabar",
    ["#22386e", "#3D5BA0", "#a9b6d6", "#efe6d2", "#e4ab9f", "#BC4538", "#6e2119"],
)


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
@dataclass
class Zenithal:
    """Lambert azimuthal equal-area projection about ``(ra0, dec0)``.

    Output coordinates are in degrees on the projection plane, x increasing
    to the east-on-sky-left convention (x grows toward decreasing RA), then
    rotated by ``rotation`` degrees.
    """

    ra0: float
    dec0: float
    rotation: float = 0.0

    def forward(self, ra, dec):
        a = (np.asarray(ra) - self.ra0) * DEG
        d = np.asarray(dec) * DEG
        d0 = self.dec0 * DEG
        cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a)
        k = np.sqrt(2 / np.clip(1 + cosc, 1e-12, None))
        x = -k * np.cos(d) * np.sin(a)
        y = k * (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a))
        return self._rotate(x / DEG, y / DEG, self.rotation)

    def inverse(self, x, y):
        x, y = self._rotate(np.asarray(x), np.asarray(y), -self.rotation)
        x, y = -x * DEG, y * DEG
        rho = np.hypot(x, y)
        valid = rho < 2
        c = 2 * np.arcsin(np.clip(rho / 2, 0, 1))
        d0 = self.dec0 * DEG
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_d = np.cos(c) * np.sin(d0) + np.where(
                rho > 0, y * np.sin(c) * np.cos(d0) / rho, 0
            )
            dec = np.arcsin(np.clip(sin_d, -1, 1))
            ra = self.ra0 * DEG + np.arctan2(
                x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c)
            )
        return np.mod(ra / DEG, 360), dec / DEG, valid

    @staticmethod
    def _rotate(x, y, angle):
        t = angle * DEG
        return x * np.cos(t) - y * np.sin(t), x * np.sin(t) + y * np.cos(t)


# ---------------------------------------------------------------------------
# Footprint parts
# ---------------------------------------------------------------------------
def footprint_parts(pixels, nside, nest=True, coarse=64, min_fraction=0.01):
    """Split occupied pixels into connected parts, largest first.

    Connectivity is judged on a ``coarse`` grid so masking holes do not split
    a patch. Returns a list of boolean masks over ``pixels``; parts smaller
    than ``min_fraction`` of the total are attached to the nearest part.
    """
    ratio = (nside // coarse) ** 2
    parent = (pixels // ratio) if nest else hp.ring2nest(nside, pixels) // ratio
    occupied = np.unique(parent)
    occ_set = set(occupied.tolist())
    label = {}
    current = 0
    for p in occupied:
        if p in label:
            continue
        stack = [p]
        label[p] = current
        while stack:
            q = stack.pop()
            for nb in hp.get_all_neighbours(coarse, q, nest=True):
                if nb >= 0 and nb in occ_set and nb not in label:
                    label[nb] = current
                    stack.append(nb)
        current += 1
    labels = np.array([label[p] for p in parent])
    sizes = np.bincount(labels)
    order = np.argsort(sizes)[::-1]
    keep = [lab for lab in order if sizes[lab] >= min_fraction * sizes.sum()]
    if len(keep) < len(order):
        # Fold small islands into the nearest kept part.
        vec = np.array(hp.pix2vec(nside, pixels, nest=nest)).T
        centres = np.array([vec[labels == lab].mean(0) for lab in keep])
        small = ~np.isin(labels, keep)
        nearest = np.argmax(vec[small] @ centres.T, axis=1)
        labels[small] = np.array(keep)[nearest]
    return [labels == lab for lab in keep]


@dataclass
class Part:
    projection: Zenithal
    xlim: tuple
    ylim: tuple

    @property
    def width(self):
        return self.xlim[1] - self.xlim[0]

    @property
    def height(self):
        return self.ylim[1] - self.ylim[0]


def _centre(ra, dec):
    v = np.array(hp.ang2vec(ra, dec, lonlat=True))
    m = v.mean(0)
    ra0, dec0 = hp.vec2ang(m / np.linalg.norm(m), lonlat=True)
    return float(ra0[0]), float(dec0[0])


def make_parts(pixels, nside, nest=True, margin=1.5):
    """Projection and extent for each disconnected part of the footprint."""
    parts = []
    for i, mask in enumerate(footprint_parts(pixels, nside, nest)):
        ra, dec = hp.pix2ang(nside, pixels[mask], nest=nest, lonlat=True)
        ra0, dec0 = _centre(ra, dec)
        proj = Zenithal(ra0, dec0)
        x, y = proj.forward(ra, dec)
        if i > 0:
            # Long axis horizontal: rotate by the principal-axis angle.
            cov = np.cov(np.vstack([x, y]))
            evals, evecs = np.linalg.eigh(cov)
            major = evecs[:, np.argmax(evals)]
            angle = np.degrees(np.arctan2(major[1], major[0]))
            angle = (angle + 90) % 180 - 90  # keep RA increasing to the left
            proj = Zenithal(ra0, dec0, -angle)
            x, y = proj.forward(ra, dec)
        parts.append(
            Part(
                proj,
                (x.min() - margin, x.max() + margin),
                (y.min() - margin, y.max() + margin),
            )
        )
    return parts


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
def coarsen(pixels, sums, nside, nside_out):
    """Sum nested-pixel sums onto ``nside_out``; returns (pixels, sums)."""
    ratio = (nside // nside_out) ** 2
    parent = pixels // ratio
    out_pix, inverse = np.unique(parent, return_inverse=True)
    return out_pix, {
        key: np.bincount(inverse, weights=value, minlength=len(out_pix))
        for key, value in sums.items()
    }


def raster(part, pixels, values, nside, resolution, nest=True):
    """Sample a partial HEALPix map on the part's projection-plane grid."""
    nx = int(np.ceil(part.width / resolution))
    ny = int(np.ceil(part.height / resolution))
    x = np.linspace(part.xlim[0], part.xlim[1], nx)
    y = np.linspace(part.ylim[0], part.ylim[1], ny)
    xx, yy = np.meshgrid(x, y)
    ra, dec, valid = part.projection.inverse(xx, yy)
    pix = hp.ang2pix(nside, ra, dec, lonlat=True, nest=nest)
    order = np.argsort(pixels)
    pos = np.searchsorted(pixels, pix, sorter=order)
    pos = np.clip(pos, 0, len(pixels) - 1)
    found = valid & (pixels[order][pos] == pix)
    img = np.full(pix.shape, np.nan)
    img[found] = values[order][pos[found]]
    return img


def graticule(ax, part, step_ra=None, step_dec=10, color=S.RULE, label=True):
    """Draw RA/Dec lines across the part and label them along the frame."""
    dec_c = part.projection.dec0
    span = max(part.width, part.height)
    step_ra = step_ra or (30 if abs(dec_c) > 65 else 15 if span > 25 else 10)
    decs = np.arange(-90 + step_dec, 90, step_dec)
    ras = np.arange(0, 360, step_ra)
    t = np.linspace(0, 360, 721)
    s = np.linspace(-89.5, 89.5, 360)
    kw = dict(color=color, lw=0.5, zorder=1)
    x0, x1 = part.xlim
    y0, y1 = part.ylim
    for d in decs:
        x, y = part.projection.forward(t, np.full_like(t, d))
        far = _far_side(part, t, np.full_like(t, d))
        x, y = np.where(far, np.nan, x), np.where(far, np.nan, y)
        ax.plot(x, y, **kw)
        if label:
            _edge_label(ax, x, y, (x0, x1, y0, y1), rf"${d:g}^\circ$", side="left")
    for r in ras:
        x, y = part.projection.forward(np.full_like(s, r), s)
        far = _far_side(part, np.full_like(s, r), s)
        x, y = np.where(far, np.nan, x), np.where(far, np.nan, y)
        ax.plot(x, y, **kw)
        if label:
            _edge_label(ax, x, y, (x0, x1, y0, y1), rf"${r:g}^\circ$", side="bottom")


def _far_side(part, ra, dec):
    v = np.array(hp.ang2vec(ra, dec, lonlat=True))
    c = hp.ang2vec(part.projection.ra0, part.projection.dec0, lonlat=True)
    return v @ c < np.cos(80 * DEG)


def _edge_label(ax, x, y, box, text, side):
    x0, x1, y0, y1 = box
    ok = np.isfinite(x) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if not ok.any():
        return
    if side == "left":
        i = np.flatnonzero(ok)[np.argmin(x[ok])]
        if x[i] - x0 > 0.08 * (x1 - x0):
            return
        ax.text(
            x0,
            y[i],
            text,
            ha="right",
            va="center",
            fontsize=7,
            color=S.INK_MUTED,
            clip_on=False,
        )
    else:
        i = np.flatnonzero(ok)[np.argmin(y[ok])]
        if y[i] - y0 < 0.08 * (y1 - y0):
            ax.text(
                x[i],
                y0,
                text,
                ha="center",
                va="top",
                fontsize=7,
                color=S.INK_MUTED,
                clip_on=False,
            )


def _scale_bar(ax, part, length=10):
    x1 = part.xlim[1] - 0.04 * part.width
    y0 = part.ylim[0] + 0.05 * part.height
    ax.plot([x1 - length, x1], [y0, y0], color=S.INK, lw=1.2, solid_capstyle="butt")
    ax.text(
        x1 - length / 2,
        y0 + 0.012 * part.height + 0.3,
        rf"${length}^\circ$",
        ha="center",
        va="bottom",
        fontsize=7,
        color=S.INK,
    )


def draw_map(
    ax,
    part,
    pixels,
    values,
    nside,
    cmap,
    norm,
    resolution=None,
    labels=True,
    scale_bar=False,
    grid_color=S.RULE,
):
    resolution = resolution or max(hp.nside2resol(nside, arcmin=True) / 60 / 2, 0.03)
    img = raster(part, pixels, values, nside, resolution)
    graticule(ax, part, color=grid_color, label=labels)
    im = ax.imshow(
        img,
        origin="lower",
        extent=(*part.xlim, *part.ylim),
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=2,
        rasterized=True,
    )
    ax.set_xlim(part.xlim)
    ax.set_ylim(part.ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    if scale_bar:
        _scale_bar(ax, part)
    return im


def map_grid(fig, spec, parts):
    """Axes for every part inside ``spec``: main part above, others below."""
    if len(parts) == 1:
        return [fig.add_subplot(spec)]
    heights = [parts[0].height] + [
        p.height * parts[0].width / max(p.width, parts[0].width) for p in parts[1:]
    ]
    sub = spec.subgridspec(len(parts), 1, height_ratios=heights, hspace=0.05)
    return [fig.add_subplot(sub[i]) for i in range(len(parts))]


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------
def _ratio(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def coverage(footprint, nside_out):
    """Fraction of each ``nside_out`` pixel covered by the occupied footprint."""
    pix = footprint["pixels"]
    nside = int(footprint["nside_sparse"])
    ratio = (nside // nside_out) ** 2
    out, counts = np.unique(pix // ratio, return_counts=True)
    return out, counts / ratio


def median_quantity(maps, name, nside_out, footprint=None):
    """Median over pixels of one map quantity."""
    return float(np.median(quantity(maps, name, nside_out, footprint)[1]))


def quantity(maps, name, nside_out, footprint=None):
    """Pixels and values of one named map quantity at ``nside_out``."""
    nside = int(maps["nside"])
    keys = [k for k in maps if k.startswith(("n_", "sum_")) and np.ndim(maps[k]) == 1]
    pix, sums = coarsen(maps["pixels"], {k: maps[k] for k in keys}, nside, nside_out)
    area = hp.nside2pixarea(nside_out, degrees=True) * 3600
    if name == "n_eff":
        frac = np.ones(len(pix))
        if footprint is not None:
            cpix, cfrac = coverage(footprint, nside_out)
            lookup = dict(zip(cpix.tolist(), cfrac.tolist()))
            frac = np.array([lookup.get(p, np.nan) for p in pix.tolist()])
        values = _ratio(sums["sum_w"] ** 2, sums["sum_w2"]) / (area * frac)
        keep = (sums["n_gal"] > 0) & (frac > 0.5)
    elif name == "sigma_e":
        values = np.sqrt(_ratio(sums["sum_w2_ee"], 2 * sums["sum_w2"]))
        keep = sums["n_gal"] > 0
    elif name == "mean_w":
        values = _ratio(sums["sum_w"], sums["n_gal"])
        keep = sums["n_gal"] > 0
    elif name in ("e1", "e2"):
        values = _ratio(sums[f"sum_w_{name}"], sums["sum_w"])
        keep = sums["n_gal"] > 0
    elif name == "star_density":
        values = sums["n_star"] / area
        keep = sums["n_star"] >= 5
    elif name == "mag":
        values = _ratio(sums["sum_mag"], sums["n_gal"])
        keep = sums["n_gal"] > 0
    elif name == "fwhm_psf":
        values = _ratio(sums["sum_w_fwhm_psf"], sums["sum_w"])
        keep = sums["n_gal"] > 0
    elif name in ("de1", "de2", "dT_over_T"):
        key = "sum_dT_over_T_psf" if name == "dT_over_T" else f"sum_{name}"
        values = _ratio(sums[key], sums["n_star"])
        keep = sums["n_star"] >= 5
    else:
        raise KeyError(name)
    # Drop sparsely filled pixels (footprint edges), whose means are noise.
    n = sums[
        "n_star" if name in ("de1", "de2", "dT_over_T", "star_density") else "n_gal"
    ]
    keep &= n >= 0.3 * np.median(n[n > 0])
    return pix[keep], values[keep]


def _robust_limits(values, symmetric=False, q=(1, 99)):
    lo, hi = np.nanpercentile(values, q)
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return lo, hi


def _colorbar(fig, im, ax, label, symmetric):
    cbar = S.colorbar(
        fig,
        im,
        ax,
        label,
        orientation="horizontal",
        fraction=0.05,
        pad=0.02,
        aspect=28,
        extend="both",
    )
    cbar.locator = mpl.ticker.MaxNLocator(5, symmetric=symmetric)
    cbar.update_ticks()
    return cbar


def footprint_map(maps, footprint, out, nside=512, stats=None):
    """Effective galaxy density over the footprint."""
    parts = make_parts(maps["pixels"][maps["n_gal"] > 0], int(maps["nside"]))
    pix, val = quantity(maps, "n_eff", nside, footprint)
    with S.figure_style():
        fig = plt.figure(figsize=(7.2, 7.2 * _aspect(parts) + 0.6))
        gs = fig.add_gridspec(1, 1)
        axes = map_grid(fig, gs[0], parts)
        norm = mpl.colors.Normalize(*_robust_limits(val, q=(0.5, 99.5)))
        for i, (ax, part) in enumerate(zip(axes, parts)):
            im = draw_map(ax, part, pix, val, nside, SEQUENTIAL, norm, scale_bar=i == 0)
        _colorbar(fig, im, axes, r"$n_{\rm eff}$ [arcmin$^{-2}$]", False)
        if stats:
            axes[0].text(
                0.0,
                1.0,
                stats,
                transform=axes[0].transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                color=S.INK_MUTED,
                linespacing=1.5,
            )
        return S.save(fig, out, "full")


def _aspect(parts):
    main = parts[0]
    h = main.height + sum(
        p.height * main.width / max(p.width, main.width) for p in parts[1:]
    )
    return h / main.width


FIELD_SETS = {
    "galaxies": [
        ("sigma_e", 256, r"$\sigma_e$ per component", False),
        ("mean_w", 256, r"$\langle w \rangle$", False),
        ("e1", 64, r"$\langle e_1 \rangle$", True),
        ("e2", 64, r"$\langle e_2 \rangle$", True),
    ],
    "psf": [
        ("fwhm_psf", 256, r"PSF FWHM at galaxies [arcsec]", False),
        (
            "dT_over_T",
            128,
            r"stars: $\langle (T_\star - T_{\rm PSF})/T_{\rm PSF} \rangle$",
            True,
        ),
        ("de1", 128, r"stars: $\langle e_{1,\star} - e_{1,{\rm PSF}} \rangle$", True),
        ("de2", 128, r"stars: $\langle e_{2,\star} - e_{2,{\rm PSF}} \rangle$", True),
    ],
}


def paper_map_pair(maps, out, which):
    """Draw the catalogue paper's two related sky-map quantities."""
    fields = {
        "stars": [
            ("star_density", "Star detections [arcmin$^{-2}$]", False),
            ("dT_over_T", r"$\langle(T_\star-T_{\rm PSF})/T_{\rm PSF}\rangle$", True),
        ],
        "magnitude": [
            ("fwhm_psf", "PSF FWHM [arcsec]", False),
            ("mag", "Mean galaxy r magnitude", False),
        ],
    }
    gal_pix = maps["pixels"][maps["n_gal"] > 0]
    parts = make_parts(gal_pix, int(maps["nside"]))
    nside = 256
    with S.figure_style():
        fig = plt.figure(figsize=(7.2, 3.6 * _aspect(parts) + 0.6))
        gs = fig.add_gridspec(1, 2, wspace=0.05)
        for cell, (name, label, symmetric) in zip(gs, fields[which]):
            pix, val = quantity(maps, name, nside)
            inside = np.isin(
                pix, np.unique(gal_pix // (int(maps["nside"]) // nside) ** 2)
            )
            pix, val = pix[inside], val[inside]
            lo, hi = _robust_limits(val, symmetric)
            norm = (
                TwoSlopeNorm(0, lo, hi) if symmetric else mpl.colors.Normalize(lo, hi)
            )
            axes = map_grid(fig, cell, parts)
            for ax, part in zip(axes, parts):
                im = draw_map(
                    ax,
                    part,
                    pix,
                    val,
                    nside,
                    DIVERGING if symmetric else SEQUENTIAL,
                    norm,
                    labels=False,
                )
            _colorbar(fig, im, axes, label, symmetric)
        return S.save(fig, out, "full")


def field_maps(maps, out, which, footprint=None, layout="wide"):
    """2 x 2 small multiples of map quantities over the main footprint part.

    Diverging quantities use a symmetric range about zero; all ranges are
    robust (1st-99th percentile) so single pixels do not set the scale.
    """
    gal_pix = maps["pixels"][maps["n_gal"] > 0]
    parts = make_parts(gal_pix, int(maps["nside"]))
    with S.figure_style():
        if layout == "wide":
            fig = plt.figure(figsize=(7.2, 7.2 * _aspect(parts) + 1.0))
            gs = fig.add_gridspec(2, 2, hspace=0.08, wspace=0.04)
        else:
            fig = plt.figure(figsize=(3.6, 4 * (3.6 * _aspect(parts) + 0.6)))
            gs = fig.add_gridspec(4, 1, hspace=0.06)
        for cell, (name, nside, label, symmetric) in zip(gs, FIELD_SETS[which]):
            pix, val = quantity(maps, name, nside, footprint)
            # Restrict star-based maps to the galaxy footprint.
            inside = np.isin(
                pix, np.unique(gal_pix // (int(maps["nside"]) // nside) ** 2)
            )
            pix, val = pix[inside], val[inside]
            lo, hi = _robust_limits(val, symmetric)
            norm = (
                TwoSlopeNorm(0, lo, hi) if symmetric else mpl.colors.Normalize(lo, hi)
            )
            axes = map_grid(fig, cell, parts)
            for ax, part in zip(axes, parts):
                im = draw_map(
                    ax,
                    part,
                    pix,
                    val,
                    nside,
                    DIVERGING if symmetric else SEQUENTIAL,
                    norm,
                    labels=False,
                    grid_color="#ddd2bb",
                )
            _colorbar(fig, im, axes, label, symmetric)
        return S.save(fig, out, "full" if layout == "wide" else "half")
