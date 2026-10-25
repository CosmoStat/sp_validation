"""Response-only diagnostics, not cosmological signal measurements.

Pair products use raw per-object R_g, not total selection/weight response.
They cannot certify adequacy of a global total response on their own (#23).
"""

import numpy as np


def pair_products(ra, dec, weight, rg, min_sep=1, max_sep=120, nbins=8):
    """Symmetrized angle-binned <R_g,ab(i) R_g,cd(j)>_ww via TreeCorr.

    Coordinates are degrees, separations arcminutes. Nonzero min_sep excludes
    self-pairs; cross-scalar correlations include both pair orientations.
    Returns all 16 tensor products, not just the scalar trace approximation.
    """
    import treecorr

    scalar = rg.reshape(4, -1)
    cats = [
        treecorr.Catalog(ra=ra, dec=dec, ra_units="deg", dec_units="deg", w=weight, k=x)
        for x in scalar
    ]
    products = np.empty((nbins, 4, 4))
    for i in range(4):
        for j in range(i, 4):
            corr = treecorr.KKCorrelation(
                min_sep=min_sep,
                max_sep=max_sep,
                nbins=nbins,
                sep_units="arcmin",
                bin_slop=0,
            )
            corr.process(cats[i], None if i == j else cats[j], num_threads=4)
            products[:, i, j] = products[:, j, i] = np.where(
                corr.weight > 0, corr.xi, np.nan
            )
    means = np.average(scalar, axis=1, weights=weight)
    return dict(
        theta=np.exp(corr.meanlogr),
        products=products,
        global_products=np.outer(means, means),
        pair_weight=corr.weight,
    )


def plot_response(gm, parent_groups, ra, dec, output):
    """Weighted/unweighted R_g, spatial split/joint samples, and patch map."""
    import matplotlib.pyplot as plt

    selected = gm.mask_dict["ns"]
    w = gm.branch_weights["ns"][selected]
    labels, jk, _ = gm.response_jackknife(parent_groups)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            x = gm.R_shear[i, j]
            limits = np.percentile(x, [0.5, 99.5])
            ax.hist(
                x,
                bins=60,
                range=limits,
                density=True,
                histtype="step",
                label="Unweighted",
            )
            ax.hist(
                x,
                bins=60,
                range=limits,
                weights=w,
                density=True,
                histtype="step",
                label="DES weighted",
            )
            ax.set(xlabel=f"Per-object Rg,{i + 1}{j + 1}", ylabel="Density")
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "response_distributions.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            for key in ("selection", "total", "joint"):
                ax.plot(np.arange(len(labels)), jk[key][:, i, j], ".", label=key)
            ax.set(xlabel="Deleted spatial group", ylabel=f"Response {i + 1}{j + 1}")
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "response_jackknife.png", dpi=160)
    plt.close(fig)
    groups = parent_groups[selected]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i in range(2):
        for j in range(2):
            values, x, y = [], [], []
            for label in labels:
                keep = groups == label
                if not keep.any():
                    continue
                x.append(np.mean(ra[keep]))
                y.append(np.mean(dec[keep]))
                values.append(np.average(gm.R_shear[i, j, keep], weights=w[keep]))
            im = axes[i, j].scatter(x, y, c=values, s=45)
            fig.colorbar(im, ax=axes[i, j], label=f"Mean Rg,{i + 1}{j + 1}")
            axes[i, j].set(xlabel="RA [deg]", ylabel="Dec [deg]")
    fig.tight_layout()
    fig.savefig(output / "response_map.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i in range(2):
        for j in range(2):
            delta = jk["total"][:, i, j] - jk["joint"][:, i, j]
            axes[i, j].plot(np.arange(len(labels)), delta, ".")
            axes[i, j].axhline(gm.R[i, j] - gm.R_joint[i, j], color="k", lw=0.8)
            axes[i, j].set(
                xlabel="Deleted spatial group", ylabel=f"Split − joint, {i + 1}{j + 1}"
            )
    fig.tight_layout()
    fig.savefig(output / "split_joint.png", dpi=160)
    plt.close(fig)
