"""PSF validation-star residual distributions."""

import matplotlib.pyplot as plt

from . import style as S


def plot(data, out):
    """Draw residual histograms with moments computed over finite detections."""
    with S.figure_style():
        fig = plt.figure(figsize=(7.2, 5.0))
        gs = fig.add_gridspec(2, 2)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :]),
        ]
        labels = [
            (r"$\Delta e_1$", r"$\Delta e_2$"),
            (r"$\Delta M^{(4)}_1$", r"$\Delta M^{(4)}_2$"),
            (r"$\sqrt{T_\star/T_{\rm PSF}}-1$",),
        ]
        for ax, indices, names in zip(axes, ((0, 1), (2, 3), (4,)), labels):
            for i, name, color in zip(indices, names, (S.COLOR_E, S.CINNABAR)):
                ax.stairs(
                    data[f"hist_{i}"],
                    data[f"edges_{i}"],
                    color=color,
                    label=f"{name}:  mean = {S.fmt_sci(float(data[f'mean_{i}']))},  "
                    rf"$\sigma$ = {S.fmt_sci(float(data[f'std_{i}']))}",
                )
            ax.set_yscale("log")
            peak = max(data[f"hist_{i}"].max() for i in indices)
            ax.set_ylim(0.5, peak * 3e3)
            ax.set_xlim((-0.15, 0.15) if indices != (4,) else (-0.3, 0.3))
            ax.set_xlabel("Residual" if len(indices) == 2 else names[0])
            ax.set_ylabel("Star detections")
            ax.legend(loc="upper center", fontsize=7)
        return S.save(fig, out, "full")
