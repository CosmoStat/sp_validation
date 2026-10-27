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
            for i, name in zip(indices, names):
                ax.stairs(
                    data[f"hist_{i}"],
                    data[f"edges_{i}"],
                    label=f"{name}: mean {float(data[f'mean_{i}']):.3g}, "
                    rf"$\sigma$ {float(data[f'std_{i}']):.3g}",
                )
            ax.set_yscale("log")
            ax.set_xlim((-0.15, 0.15) if indices != (4,) else (-0.3, 0.3))
            ax.set_xlabel("Residual" if len(indices) == 2 else names[0])
            ax.set_ylabel("Star detections")
            ax.legend(fontsize=7)
        return S.save(fig, out, "full")
