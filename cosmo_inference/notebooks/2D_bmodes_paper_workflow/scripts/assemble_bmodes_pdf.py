# %%
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    snakemake  # type: ignore  # noqa: F821
except NameError as exc:  # pragma: no cover
    raise RuntimeError("assemble_bmodes_pdf must be executed via Snakemake") from exc


def load_inputs():
    try:
        ordered_paths = [Path(path) for path in snakemake.params["ordered_paths"]]
    except KeyError as exc:
        raise KeyError("Missing 'ordered_paths' parameter for assemble_bmodes_pdf") from exc

    if not ordered_paths:
        raise ValueError("Received empty ordered path list for PDF assembly")

    for path in ordered_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required plot is missing: {path}")

    output_path = Path(snakemake.output["pdf"])
    return ordered_paths, output_path


def add_image_page(pdf: PdfPages, image_path: Path, dpi: float = 100.0) -> None:
    """Embed a single image into the PDF while preserving aspect ratio."""
    image = mpimg.imread(image_path)
    height, width = image.shape[:2]
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, dpi=dpi)
    plt.close(fig)


def main():
    ordered_paths, output_path = load_inputs()
    with PdfPages(output_path) as pdf:
        for image_path in ordered_paths:
            add_image_page(pdf, image_path)


if __name__ == "__main__":
    main()
