#!/usr/bin/env python
"""Read matrix entries from FITS header and print in tex or pdf format."""

from astropy.io import fits
import numpy as np
import subprocess
import tempfile
import os

from cs_util.args import parse_options


def format_scientific_latex(value, prec, exp_thresh=3):
    """Format number in scientific notation for LaTeX.

    Parameters
    ----------
    value : float
        input value
    prec : int
        number of significant digits
    exp_thresh : int, optional
        exponent threshold for switching to scientific notation;
        use scientific notation if |exponent| >= exp_thresh;
        default is 3

    Returns
    -------
    str
        LaTeX-formatted string, e.g. "1.23 \\times 10^{-4}"

    """
    if value == 0:
        return "0"

    # Get the exponent
    exp = int(np.floor(np.log10(np.abs(value))))

    # Use regular notation if exponent is small
    if abs(exp) < exp_thresh:
        return f"{value:.{prec}g}"

    # Get the mantissa
    mantissa = value / 10**exp

    # Format mantissa with specified precision
    mantissa_str = f"{mantissa:.{prec-1}f}"

    return f"{mantissa_str} \\times 10^{{{exp}}}"


def get_matrix_from_header(header, prefix):
    """Extract 2x2 matrix from FITS header using prefix.

    Looks for keys: prefix11, prefix12, prefix21, prefix22
    """
    matrix = np.zeros((2, 2))
    for i in range(1, 3):
        for j in range(1, 3):
            key = f"{prefix}{i}{j}"
            if key not in header:
                raise KeyError(f"Key '{key}' not found in FITS header")
            matrix[i-1, j-1] = header[key]
    return matrix


def matrix_to_tex(matrix, prec, exp_thresh):
    """Convert matrix to LaTeX format."""
    rows = []
    for i in range(2):
        row_vals = [format_scientific_latex(matrix[i, j], prec, exp_thresh) for j in range(2)]
        rows.append(" & ".join(row_vals))

    tex = r"\begin{pmatrix}" + "\n"
    tex += (" " + r" \\" + "\n").join(rows) + "\n"
    tex += r"\end{pmatrix}"
    return tex


def generate_pdf(tex_content, output_path):
    """Generate PDF from LaTeX content."""
    full_tex = r"""\documentclass[preview,border=2pt]{article}
\usepackage{amsmath}
\begin{document}
$""" + tex_content + r"""$
\end{document}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, "matrix.tex")
        with open(tex_file, "w") as f:
            f.write(full_tex)

        # Run pdflatex
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_file],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"pdflatex failed: {result.stderr}")

        pdf_file = os.path.join(tmpdir, "matrix.pdf")
        if output_path:
            import shutil
            shutil.copy(pdf_file, output_path)
            print(f"PDF written to {output_path}")
        else:
            print(f"PDF generated at {pdf_file}")


def main():
    # Default parameter values
    p_def = {
        "input": None,
        "prefix": "R_S",
        "format": "tex",
        "prec": 3,
        "exp_thresh": 3,
        "output": None,
        "ext": 0,
    }

    # Short options
    short_options = {
        "input": "-i",
        "prefix": "-r",
        "prec": "-n",
        "exp_thresh": "-t",
        "output": "-o",
        "ext": "-e",
    }

    # Types
    types = {
        "input": "string",
        "prefix": "string",
        "format": "string",
        "prec": "int",
        "exp_thresh": "int",
        "output": "string",
        "ext": "int",
    }

    # Help strings
    help_strings = {
        "input": "Input FITS file",
        "prefix": "Prefix for matrix entries (default: {})",
        "format": "Output format: tex or pdf (default: {})",
        "prec": "Number of significant digits (default: {})",
        "exp_thresh": "Exponent threshold for scientific notation (default: {})",
        "output": "Output file path (extension added based on format)",
        "ext": "FITS extension to read header from (default: {})",
    }

    options = parse_options(p_def, short_options, types, help_strings)

    if options["input"] is None:
        raise ValueError("Input FITS file (-i) is required")

    if options["format"] not in ["tex", "pdf"]:
        raise ValueError("Format must be 'tex' or 'pdf'")

    # Read FITS header
    with fits.open(options["input"]) as hdu:
        header = hdu[options["ext"]].header

    # Extract matrix
    matrix = get_matrix_from_header(header, options["prefix"])

    # Generate output
    tex_content = matrix_to_tex(matrix, options["prec"], options["exp_thresh"])

    # Determine output path with correct extension
    output_path = None
    if options["output"]:
        output_path = f"{options['output']}.{options['format']}"

    if options["format"] == "tex":
        full_tex = r"""\documentclass[preview,border=2pt]{article}
\usepackage{amsmath}
\begin{document}
$""" + tex_content + r"""$
\end{document}
"""
        print(full_tex)
        if output_path:
            with open(output_path, "w") as f:
                f.write(full_tex)
            print(f"Written to {output_path}")
    elif options["format"] == "pdf":
        generate_pdf(tex_content, output_path)


if __name__ == "__main__":
    main()
