"""Value-level leak detector for *test-generated*, trusted run directories.

This is a regression oracle, not an information-flow proof: it detects copied
values (even single-element subsets, permutations and rounded text), not arbitrary
encodings or transformations. Image pixels cannot be inverted into data; the
integration test separately records the inputs to the real renderers.
"""

import json
import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits

_NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def arrays(value, label="value"):
    """Walk arrays, structured columns and nested objects without flattening keys."""
    if type(value).__module__.startswith("sacc") and hasattr(value, "mean"):
        yield from arrays(value.mean, f"{label}.mean")
    elif type(value).__module__.startswith("treecorr"):
        # A raw GGCorrelation is the most relevant accidental producer return.
        # Inspect signal properties, not its C++ handle or recursive patch tree.
        for name in ("xip", "xim", "xip_im", "xim_im", "xi"):
            if hasattr(value, name):
                yield from arrays(getattr(value, name), f"{label}.{name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from arrays(item, f"{label}.{key}")
    elif isinstance(value, SimpleNamespace):
        yield from arrays(vars(value), label)
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from arrays(item, f"{label}[{i}]")
    elif isinstance(value, (np.ndarray, np.number, float, int)):
        data = np.asarray(value)
        if data.dtype.names:
            for name in data.dtype.names:
                yield from arrays(data[name], f"{label}.{name}")
        elif data.dtype.kind == "O":
            yield from arrays(data.tolist(), label)
        elif data.dtype.kind in "fciu":
            yield label, data.ravel()
    elif isinstance(value, str):
        yield label, np.array([float(x) for x in _NUMBER.findall(value)])


def file_arrays(path):
    """Decode every file in the synthetic run; unknown binary formats fail closed."""
    suffix = path.suffix.lower()
    if suffix in (".sacc", ".fits", ".fit", ".fz"):
        with fits.open(path) as hdus:
            for i, hdu in enumerate(hdus):
                yield from arrays(hdu.data, f"HDU[{i}]")
                yield from arrays(dict(hdu.header), f"header[{i}]")
    elif suffix in (".npy", ".npz"):
        data = np.load(path, allow_pickle=True)  # Only our own trusted tmp files.
        if isinstance(data, np.lib.npyio.NpzFile):
            with data:
                for key in data.files:
                    yield from arrays(data[key], key)
        else:
            yield from arrays(data)
    elif suffix in (".pkl", ".pickle"):
        yield from arrays(pickle.loads(path.read_bytes()))
    elif suffix == ".json":
        yield from arrays(json.loads(path.read_text()))
    elif suffix == ".png":
        from PIL import Image

        with Image.open(path) as image:
            yield from arrays(image.info, "image metadata")
    else:
        # Includes txt/csv/log/yaml and extensionless text. Do not silently skip
        # an unexpected product just because the scanner lacks its decoder.
        try:
            yield from arrays(path.read_text())
        except UnicodeDecodeError as error:
            raise AssertionError(f"No leak decoder for {path}") from error


def _json_precision(text):
    """Split a JSON document's numbers into rounded (<=10 digits) and exact."""
    rounded, exact = [], []

    def parse(token):
        digits = token.lower().split("e")[0].replace("-", "").replace(".", "")
        (rounded if len(digits.lstrip("0")) <= 10 else exact).append(float(token))
        return float(token)

    json.loads(text, parse_float=parse, parse_int=parse)
    return np.array(rounded), np.array(exact)


def assert_no_raw_values(root, references, *, expected=None, objects=(), logs=""):
    """Find even reordered/single-value subsets using scale-sensitive allclose.

    References contain finite, nonzero cosmological values. Optional ``expected``
    supplies their concealed twins: values indistinguishable at each decoder's
    precision are excluded for that decoder only, so rounded text cannot cause a
    false leak while weakly shifted binary values remain checked. B modes,
    covariance, coordinates and weights are deliberately not references.
    A fixed 1e-8 absolute tolerance would incorrectly
    identify essentially every C_ell as zero, so the absolute tolerance is tiny.
    """
    references = {key: np.asarray(value).ravel() for key, value in references.items()}

    def check(source, candidates, rtol=1e-10):
        for label, values in candidates:
            values = np.asarray(values, float)
            values = values[np.isfinite(values)]
            for name, reference in references.items():
                raw = reference
                if expected is not None:
                    raw = raw[~np.isclose(raw, expected[name], rtol=rtol, atol=1e-25)]
                if not len(raw):
                    continue
                # isclose(a, b) requires |a| within the extrema of
                # |b| ± (atol + rtol*|b|). Geometry maps often contain millions
                # of zeros/counts but no values on the shear signal's scale.
                absolute = np.abs(np.asarray(raw, dtype=float))
                lower = np.nextafter(
                    max(0.0, absolute.min() * (1 - rtol) - 1e-25), -np.inf
                )
                upper = np.nextafter(absolute.max() * (1 + rtol) + 1e-25, np.inf)
                possible = values[(np.abs(values) >= lower) & (np.abs(values) <= upper)]
                # Chunk large FITS columns so a future larger fixture stays bounded.
                for start in range(0, len(possible), 4096):
                    match = np.isclose(
                        possible[start : start + 4096, None],
                        raw[None, :],
                        rtol=rtol,
                        atol=1e-25,
                    )
                    if np.any(match):
                        raise AssertionError(
                            f"Unblinded {name} value in {source} ({label})"
                        )

    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        source = str(path.relative_to(root))
        suffix = path.suffix.lower()
        if suffix == ".json":
            # JSON keeps each float's written digits: full-precision floats are
            # binary values, short ones were rounded by their writer.
            rounded, exact = _json_precision(path.read_text())
            check(source, [("rounded JSON numbers", rounded)], rtol=5e-6)
            check(source, [("JSON numbers", exact)], rtol=1e-10)
            continue
        text = suffix in (".txt", ".dat", ".csv", ".log", ".yaml")
        check(source, file_arrays(path), rtol=5e-6 if text else 1e-10)
    for label, value in objects:
        check(label, arrays(value))
    check("captured stdout/stderr/logging", arrays(logs), rtol=5e-6)
    # The real COSEBIs producer logs chi2(E) with only two decimal places.
    # A relative allclose cannot detect that rounding. Compare its actual
    # format, excluding values whose concealed twin prints identically.
    if expected is not None:
        tokens = set(_NUMBER.findall(logs))
        for name, raw in references.items():
            if name.endswith("chi2_E"):
                for public, sealed in zip(raw, expected[name], strict=True):
                    token = f"{public:.2f}"
                    if token != f"{sealed:.2f}" and token in tokens:
                        raise AssertionError(
                            f"Unblinded {name} value in captured stdout/stderr/logging "
                            "(two-decimal chi2_E)"
                        )
