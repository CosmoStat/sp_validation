"""SPATIAL MASKS.

This module implements the healsparse-backed ``Mask`` class and the
spatial-masking helpers (config-driven mask construction, mask statistics,
correlation/confusion matrices) used when building catalogues.

:Author: Martin Kilbinger
"""

import healsparse as hsp
import numpy as np
from astropy.io import fits
from scipy import stats

_KIND_ALIASES = {"smaller_equal": "less_equal"}

_KIND_OPS = {
    "equal": lambda a, v: a == v,
    "not_equal": lambda a, v: a != v,
    "greater": lambda a, v: a > v,
    "greater_equal": lambda a, v: a >= v,
    "less": lambda a, v: a < v,
    "less_equal": lambda a, v: a <= v,
    "range": lambda a, v: (a >= v[0]) & (a <= v[1]),
}


def apply_condition(array, kind, value):
    """Apply Condition.

    Evaluate one mask condition (``kind``/``value``, as specified in mask
    config YAML files) against an array and return a boolean mask. Single
    shared grammar for the object-selection ``Mask`` class and the
    cosmo_inference footprint builder (see issue #181).

    Parameters
    ----------
    array : numpy.ndarray
        input data
    kind : str
        operation type, one of "equal", "not_equal", "greater",
        "greater_equal", "less", "less_equal" (alias "smaller_equal"),
        "range"
    value : float or list
        value(s) to be used in mask operation; two-element list for "range"

    Returns
    -------
    numpy.ndarray
        boolean mask

    """
    kind = _KIND_ALIASES.get(kind, kind)
    if kind not in _KIND_OPS:
        raise ValueError(f"Unknown mask condition kind: {kind!r}")
    return _KIND_OPS[kind](array, value)


def correlation_matrix(masks, confidence_level=0.9):

    n_key = len(masks)
    print(n_key)

    cm = np.empty((n_key, n_key))
    r_val = np.zeros_like(cm)
    r_cl = np.empty((n_key, n_key, 2))

    for idx, mask_idx in enumerate(masks):
        for jdx, mask_jdx in enumerate(masks):
            res = stats.pearsonr(mask_idx._mask, mask_jdx._mask)
            r_val[idx][jdx] = res.statistic
            r_cl[idx][jdx] = res.confidence_interval(confidence_level=confidence_level)

    return r_val, r_cl


def confusion_matrix(prediction, observation):

    result = {}

    result["true_pos"] = sum(prediction & observation)
    result["true_neg"] = sum(np.logical_not(prediction) & np.logical_not(observation))
    result["false_neg"] = sum(prediction & np.logical_not(observation))
    result["false_pos"] = sum(np.logical_not(prediction) & observation)
    result["false_pos_rate"] = result["false_pos"] / (
        result["false_pos"] + result["true_neg"]
    )
    result["false_neg_rate"] = result["false_neg"] / (
        result["false_neg"] + result["true_pos"]
    )
    result["sensitivity"] = result["true_pos"] / (
        result["true_pos"] + result["false_neg"]
    )
    result["specificity"] = result["true_neg"] / (
        result["true_neg"] + result["false_pos"]
    )

    cm = np.zeros((2, 2))
    cm[0][0] = result["true_pos"]
    cm[1][1] = result["true_neg"]
    cm[0][1] = result["false_neg"]
    cm[1][0] = result["false_pos"]
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    result["cmn"] = cmn

    return result


class Mask:
    """Mask.

    Class to handle masking of catalogues.

    Parameters
    ----------
    col_name : str
        name of column to use for mask
    label : str
        mask label
    kind : str
        operation type; see :func:`apply_condition` for the allowed values,
        plus "not_equal_2bands", which keeps objects whose value differs
        from ``value`` in ``col_name`` *or* ``col_name2`` (two-band OR)
    value : float or list
        value(s) to be used in mask operation
    col_name2 : str, optional
        name of second column; required for (and only used by) kind
        "not_equal_2bands"
    dat : numpy.ndarray, optional
        input data, default is `None`; apply mask if given
    verbose : bool, optional
        verbose output if ``True``; default is ``False``

    """

    def __init__(
        self,
        col_name,
        label,
        kind=None,
        value=0,
        col_name2=None,
        dat=None,
        verbose=False,
    ):

        self._col_name = col_name
        self._label = label
        self._value = value
        self._kind = kind
        if kind == "not_equal_2bands" and col_name2 is None:
            raise ValueError("kind 'not_equal_2bands' requires col_name2")
        self._col_name2 = col_name2
        self._num_ok = None
        self._verbose = verbose

        if self._verbose:
            print("Initialising mask:", self)

        if dat is not None:
            self.apply(dat)

    def __repr__(self):

        return (
            f"Mask(col_name={self._col_name}, label={self._label}, kind={self._kind},"
            + f" value={self._value})"
        )

    @classmethod
    def from_list(cls, masks, label="combined", verbose=False):

        if verbose:
            print(f"Combining {len(masks)} masks")

        my_mask = cls(label, label, kind="combined", value=None)

        my_mask._mask = np.logical_and.reduce([m._mask for m in masks])

        return my_mask

    def apply(self, dat):

        if self._kind == "not_equal_2bands":
            self._mask = apply_condition(
                dat[self._col_name], "not_equal", self._value
            ) | apply_condition(dat[self._col_name2], "not_equal", self._value)
        else:
            self._mask = apply_condition(dat[self._col_name], self._kind, self._value)

    def to_bool(self, hsp_mask):

        if self._verbose:
            print("to_bool: get valid pixels")
        valid_pixels = hsp_mask.valid_pixels

        # Abuse of col_name
        self._col_name = valid_pixels

        if self._verbose:
            print("to_bool: apply mask")
        self.apply(hsp_mask)
        mask_bool = hsp.HealSparseMap.make_empty(
            hsp_mask.nside_coverage,
            hsp_mask.nside_sparse,
            dtype="bool",
        )
        mask_bool[valid_pixels] = self._mask
        return mask_bool

    @classmethod
    def print_strings(cls, coln, lab, num, fnum, f_out=None):
        msg = f"{coln:30s} {lab:30s} {num:10s} {fnum:10s}"
        print(msg)
        if f_out:
            print(msg, file=f_out)

    def print_stats(self, num_obj, f_out=None):
        if self._num_ok is None:
            self._num_ok = sum(self._mask)

        si = f"{self._num_ok:10d}"
        sf = f"{self._num_ok / num_obj:10.2%}"
        self.print_strings(self._col_name, self._label, si, sf, f_out=f_out)

    def get_sign(self, latex=False):

        sign = None
        if self._kind == "equal":
            sign = "$=$" if latex else "="
        elif self._kind == "not_equal":
            sign = r"$\ne$" if latex else "!="
        elif self._kind in ("greater_equal", "range"):
            sign = r"$\leq$" if latex else ">="
        elif self._kind == "smaller_equal":
            sign = r"$\geq$" if latex else "<="
        return sign

    def print_condition(self, f_out, latex=False):

        if self._value is None:
            return ""

        sign = self.get_sign(latex=latex)

        name = self._label if latex else self._col_name

        if sign is not None:
            print(f"{name} {sign} {self._value}", file=f_out)

        if self._kind == "not_equal_2bands":
            print(
                f"{name} != {self._value} or {self._col_name2} != {self._value}",
                file=f_out,
            )

        if self._kind == "range":
            print(f"{self._value[0]} {sign} {name} {sign} {self._value[1]}", file=f_out)

    def print_summary(self, f_out):
        print(f"[{self._label}]\t\t\t", file=f_out, end="")
        self.print_condition(f_out)

    def create_descr(self):
        """Create Descr.

        Create description of mask for later use in output file header.

        Returns
        -------
        str
            description

        """
        sign = self.get_sign()
        if sign is not None:
            descr = f"{sign}{self._value}"
        if self._kind == "range":
            descr = f"{self._value[0]}<={self._col_name}<={self._value[1]}"
        if self._kind == "not_equal_2bands":
            descr = f"!={self._value} in {self._col_name} or {self._col_name2}"
        self._descr = descr

        # Create description for FITS header

    def add_summary_to_FITS_header(self, header):

        header_new = fits.Header()

        self.create_descr()

        header_new[self._col_name] = (self._descr, self._label)

        header.update(header_new)


def print_mask_stats(num_obj, masks, mask_combined):
    """Print Mask Stats.

    Print mask statistics.

    Parameters
    ----------
    num_obj

    """
    Mask.print_strings("flag", "label", f"{'num_ok':>10}", f"{'num_ok[%]':>10}")
    for my_mask in masks:
        my_mask.print_stats(num_obj)

    mask_combined.print_stats(num_obj)


def get_masks_from_config(config, dat, dat_ext, masks_to_apply=None, verbose=False):
    """Get Masks From Config.

    Return mask information from yaml config structure.

    Parameters
    ----------
    config : dict
        config information
    dat : numpy.ndarray
        input data
    det_ext : numpy.ndarray
        input extended data
    masks_to_apply: list, optional
        masks to apply exclusively; if `None` (default), use all masks
    verbose : bool, optional
        verbose output if ``True``; default is ``False``

    Returns
    -------
    list
        list of masks
    dict
        list of indices for given mask column name (label)

    """
    # List to store all mask objects
    masks = []

    # Dict to associate labels with index in mask list
    labels = {}

    # Loop over mask sections from config file
    config_data = {key: config[key] for key in ["dat", "dat_ext"] if key in config}
    idx = 0
    for section, mask_list in config_data.items():
        # Set data source
        dat_source = dat if section == "dat" else dat_ext

        # Loop over mask information in this section
        for mask_params in mask_list:
            use_this_mask = False
            if masks_to_apply is not None:
                if mask_params["col_name"] in masks_to_apply:
                    use_this_mask = True
            else:
                use_this_mask = True

            if use_this_mask:
                # Ensure 'range' kind has exactly two values
                value = mask_params["value"]
                if mask_params["kind"] == "range" and (
                    not isinstance(value, list) or len(value) != 2
                ):
                    raise ValueError(
                        f"Range kind requires a list of two values, got {value}"
                    )

                # Create mask instance and append to list
                my_mask = Mask(**mask_params, dat=dat_source, verbose=verbose)
                masks.append(my_mask)
                labels[my_mask._col_name] = idx
                idx += 1
            else:
                if verbose:
                    print(f"Skipping mask {mask_params['col_name']}")
                continue

    return masks, labels
